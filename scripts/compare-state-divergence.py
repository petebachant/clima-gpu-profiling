"""Does the fused radiation diverge from the baseline by more than a reseed does?

Fusing the all-sky and clear-sky solves changes how McICA randomness is
consumed, so the fused run cannot reproduce the baseline state step for step.
A bare fused-vs-baseline diff is therefore uninterpretable on its own: the
question is whether the fused run diverges by more than a different cloud draw
would.

The control is the fused code with the radiation seed shifted -- same physics,
different draw. One shift gives one sample, which is not enough: two chaotic
trajectories cross, so at any single step the control divergence can dip near
zero and drive the ratio arbitrarily high for no physical reason. Seed offsets
{0, 1, 2} give three control pairs, and the fusion number is asked to sit
inside that spread.
"""

import itertools
import json
import statistics
import tomllib

FUSION = ("unfused", "fused")
CONTROL_ARMS = ["fused", "fused_reseeded", "fused_reseeded2"]
RUNS = {
    "unfused": "results/state-unfused.toml",
    "fused": "results/state-fused.toml",
    "fused_reseeded": "results/state-fused-reseeded.toml",
    "fused_reseeded2": "results/state-fused-reseeded2.toml",
}
state = {name: tomllib.load(open(path, "rb")) for name, path in RUNS.items()}


def divergence(a, b):
    """Per-step worst relative difference in field sums, over shared fields.

    `sum` is normalized by `sumabs`: a field's sum can pass through zero, and
    its magnitude cannot.
    """
    out = {}
    snaps_a, snaps_b = a["snapshots"], b["snapshots"]
    for step in sorted(set(snaps_a) & set(snaps_b), key=lambda s: int(s.split("_")[1])):
        worst, worst_field = 0.0, None
        for field in set(snaps_a[step]) & set(snaps_b[step]):
            fa, fb = snaps_a[step][field], snaps_b[step][field]
            scale = abs(fa["sumabs"])
            if scale == 0:
                continue
            rel = abs(fa["sum"] - fb["sum"]) / scale
            if rel > worst:
                worst, worst_field = rel, field
        out[step] = {"worst_rel_diff": worst, "field": worst_field}
    return out


fusion = divergence(*(state[n] for n in FUSION))
controls = {
    f"{a}_vs_{b}": divergence(state[a], state[b])
    for a, b in itertools.combinations(CONTROL_ARMS, 2)
}

steps = sorted(
    set(fusion).intersection(*controls.values()), key=lambda s: int(s.split("_")[1])
)
per_step = {}
for step in steps:
    f = fusion[step]["worst_rel_diff"]
    c = [controls[k][step]["worst_rel_diff"] for k in controls]
    mid = statistics.median(c)
    per_step[step] = {
        "fusion_vs_unfused": f,
        "control_min": min(c),
        "control_median": mid,
        "control_max": max(c),
        "controls": {k: controls[k][step]["worst_rel_diff"] for k in controls},
        "field": fusion[step]["field"],
        # Against the middle of the control spread, not its top: one control
        # pair diverging unusually far would otherwise widen the envelope
        # enough to clear a genuinely broken change.
        "ratio_to_control_median": (f / mid) if mid > 0 else None,
        # Both zero at the early steps, before sampling has had an effect
        "inside_control_spread": min(c) <= f <= max(c) if mid > 0 else f == 0.0,
    }

ratios = [v["ratio_to_control_median"] for v in per_step.values()
          if v["ratio_to_control_median"] is not None]
final = steps[-1]
results = {
    "note": "fused-vs-unfused state divergence against the same code reseeded. "
            "Fusing changes how McICA randomness is consumed, so the fused run "
            "cannot match the baseline step for step; the test is whether it "
            "diverges by more than a different cloud draw does.",
    "steps": int(state["fused"]["steps"]),
    "control_pairs": sorted(controls),
    "per_step": per_step,
    "max_ratio_to_control_median": max(ratios) if ratios else None,
    "final_step": final,
    "final_fusion_rel_diff": per_step[final]["fusion_vs_unfused"],
    "final_control_min": per_step[final]["control_min"],
    "final_control_median": per_step[final]["control_median"],
    "final_control_max": per_step[final]["control_max"],
    "final_ratio_to_control_median": per_step[final]["ratio_to_control_median"],
}

# The three control pairs come from three seeds of the same code, so they must
# agree with each other to within the spread of a cloud draw. If one does not,
# the median cannot be trusted either: an arm that is wrong -- a stale file, a
# seed that did not take -- sits in two of the three pairs, so it moves the
# median as readily as the maximum, and the comparison would be against noise
# rather than against resampling. Checked at the last step, where the signal is
# largest.
spread = [per_step[final]["controls"][k] for k in sorted(controls)]
results["control_spread_ratio"] = (max(spread) / min(spread)) if min(spread) > 0 else None
results["controls_consistent"] = (
    results["control_spread_ratio"] is not None
    and results["control_spread_ratio"] <= 10.0
)
# The claim is that fusing is no worse than resampling. A factor of two on the
# median control absorbs the crossings that make any single-step ratio noisy; a
# real defect grows away from the controls rather than tracking them at a
# constant offset.
results["passes"] = (
    bool(ratios) and max(ratios) <= 2.0 and results["controls_consistent"]
)
json.dump(results, open("results/state-divergence.json", "w"), indent=2)

for step in steps:
    v = per_step[step]
    r = v["ratio_to_control_median"]
    ratio = "n/a" if r is None else f"{r:.2f}"
    print(f"{step:>10}  fusion {v['fusion_vs_unfused']:.3e}  "
          f"control {v['control_min']:.3e}/{v['control_median']:.3e}/"
          f"{v['control_max']:.3e}  ratio {ratio:>5}  ({v['field']})")
agree = results["control_spread_ratio"]
print(f"control pairs agree to {'n/a' if agree is None else f'{agree:.2f}x'} "
      f"-> {'consistent' if results['controls_consistent'] else 'INCONSISTENT, median untrustworthy'}")
print(f"worst ratio to the median control {results['max_ratio_to_control_median']} "
      f"-> {'PASS' if results['passes'] else 'FAIL'}")
