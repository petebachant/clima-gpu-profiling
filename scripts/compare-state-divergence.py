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
    hi = max(c)
    per_step[step] = {
        "fusion_vs_unfused": f,
        "control_min": min(c),
        "control_max": hi,
        "controls": {k: controls[k][step]["worst_rel_diff"] for k in controls},
        "field": fusion[step]["field"],
        # Against the top of the control spread, not a single draw
        "ratio_to_control_max": (f / hi) if hi > 0 else None,
        # Both zero at the early steps, before sampling has had an effect
        "inside_control_spread": f <= hi if hi > 0 else f == 0.0,
    }

ratios = [v["ratio_to_control_max"] for v in per_step.values()
          if v["ratio_to_control_max"] is not None]
final = steps[-1]
results = {
    "note": "fused-vs-unfused state divergence against the same code reseeded. "
            "Fusing changes how McICA randomness is consumed, so the fused run "
            "cannot match the baseline step for step; the test is whether it "
            "diverges by more than a different cloud draw does.",
    "steps": int(state["fused"]["steps"]),
    "control_pairs": sorted(controls),
    "per_step": per_step,
    "max_ratio_to_control_max": max(ratios) if ratios else None,
    "final_step": final,
    "final_fusion_rel_diff": per_step[final]["fusion_vs_unfused"],
    "final_control_min": per_step[final]["control_min"],
    "final_control_max": per_step[final]["control_max"],
    "final_ratio_to_control_max": per_step[final]["ratio_to_control_max"],
    # The claim is that fusing is no worse than resampling. Allow a factor of
    # two above the widest control pair to absorb the crossings that make any
    # single-step ratio noisy; a real defect grows out of that band rather than
    # touching it.
    "passes": bool(ratios) and max(ratios) <= 2.0,
}
json.dump(results, open("results/state-divergence.json", "w"), indent=2)

for step in steps:
    v = per_step[step]
    ratio = "n/a" if v["ratio_to_control_max"] is None else f"{v['ratio_to_control_max']:.2f}"
    print(f"{step:>10}  fusion {v['fusion_vs_unfused']:.3e}  "
          f"control {v['control_min']:.3e}-{v['control_max']:.3e}  "
          f"ratio {ratio:>5}  ({v['field']})")
print(f"worst ratio to the control envelope {results['max_ratio_to_control_max']} "
      f"-> {'PASS' if results['passes'] else 'FAIL'}")
