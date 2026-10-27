"""Append one unprofiled AMIP pair to the SYPD repeat record.

The two AMIP stages declare their scheduler logs as outputs, so a repeat run
deletes and rewrites them: the working tree only ever holds the latest pair, and
a single pair cannot be told apart from noise. The null test on 2026-10-22
measured that noise at a few points of wall clock, so a claimed speedup near
that size is only a result once it has repeated.

Each run appends here, keyed on a hash of the two logs so re-running the script
without a new measurement cannot inflate the count, and stamped with the
treatment digest and the whole submodule pointer map so pairs from different
code are never averaged together.

A stage, but only safely so because its output is declared
`delete_before_run: false`. DVC deletes a stage's outputs before running it,
which for an append-only record would mean losing every earlier pair on the
first re-run; persisting it is what lets the accumulation and the pipeline
coexist. It also always runs, since a new pair exists whenever the logs are
newer and nothing about the file contents says so.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

OUT = Path("results/sypd-repeats.json")
LOGS = {arm: Path(f".calkit/scheduler/logs/amip-{arm}.out") for arm in ("baseline", "mod")}
NULL = Path("results/null-record/summary.json")
DAYS_PER_YEAR = 365.25

PATTERNS = {
    "sypd": r"Info: SYPD: ([\d.eE+-]+)",
    "wall_s": r"Simulation took ([\d.eE+-]+) seconds",
    "step_s": r"Walltime per coupling step: ([\d.eE+-]+)",
}
GPU_LINE = r"gpu-guard: running on GPU\(s\) \[([\d,]*)\]"


def gpus(text):
    """Which devices the guard reported, or None if the log predates it."""
    found = re.findall(GPU_LINE, text)
    return [int(x) for x in found[-1].split(",") if x] if found else None


def pair_order():
    """Which arm ran second, from the logs' own mtimes.

    The scheduler writes each log as its job finishes, so the later mtime is the
    arm that ran second. Nothing else on disk says: jobs.json keeps the ids of
    the last scheduled pass and a forced resubmission does not update it.
    """
    stamps = {arm: path.stat().st_mtime for arm, path in LOGS.items()}
    return min(stamps, key=stamps.get), max(stamps, key=stamps.get)


def recover_from_history(run_id):
    """The GPU assignment of an earlier pair, read out of the commit that holds it.

    `run_id` is a hash of both logs, so the commit whose committed logs hash to
    it is the one that recorded that pair -- which makes anything else those logs
    said recoverable without a figure being typed in here.
    """
    paths = [str(p) for p in LOGS.values()]
    revs = subprocess.run(
        ["git", "log", "--format=%H", "--", paths[0]],
        capture_output=True, text=True,
    ).stdout.split()
    for rev in revs:
        blobs = []
        for path in paths:
            out = subprocess.run(
                ["git", "show", f"{rev}:{path}"], capture_output=True
            )
            if out.returncode != 0:
                break
            blobs.append(out.stdout)
        if len(blobs) != len(paths):
            continue
        digest = hashlib.sha256()
        for blob in blobs:
            digest.update(blob)
        if digest.hexdigest()[:12] == run_id:
            return {
                arm: gpus(blob.decode(errors="replace"))
                for arm, blob in zip(LOGS, blobs)
            }
    return None


def read_arm(path):
    text = path.read_text(errors="replace")
    out = {"gpus": gpus(text)}
    for key, pat in PATTERNS.items():
        found = re.findall(pat, text)
        if not found:
            sys.exit(f"{path}: no match for {key!r} -- did the run finish?")
        # A log holds one timed window; take the last in case of a requeue.
        out[key] = float(found[-1])
    out["n_steps_timed"] = round(out["wall_s"] / out["step_s"])
    # Recover the coupling timestep the run actually took, the check that caught
    # the --config_file override in 2026-09: a speedup that moves dt is not one.
    out["dt_cpl_s"] = round(out["sypd"] * DAYS_PER_YEAR * out["wall_s"] / out["n_steps_timed"], 3)
    return out


def main():
    treatment = json.loads(Path("results/treatment.json").read_text())
    arms = {arm: read_arm(path) for arm, path in LOGS.items()}

    digest = hashlib.sha256()
    for path in LOGS.values():
        digest.update(path.read_bytes())
    run_id = digest.hexdigest()[:12]

    pointers = {}
    for line in subprocess.check_output(["git", "ls-tree", "HEAD"], text=True).splitlines():
        meta, _, path = line.partition("\t")
        _, kind, sha = meta.split()
        if kind == "commit":
            pointers[path] = sha

    base, mod = arms["baseline"], arms["mod"]
    first, second = pair_order()
    rec = {
        "run_id": run_id,
        # Both are confounds a single pair cannot see past, so both are recorded
        # per pair: the arms run one after the other on a shared machine, and
        # SLURM hands out whichever device is free.
        "order": f"{first}_first",
        "order_source": "log mtimes",
        "ran_second": second,
        "gpus_by_arm": {arm: a["gpus"] for arm, a in arms.items()},
        "baseline": base,
        "mod": mod,
        # The project's convention, as results/summary.json reports it: the
        # fraction of baseline wall clock the treatment removes.
        "speedup_pct": round((1 - base["sypd"] / mod["sypd"]) * 100, 3),
        "step_change_pct": round(100 * (mod["step_s"] - base["step_s"]) / base["step_s"], 3),
        "treatment": treatment["arms_differ_in"],
        "diff_digest": treatment["diff_digest"],
        "reproducible": treatment["reproducible"],
        "dirty_trees": treatment["dirty_trees"],
        "stack": pointers,
    }

    doc = json.loads(OUT.read_text()) if OUT.exists() else {"pairs": [], "note": (
        "One entry per unprofiled AMIP pair, each an independent pair of runs of "
        "the same committed code. `speedup_pct` follows results/summary.json: "
        "1 - baseline_sypd/mod_sypd. `spread_points` across pairs of one treatment "
        "is this measurement's own repeatability, to be read against the null "
        "test's noise floor in results/null-record/."
    )}
    already = any(p["run_id"] == run_id for p in doc["pairs"])
    if already:
        # Not an error: the derived view below is recomputed and rewritten, so
        # the comparison can be refreshed without a new measurement. The append
        # is what stays guarded, since repeating it would inflate the count.
        print(f"pair {run_id} is already recorded; recomputing the derived view only")
    else:
        doc["pairs"].append(rec)

    groups = {}
    for p in doc["pairs"]:
        groups.setdefault(p["diff_digest"], []).append(p)
        if p.get("gpus_by_arm") is None:
            # An older pair recorded before the assignment was being kept. Its
            # logs are long overwritten, but the commit that holds them is
            # findable from its own run_id.
            p["gpus_by_arm"] = recover_from_history(p["run_id"])
        if p.get("order") is None:
            # Order is NOT recoverable the same way. It comes from the logs'
            # mtimes, which Git does not keep, and .calkit/stage-runs.json is no
            # substitute: a forced resubmission does not update it, so at a
            # forced pair's commit it still holds the previous pass's times and
            # would hand back an order that pair never ran in.
            p["order"] = rec["order"] if p["run_id"] == run_id else None
            p["order_source"] = "log mtimes" if p["order"] else "not recorded"
        # Derived here rather than stored, so it is right for the pairs whose
        # assignment was recovered after the fact as well as for new ones.
        by_arm = p.get("gpus_by_arm") or {}
        both = [by_arm.get("baseline"), by_arm.get("mod")]
        p["same_gpu"] = None if None in both else both[0] == both[1]
    doc["by_treatment"] = {
        k: {
            "n_pairs": len(v),
            "treatment": v[0]["treatment"],
            "speedups_pct": [p["speedup_pct"] for p in v],
            "mean_pct": round(sum(p["speedup_pct"] for p in v) / len(v), 3),
            "min_pct": min(p["speedup_pct"] for p in v),
            "max_pct": max(p["speedup_pct"] for p in v),
            "spread_points": round(max(p["speedup_pct"] for p in v)
                                   - min(p["speedup_pct"] for p in v), 3),
            "stacks": len({json.dumps(p["stack"], sort_keys=True) for p in v}),
            "orders": sorted({p.get("order") or "unknown" for p in v}),
            "gpu_assignments": [p.get("gpus_by_arm") for p in v],
            "same_gpu_pairs": sum(1 for p in v if p.get("same_gpu")),
        }
        for k, v in groups.items()
    }
    for k, v in groups.items():
        by_order = {}
        for p in v:
            by_order.setdefault(p.get("order") or "unknown", []).append(p["speedup_pct"])
        g = doc["by_treatment"][k]
        g["by_order"] = {
            o: {"n": len(xs), "mean_pct": round(sum(xs) / len(xs), 3)}
            for o, xs in by_order.items()
        }
        fwd, rev = g["by_order"].get("baseline_first"), g["by_order"].get("mod_first")
        if fwd and rev:
            # If running second is worth A, one order measures T + A and the
            # other T - A, so the half-difference is the confound and the mean
            # is the treatment with it removed.
            g["order_effect_points"] = round((fwd["mean_pct"] - rev["mean_pct"]) / 2, 3)
            g["order_corrected_pct"] = round((fwd["mean_pct"] + rev["mean_pct"]) / 2, 3)
    # The floor a repeat is read against: two IDENTICAL arms, measured once, came
    # out 2.5 points apart. So a margin over that is what makes a pair a result,
    # and the spread between repeats is what makes it a measurement.
    null = json.loads(NULL.read_text())
    doc["null_test"] = {
        "speedup_pct": round(null["speedup_pct"], 3),
        "pairs": 1,
        "note": "results/null-record/, both arms on main at "
                "exp/2026-10-22-null-test-on-main. One pair, so this is a single "
                "draw from the wall-clock noise and not an estimate of its width.",
    }
    for g in doc["by_treatment"].values():
        g["margin_over_null_points"] = round(g["mean_pct"] - doc["null_test"]["speedup_pct"], 3)
        g["margin_exceeds_repeat_spread"] = g["margin_over_null_points"] > g["spread_points"]
    # A citation cannot name a treatment by its digest without going stale the
    # next time the arms differ in something else, so the group holding the most
    # recent pair is also published under a fixed key.
    # Written before the reversed-order pair was run, so it is a prediction and
    # not a description: if the null test's figure is entirely the advantage of
    # running second, reversing the order moves the headline by twice it.
    for g in doc["by_treatment"].values():
        fwd = g["by_order"].get("baseline_first")
        if fwd:
            g["predicted_if_order_explains_null_pct"] = round(
                fwd["mean_pct"] - 2 * doc["null_test"]["speedup_pct"], 3
            )

    newest = doc["pairs"][-1]["diff_digest"]
    doc["latest_treatment"] = dict(doc["by_treatment"][newest], diff_digest=newest)
    # Run order, GPU assignment and the spread between repeats can only be read
    # off a group with several pairs, and a re-baseline starts a fresh group, so
    # the richest group is named separately from the current one. Reversed so a
    # tie goes to the later group, since by_treatment is in pair order.
    best = max(reversed(list(doc["by_treatment"])),
               key=lambda k: doc["by_treatment"][k]["n_pairs"])
    doc["most_repeated_treatment"] = dict(doc["by_treatment"][best], diff_digest=best)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")

    verb = "re-read" if already else "recorded"
    print(f"{verb} pair {run_id}: baseline {base['sypd']:.4f} -> mod {mod['sypd']:.4f} "
          f"SYPD = {rec['speedup_pct']:+.2f}%  (per-step {rec['step_change_pct']:+.2f}%)")
    if not rec["reproducible"]:
        print(f"  WARNING: measured against dirty submodules {rec['dirty_trees']}, "
              "so this pair is not reproducible from the commit")
    if abs(base["dt_cpl_s"] - mod["dt_cpl_s"]) > 0.01 * base["dt_cpl_s"]:
        print(f"  WARNING: the arms took different timesteps "
              f"({base['dt_cpl_s']} vs {mod['dt_cpl_s']} s): not one problem, not a speedup")
    for k, g in doc["by_treatment"].items():
        tag = "+".join(g["treatment"]) or "null test"
        print(f"  {tag} [{k[:8]}]: {g['n_pairs']} pair(s) "
              f"{[f'{v:+.2f}' for v in g['speedups_pct']]} "
              f"mean {g['mean_pct']:+.2f}%  spread {g['spread_points']:.2f} points"
              + ("  (MIXED STACKS)" if g["stacks"] > 1 else ""))
        orders = ", ".join(f"{o}: {d['mean_pct']:+.2f}% (n={d['n']})"
                           for o, d in sorted(g["by_order"].items()))
        print(f"    by order -- {orders}")
        if "order_effect_points" in g:
            print(f"    running second is worth {g['order_effect_points']:+.2f} points; "
                  f"order-corrected {g['order_corrected_pct']:+.2f}%")
        elif "predicted_if_order_explains_null_pct" in g:
            print(f"    if order explains the null, the reverse order gives "
                  f"{g['predicted_if_order_explains_null_pct']:+.2f}%")
        print(f"    {g['margin_over_null_points']:+.2f} points over the null test's "
              f"{doc['null_test']['speedup_pct']:+.2f}%, "
              f"{'wider' if g['margin_exceeds_repeat_spread'] else 'NOT wider'} "
              "than the spread between repeats")


if __name__ == "__main__":
    main()
