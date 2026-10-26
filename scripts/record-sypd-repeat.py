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

Not a pipeline stage: its input is a stage output that the next repeat erases,
and its own output has to survive that (AGENTS.md, "a transient artifact cannot
be a stage dependency").
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

OUT = Path("results/sypd-repeats.json")
LOGS = {arm: Path(f".calkit/scheduler/logs/amip-{arm}.out") for arm in ("baseline", "mod")}
DAYS_PER_YEAR = 365.25

PATTERNS = {
    "sypd": r"Info: SYPD: ([\d.eE+-]+)",
    "wall_s": r"Simulation took ([\d.eE+-]+) seconds",
    "step_s": r"Walltime per coupling step: ([\d.eE+-]+)",
}


def read_arm(path):
    text = path.read_text(errors="replace")
    out = {}
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
    rec = {
        "run_id": run_id,
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
    if any(p["run_id"] == run_id for p in doc["pairs"]):
        sys.exit(f"{run_id} is already recorded: these logs hold a pair that was "
                 "already appended, so there is nothing new to record")
    doc["pairs"].append(rec)

    groups = {}
    for p in doc["pairs"]:
        groups.setdefault(p["diff_digest"], []).append(p)
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
        }
        for k, v in groups.items()
    }
    OUT.write_text(json.dumps(doc, indent=2) + "\n")

    print(f"recorded pair {run_id}: baseline {base['sypd']:.4f} -> mod {mod['sypd']:.4f} "
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


if __name__ == "__main__":
    main()
