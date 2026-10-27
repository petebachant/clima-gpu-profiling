"""Collect every null test in this repository's committed history.

A null test is a pass where `record-treatment` found the arms identical in
content, so whatever it measures is zero. Each one is a draw from the
wall-clock noise, and the collection is the only evidence this project has about
how wide that noise is -- the alternative is to believe a single pair.

The pairing is the whole difficulty. `results/treatment.json` and
`results/summary.json` are written by different stages, and before
`record-treatment` carried `always_run` it could be skipped while the pass it
was supposed to describe went ahead, leaving a commit whose treatment record
belongs to an earlier run. One such commit claims identical arms beside a
+12.6% speedup. So a commit counts only if BOTH files changed in it, which is
the signature of one pass writing both; a commit that rewrote only the treatment
record is excluded and counted as such, rather than quietly averaged in.

Its source is git history, which DVC cannot hash, so the stage always runs.
"""

import json
import subprocess
from pathlib import Path

TREATMENT = "results/treatment.json"
SUMMARY = "results/summary.json"
OUT = Path("results/null-census.json")
# The window every current figure is measured on: four simulated days at a
# 120 s coupled timestep. Earlier passes used shorter windows and coarser
# timesteps, so their noise is not this window's noise.
CURRENT_STEPS = 2880
CURRENT_DT_S = 120.0


def git(*args):
    out = subprocess.run(["git", *args], capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def blob(rev, path):
    text = git("show", f"{rev}:{path}")
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def main():
    revs = (git("log", "--format=%H %ct", "--", TREATMENT) or "").splitlines()

    nulls, skipped_unpaired, seen = [], 0, set()
    for line in revs:
        rev, ts = line.split()
        changed = (git("diff", "--name-only", f"{rev}^", rev, "--", TREATMENT, SUMMARY) or "")
        if TREATMENT not in changed or SUMMARY not in changed:
            # Either the pass did not rewrite both records, or this commit only
            # re-committed one of them. Cannot be paired; see the docstring.
            treatment = blob(rev, TREATMENT)
            if treatment is not None and treatment.get("arms_differ_in") == []:
                skipped_unpaired += 1
            continue
        treatment, summary = blob(rev, TREATMENT), blob(rev, SUMMARY)
        if treatment is None or summary is None:
            continue
        if treatment.get("arms_differ_in") != []:
            continue
        base, mod = summary.get("baseline", {}), summary.get("mod", {})
        key = (base.get("sypd"), mod.get("sypd"))
        if key in seen:
            continue
        seen.add(key)
        steps, dt = base.get("n_steps_total"), base.get("dt_cpl_s")
        nulls.append({
            "commit": rev[:9],
            "date": git("show", "-s", "--format=%cs", rev).strip(),
            "unix_time": int(ts),
            "speedup_pct": round(summary["speedup_pct"], 3),
            "baseline_sypd": base.get("sypd"),
            "mod_sypd": mod.get("sypd"),
            "n_steps_total": steps,
            "dt_cpl_s": dt,
            "current_window": steps == CURRENT_STEPS
            and dt is not None
            and abs(dt - CURRENT_DT_S) < 0.01 * CURRENT_DT_S,
        })

    nulls.sort(key=lambda n: n["unix_time"])
    current = [n for n in nulls if n["current_window"]]
    earlier = [n for n in nulls if not n["current_window"]]

    def extreme(rows):
        return round(max((abs(r["speedup_pct"]) for r in rows), default=0.0), 3)

    out = {
        "nulls": nulls,
        "n_nulls": len(nulls),
        "speedups_pct": [n["speedup_pct"] for n in nulls],
        "abs_max_pct": extreme(nulls),
        "n_current_window": len(current),
        "current_window_pct": [n["speedup_pct"] for n in current],
        "n_earlier_windows": len(earlier),
        "earlier_abs_max_pct": extreme(earlier),
        "skipped_unpaired": skipped_unpaired,
        "note": (
            "A null test is a pass whose arms record-treatment found identical in "
            "content, so its speedup is a measurement of nothing. Counted only where "
            "one pass wrote both results/treatment.json and results/summary.json, "
            "since before record-treatment carried always_run a commit could pair a "
            "stale treatment record with a real measurement; skipped_unpaired counts "
            "the commits rejected on that test, which includes the harmless case of a "
            "later commit re-committing an already counted pass. earlier_abs_max_pct "
            "covers the passes run on shorter windows and coarser timesteps than the "
            "current benchmark, whose noise is not interchangeable with this one's -- "
            "n_current_window says how many nulls exist at the window in use now."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"{len(nulls)} null test(s) in history, {skipped_unpaired} rejected as unpaired")
    for n in nulls:
        mark = "current window" if n["current_window"] else f"dt={n['dt_cpl_s']}"
        print(f"  {n['date']} {n['commit']} {n['speedup_pct']:+.3f}%  ({mark})")
    print(
        f"largest magnitude {out['abs_max_pct']:.3f}%; "
        f"{len(current)} at the current window, "
        f"largest elsewhere {out['earlier_abs_max_pct']:.3f}%"
    )


main()
