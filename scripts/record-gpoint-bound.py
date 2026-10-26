"""Record one point of the g-point parallelism bound before the next run erases it.

results/nsys/mod.sqlite holds whichever experiment profiled last, so a series of
runs that differ only in NGPT_GROUPS would leave only its final point. Each run
appends here instead, keyed by the group count read out of the mod arm's source
so it cannot be mislabeled by hand.

Not a pipeline stage: the fluxes these runs produce are wrong by construction
(the threads of a column race on the per-column scratch), so this prices a
restructure and must never be read as a result.
"""

import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

OUT = Path("results/experiments/gpoint-bound/bound.json")
SRC = Path("RRTMGP.jl-mod/ext/cuda/rte_longwave_2stream.jl")
Q = """SELECT s.value, k.registersPerThread, k.gridX*k.gridY*k.gridZ,
       COUNT(*), SUM(k.end-k.start)/1e6
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id
WHERE s.value LIKE '%rte_%2stream_solve_both%' GROUP BY s.value"""


def radiation(db):
    c = sqlite3.connect(db)
    out = {}
    for name, regs, blocks, n, ms in c.execute(Q):
        out["lw" if "_lw_" in name else "sw"] = {
            "registers": regs, "blocks": blocks, "launches": n, "total_ms": round(ms, 1)
        }
    out["device_ms"] = round(
        c.execute("SELECT SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0], 1
    )
    c.close()
    if "lw" not in out or "sw" not in out:
        sys.exit(f"{db}: did not find both fused radiation kernels")
    return out


def main():
    m = re.search(r"^const NGPT_GROUPS = (\d+)$", SRC.read_text(), re.M)
    groups = int(m.group(1)) if m else 1

    base = radiation("results/nsys/baseline.sqlite")
    mod = radiation("results/nsys/mod.sqlite")

    rec = {
        "groups": groups,
        "mod_commit": subprocess.check_output(
            ["git", "-C", "RRTMGP.jl-mod", "rev-parse", "HEAD"], text=True
        ).strip(),
        "baseline": base,
        "mod": mod,
        "lw_change_pct": round(100 * (mod["lw"]["total_ms"] - base["lw"]["total_ms"])
                               / base["lw"]["total_ms"], 2),
        "sw_control_change_pct": round(100 * (mod["sw"]["total_ms"] - base["sw"]["total_ms"])
                                       / base["sw"]["total_ms"], 2),
        "device_change_pct": round(100 * (mod["device_ms"] - base["device_ms"])
                                    / base["device_ms"], 2),
    }

    doc = json.loads(OUT.read_text()) if OUT.exists() else {"points": [], "note": (
        "Longwave g-point loop split across `groups` threads per column, shortwave "
        "untouched as the control. The fluxes are WRONG: no per-group scratch and no "
        "cross-group reduction, so a column's threads race. Prices the restructure "
        "described in docs/learnings.md section 4l; never a result."
    )}
    doc["points"] = [p for p in doc["points"] if p["groups"] != groups] + [rec]
    doc["points"].sort(key=lambda p: p["groups"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n")

    print(f"recorded groups={groups}")
    for p in doc["points"]:
        print(f"  groups {p['groups']:>3}: LW {p['baseline']['lw']['total_ms']:>7.1f} -> "
              f"{p['mod']['lw']['total_ms']:<7.1f} {p['lw_change_pct']:>+7.2f}%   "
              f"blocks {p['mod']['lw']['blocks']:>5}   "
              f"SW control {p['sw_control_change_pct']:>+6.2f}%   "
              f"device {p['device_change_pct']:>+6.2f}%")


if __name__ == "__main__":
    main()
