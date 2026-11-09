"""How much register headroom the model's device time actually has.

Run by hand after a profiling pass, for the reason `snapshot-quadrature-profile.py`
gives: `results/nsys/*.sqlite` holds whichever experiment profiled last, so a
document cannot depend on it. The census is a few kilobytes and is committed; the
650 MB profile it came from may be garbage-collected.

The question it answers is the precondition for kernel fusion (ClimaAtmos #4261).
An A100 hands registers out per scheduler in 256-register granules and rounds each
warp's allocation up to a whole granule, so resident warps fall in steps and a
kernel's HEADROOM is how many registers it can gain before crossing the next
boundary. `ptxas` spends the granule it has already been charged for, so register
counts cluster ON the boundaries and headroom is usually zero -- which is why
fusing work into a kernel, and so adding live state, so often costs a fifth to a
quarter of its occupancy before it saves anything.

Usage:
    python scripts/snapshot-register-headroom.py <arm.sqlite> <out.json> [arm]
"""

import hashlib
import json
import math
import sqlite3
import subprocess
import sys

REGS_PER_SM = 65536
REGS_PER_ALLOCATION = 256
SCHEDULERS_PER_SM = 4
WARPS_PER_SM = 64


def granule_warps(regs):
    """Resident warps per SM the register count alone allows."""
    if regs <= 0:
        return WARPS_PER_SM
    per_warp = math.ceil(regs * 32 / REGS_PER_ALLOCATION) * REGS_PER_ALLOCATION
    per_sched = (REGS_PER_SM // SCHEDULERS_PER_SM) // per_warp
    return min(WARPS_PER_SM, per_sched * SCHEDULERS_PER_SM)


def headroom(regs):
    """Registers gainable before the next granule boundary is crossed."""
    granules = math.ceil(regs * 32 / REGS_PER_ALLOCATION)
    return (granules * REGS_PER_ALLOCATION) // 32 - regs


def gitref():
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main(db_path, out_path, arm="mod"):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    (total,) = cur.execute(
        "SELECT SUM(end - start) / 1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()

    rows = cur.execute(
        """SELECT s.value, k.registersPerThread, COUNT(*), SUM(k.end - k.start) / 1e6
           FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
           GROUP BY s.value, k.registersPerThread"""
    ).fetchall()

    by_room, variants = {}, 0
    for _name, regs, _n, ms in rows:
        room = headroom(regs)
        slot = by_room.setdefault(room, {"ms": 0.0, "variants": 0})
        slot["ms"] += ms
        slot["variants"] += 1
        variants += 1

    at_zero = by_room.get(0, {"ms": 0.0})["ms"]
    at_most_one = sum(v["ms"] for k, v in by_room.items() if k <= 1)

    doc = {
        "provenance": {
            "source_commit": gitref(),
            "arm": arm,
            "profile": db_path,
            "md5": hashlib.md5(open(db_path, "rb").read(1 << 20)).hexdigest(),
            "note": (
                "md5 is of the first MiB only, enough to tell two profiles "
                "apart without reading 650 MB. The profile is transient; this "
                "census is the record."
            ),
        },
        "device": {
            "regs_per_allocation": REGS_PER_ALLOCATION,
            "schedulers_per_sm": SCHEDULERS_PER_SM,
            "regs_per_scheduler": REGS_PER_SM // SCHEDULERS_PER_SM,
            "device_total_ms": round(total, 1),
            "kernel_variants": variants,
        },
        "by_headroom": {
            str(k): {
                "ms": round(v["ms"], 1),
                "pct_of_device": round(100.0 * v["ms"] / total, 2),
                "variants": v["variants"],
                "warps_per_sm_if_crossed": granule_warps(
                    # one register past the boundary
                    next(r for r in range(1, 257) if headroom(r) == k) + k + 1
                ),
            }
            for k, v in sorted(by_room.items())
        },
        "summary": {
            "zero_headroom_ms": round(at_zero, 1),
            "zero_headroom_pct": round(100.0 * at_zero / total, 1),
            "at_most_one_register_ms": round(at_most_one, 1),
            "at_most_one_register_pct": round(100.0 * at_most_one / total, 1),
            # The claim fusion has to clear.
            "most_of_device_cannot_absorb_one_register": at_most_one > total / 2,
            "useful_cap_targets": [
                r for r in range(1, 257)
                if headroom(r) == 0 and granule_warps(r) != granule_warps(r + 1)
            ],
        },
    }
    json.dump(doc, open(out_path, "w"), indent=2)
    s = doc["summary"]
    print(f"{arm}: {s['zero_headroom_pct']}% of device time has zero register "
          f"headroom, {s['at_most_one_register_pct']}% has at most one "
          f"({variants} variants, {total:.1f} ms)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
