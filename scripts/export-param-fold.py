"""What folding the microphysics parameters into `Val` did, kernel by kernel.

Reads the two nsys profiles directly rather than `results/top-kernels.csv`,
for two reasons: registers per thread are recorded per launch, so the profile
says what the compiler produced in the run that was timed, and one kernel name
covers two distinct kernels that must not be averaged together
(`set_sgs_moments_and_cloud_fraction` appears at two register counts, one of
which this treatment does not touch).

The sqlite databases are 1.9 GB and regenerable, so the durable record is this
JSON.
"""

import json
import sqlite3
import sys
from pathlib import Path

BASELINE = Path("results/nsys/baseline.sqlite")
MOD = Path("results/nsys/mod.sqlite")
OUT = Path("results/param-fold.json")

# The kernel names carry their source path and line, so the two arms spell the
# same kernel differently and the line numbers moved with the edit.
TARGETS = {
    "microphysics_quadrature": (
        "%microphysics_cache_jl_L1013", "%microphysics_cache_jl_L1016"),
    "microphysics_updraft": (
        "%microphysics_cache_jl_L970", "%microphysics_cache_jl_L973"),
    "cloud_fraction": ("set_cloud_fraction__NVTX", "set_cloud_fraction__NVTX"),
    "sgs_moments": (
        "set_sgs_moments_and_cloud_fraction__NVTX",
        "set_sgs_moments_and_cloud_fraction__NVTX"),
}

Q = """
SELECT k.registersPerThread, k.blockX * k.blockY * k.blockZ,
       k.localMemoryPerThread, COUNT(*), SUM(k.end - k.start)
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
WHERE s.value LIKE ? GROUP BY k.registersPerThread, k.blockX
"""
Q_TOTAL = "SELECT SUM(end - start), COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL"

# A100 sm_80, per SM.
REGS_PER_SM = 65536
MAX_WARPS_PER_SM = 64
MAX_BLOCKS_PER_SM = 32


def occupancy(registers, block_size):
    """Warps resident per SM, as ncu's occupancy limits compute it."""
    if registers == 0 or block_size == 0:
        return None, None
    blocks = min(REGS_PER_SM // (block_size * registers), MAX_BLOCKS_PER_SM)
    warps = min(blocks * (block_size // 32), MAX_WARPS_PER_SM)
    return warps, round(100 * warps / MAX_WARPS_PER_SM, 2)


def next_step_registers(registers, block_size):
    """The register count the kernel must reach for one more block per SM."""
    blocks = REGS_PER_SM // (block_size * registers)
    target = REGS_PER_SM // (block_size * (blocks + 1))
    return target if target >= 1 else None


def heaviest(db, pattern):
    """The variant that dominates, when one name covers several kernels."""
    rows = db.execute(Q, (pattern,)).fetchall()
    if not rows:
        return None
    regs, block, spill, launches, total_ns = max(rows, key=lambda r: r[4])
    warps, pct_occ = occupancy(regs, block)
    return {
        "registers": regs,
        "block_size": block,
        "spill_bytes_per_thread": spill,
        "warps_per_sm": warps,
        "occupancy_pct": pct_occ,
        "registers_for_next_step": next_step_registers(regs, block),
        "launches": launches,
        "total_ms": round(total_ns / 1e6, 3),
        "mean_us": round(total_ns / launches / 1e3, 1),
        "name_covers_variants": len(rows),
    }


def pct(old, new):
    return round(100 * (new - old) / old, 2)


def main():
    for p in (BASELINE, MOD):
        if not p.exists():
            sys.exit(f"missing {p}")
    bdb, mdb = sqlite3.connect(BASELINE), sqlite3.connect(MOD)

    out = {"kernels": {}}
    for key, (bpat, mpat) in TARGETS.items():
        b, m = heaviest(bdb, bpat), heaviest(mdb, mpat)
        if b is None or m is None:
            sys.exit(f"{key}: no launches matched ({bpat!r}, {mpat!r})")
        out["kernels"][key] = {
            "baseline": b,
            "mod": m,
            "mean_change_pct": pct(b["mean_us"], m["mean_us"]),
            "total_ms_saved": round(b["total_ms"] - m["total_ms"], 3),
            "registers_saved": b["registers"] - m["registers"],
            "crossed_occupancy_step": m["warps_per_sm"] > b["warps_per_sm"],
        }

    bt, bn = bdb.execute(Q_TOTAL).fetchone()
    mt, mn = mdb.execute(Q_TOTAL).fetchone()
    out["device_total"] = {
        "baseline_ms": round(bt / 1e6, 1),
        "mod_ms": round(mt / 1e6, 1),
        "baseline_launches": bn,
        "mod_launches": mn,
        "change_pct": pct(bt, mt),
        "ms_saved": round((bt - mt) / 1e6, 1),
    }
    crossed = [k for k, v in out["kernels"].items() if v["crossed_occupancy_step"]]
    out["kernels_crossing_a_step"] = len(crossed)
    out["crossed"] = sorted(crossed)
    out["note"] = (
        "Registers per thread and block size as the profiled run chose them; "
        "ClimaCore sizes the block from the register count, so both move "
        "together and occupancy is derived from the pair. A kernel's mean is "
        "over its own launches, and `name_covers_variants` > 1 means the "
        "reported figures are the heaviest variant of a shared name. "
        "`registers_for_next_step` is what the kernel would have to reach for "
        "one more block per SM, which is the gap a fold has to close to pay."
    )
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    for key, v in out["kernels"].items():
        print(f"{key:24s} {v['baseline']['registers']:3d} -> {v['mod']['registers']:3d} regs  "
              f"{v['baseline']['warps_per_sm']:2d} -> {v['mod']['warps_per_sm']:2d} warps/SM  "
              f"{v['baseline']['mean_us']:9.1f} -> {v['mod']['mean_us']:9.1f} us  "
              f"{v['mean_change_pct']:+6.2f}%  ({v['total_ms_saved']:+.1f} ms)  "
              f"next step at {v['mod']['registers_for_next_step']} regs")
    d = out["device_total"]
    print(f"all kernels: {d['baseline_ms']} -> {d['mod_ms']} ms ({d['change_pct']:+.2f}%)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
