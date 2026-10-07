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
import re
import sqlite3
import sys
from pathlib import Path

BASELINE = Path("results/nsys/baseline.sqlite")
MOD = Path("results/nsys/mod.sqlite")
OUT = Path("results/param-fold.json")
# The same treatment measured against an earlier upstream stack, kept as a
# committed snapshot because an upstream edit moved two of these kernels
# across an occupancy step in opposite directions. See its `provenance`.
PRIOR = Path("results/param-fold-prior.json")

# A kernel name carries its source path and line, so the two arms spell the
# same kernel differently and any upstream edit above it renumbers it. Select
# by name pattern and time rank instead: `rank` is the position among the
# matching kernel names ordered by total device time, descending.
TARGETS = {
    "microphysics_quadrature": ("set_microphysics_tendency_cache%", 0),
    "microphysics_updraft": ("set_microphysics_tendency_cache%", 1),
    "cloud_fraction": ("set_cloud_fraction__NVTX", 0),
    "sgs_moments": ("set_sgs_moments_and_cloud_fraction__NVTX", 0),
}

# Ranking is only safe while the ranks a pattern claims stand well clear of
# the first one it does not; below this ratio the script refuses rather than
# mislabel. The claimed ranks may sit arbitrarily close to each other --- the
# two microphysics kernels differ by about 4x --- so the check applies at the
# boundary, not between neighbors.
MIN_RANK_SEPARATION = 5.0
DEEPEST_RANK = {p: max(r for q, r in TARGETS.values() if q == p)
                for p, _ in TARGETS.values()}

Q = """
SELECT s.value, k.registersPerThread, k.blockX * k.blockY * k.blockZ,
       k.localMemoryPerThread, COUNT(*), SUM(k.end - k.start)
FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName
WHERE s.value LIKE ? GROUP BY s.value, k.registersPerThread, k.blockX
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


def by_total_time(db, pattern):
    """The matching kernel names, heaviest first, with their launch variants."""
    names = {}
    for name, regs, block, spill, launches, total_ns in db.execute(Q, (pattern,)):
        names.setdefault(name, []).append((regs, block, spill, launches, total_ns))
    ranked = sorted(names.items(), key=lambda kv: -sum(r[4] for r in kv[1]))
    return [(name, sum(r[4] for r in rows), rows) for name, rows in ranked]


def select(db, pattern, rank):
    """The rank-th heaviest matching name, reduced to its dominant variant.

    Within one name the heaviest variant is taken, not the mean: a name can
    cover several distinct kernels at different register counts, and averaging
    them hides the one a treatment touched.
    """
    ranked = by_total_time(db, pattern)
    if len(ranked) <= rank:
        return None
    last = DEEPEST_RANK[pattern]
    if rank == last and len(ranked) > last + 1:
        margin = ranked[last][1] / max(ranked[last + 1][1], 1)
        if margin < MIN_RANK_SEPARATION:
            sys.exit(
                f"{pattern!r} rank {last} ({ranked[last][0]}) is only "
                f"{margin:.1f}x the first unclaimed name down; "
                "ranking is not safe"
            )
    name, _, rows = ranked[rank]
    regs, block, spill, launches, total_ns = max(rows, key=lambda r: r[4])
    warps, pct_occ = occupancy(regs, block)
    return {
        "kernel": name,
        "source_line": source_line(name),
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


def source_line(name):
    """The source line a ClimaCore-generated kernel name ends with, if any."""
    m = re.search(r"_L(\d+)$", name)
    return int(m.group(1)) if m else None


def pct(old, new):
    return round(100 * (new - old) / old, 2)


def compare_to_prior(kernels):
    """How each kernel's payoff moved when the upstream stack moved under it.

    The occupancy step is the claim being tested, so the kernels that changed
    step status between the two stacks are the ones that carry evidence: a
    kernel that lost its step should give most of its payoff back, and one that
    gained a step should pick one up.
    """
    prior = json.loads(PRIOR.read_text())
    out = {"prior_commit": prior["provenance"]["source_commit"],
           "kernels": {}, "switched": {}}
    for key, now in kernels.items():
        was = prior["kernels"].get(key)
        if was is None:
            continue
        entry = {
            "prior_change_pct": was["mean_change_pct"],
            "now_change_pct": now["mean_change_pct"],
            "prior_crossed_step": was["crossed_occupancy_step"],
            "now_crossed_step": now["crossed_occupancy_step"],
            "prior_registers_saved": was["registers_saved"],
            "now_registers_saved": now["registers_saved"],
        }
        entry["step_changed"] = (
            was["crossed_occupancy_step"] != now["crossed_occupancy_step"])
        # Payoff is negative, so a payoff that shrank is a positive delta.
        entry["payoff_delta_points"] = round(
            now["mean_change_pct"] - was["mean_change_pct"], 2)
        out["kernels"][key] = entry
        if entry["step_changed"]:
            out["switched"][key] = {
                "direction": "lost" if was["crossed_occupancy_step"] else "gained",
                "payoff_points": abs(entry["payoff_delta_points"]),
            }
    out["n_switched"] = len(out["switched"])
    lost = [v["payoff_points"] for v in out["switched"].values()
            if v["direction"] == "lost"]
    gained = [v["payoff_points"] for v in out["switched"].values()
              if v["direction"] == "gained"]
    out["points_lost_losing_a_step"] = round(sum(lost) / len(lost), 1) if lost else None
    out["points_gained_gaining_a_step"] = (
        round(sum(gained) / len(gained), 1) if gained else None)
    # With no step, what is left tracks how many registers came off.
    nostep = [(v["now_registers_saved"], abs(v["now_change_pct"]))
              for v in out["kernels"].values() if not v["now_crossed_step"]]
    nostep += [(v["prior_registers_saved"], abs(v["prior_change_pct"]))
               for v in out["kernels"].values() if not v["prior_crossed_step"]]
    out["without_a_step"] = [
        {"registers_saved": r, "payoff_pct": c} for r, c in sorted(nostep)]
    out["note"] = (
        "Not a controlled A/B on the step alone: the upstream change that "
        "moved the step also changed the kernel bodies (an argument was added "
        "to the microphysics path). The evidence is that both switchers moved "
        "the way the step predicts, in opposite directions, by a similar "
        "number of points."
    )
    return out


def main():
    for p in (BASELINE, MOD):
        if not p.exists():
            sys.exit(f"missing {p}")
    bdb, mdb = sqlite3.connect(BASELINE), sqlite3.connect(MOD)

    out = {"kernels": {}}
    for key, (pattern, rank) in TARGETS.items():
        b, m = select(bdb, pattern, rank), select(mdb, pattern, rank)
        if b is None or m is None:
            sys.exit(f"{key}: nothing at rank {rank} of {pattern!r}")
        out["kernels"][key] = {
            "baseline": b,
            "mod": m,
            "mean_change_pct": pct(b["mean_us"], m["mean_us"]),
            "total_ms_saved": round(b["total_ms"] - m["total_ms"], 3),
            "registers_saved": b["registers"] - m["registers"],
            "crossed_occupancy_step": m["warps_per_sm"] > b["warps_per_sm"],
        }

    # The two microphysics kernels are told apart by time rank, so check the
    # ranking against the source order, which is the thing it stands in for:
    # the quadrature closure is below the updraft one in both arms.
    for arm in ("baseline", "mod"):
        quad = out["kernels"]["microphysics_quadrature"][arm]["source_line"]
        upd = out["kernels"]["microphysics_updraft"][arm]["source_line"]
        if quad is None or upd is None or quad <= upd:
            sys.exit(
                f"{arm}: microphysics ranking disagrees with source order "
                f"(quadrature L{quad}, updraft L{upd})"
            )

    if PRIOR.exists():
        out["vs_prior_stack"] = compare_to_prior(out["kernels"])

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
    # device_total covers every kernel in the arm, so it also carries any OTHER
    # treatment the mod arm happens to be running. The fold's own contribution
    # is the sum over the kernels it was applied to, against the same
    # whole-device denominator.
    fb = sum(v["baseline"]["total_ms"] for v in out["kernels"].values())
    fm = sum(v["mod"]["total_ms"] for v in out["kernels"].values())
    out["fold_attributable"] = {
        "kernels": len(out["kernels"]),
        "baseline_ms": round(fb, 1),
        "mod_ms": round(fm, 1),
        "ms_saved": round(fb - fm, 1),
        "change_pct": pct(fb, fm),
        "pct_of_device_total": pct(bt / 1e6, bt / 1e6 - (fb - fm)),
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
        "one more block per SM, which is the gap a fold has to close to pay. "
        "`device_total` is every kernel in the arm and so includes any other "
        "treatment the mod arm carries; `fold_attributable` is the subtotal "
        "over the kernels listed here, which is the fold's own share."
    )
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    for key, v in out["kernels"].items():
        step = "step" if v["crossed_occupancy_step"] else "    "
        print(f"{key:24s} {v['baseline']['registers']:3d} -> {v['mod']['registers']:3d} regs  "
              f"{v['baseline']['warps_per_sm']:2d} -> {v['mod']['warps_per_sm']:2d} warps/SM  "
              f"{v['baseline']['mean_us']:9.1f} -> {v['mod']['mean_us']:9.1f} us  "
              f"{v['mean_change_pct']:+6.2f}%  ({v['total_ms_saved']:+.1f} ms)  "
              f"{step}  next at {v['mod']['registers_for_next_step']} regs")
    f = out["fold_attributable"]
    print(f"these {f['kernels']} kernels: {f['baseline_ms']} -> {f['mod_ms']} ms "
          f"({f['change_pct']:+.2f}%), {f['pct_of_device_total']:+.2f}% of the device")
    d = out["device_total"]
    print(f"whole arm:   {d['baseline_ms']} -> {d['mod_ms']} ms "
          f"({d['change_pct']:+.2f}%)  [all treatments]")
    v = out.get("vs_prior_stack")
    for key, sw in (v or {}).get("switched", {}).items():
        e = v["kernels"][key]
        print(f"{key:24s} {sw['direction']} its step since "
              f"{v['prior_commit'][:7]}: {e['prior_change_pct']:+.2f}% -> "
              f"{e['now_change_pct']:+.2f}% ({sw['payoff_points']:.1f} points)")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
