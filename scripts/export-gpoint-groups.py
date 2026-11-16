"""Price splitting the longwave g-point loop across the threads of one grid.

Section 4l of `docs/learnings.md` established that radiation is a single
resident wave -- 96 blocks of 256 threads on 108 SMs, 8 warps each -- so there
is no second wave to hide its latency behind, every tuning lever was tried and
rejected, and the only remaining change was to restructure the g-point loop.
A first attempt ran the groups as concurrent launches instead. It measures the
SAME saving as this one, to inside the control's noise, because independent
blocks do not care which grid they were launched in: the scheduler backfills an
SM across a stream boundary as readily as within a grid. Its low overlap factor
counted resident grids, not idle SMs. Section 8m of docs/learnings.md has it.
`results/experiments/gpoint-bound/bound.json` priced that restructure with a
deliberately incorrect kernel (the threads of a column raced). This measures the
correct one.

Three things have to be read separately, and only the first is the treatment:

  * The longwave solve's device time, which is what the split changes. The mod
    arm's kernel has a DIFFERENT NAME -- it is a new kernel, not an edit of the
    old one -- so the arms are paired by role, not by name. Pairing by name
    would report the baseline kernel as vanished and the mod one as new.
  * The overhead the split adds: a reduction over the per-group partials and the
    hoisted aerosol mask. These are longwave cost too, so they are charged to
    the treatment rather than left out of it.
  * The shortwave solve, which is untouched. It runs the same code in both arms
    at the same launch shape, so whatever it moves by is this pass's noise, and
    it is the control that says whether the longwave change is resolved.

The nsys databases are transient -- the next experiment overwrites them -- so
the derived numbers are snapshotted here with their provenance, as AGENTS.md
requires of anything a write-up will cite.
"""

import hashlib
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

BASE = "results/nsys/baseline.sqlite"
MOD = "results/nsys/mod.sqlite"
OUT = Path("results/gpoint-groups.json")

# Matched by role, and each role is a list because one arm's role can be
# several kernels. The baseline launches one fused solve per band; the mod arm
# launched one grouped solve per band and now launches an optics kernel and a
# sweeps kernel per band instead. A pattern pinned to one of those names reports
# the band as having vanished, which is the failure
# `analysis-stages-pinned-to-kernel-identity` describes and which costs a
# reservation to discover. The overhead kernels exist only in the mod arm.
LW_SOLVE = ("%rte_lw_2stream_solve%", "%lw_group_optics%", "%lw_group_sweeps%")
SW_SOLVE = ("%rte_sw_2stream_solve%", "%sw_group_optics%", "%sw_group_sweeps%")
LW_REDUCE = ("%reduce_lw_groups%",)
LW_AERO = ("%lw_aero_mask%",)
SW_REDUCE = ("%reduce_sw_groups%",)
SW_AERO = ("%sw_aero_mask%",)

# Everything either band launches, so the noise band cannot pick up a treated
# kernel. `shared` already drops a name present in one arm only, which is every
# name here, so this is belt and braces rather than the guard -- but a role that
# does appear in both arms would otherwise be compared against itself.
TREATED_FRAGMENTS = (
    "rte_lw", "rte_sw", "lw_group_", "sw_group_",
    "reduce_lw_groups", "reduce_sw_groups", "lw_aero_mask", "sw_aero_mask",
)

A100_SMS = 108
A100_REGS_PER_SM = 65536
A100_THREADS_PER_SM = 2048
A100_BLOCKS_PER_SM = 32


# ClimaCore's `max_active_blocks`, a port of NVIDIA's occupancy calculator.
# Dividing the register file by `regs * block` instead is wrong at several A100
# steps -- it puts the 128-thread step at 170 registers where the real one is
# 168 -- because registers are allocated per warp in 256-register units, a
# block's warps round up to the scheduler count, and the file is divided per
# scheduler rather than per multiprocessor. The CUDA driver's own occupancy
# call agrees with this model on every kernel in this solve; see
# docs/experiments/radiation-phase-split.md.
A100_REGS_PER_ALLOCATION = 256
A100_MAX_REGS_PER_BLOCK = 65536
A100_SCHEDULERS_PER_SM = 4


def round_up(n, divisor):
    return -(-n // divisor) * divisor


def max_active_blocks(regs, block):
    warps_per_block = -(-block // 32)
    max_warps = A100_THREADS_PER_SM // 32
    if regs > 0:
        allocated_warps = round_up(warps_per_block, A100_SCHEDULERS_PER_SM)
        per_warp = round_up(regs * 32, A100_REGS_PER_ALLOCATION)
        if allocated_warps * per_warp > A100_MAX_REGS_PER_BLOCK:
            return 0
        per_scheduler = (A100_REGS_PER_SM // A100_SCHEDULERS_PER_SM) // per_warp
        max_warps = min(max_warps, per_scheduler * A100_SCHEDULERS_PER_SM)
    blocks = max_warps // warps_per_block
    # Blocks of two warps are dealt unevenly to four schedulers unless each
    # gets the same number.
    if warps_per_block == 2:
        blocks = 4 * (blocks // 4)
    return min(blocks, A100_BLOCKS_PER_SM)


def warps_per_sm(regs, block):
    return max_active_blocks(regs, block) * ((block + 31) // 32)


def connect(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def union_ms(rows):
    """Elapsed time covered by a set of [start, end] intervals, merged.

    `SUM(end-start)` is elapsed GPU time only while kernels do not overlap,
    which held for every measurement in this project until a treatment started
    launching concurrent kernels. It then counts a moment twice for every kernel
    resident in it: for the longwave split, summing reported +78% where the
    merged intervals say -9%, the opposite sign. Every device share here is
    merged, and the summed figure is kept beside it only to show the gap.
    """
    rows = sorted(rows)
    total, cur_s, cur_e = 0, None, None
    for st, en in rows:
        if cur_s is None:
            cur_s, cur_e = st, en
        elif st <= cur_e:
            cur_e = max(cur_e, en)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = st, en
    if cur_s is not None:
        total += cur_e - cur_s
    return total / 1e6


def device_total(con):
    rows = con.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchall()
    summed = sum(e - s for s, e in rows) / 1e6
    return union_ms(rows), summed


def family(con, likes):
    """Every launch of one kernel role, per kernel and in aggregate.

    A role is several kernels when an arm splits it. The shape check stays
    per NAME rather than per role: one name standing for two launch shapes
    would put one launch's registers beside another's grid and invent a kernel
    that never ran, which is the error section 8g of docs/learnings.md was
    written for. Two different names with different shapes is the normal case
    now and is reported rather than rejected.
    """
    clause = " OR ".join(["s.value LIKE ?"] * len(likes))
    rows = con.execute(f"""
        SELECT s.value, k.registersPerThread, k.blockX*k.blockY*k.blockZ,
               k.gridX*k.gridY*k.gridZ, k.start, k.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE {clause}
        ORDER BY k.start
    """, tuple(likes)).fetchall()
    if not rows:
        return None

    by_name = {}
    for name, regs, block, grid, start, end in rows:
        by_name.setdefault(name, []).append((regs, block, grid, start, end))

    kernels = []
    for name, launches in sorted(by_name.items()):
        shapes = {(r[0], r[1], r[2]) for r in launches}
        if len(shapes) != 1:
            sys.exit(
                f"{name} ran at {len(shapes)} distinct shapes: {sorted(shapes)}"
            )
        regs, block, grid = launches[0][0], launches[0][1], launches[0][2]
        kernels.append({
            "name": name,
            "launches": len(launches),
            "elapsed_ms": round(
                union_ms([(s, e) for *_, s, e in launches]), 1
            ),
            "registers": regs,
            "block": block,
            "grid": grid,
            "threads": block * grid,
            "warps_per_sm": warps_per_sm(regs, block),
            # Fewer blocks than SMs means part of the device is idle for the
            # whole kernel, which is the condition the split exists to remove.
            "blocks_per_sm": round(grid / A100_SMS, 2),
        })

    summed = sum(e - s for *_, s, e in rows) / 1e6
    ms = union_ms([(s, e) for *_, s, e in rows])
    # The shape reported for the role is the kernel carrying its time, so a
    # register count or a grid read off this role describes where the time is.
    big = max(kernels, key=lambda k: k["elapsed_ms"])
    return {
        "name": " + ".join(k["name"] for k in kernels),
        "kernels": kernels,
        "launches": len(rows),
        "elapsed_ms": round(ms, 1),
        "summed_ms": round(summed, 1),
        # Above one means launches of this role overlapped each other, so the
        # summed figure is not elapsed time.
        "summed_over_elapsed": round(summed / ms, 2) if ms else None,
        "registers": big["registers"],
        "block": big["block"],
        "grid": big["grid"],
        "threads": big["threads"],
        # A role whose kernels cover different thread counts has no single
        # amount of work per launch, so the group count cannot be read off it.
        "threads_agree": len({k["threads"] for k in kernels}) == 1,
        "warps_per_sm": big["warps_per_sm"],
        "blocks_per_sm": big["blocks_per_sm"],
        "_rows": rows,
    }


def waves(fam):
    """How many grid-fulls of blocks the device holds, and how many there are.

    Registers decide how many blocks fit on an SM at once, and growing the grid
    does not change that number. What the grid changes is whether there are
    blocks left over: at one block per SM the whole grid is resident at once,
    every SM waits for its own column to finish, and the launch is as long as
    its slowest column. Past one wave the scheduler hands a finished SM
    another block, which is what absorbs the tail.

    Read it as a prediction to test, not a reason. The g-point grouping moved
    waves from below one to many with registers and blocks per SM unchanged,
    and the kernel time moved by about what the wave count predicted and no
    more. The optics split then changed blocks per SM as well, which is a
    different mechanism and a larger effect, so the two arms no longer share an
    occupancy and `blocks_resident_per_sm` has to be read for each. A wave
    count explains where work goes, never how long it takes.
    """
    regs, block, grid = fam["registers"], fam["block"], fam["grid"]
    per_sm = max_active_blocks(regs, block)
    resident = A100_SMS * per_sm
    return {
        "blocks": grid,
        "blocks_resident_per_sm": per_sm,
        "blocks_resident": resident,
        "waves": round(grid / resident, 2) if resident else None,
        "launches": fam["launches"],
    }


def controls(con, exclude):
    """The largest kernels outside radiation, as this pass's noise band."""
    rows = con.execute("""
        SELECT s.value, SUM(k.end-k.start)/1e6
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        GROUP BY s.value ORDER BY 2 DESC LIMIT 25
    """).fetchall()
    return {n.replace("_jl_mod_", "_jl_"): ms for n, ms in rows
            if not any(x in n for x in exclude)}


def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    bcon, mcon = connect(BASE), connect(MOD)
    btotal, bsummed = device_total(bcon)
    mtotal, msummed = device_total(mcon)

    b_lw, m_lw = family(bcon, LW_SOLVE), family(mcon, LW_SOLVE)
    b_sw, m_sw = family(bcon, SW_SOLVE), family(mcon, SW_SOLVE)
    for label, got in (("baseline longwave", b_lw), ("mod longwave", m_lw),
                       ("baseline shortwave", b_sw), ("mod shortwave", m_sw)):
        if got is None:
            sys.exit(f"no {label} launches found; the kernel family moved")

    reduce_k = family(mcon, LW_REDUCE)
    aero_k = family(mcon, LW_AERO)
    overhead_ms = sum(k["elapsed_ms"] for k in (reduce_k, aero_k) if k)
    # The split is only worth what it nets after its own reduction.
    lw_mod_charged = m_lw["elapsed_ms"] + overhead_ms
    lw_saved = b_lw["elapsed_ms"] - lw_mod_charged

    # Shortwave is treated now too, and is charged the same way: its own
    # reduction and hoisted aerosol mask. Until 2026-11-05 it was the untouched
    # in-arm control, and this file reported it as one.
    sw_reduce_k = family(mcon, SW_REDUCE)
    sw_aero_k = family(mcon, SW_AERO)
    sw_overhead_ms = sum(
        k["elapsed_ms"] for k in (sw_reduce_k, sw_aero_k) if k
    )
    sw_mod_charged = m_sw["elapsed_ms"] + sw_overhead_ms
    sw_saved = b_sw["elapsed_ms"] - sw_mod_charged
    sw_is_treated = sw_reduce_k is not None

    # Threads, not blocks: the mod arm launches a narrower block than the
    # baseline since the optics split, so a ratio of grids is not a ratio of
    # work and read 32 where the group count is 16.
    for label, fam in (("baseline longwave", b_lw), ("mod longwave", m_lw)):
        if not fam["threads_agree"]:
            sys.exit(
                f"{label} covers several thread counts "
                f"({sorted({k['threads'] for k in fam['kernels']})}), so the "
                "group count cannot be read off it"
            )
    ngroups = m_lw["threads"] // b_lw["threads"]
    w_base, w_mod = waves(b_lw), waves(m_lw)

    # The noise band is the untreated kernels alone. Shortwave used to be in
    # it; a band that is itself treated is not a control, and leaving it in
    # compared one treated band's saving against the other's.
    bc, mc = controls(bcon, TREATED_FRAGMENTS), controls(mcon, TREATED_FRAGMENTS)
    shared = sorted(set(bc) & set(mc))
    worst_control = max((abs(mc[k] - bc[k]) for k in shared), default=0.0)

    doc = {
        "longwave": {
            "baseline": {k: v for k, v in b_lw.items() if k != "_rows"},
            "mod": {k: v for k, v in m_lw.items() if k != "_rows"},
            "mod_overhead_ms": round(overhead_ms, 1),
            "mod_charged_ms": round(lw_mod_charged, 1),
            "saved_ms": round(lw_saved, 1),
            "change_pct": round(100 * -lw_saved / b_lw["elapsed_ms"], 2),
            "change_pct_if_summed": round(
                100 * (m_lw["summed_ms"] + overhead_ms - b_lw["summed_ms"])
                / b_lw["summed_ms"], 2),
            "pct_of_device_baseline": round(100 * b_lw["elapsed_ms"] / btotal, 2),
            "saved_pct_of_device": round(100 * lw_saved / btotal, 3),
        },
        "overhead": {
            "reduction": {k: v for k, v in (reduce_k or {}).items() if k != "_rows"},
            "aero_mask": {k: v for k, v in (aero_k or {}).items() if k != "_rows"},
        },
        "waves": {
            "groups": ngroups,
            "baseline": w_base,
            "mod": w_mod,
            # False while the only treatment was the grouping, and true since
            # the optics split: the arms then differ in occupancy as well as in
            # grid, so a wave count no longer isolates placement.
            "occupancy_differs": (
                w_base["blocks_resident_per_sm"]
                != w_mod["blocks_resident_per_sm"]
            ),
        },
        "shortwave": {
            "baseline": {k: v for k, v in b_sw.items() if k != "_rows"},
            "mod": {k: v for k, v in m_sw.items() if k != "_rows"},
            "mod_overhead_ms": round(sw_overhead_ms, 1),
            "mod_charged_ms": round(sw_mod_charged, 1),
            "saved_ms": round(sw_saved, 1),
            "change_pct": round(100 * -sw_saved / b_sw["elapsed_ms"], 2),
            "change_pct_if_summed": round(
                100 * (m_sw["summed_ms"] + sw_overhead_ms - b_sw["summed_ms"])
                / b_sw["summed_ms"], 2),
            "pct_of_device_baseline": round(100 * b_sw["elapsed_ms"] / btotal, 2),
            "saved_pct_of_device": round(100 * sw_saved / btotal, 3),
            # False would mean the mod arm has no shortwave reduction kernel,
            # so this band is untreated and IS a usable control again.
            "treated": sw_is_treated,
        },
        "overhead_sw": {
            "reduction": {
                k: v for k, v in (sw_reduce_k or {}).items() if k != "_rows"
            },
            "aero_mask": {
                k: v for k, v in (sw_aero_k or {}).items() if k != "_rows"
            },
        },
        "noise": {
            "worst_untreated_kernel_delta_ms": round(worst_control, 1),
            "untreated_kernels_compared": len(shared),
        },
        # Two separate claims. A saving is resolved if it is larger than the
        # untreated kernels moved; the arm total is a different question and the
        # mod arm also carries the sedimentation fold, so it is not this
        # treatment's number.
        #
        # Each band is compared against the untreated noise band ONLY. Until
        # 2026-11-05 this also took the max with the shortwave movement, which
        # was right while shortwave was untouched and became wrong the moment it
        # was treated: it then compared one band's saving against the other's,
        # and would have reported the longwave as unresolved had the shortwave
        # saved more.
        "saving_resolved_against_controls":
            abs(lw_saved) > worst_control,
        "sw_saving_resolved_against_controls":
            abs(sw_saved) > worst_control,
        "device_total": {
            "baseline_elapsed_ms": round(btotal, 1),
            "mod_elapsed_ms": round(mtotal, 1),
            "change_pct": round(100 * (mtotal - btotal) / btotal, 2),
            "baseline_summed_ms": round(bsummed, 1),
            "mod_summed_ms": round(msummed, 1),
            "change_pct_if_summed": round(100 * (msummed - bsummed) / bsummed, 2),
            # The baseline launches nothing concurrently, so its two figures
            # agree; the gap in the mod arm is the treatment's own overlap.
            "baseline_summed_equals_elapsed":
                round(bsummed, 1) == round(btotal, 1),
            "mod_concurrency_inflation_ms": round(msummed - mtotal, 1),
            "carries_other_treatments": True,
        },
        "provenance": {
            "baseline_ref": git("-C", "RRTMGP.jl", "rev-parse", "HEAD"),
            "mod_ref": git("-C", "RRTMGP.jl-mod", "rev-parse", "HEAD"),
            "source_commit": git("rev-parse", "HEAD"),
            "baseline_db_sha256_16": digest(BASE),
            "mod_db_sha256_16": digest(MOD),
        },
        "note": (
            "All device shares are ELAPSED time -- the union of the kernel "
            "intervals -- not `SUM(end-start)`, which counts a moment once per "
            "resident kernel and so is not elapsed time once a treatment "
            "launches concurrently. `change_pct_if_summed` is what the summed "
            "figure would have claimed, and for longwave it has the opposite "
            "sign. Both bands are treated as of 2026-11-05, and each is "
            "charged for its own reduction and hoisted aerosol mask; neither "
            "is a control any more, so each saving is resolved against the "
            "untreated kernels alone. The mod arm's solve kernels are "
            "different kernels from the baseline's, so the arms are paired by "
            "role. `waves` is a tested prediction, not an "
            "explanation: it rose from below one to many at unchanged blocks per "
            "SM and the longwave time did not move, and a concurrent-launch "
            "build of the same split measured the same saving at a low overlap "
            "factor. The device total carries the sedimentation "
            "fold as well and is not this treatment's figure."
        ),
    }
    bcon.close()
    mcon.close()
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    lw = doc["longwave"]
    print(f"wrote {OUT}")
    print(f"  longwave {lw['baseline']['elapsed_ms']} -> {lw['mod_charged_ms']} ms elapsed "
          f"({lw['change_pct']:+.2f}%, {lw['saved_pct_of_device']:+.3f} points of device)")
    print(f"  waves {w_base['waves']} -> {w_mod['waves']} over {ngroups} groups, "
          f"{w_base['blocks_resident_per_sm']} -> "
          f"{w_mod['blocks_resident_per_sm']} block(s)/SM")
    print(f"  shortwave {doc['shortwave']['baseline']['elapsed_ms']} -> "
          f"{doc['shortwave']['mod_charged_ms']} ms elapsed "
          f"({doc['shortwave']['change_pct']:+.2f}%)")
    print(f"  noise band {doc['noise']['worst_untreated_kernel_delta_ms']} ms; "
          f"resolved lw={doc['saving_resolved_against_controls']} "
          f"sw={doc['sw_saving_resolved_against_controls']}")


if __name__ == "__main__":
    main()
