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

# Matched by role. The longwave family covers both arms' solve kernels, whose
# names differ; the overhead kernels exist only in the mod arm.
LW_SOLVE = "%rte_lw_2stream_solve%"
SW_SOLVE = "%rte_sw_2stream_solve%"
LW_REDUCE = "%reduce_lw_groups%"
LW_AERO = "%lw_aero_mask%"
SW_REDUCE = "%reduce_sw_groups%"
SW_AERO = "%sw_aero_mask%"

A100_SMS = 108
A100_REGS_PER_SM = 65536
A100_THREADS_PER_SM = 2048
A100_BLOCKS_PER_SM = 32


def warps_per_sm(regs, block):
    blocks = min(A100_REGS_PER_SM // max(regs * block, 1),
                 A100_THREADS_PER_SM // max(block, 1), A100_BLOCKS_PER_SM)
    return blocks * ((block + 31) // 32)


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


def family(con, like):
    """Every launch of one kernel family, with shape and raw start/end times."""
    rows = con.execute("""
        SELECT s.value, k.registersPerThread, k.blockX*k.blockY*k.blockZ,
               k.gridX*k.gridY*k.gridZ, k.start, k.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.shortName = s.id
        WHERE s.value LIKE ?
        ORDER BY k.start
    """, (like,)).fetchall()
    if not rows:
        return None
    summed = sum(e - s for *_, s, e in rows) / 1e6
    ms = union_ms([(s, e) for *_, s, e in rows])
    # One name must not stand for two launch shapes: reporting one launch's
    # registers beside another's grid invents a kernel that never ran, which is
    # the error section 8g of docs/learnings.md was written for.
    shapes = {(r[1], r[2], r[3]) for r in rows}
    if len(shapes) != 1:
        sys.exit(f"{rows[0][0]} ran at {len(shapes)} distinct shapes: {sorted(shapes)}")
    regs, block, grid = rows[0][1], rows[0][2], rows[0][3]
    return {
        "name": rows[0][0],
        "launches": len(rows),
        "elapsed_ms": round(ms, 1),
        "summed_ms": round(summed, 1),
        # Above one means launches of this family overlapped each other, so the
        # summed figure is not elapsed time.
        "summed_over_elapsed": round(summed / ms, 2) if ms else None,
        "registers": regs,
        "block": block,
        "grid": grid,
        "warps_per_sm": warps_per_sm(regs, block),
        # Fewer blocks than SMs means part of the device is idle for the whole
        # kernel, which is the condition the split exists to remove.
        "blocks_per_sm": round(grid / A100_SMS, 2),
        "_rows": rows,
    }


def waves(fam):
    """How many grid-fulls of blocks the device holds, and how many there are.

    This is the quantity the split is for. Registers decide how many blocks fit
    on an SM at once, and that number does not change when the grid grows, so
    occupancy per SM is identical grouped or not. What changes is whether there
    are blocks left over: at one block per SM the whole grid is resident at
    once, every SM waits for its own column to finish, and the launch is as
    long as its slowest column. Past one wave the scheduler hands a finished SM
    another block, which is what absorbs the tail.

    Read it as a prediction to test, not a reason. This treatment moved waves
    from below one to many, with registers and blocks per SM unchanged, and the
    kernel time did not move -- so placement was never the limiter here. A wave
    count explains where work goes, never how long it takes.
    """
    regs, block, grid = fam["registers"], fam["block"], fam["grid"]
    per_sm = min(A100_REGS_PER_SM // max(regs * block, 1),
                 A100_THREADS_PER_SM // max(block, 1), A100_BLOCKS_PER_SM)
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

    # The split is one launch over a wider grid now, so the group count comes
    # from the grid ratio; launch counts are equal in both arms.
    ngroups = m_lw["grid"] // b_lw["grid"]
    w_base, w_mod = waves(b_lw), waves(m_lw)

    # The noise band is the untreated kernels alone. Shortwave used to be in
    # it; a band that is itself treated is not a control, and leaving it in
    # compared one treated band's saving against the other's.
    bc, mc = controls(bcon, ("rte_lw", "rte_sw")), controls(mcon, ("rte_lw", "rte_sw"))
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
    print(f"  waves {w_base['waves']} -> {w_mod['waves']} over {ngroups} groups "
          f"at {w_mod['blocks_resident_per_sm']} block(s)/SM either way")
    print(f"  shortwave {doc['shortwave']['baseline']['elapsed_ms']} -> "
          f"{doc['shortwave']['mod_charged_ms']} ms elapsed "
          f"({doc['shortwave']['change_pct']:+.2f}%)")
    print(f"  noise band {doc['noise']['worst_untreated_kernel_delta_ms']} ms; "
          f"resolved lw={doc['saving_resolved_against_controls']} "
          f"sw={doc['sw_saving_resolved_against_controls']}")


if __name__ == "__main__":
    main()
