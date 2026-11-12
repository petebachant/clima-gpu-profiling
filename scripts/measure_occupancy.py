"""The A100 launch model ClimaCore uses, and what a register costs under it.

ClimaCore sizes every slice-loop grid from the kernel's own register count: one
occupancy wave (`max_waves = 1`), so the resident thread count is whatever the
register file allows and the loop strides over the rest of its slices. The model
below is transcribed from `ClimaCore.jl/ext/cuda/cuda_utils.jl`
(`max_active_blocks` and `uncached_launch_configuration`) for an A100, and is
validated by reproducing observed launch geometry rather than by querying the
device.
"""

import hashlib
import json
import os
import sqlite3
import subprocess

# uncached_device_attributes() for Compute Capability 8.0. The values below the
# API queries come from CUDA's own occupancy headers, as the Julia does.
A100 = dict(
    threads_per_warp=32,
    max_block_dim_x=1024,
    sm_count=108,
    max_threads_per_block=1024,
    max_threads_per_sm=2048,
    max_regs_per_block=65536,
    max_regs_per_sm=65536,
    max_shmem_per_sm=167936,
    reserved_shmem_per_block=1024,
    max_shmem_per_block_optin=166912,
    regs_per_allocation=256,
    shmem_per_allocation=128,
    max_regs_per_thread=256,
    schedulers_per_sm=4,
    max_blocks_per_sm=32,
    max_schedulers_per_sm=4,
)

WARPS_PER_SM = A100["max_threads_per_sm"] // A100["threads_per_warp"]


def _round_up(n, divisor):
    return -(-n // divisor) * divisor


def max_active_blocks(threads_per_block, regs_per_thread, shmem_per_block, a=A100):
    """cudaOccMaxActiveBlocksPerMultiprocessor times the SM count, and its limiter."""
    max_blocks_per_sm, limit = a["max_blocks_per_sm"], "num_blocks"
    if threads_per_block:
        if threads_per_block > a["max_threads_per_block"]:
            return 0, "num_threads"
        warps_per_block = -(-threads_per_block // a["threads_per_warp"])
        max_warps_per_sm, warp_limit = WARPS_PER_SM, "num_threads"
        if regs_per_thread:
            if regs_per_thread > a["max_regs_per_thread"]:
                return 0, "registers"
            allocated_warps = _round_up(warps_per_block, a["max_schedulers_per_sm"])
            allocated_regs = _round_up(
                regs_per_thread * a["threads_per_warp"], a["regs_per_allocation"]
            )
            if allocated_warps * allocated_regs > a["max_regs_per_block"]:
                return 0, "registers"
            per_scheduler = a["max_regs_per_sm"] // a["schedulers_per_sm"]
            by_regs = (per_scheduler // allocated_regs) * a["schedulers_per_sm"]
            if by_regs < max_warps_per_sm:
                max_warps_per_sm, warp_limit = by_regs, "registers"
        by_warps = max_warps_per_sm // warps_per_block
        # Blocks of two warps are dealt unevenly to the four schedulers of an SM.
        if warps_per_block == 2 and a["schedulers_per_sm"] == 4:
            by_warps = 4 * (by_warps // 4)
        if by_warps < max_blocks_per_sm:
            max_blocks_per_sm, limit = by_warps, warp_limit
    if shmem_per_block:
        allocated = _round_up(
            shmem_per_block + a["reserved_shmem_per_block"], a["shmem_per_allocation"]
        )
        if allocated > a["max_shmem_per_block_optin"] + a["reserved_shmem_per_block"]:
            return 0, "shared_memory"
        by_shmem = a["max_shmem_per_sm"] // allocated
        if by_shmem < max_blocks_per_sm:
            max_blocks_per_sm, limit = by_shmem, "shared_memory"
    return max_blocks_per_sm * a["sm_count"], limit


def launch_config(
    regs, shmem=lambda threads: 0, max_slices=None, granularity=None,
    strict=False, max_waves=1, a=A100,
):
    """uncached_launch_configuration: (threads_per_block, blocks, limiter)."""
    max_threads_per_block = a["max_block_dim_x"]
    unit = min(max(granularity or 0, a["threads_per_warp"]), max_threads_per_block)
    if granularity is None:
        sizes = [
            min(u * unit, max_threads_per_block)
            for u in range(1, -(-max_threads_per_block // unit) + 1)
        ]
    else:
        sizes = [u * unit for u in range(1, max_threads_per_block // unit + 1)]
    best = (0, 0, "user")
    for threads in sizes:
        per_wave, active_limit = max_active_blocks(threads, regs, shmem(threads), a)
        if max_slices is None:
            blocks, limit = per_wave * max_waves, active_limit
        else:
            user = (max_slices // threads) if strict else -(-max_slices // threads)
            blocks, limit = (
                (user, "user")
                if user <= per_wave * max_waves
                else (per_wave * max_waves, active_limit)
            )
        if threads * blocks >= best[0] * best[1] and (
            blocks >= a["sm_count"] or blocks >= best[1]
        ):
            best = (threads, blocks, limit)
    return best


def warps_per_sm(regs, a=A100):
    """Resident warps per SM of a point loop with more slices than threads."""
    threads, blocks, _ = launch_config(regs, a=a)
    return blocks // a["sm_count"] * (threads // a["threads_per_warp"])


def step_table(a=A100):
    """Contiguous register bands, each with the warps per SM it buys."""
    bands = []
    for regs in range(1, a["max_regs_per_thread"] + 1):
        threads, blocks, limit = launch_config(regs, a=a)
        warps = blocks // a["sm_count"] * (threads // a["threads_per_warp"])
        if bands and bands[-1]["warps_per_sm"] == warps:
            bands[-1]["max_registers"] = regs
        else:
            bands.append(
                dict(
                    min_registers=regs,
                    max_registers=regs,
                    warps_per_sm=warps,
                    occupancy_pct=round(100 * warps / WARPS_PER_SM, 2),
                    block_size=threads,
                    grid_size=blocks,
                    threads=threads * blocks,
                    limiter=limit,
                )
            )
    return bands


def observed_blocks_per_sm(grid, a=A100):
    """Blocks per SM of a launch, or None when the grid is not whole waves."""
    return None if grid % a["sm_count"] else grid // a["sm_count"]


def predicted_warps_per_sm(regs, block, shmem, a=A100):
    """Resident warps per SM, and the limiter, of one occupancy wave."""
    blocks, limit = max_active_blocks(block, regs, shmem, a)
    return blocks // a["sm_count"] * (block // a["threads_per_warp"]), limit


def waves(grid, regs, block, shmem, a=A100):
    """Grid size over the blocks that one occupancy wave holds.

    A ClimaCore slice loop launches exactly one wave (`max_waves = 1`), so this
    is 1.0 for every kernel ClimaCore launches, and larger for the kernels that
    GPUArrays and RRTMGP launch with a thread per element.
    """
    per_wave, _ = max_active_blocks(block, regs, shmem, a)
    return None if not per_wave else grid / per_wave


INVENTORY_QUERY = """
SELECT s.value, COUNT(*), SUM(k.end - k.start) / 1e6,
       k.registersPerThread, k.staticSharedMemory, k.dynamicSharedMemory,
       k.gridX * k.gridY * k.gridZ, k.blockX * k.blockY * k.blockZ,
       k.localMemoryPerThread
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON s.id = k.shortName
GROUP BY s.value, k.registersPerThread, k.staticSharedMemory,
         k.dynamicSharedMemory, k.gridX, k.blockX
ORDER BY 3 DESC
"""


def inventory(sqlite_path, top=None):
    """Launch geometry per kernel variant, grouped so no field is averaged.

    Grouping by (name, registers, shared memory, grid, block) is what keeps the
    shape fields honest: a MAX over a shared short name invents a launch that
    cannot exist (docs/learnings.md section 8i).
    """
    con = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        rows = con.execute(INVENTORY_QUERY).fetchall()
    finally:
        con.close()
    out = []
    for name, n, ms, regs, ssm, dsm, grid, block, spill in rows:
        shmem = (ssm or 0) + (dsm or 0)
        predicted, limit = predicted_warps_per_sm(regs, block, shmem)
        n_waves = waves(grid, regs, block, shmem)
        fills_one_wave = n_waves == 1.0
        out.append(
            dict(
                kernel=name,
                launches=n,
                total_ms=round(ms, 1),
                registers=regs,
                shared_memory_bytes=shmem,
                # nsys leaves `localMemoryPerThread` at 0 for EVERY kernel in
                # these profiles -- one distinct value across the whole table --
                # so it is not a measurement of spill and a 0 here never meant
                # "does not spill". The quadrature kernel reads 0 from nsys and
                # 328 B from the compile-time `CUDA.memory(k).local`, and the
                # radiation solvers read 0 from nsys and 216-288 B compiled.
                # Spill comes from a compile (the register probes) or from ncu,
                # never from here; `nsys_reports_spill` records whether a future
                # nsys populates the field, so the claim is rechecked each pass.
                spill_bytes_per_thread=None,
                nsys_local_memory_per_thread=spill or 0,
                grid_size=grid,
                block_size=block,
                threads=grid * block,
                blocks_per_sm=observed_blocks_per_sm(grid),
                waves=round(n_waves, 3) if n_waves is not None else None,
                fills_one_wave=fills_one_wave,
                warps_per_sm=predicted if fills_one_wave else None,
                predicted_warps_per_sm=predicted,
                occupancy_pct=(
                    round(100 * predicted / WARPS_PER_SM, 2) if fills_one_wave else None
                ),
                limiter=limit,
            )
        )
    return out[:top] if top else out


def _sha256_16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def _git(*args):
    return subprocess.run(
        ("git",) + args, capture_output=True, text=True, check=False
    ).stdout.strip()


def snapshot(arms, out_path, top=24, note=""):
    """Commit the launch geometry of a transient profile pair as its own record.

    `results/nsys/*.sqlite` holds whichever experiment profiled last, so it
    cannot be a stage dependency (AGENTS.md). The provenance block is what makes
    this file readable later: the profile it came from is identified by hash.
    """
    out = {
        "note": note,
        "provenance": {
            "superproject_commit": _git("rev-parse", "HEAD"),
            "model_source": "ClimaCore.jl/ext/cuda/cuda_utils.jl",
            "device": "NVIDIA A100 (Compute Capability 8.0)",
            "max_waves": 1,
        },
        "arms": {},
    }
    for arm, path in arms.items():
        out["provenance"][f"{arm}_sqlite_sha256_16"] = _sha256_16(path)
        out["provenance"][f"{arm}_sqlite_mtime"] = int(os.path.getmtime(path))
        rows = inventory(path)
        device_ms = round(sum(r["total_ms"] for r in rows), 1)
        at_capacity = [r for r in rows if r["fills_one_wave"]]
        below = [r for r in rows if r["waves"] is not None and r["waves"] < 1.0]
        above = [r for r in rows if r["waves"] is not None and r["waves"] > 1.0]
        ms = lambda g: round(sum(r["total_ms"] for r in g), 1)
        out["arms"][arm] = {
            "device_total_ms": device_ms,
            "kernel_variants": len(rows),
            "fills_one_wave_variants": len(at_capacity),
            "fills_one_wave_ms": ms(at_capacity),
            "fills_one_wave_pct_of_device": round(100 * ms(at_capacity) / device_ms, 2),
            "below_one_wave_variants": len(below),
            "below_one_wave_ms": ms(below),
            "above_one_wave_variants": len(above),
            "above_one_wave_ms": ms(above),
            "above_one_wave_kernels": sorted({r["kernel"][:44] for r in above}),
            "register_limited_ms": ms(
                [r for r in at_capacity if r["limiter"] == "registers"]
            ),
            "register_limited_variants": len(
                [r for r in at_capacity if r["limiter"] == "registers"]
            ),
            "register_limited_pct_of_device": round(
                100
                * ms([r for r in at_capacity if r["limiter"] == "registers"])
                / device_ms,
                2,
            ),
            "validation": validate(rows),
            "top_kernels": rows[:top],
        }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    return out


# --- What a register buys -------------------------------------------------
#
# Occupancy is not time: a kernel with more resident warps hides latency better
# but runs the same instructions, so time falls by less than warps rise. The
# exponent below is fitted from treatments that crossed an occupancy step, and
# it is an UPPER BOUND on the occupancy effect: every such treatment also
# removed work, and nothing here separates the two.


def elasticity(warps_before, warps_after, ms_before, ms_after):
    """d(log time) / d(log resident warps) across one treatment."""
    import math

    return math.log(ms_after / ms_before) / math.log(warps_after / warps_before)


def predict_step(total_ms, warps_before, warps_after, exponent):
    """Time after a change in resident warps, under a fitted power law."""
    factor = (warps_after / warps_before) ** exponent
    return dict(
        warps_before=warps_before,
        warps_after=warps_after,
        warps_factor=round(warps_after / warps_before, 4),
        predicted_ms=round(total_ms * factor, 1),
        saved_ms=round(total_ms * (1 - factor), 1),
        change_pct=round(100 * (factor - 1), 2),
    )


def calibration(fold_path="results/retired/sedimentation-fold.json"):
    """Occupancy-step calibration points from the sedimentation fold.

    The fold is the only treatment this repo has measured that moved a kernel
    across an occupancy step while leaving a control population to resolve it
    against. Its two treated kernels are read from the committed snapshot, and
    their recorded warps per SM are checked against the launch model.
    """
    fold = json.load(open(fold_path))
    points = []
    for name, k in fold["kernels"].items():
        wb, wa = k["warps_per_sm"]["baseline"], k["warps_per_sm"]["mod"]
        mb, ma = k["total_ms"]["baseline"], k["total_ms"]["mod"]
        points.append(
            dict(
                kernel=name,
                registers_before=k["registers"]["baseline"],
                registers_after=k["registers"]["mod"],
                warps_before=wb,
                warps_after=wa,
                model_warps_before=warps_per_sm(k["registers"]["baseline"]),
                model_warps_after=warps_per_sm(k["registers"]["mod"]),
                ms_before=mb,
                ms_after=ma,
                change_pct=k["change_pct"],
                exponent=round(elasticity(wb, wa, mb, ma), 4),
            )
        )
    for p in points:
        p["model_agrees"] = (
            p["model_warps_before"] == p["warps_before"]
            and p["model_warps_after"] == p["warps_after"]
        )
    return dict(
        source=fold_path,
        treatment=fold["provenance"]["treatment"],
        resolved_against_controls=fold["saving_resolved_against_controls"],
        worst_control_delta_ms=fold["worst_control_delta_ms"],
        null_floor_pct=fold["end_to_end"]["floor_pct"],
        points=points,
        all_model_agrees=all(p["model_agrees"] for p in points),
        exponent_mean=round(sum(p["exponent"] for p in points) / len(points), 4),
        exponent_min=round(min(p["exponent"] for p in points), 4),
        exponent_max=round(max(p["exponent"] for p in points), 4),
    )


def validate(rows):
    """Predict (block, grid) from the register count alone and compare.

    This is the model's falsifiable claim, and it has to be made without
    reference to the launch being checked: `launch_config` is given the register
    count and nothing else, so it picks the block size too. Kernels that
    allocate shared memory are excluded because their per-block requirement is a
    function of the block size that nsys records only at the size chosen, and
    kernels whose loop had fewer slices than one wave are excluded because their
    grid is bounded by the slice count rather than by occupancy.
    """
    checked, shape_agree, thread_agree, disagree = 0, 0, 0, []
    for r in rows:
        if r["shared_memory_bytes"] or not r["fills_one_wave"]:
            continue
        threads, blocks, _ = launch_config(r["registers"])
        checked += 1
        shape_agree += (threads, blocks) == (r["block_size"], r["grid_size"])
        thread_agree += threads * blocks == r["threads"]
        if (threads, blocks) != (r["block_size"], r["grid_size"]):
            disagree.append(
                dict(
                    kernel=r["kernel"][:60],
                    registers=r["registers"],
                    predicted=f"{threads}x{blocks}",
                    observed=f"{r['block_size']}x{r['grid_size']}",
                    threads_agree=threads * blocks == r["threads"],
                )
            )
    # The resident thread count is what occupancy is, and the model gets it for
    # every launch. It misses the block shape of the loops that hand a slice to
    # a sub-block, which ask launch_configuration for a coarser granularity than
    # a warp, so the same threads are spread over more, smaller blocks.
    return dict(
        checked=checked,
        thread_count_agree=thread_agree,
        thread_count_agree_pct=round(100 * thread_agree / checked, 2) if checked else None,
        block_shape_agree=shape_agree,
        block_shape_agree_pct=round(100 * shape_agree / checked, 2) if checked else None,
        shape_only_mismatches=len([d for d in disagree if d["threads_agree"]]),
        disagree=disagree[:12],
    )
