"""Which kernels cannot fill this device, and which of them splitting can help.

The longwave g-point split came out of one observation in section 4l of the
profiling study: at one thread per column the radiation solve's grid is 96
blocks against 108 SMs, so the whole problem is resident at once, no block is
queued behind it, and an SM that finishes early idles until the slowest column
is done. Splitting the g-point loop multiplied the blocks and the saving
followed. This census asks the same question of every kernel in the profile, so
the next member of that family is chosen from a measurement rather than guessed.

It is deliberately name-agnostic. The other analysis stages here match kernels by
name and break when something upstream is renamed or fused; this one enumerates
whatever the profile holds and classifies it by launch geometry, so a rename
changes a row's label and nothing else.

One loss formula covers every shape. A kernel's blocks occupy
`ceil(waves)` wave-times to do `waves` wave-times of work, so the geometric
waste is `1 - waves / ceil(waves)` whether the grid is smaller than the device
or merely leaves a ragged last wave. The categories below only name the shape:

  * `idle_sms`         -- fewer blocks than SMs, so some SMs get nothing at all.
  * `partial_wave`     -- every SM has a block but the device could hold more of
                          them at once. Same arithmetic as above: the first pass
                          of this script scored it zero, which was wrong, and it
                          is the shape `non_orographic_gravity_wave_forcing` has.
  * `exactly_resident` -- the grid fills the device once, so there is no
                          geometric waste left to collect.
  * `multi_wave`       -- more blocks than fit at once, so the last wave is
                          partly empty.

What is NOT costed here is occupancy. A kernel can have a block on every SM and
still hold far fewer warps than its registers allow; that is reported as
`warp_capacity_pct` and deliberately left out of the loss, because section 4l
records ncu's occupancy recommendation being the wrong lever four times in this
project, including a +3.93% regression from doubling occupancy on these very
radiation kernels.

So `quantization_loss_ms` is a floor on what launch geometry alone is wasting,
not an estimate of what any treatment would return.
"""

import hashlib
import json
import math
import os
import sqlite3
import subprocess
import tomllib

BASE = "results/nsys/baseline.sqlite"
SWEEP = "results/gpoint-sweep.toml"
OUT = "results/wave-census.json"

# A100
SMS = 108
REGS_PER_SM = 65536
THREADS_PER_SM = 2048
MAX_BLOCKS_PER_SM = 32
WARPS_PER_SM = 64
REG_GRANULARITY = 8  # per-thread register allocation granularity
REGISTER_CAP = 255

# Rows below this share of device time are noise for this question.
MIN_MS = 25.0


def connect(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def union_ms(rows):
    """Elapsed time covered by a set of [start, end] intervals, merged.

    `SUM(end-start)` is elapsed GPU time only while kernels do not overlap. Kept
    beside the merged figure so the gap is visible rather than assumed away.
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
    rows = con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    return union_ms(rows), sum(e - s for s, e in rows) / 1e6


def blocks_per_sm(regs, threads):
    """How many of this kernel's blocks an SM holds at once.

    Registers are allocated per warp in a fixed granularity, so the usable
    figure rounds the per-thread count up before multiplying.
    """
    warps = math.ceil(threads / 32)
    per_block = math.ceil(regs / REG_GRANULARITY) * REG_GRANULARITY * 32 * warps
    return max(
        1,
        min(
            REGS_PER_SM // max(per_block, 1),
            THREADS_PER_SM // max(threads, 1),
            MAX_BLOCKS_PER_SM,
        ),
    )


def configs(con):
    """Every (kernel, launch geometry) the profile holds, largest first.

    Grouped by geometry as well as name because one kernel can be launched at
    more than one grid, and a census that averaged them would describe neither.
    """
    return con.execute(
        """
        SELECT s.value,
               k.gridX * k.gridY * k.gridZ,
               k.blockX * k.blockY * k.blockZ,
               k.registersPerThread,
               COUNT(*),
               SUM(k.end - k.start) / 1e6
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON s.id = k.shortName
        GROUP BY s.value, 2, 3, 4
        ORDER BY 6 DESC
        """
    ).fetchall()


def classify(blocks, threads, regs, ms, ncol):
    per_sm = blocks_per_sm(regs, threads)
    resident = SMS * per_sm
    waves = blocks / resident
    blocks_on_an_sm = min(blocks / SMS, per_sm)
    warps = blocks_on_an_sm * math.ceil(threads / 32)

    # The blocks occupy ceil(waves) wave-times to do waves wave-times of work.
    loss = 1.0 - waves / math.ceil(waves)
    if blocks < SMS:
        category = "idle_sms"
    elif waves < 1.0:
        category = "partial_wave"
    elif waves == 1.0:
        category = "exactly_resident"
    else:
        category = "multi_wave"

    return {
        "blocks": blocks,
        "threads_per_block": threads,
        "registers": regs,
        "blocks_resident_per_sm": per_sm,
        "blocks_resident": resident,
        "waves": round(waves, 2),
        "warps_per_sm": round(warps, 1),
        "warp_capacity_pct": round(100.0 * warps / WARPS_PER_SM, 1),
        "at_register_cap": regs >= REGISTER_CAP,
        # True when the grid covers the column count once with one thread each,
        # which is the shape section 4l diagnosed and the g-point split escaped.
        "one_thread_per_column": ncol <= blocks * threads < ncol + threads,
        "category": category,
        "quantization_loss_pct": round(100.0 * loss, 1),
        "quantization_loss_ms": round(ms * loss, 1),
    }


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
    with open(SWEEP, "rb") as f:
        ncol = tomllib.load(f)["ncol"]

    con = connect(BASE)
    elapsed, summed = device_total(con)

    kernels = []
    for name, blocks, threads, regs, launches, ms in configs(con):
        if ms < MIN_MS:
            continue
        row = {
            "kernel": name.replace("_jl_mod_", "_jl_"),
            "launches": launches,
            "ms": round(ms, 1),
            "pct_of_device": round(100.0 * ms / elapsed, 2),
        }
        row.update(classify(blocks, threads, regs, ms, ncol))
        kernels.append(row)

    censused = sum(k["ms"] for k in kernels)
    categories = {}
    for cat in ("idle_sms", "partial_wave", "exactly_resident", "multi_wave"):
        sel = [k for k in kernels if k["category"] == cat]
        categories[cat] = {
            "configs": len(sel),
            "ms": round(sum(k["ms"] for k in sel), 1),
            "pct_of_censused": round(100.0 * sum(k["ms"] for k in sel) / censused, 1),
            "quantization_loss_ms": round(
                sum(k["quantization_loss_ms"] for k in sel), 1
            ),
        }

    capped = [k for k in kernels if k["at_register_cap"]]
    starved = [k for k in kernels if k["warp_capacity_pct"] < 40.0]

    results = {
        "source": {
            "sqlite": BASE,
            "sqlite_digest": digest(BASE),
            "commit": git("rev-parse", "HEAD"),
            "columns_from": SWEEP,
        },
        "machine": {
            "sms": SMS,
            "registers_per_sm": REGS_PER_SM,
            "threads_per_sm": THREADS_PER_SM,
            "max_blocks_per_sm": MAX_BLOCKS_PER_SM,
            "warps_per_sm": WARPS_PER_SM,
            "register_cap": REGISTER_CAP,
            "resident_thread_slots": SMS * THREADS_PER_SM,
        },
        "problem": {
            "columns": ncol,
            # The ceiling on every per-column kernel in this model at this
            # resolution: one thread per column can occupy no more of the
            # device than this, however cheap the thread is.
            "column_threads_as_pct_of_slots": round(
                100.0 * ncol / (SMS * THREADS_PER_SM), 2
            ),
        },
        "device": {
            "elapsed_ms": round(elapsed, 1),
            "summed_ms": round(summed, 1),
            "summed_over_elapsed": round(summed / elapsed, 3) if elapsed else None,
            "censused_ms": round(censused, 1),
            "censused_pct_of_device": round(100.0 * censused / elapsed, 1),
            "min_ms_per_config": MIN_MS,
        },
        "categories": categories,
        "at_register_cap": {
            "configs": len(capped),
            "ms": round(sum(k["ms"] for k in capped), 1),
            "pct_of_device": round(100.0 * sum(k["ms"] for k in capped) / elapsed, 1),
            "kernels": [k["kernel"] for k in capped],
        },
        "warp_starved": {
            "threshold_pct": 40.0,
            "configs": len(starved),
            "ms": round(sum(k["ms"] for k in starved), 1),
            "pct_of_censused": round(
                100.0 * sum(k["ms"] for k in starved) / censused, 1
            ),
        },
        "kernels": kernels,
        "note": (
            "Device shares are ELAPSED time, the union of the kernel intervals, "
            "with the summed figure beside them to show the gap. "
            "quantization_loss_ms is a FLOOR on what launch geometry alone "
            "wastes: the blocks occupy ceil(waves) wave-times to do waves "
            "wave-times of work, which is the same arithmetic whether the grid "
            "is smaller than the device or merely leaves a ragged last wave. "
            "Occupancy is NOT costed -- a kernel can have a block on every SM "
            "and still hold a fraction of the warps its registers allow, and "
            "section 4l of the profiling study records ncu's occupancy "
            "recommendation being the wrong lever four times here, including a "
            "+3.93% regression from doubling it on these very radiation "
            "kernels. Read warp_capacity_pct beside the loss, not as part of "
            "it. at_register_cap is the family where the lever "
            "is splitting the kernel rather than freeing registers, because a "
            "kernel at the cap spends any freed register on live state. "
            "one_thread_per_column marks the launch shape section 4l "
            "diagnosed. Rows under min_ms_per_config are omitted."
        ),
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"wrote {OUT}")
    print(f"  device elapsed {elapsed:,.0f} ms, censused {censused:,.0f} ms")
    print(f"  at the {REGISTER_CAP}-register cap: {len(capped)} configs, "
          f"{sum(k['ms'] for k in capped):,.0f} ms")
    for k in capped:
        print(f"    {k['ms']:8.1f} ms  {k['waves']:5.2f} waves  "
              f"{k['warp_capacity_pct']:5.1f}% warps  {k['kernel'][:58]}")
    print(f"  one thread per column: "
          f"{sum(1 for k in kernels if k['one_thread_per_column'])} configs")


if __name__ == "__main__":
    main()
