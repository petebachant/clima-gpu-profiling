"""Price the removal of the SGS quadrature kernel's redundant saturation work.

The environment quadrature broadcast of `set_microphysics_tendency_cache!` is the
heaviest kernel on the device. It cannot be made more occupant: a frame census of
its call graph puts CloudMicrophysics' bulk tendency alone at 102 registers,
which is inside the same A100 occupancy band as the whole kernel's 128, so no
ClimaAtmos-side register treatment moves its resident warps at all.

What it can be made to do is less arithmetic at the same answer. Two saturation
specific humidities per node were computed and thrown away: the node carried a
total water that nothing read, and the node function recomputed the saturation
`_point_value` had just evaluated at the same temperature and density. Removing
both is bit-for-bit, and the compile-only probe measures it that way -- the two
arms agree on an FNV-1a hash of every output bit.

The nsys databases are transient, so the derived numbers are snapshotted here
with their provenance, as AGENTS.md requires of anything a write-up will cite.
"""

import hashlib
import json
import re
import sqlite3
import subprocess
import sys
import tomllib
from pathlib import Path

BASE = "results/nsys/baseline.sqlite"
MOD = "results/nsys/mod.sqlite"
OCCUPANCY = "results/experiments/occupancy-launch-model/stats.json"
PROBES = {
    "untreated": "results/quadrature-registers-work-baseline.toml",
    "saturation_reuse": "results/quadrature-registers-work-mod-satreuse-only.toml",
    "treated": "results/quadrature-registers-work-mod.toml",
}
OUT = Path("results/experiments/quadrature-saturation/stats.json")

# The treated broadcast, identified by shape rather than by source line. A
# kernel's name embeds the line of the statement that built it, and the mod arm
# also carries the sedimentation fold, which inserts lines into
# `cache/microphysics_cache.jl` -- so the quadrature broadcast is L1297 in one
# arm and L1313 in the other, and pairing by name silently finds nothing.
# What does not move: it is the heaviest kernel the cache launches, and it runs
# once per coupling step over the profiled window.
TARGET_FAMILY = "set_microphysics_tendency_cache"
TARGET_LAUNCHES = 240
# The sedimentation fold is a standing difference between the arms, so its
# kernels are not a control on run-to-run noise.
NOT_A_CONTROL = re.compile(r"precipitation_velocit|terminal_velocit")
PROBE_TARGET = "quadrature broadcast (L1297)"
PROBE_CONTROL = "non-quadrature sibling"


A100_REGS_PER_SM = 65536
A100_THREADS_PER_SM = 2048
A100_BLOCKS_PER_SM = 32


def warps_per_sm(regs, block):
    """Resident warps per SM for one block shape, register- and slot-limited."""
    blocks = min(A100_REGS_PER_SM // max(regs * block, 1),
                 A100_THREADS_PER_SM // max(block, 1), A100_BLOCKS_PER_SM)
    return blocks * ((block + 31) // 32)


def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def normalize(name):
    """A kernel name with the arm tag removed, so the two arms pair up."""
    return re.sub(
        r"(Clima\w+?|RRTMGP|CloudMicrophysics)_jl_mod_", r"\1_jl_", name
    )


def control_key(name):
    """A kernel name with the arm tag AND the source line removed.

    A name embeds the line of the statement that built the kernel, and the mod
    arm's sedimentation fold inserts lines into the same file the treated
    broadcast lives in. Pairing controls by exact name therefore compares one
    arm's statement with whatever happens to sit at that line in the other,
    which reported a 99% change on a kernel nothing touched. Aggregating every
    launch of one function into a single key is the comparison that survives an
    edit above it.
    """
    return re.sub(r"_L\d+$", "_L*", normalize(name))


def arm(path):
    """Device total, per-kernel totals, and the target's launch populations."""
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    with con:
        total = con.execute(
            "SELECT SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
        rows = con.execute("""
            SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6,
                   k.registersPerThread, k.blockX*k.blockY*k.blockZ,
                   MAX(k.localMemoryPerThread)
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id
            GROUP BY s.value, k.registersPerThread,
                     k.blockX*k.blockY*k.blockZ
        """).fetchall()
    con.close()

    by_kernel, target = {}, []
    for name, n, ms, regs, block, spill in rows:
        key = normalize(name)
        ckey = control_key(name)
        by_kernel[ckey] = by_kernel.get(ckey, 0.0) + ms
        if TARGET_FAMILY in key and n == TARGET_LAUNCHES:
            target.append({
                "name": key, "launches": n, "total_ms": round(ms, 1),
                "registers": regs, "block_size": block,
                "spill_bytes": spill,
            })
    return {"device_total_ms": round(total, 1), "by_kernel": by_kernel,
            "target_populations": target}


def one_target(a, tag):
    """The heaviest once-per-step kernel the cache launches.

    The cache launches two of them: the environment quadrature broadcast and the
    `foreach_column` precompute, which is two orders of magnitude lighter. Take
    the heavier and assert the gap, so a future fusion that makes them
    comparable fails here rather than quietly reporting the wrong one.
    """
    pops = sorted(a["target_populations"], key=lambda p: -p["total_ms"])
    if len(pops) < 2:
        sys.exit(f"{tag}: expected at least two populations, found {pops}")
    if pops[0]["total_ms"] < 10 * pops[1]["total_ms"]:
        sys.exit(
            f"{tag}: the heaviest cache kernel is not clearly the quadrature "
            f"one: {pops[0]} vs {pops[1]}"
        )
    return pops[0]


def probe_case(path, case):
    d = tomllib.load(open(path, "rb"))
    if d["climaatmos_dirty"]:
        sys.exit(f"{path}: probe ran on a dirty ClimaAtmos tree")
    return d["cases"][case], d["climaatmos_commit"], d["layout"]["points"]


def main(out=OUT):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    base, mod = arm(BASE), arm(MOD)
    bt, mt = one_target(base, "baseline"), one_target(mod, "mod")

    saved_ms = bt["total_ms"] - mt["total_ms"]
    # ClimaCore sizes the grid from the register count, so a register is also a
    # block size and a resident warp count. This kernel compiles to exactly the
    # top of its occupancy band, which leaves it no headroom at all.
    bw = warps_per_sm(bt["registers"], bt["block_size"])
    mw = warps_per_sm(mt["registers"], mt["block_size"])
    kernel = {
        "baseline": bt,
        "mod": mt,
        "baseline_warps_per_sm": bw,
        "mod_warps_per_sm": mw,
        "warps_lost": bw - mw,
        "occupancy_lost_pct": round(100 * (bw - mw) / bw, 1),
        "band_crossed": bw != mw,
        "saved_ms": round(saved_ms, 1),
        "saved_pct": round(100 * saved_ms / bt["total_ms"], 2),
        "launches_agree": bt["launches"] == mt["launches"],
        "registers_moved": mt["registers"] - bt["registers"],
        "spill_moved_bytes": mt["spill_bytes"] - bt["spill_bytes"],
    }

    # Every other paired kernel is a control: the treatment is one broadcast, so
    # anything else that moves is run-to-run noise or a standing arm difference.
    controls = []
    for key, bms in base["by_kernel"].items():
        if key not in mod["by_kernel"] or bms < 100.0:
            continue
        # The treated broadcast's own function, and the standing arm difference.
        if TARGET_FAMILY in key or NOT_A_CONTROL.search(key):
            continue
        mms = mod["by_kernel"][key]
        controls.append({"kernel": key, "baseline_ms": round(bms, 1),
                         "mod_ms": round(mms, 1),
                         "change_pct": round(100 * (mms - bms) / bms, 2)})
    controls.sort(key=lambda c: -abs(c["change_pct"]))
    control = {
        "kernels_over_100ms": len(controls),
        "worst_change_pct": controls[0]["change_pct"] if controls else None,
        "worst_kernel": controls[0]["kernel"] if controls else None,
        "median_abs_change_pct": round(
            sorted(abs(c["change_pct"]) for c in controls)[len(controls) // 2], 2
        ) if controls else None,
        "rows": controls[:10],
    }

    # The device-ms to SYPD conversion measured on this stack, and the floor two
    # runs of identical code disagree by, both from the launch-model experiment.
    occ = json.load(open(OCCUPANCY))
    pct_per_ms = occ["prediction"]["sypd_pct_per_device_ms"]
    null_floor = occ["prediction"]["null_floor_pct"]
    end_to_end = {
        "sypd_pct_per_device_ms": pct_per_ms,
        "sypd_points": round(saved_ms * pct_per_ms, 2),
        "null_floor_pct": null_floor,
        "clears_null_floor": saved_ms * pct_per_ms > null_floor,
        "device_share_pct": round(100 * saved_ms / base["device_total_ms"], 2),
    }

    # The compile-only probe, which prices the same treatment on a state in which
    # almost every cell carries condensate, and is where the bit-for-bit check
    # lives. Its percentage is an upper bound on the profile's: a cell that takes
    # an early-out never pays the saturation evaluations the treatment removes.
    probe, commits = {}, {}
    for tag, path in PROBES.items():
        case, ref, points = probe_case(path, PROBE_TARGET)
        ctrl, _, _ = probe_case(path, PROBE_CONTROL)
        probe[tag] = {
            "device_ms": case["device_ms"],
            "registers": case["registers"],
            "spill_bytes": case["local_bytes"],
            "out_bits_hash": case["out_bits_hash"],
            "block_size": case["maxthreads"],
            "kernels_launched": case["kernels_launched"],
            "control_device_ms": ctrl["device_ms"],
            "control_out_bits_hash": ctrl["out_bits_hash"],
        }
        commits[tag] = ref
    u, s, t = probe["untreated"], probe["saturation_reuse"], probe["treated"]
    probe_summary = {
        "saved_pct_total": round(100 * (u["device_ms"] - t["device_ms"])
                                 / u["device_ms"], 2),
        "saved_pct_saturation_reuse": round(
            100 * (u["device_ms"] - s["device_ms"]) / u["device_ms"], 2),
        "saved_pct_dead_field": round(
            100 * (s["device_ms"] - t["device_ms"]) / u["device_ms"], 2),
        "control_change_pct": round(
            100 * (t["control_device_ms"] - u["control_device_ms"])
            / u["control_device_ms"], 2),
        # The point of the treatment: identical output, not nearly identical.
        "bit_for_bit": len({p["out_bits_hash"] for p in probe.values()}) == 1,
        "control_bit_for_bit": len(
            {p["control_out_bits_hash"] for p in probe.values()}) == 1,
        "spill_moved_bytes": t["spill_bytes"] - u["spill_bytes"],
        "points": points,
        # The saving cannot be an occupancy effect: the probe compiles the same
        # register count into the same block size in every state, so the launch
        # geometry is identical and only the arithmetic inside it changed.
        "registers_unchanged": len({p["registers"] for p in probe.values()}) == 1,
        "block_size_unchanged": len(
            {p["block_size"] for p in probe.values()}) == 1,
        "one_kernel_each": {p["kernels_launched"] for p in probe.values()} == {1},
        "registers": t["registers"],
        "block_size": t["block_size"],
    }

    # The saturation evaluations one cloudy cell performs in this kernel, by
    # where they sit: one per node to fill the node's unread total water, one
    # per temperature point in `_point_value` to build the state the evaluator
    # receives, and one per temperature point inside the node function to
    # recover the saturation excess from that state. The first and third are
    # what the treatment removes; the second is the only one that has to happen,
    # because each point is at its own temperature.
    nodes, points_max = 8, 16
    work = {
        "nodes": nodes,
        "points_max": points_max,
        "calls_node_table": nodes,
        "calls_point_state": points_max,
        "calls_node_function": points_max,
        "calls_before": nodes + 2 * points_max,
        "calls_after": points_max,
    }
    work["calls_removed"] = work["calls_before"] - work["calls_after"]
    work["removed_pct"] = round(
        100 * work["calls_removed"] / work["calls_before"], 1)

    # The frame census, which is why no register treatment is attempted here.
    # It is computed by the launch-model experiment, which owns the occupancy
    # question, and carried through to this file so the prose below has one
    # source rather than two readings of the same probe.
    frames = occ["frames"]

    doc = {
        "note": __doc__.strip().splitlines()[0],
        "provenance": {
            "baseline_ref": git("-C", "ClimaAtmos.jl", "rev-parse", "HEAD"),
            "mod_ref": git("-C", "ClimaAtmos.jl-mod", "rev-parse", "HEAD"),
            "mod_branch": git("-C", "ClimaAtmos.jl-mod", "rev-parse",
                              "--abbrev-ref", "HEAD"),
            "superproject_commit": git("rev-parse", "HEAD"),
            "baseline_sqlite_sha256_16": digest(BASE),
            "mod_sqlite_sha256_16": digest(MOD),
            "probe_commits": commits,
            "profiled_steps": 120,
        },
        "kernel": kernel,
        "control": control,
        "end_to_end": end_to_end,
        "probe": probe,
        "probe_summary": probe_summary,
        "work": work,
        "frames": frames,
    }
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {out}")
    ret = doc
    print(f"  target {bt['total_ms']} -> {mt['total_ms']} ms "
          f"({kernel['saved_pct']}%), {end_to_end['sypd_points']} SYPD points")
    print(f"  worst control {control['worst_change_pct']}% "
          f"({control['worst_kernel']})")
    print(f"  probe {probe_summary['saved_pct_total']}%, bit-for-bit "
          f"{probe_summary['bit_for_bit']}")
    return ret


if __name__ == "__main__":
    main()
