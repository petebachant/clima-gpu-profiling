"""Price the terminal-velocity fold (PR #4947) on the fused inventory.

`export-sedimentation-fold` is frozen and cannot be restated: it is built
around the environment/updraft kernel pair, and upstream's 2026-11-03 pull
fused those into one `set_precipitation_velocities`, so that stage dies on an
empty match. A frozen experiment record is the measurement as it was run, so it
stays as it is and this stage prices the same treatment on the current stack.

What changed is not the treatment but its worth. The pull pushed the unfolded
kernel from 19 to 115 registers, past an occupancy step, so the fold now
recovers an order of magnitude more than when it was written. See
results/regression-record/attribution-1103.json.

Two independent routes are reported and neither is hidden:

  * `device` converts the measured device saving through the anchor the census
    establishes -- the longwave g-point split, where both the device saving and
    the SYPD gain are measured on real coupler steps.
  * `end_to_end_residual` takes the AMIP pair and subtracts the radiation
    splits' own in-process measurements, leaving what the rest of the mod arm
    accounts for. The arm carries only this fold besides radiation.

The residual does not resolve against the null floor on its own; the agreement
between the two routes is the evidence, and `routes_agree_within_pts` records
the gap rather than asserting it is small.
"""

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
import tomllib

R = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TREATED = "set_precipitation_velocities"
# The mod arm also carries the two g-point splits, so the radiation solves are
# treated kernels and cannot serve as controls. Everything else that pairs
# across the arms is untouched by either treatment, and its worst move bounds
# run-to-run noise at the same scale as the treated kernel.
#
# An earlier version took the controls from the fold's own source file instead.
# That set is empty now: upstream's fusion turned every kernel in
# microphysics_cache.jl into a named function, and a named kernel's group key is
# the function, not the file. It reported zero controls and a "resolved by"
# factor of 6.6e11 from dividing by an epsilon.
ALSO_TREATED = (
    "rte_lw_2stream_solve_both_skies_CUDA_",
    "rte_sw_2stream_solve_both_skies_CUDA_",
)
LINE_SUFFIX = re.compile(r"_L\d+$")


def rel(*p):
    return os.path.join(R, *p)


def group(name):
    head = name.split("__FILE")[0]
    if head:
        return head.replace("__NVTX", "")
    return LINE_SUFFIX.sub("_L*", name)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = con.execute(
        """SELECT s.value, COUNT(*), SUM(k.end - k.start)/1e6,
                  MAX(k.registersPerThread), MAX(k.blockX*k.blockY*k.blockZ)
           FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON s.id = k.shortName
           GROUP BY s.value"""
    ).fetchall()
    con.close()
    out = {}
    for name, n, ms, regs, threads in rows:
        # The mod arm's own packages carry a `_mod` path segment; drop it so a
        # kernel pairs with its baseline counterpart.
        g = group(name.replace("_jl_mod_", "_jl_"))
        a = out.setdefault(
            g, {"ms": 0.0, "launches": 0, "registers": 0, "threads": 0, "names": 0}
        )
        a["ms"] += ms or 0.0
        a["launches"] += n
        a["registers"] = max(a["registers"], regs or 0)
        a["threads"] = max(a["threads"], threads or 0)
        a["names"] += 1
    return out


# The residual route subtracts radiation from the AMIP pair and attributes the
# rest to this fold, which is only valid while the mod arm carries nothing else.
# PR #4947 touches exactly these two files.
FOLD_FILES = {
    "ClimaAtmos.jl-mod/src/cache/microphysics_cache.jl",
    "ClimaAtmos.jl-mod/src/parameterized_tendencies/microphysics/"
    "terminal_velocity_utils.jl",
}


def atmos_arm_files():
    """Which files the mod arm changes in ClimaAtmos, from the recorded diff.

    Derived rather than asserted: if a later experiment adds a second treatment
    to this arm, the residual route stops being attributable and must say so
    itself. A hardcoded `true` here would keep claiming otherwise.
    """
    path = rel("diffs/ClimaAtmos.jl.diff")
    if not os.path.exists(path):
        return None
    files = set()
    with open(path) as f:
        for line in f:
            if line.startswith("+++ b/"):
                files.add(line[len("+++ b/"):].strip())
    return files


def gitref(path):
    try:
        return subprocess.run(
            ["git", "-C", rel(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main():
    bpath, mpath = rel("results/nsys/baseline.sqlite"), rel("results/nsys/mod.sqlite")
    base, mod = load(bpath), load(mpath)
    census = json.load(open(rel("results/docs/wave-census.json")))
    summary = json.load(open(rel("results/summary.json")))
    nulls = json.load(open(rel("results/null-census.json")))
    with open(rel("results/gpoint-wallclock.toml"), "rb") as f:
        lw_wall = tomllib.load(f)
    with open(rel("results/sw-gpoint-wallclock.toml"), "rb") as f:
        sw_wall = tomllib.load(f)

    b, m = base.get(TREATED), mod.get(TREATED)
    if not b or not m:
        sys.exit(f"{TREATED} missing from an arm: baseline={bool(b)} mod={bool(m)}")
    saved = b["ms"] - m["ms"]

    # Controls: every kernel that pairs across the arms and is treated by
    # neither the fold nor the g-point splits. An unpaired kernel is excluded
    # rather than charged, since a name present in one arm only carries no
    # difference to measure.
    controls = []
    for g in sorted(set(base) & set(mod)):
        if g == TREATED or g in ALSO_TREATED:
            continue
        cb, cm = base[g]["ms"], mod[g]["ms"]
        controls.append({"kernel": g, "baseline_ms": round(cb, 1),
                         "mod_ms": round(cm, 1), "delta_ms": round(cm - cb, 1)})
    controls.sort(key=lambda c: -abs(c["delta_ms"]))
    assert controls, "no paired untreated kernels, so nothing bounds the noise"
    worst_control = max(abs(c["delta_ms"]) for c in controls)

    # Two anchors, because the census's conversion and the AMIP pair are not on
    # the same basis. The census divides the longwave's STEADY-STATE gain by its
    # device saving; the AMIP pair is a full-run comparison, so the residual
    # below is full-run. Both conversions are reported rather than one chosen,
    # since the gap between them is the same size as the agreement being
    # claimed and hiding it would overstate the agreement.
    per_ms = census["conversion"]["sypd_pct_per_device_ms"]
    lw_saved_ms = census["conversion"]["lw_saved_ms"]
    device_sypd = saved * per_ms

    lw_arm = lw_wall["arms"][str(lw_wall["default_ngpt_groups"])]
    lw_sypd = lw_arm["sypd_gain_pct"]
    per_ms_full = lw_sypd / lw_saved_ms
    device_sypd_full = saved * per_ms_full
    sw_sypd = sw_wall["arms"][str(sw_wall["default_ngpt_groups"])]["sypd_gain_pct"]
    pair = summary["speedup_pct"]
    residual = pair - (lw_sypd + sw_sypd)
    arm_files = atmos_arm_files()

    device_total_base = sum(v["ms"] for v in base.values())
    out = {
        "treatment": (
            "Val-fold the terminal-velocity parameters at both "
            "set_precipitation_velocities! call sites (CliMA/ClimaAtmos.jl#4947)"
        ),
        "treated": {
            "kernel": TREATED,
            "baseline_ms": round(b["ms"], 1),
            "mod_ms": round(m["ms"], 1),
            "saved_ms": round(saved, 1),
            "change_pct": round(100.0 * (m["ms"] / b["ms"] - 1), 2),
            "baseline_registers": b["registers"],
            "mod_registers": m["registers"],
            "threads_per_block": b["threads"],
            "launches": b["launches"],
            "launches_match": b["launches"] == m["launches"],
            "pct_of_device": round(100.0 * saved / device_total_base, 3),
        },
        "controls": {
            "n": len(controls),
            "worst_delta_ms": round(worst_control, 1),
            "resolved_against_controls": abs(saved) > worst_control,
            # None rather than a number divided by an epsilon: a zero-move
            # control set bounds nothing, and saying so beats a huge ratio.
            "resolved_by_factor": (
                round(abs(saved) / worst_control, 1) if worst_control else None
            ),
            "largest_movers": controls[:8],
            "excluded_as_also_treated": list(ALSO_TREATED),
            "unpaired_excluded": sorted(set(base) ^ set(mod)),
        },
        "device": {
            "sypd_pct_per_device_ms": per_ms,
            "sypd_pct": round(device_sypd, 3),
            "anchor_basis": "longwave steady-state gain per device ms",
            "sypd_pct_full_run_anchor": round(device_sypd_full, 3),
            "full_run_anchor_basis": "longwave full-run gain per device ms",
            "anchor_spread_pts": round(abs(device_sypd_full - device_sypd), 3),
            "anchor": (
                "The longwave g-point split, the one treatment whose device "
                "saving and SYPD gain are both measured; the shortwave then "
                "tested the conversion and came in "
                f"{census['prediction']['error_pts']} points off."
            ),
        },
        "end_to_end_residual": {
            "amip_pair_pct": round(pair, 3),
            "longwave_in_process_pct": round(lw_sypd, 3),
            "shortwave_in_process_pct": round(sw_sypd, 3),
            "residual_pct": round(residual, 3),
            "null_floor_pct": nulls["current_window_mean_pct"],
            "null_spread_points": nulls["current_window_spread_points"],
            # The pair cannot resolve a treatment this size on this machine, so
            # the residual is a cross-check on the conversion, not a result.
            "residual_clears_null_floor": residual > nulls["current_window_mean_pct"],
            "arm_carries_only_this_besides_radiation": (
                None if arm_files is None else arm_files == FOLD_FILES
            ),
            "atmos_arm_changed_files": (
                None if arm_files is None else sorted(arm_files)
            ),
        },
        "routes_agree_within_pts": round(abs(device_sypd - residual), 3),
        "routes_agree_within_pts_same_basis": round(
            abs(device_sypd_full - residual), 3
        ),
        "sypd_range_pct": [
            round(min(device_sypd, device_sypd_full, residual), 3),
            round(max(device_sypd, device_sypd_full, residual), 3),
        ],
        "vs_as_written": {
            "frozen_snapshot_saved_ms": 48.9,
            "frozen_snapshot_source": "results/sedimentation-fold.json",
            "times_larger_now": round(saved / 48.9, 1),
            "why": (
                "Upstream's 2026-11-03 broadcast fusion took the unfolded kernel "
                "from 19 to 115 registers, past an occupancy step. The fold did "
                "not change; what it is worth did."
            ),
        },
        "provenance": {
            "superproject_commit": gitref("."),
            "mod_ref": gitref("ClimaAtmos.jl-mod"),
            "baseline_ref": gitref("ClimaAtmos.jl"),
            "baseline_sqlite_sha256_16": digest(bpath),
            "mod_sqlite_sha256_16": digest(mpath),
            "profiled_steps": 120,
        },
    }

    path = rel("results/fold-on-current-stack.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    t = out["treated"]
    print(f"{TREATED}: {t['baseline_ms']} -> {t['mod_ms']} ms "
          f"({t['change_pct']}%), {t['baseline_registers']} -> "
          f"{t['mod_registers']} registers")
    print(f"saved {t['saved_ms']} ms = {out['device']['sypd_pct']}% of SYPD converted; "
          f"AMIP residual {out['end_to_end_residual']['residual_pct']}%, "
          f"routes {out['routes_agree_within_pts']} pts apart")
    print(f"resolved against {out['controls']['n']} in-block controls by "
          f"{out['controls']['resolved_by_factor']}x")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
