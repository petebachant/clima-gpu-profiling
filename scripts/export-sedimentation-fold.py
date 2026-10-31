"""Price the terminal-velocity parameter fold, and separate two predictors.

`docs/fold-decision-rule.md` records four fold sites whose payoffs and flat
results are confounded: every site that paid started at 8 warps per SM AND held
at least 3.4% of device time, and every flat one started at 32 warps AND held at
most 0.4%. So occupancy headroom and device share could not be told apart.

This fold is the missing cell. The Chen 2022 snow terminal velocity has the
headroom of a paying site and the device share of a flat one, so whichever
predictor is doing the work, this measures which.

Both arms are profiled on one stack and differ only in the treatment, but the
nsys databases are transient -- the next experiment overwrites them -- so the
derived numbers are snapshotted here with their provenance, as
AGENTS.md requires of anything a write-up will cite.
"""

import hashlib
import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

BASE = "results/nsys/baseline.sqlite"
MOD = "results/nsys/mod.sqlite"
SUMMARY = "results/summary.json"
CENSUS = "results/null-census.json"
AMIP_LOGS = {"baseline": ".calkit/scheduler/logs/amip-baseline.out",
             "mod": ".calkit/scheduler/logs/amip-mod.out"}
OUT = Path("results/sedimentation-fold.json")

# The treated broadcasts, by baseline source line, with the mod line they move
# to: the fold inserts 8 lines at each of the two call sites, so everything
# below the second one shifts by 16. Named by what the statement computes.
TREATED = {
    221: ("env_terminal_velocity", 237),
    233: ("updraft_terminal_velocity", 249),
}
# The rest of the same block, untreated, as the control on run-to-run noise.
CONTROL = {220: 236, 232: 248, 234: 250, 235: 251,
           239: 255, 246: 262, 254: 270, 256: 272, 261: 277}

A100_REGS_PER_SM = 65536
A100_THREADS_PER_SM = 2048
A100_BLOCKS_PER_SM = 32


def warps_per_sm(regs, block):
    """Resident warps per SM for one block shape, register- and slot-limited."""
    blocks = min(A100_REGS_PER_SM // max(regs * block, 1),
                 A100_THREADS_PER_SM // max(block, 1), A100_BLOCKS_PER_SM)
    return blocks * ((block + 31) // 32)


def populations(path):
    """Every launch population keyed by source line, split by register count.

    Split rather than aggregated: one short name covers four tracers here, and
    three of them compile to a different register count than the fourth. Taking
    a MAX over the group reports one tracer's registers beside another's block
    size and invents a kernel that never ran.
    """
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    out, total = {}, None
    with con:
        total = con.execute(
            "SELECT SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
        rows = con.execute("""
            SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6,
                   k.registersPerThread, k.blockX*k.blockY*k.blockZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id
            WHERE s.value LIKE '%microphysics\\_cache%' ESCAPE '\\'
            GROUP BY s.value, k.registersPerThread, k.blockX*k.blockY*k.blockZ
        """).fetchall()
    for name, n, ms, regs, block in rows:
        m = re.search(r"_L(\d+)$", name)
        if m:
            out.setdefault(int(m.group(1)), []).append(
                {"launches": n, "total_ms": ms, "registers": regs,
                 "block_size": block}
            )
    con.close()
    return out, total


def pick(pops, line, launches):
    """The population at `line` with exactly `launches` launches."""
    hit = [p for p in pops.get(line, []) if p["launches"] == launches]
    if len(hit) != 1:
        sys.exit(
            f"L{line}: expected one population with {launches} launches, "
            f"found {len(hit)} among {pops.get(line)}"
        )
    return hit[0]


def arm_totals(path):
    """Device time by kernel, with the arm tag and line number normalized away.

    The fold shifts line numbers, and a kernel's name embeds them, so pairing
    the arms by raw name would report every kernel below the edit as having
    vanished and a new one appeared.
    """
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    agg = {}
    with con:
        rows = con.execute("""
            SELECT s.value, SUM(k.end-k.start)/1e6
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id GROUP BY s.value
        """).fetchall()
    con.close()
    for name, ms in rows:
        key = re.sub(r"_L\d+$", "_L*",
                     re.sub(r"(Clima\w+?|RRTMGP|CloudMicrophysics)_jl_mod_",
                            r"\1_jl_", name))
        agg[key] = agg.get(key, 0.0) + ms
    return agg


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
    bpop, btotal = populations(BASE)
    mpop, mtotal = populations(MOD)

    kernels, treated_b, treated_m = {}, 0.0, 0.0
    for line, (label, mline) in sorted(TREATED.items()):
        # The treated tracer is the register-heavy population at this line.
        # Launch count then pairs the arms, since it is set by the call site,
        # which the fold does not touch, and registers are what changed.
        b = max(bpop.get(line, []), key=lambda p: p["registers"])
        m = pick(mpop, mline, b["launches"])
        treated_b += b["total_ms"]
        treated_m += m["total_ms"]
        kernels[label] = {
            "baseline_line": line,
            "mod_line": mline,
            "launches": b["launches"],
            "registers": {"baseline": b["registers"], "mod": m["registers"]},
            "registers_saved": b["registers"] - m["registers"],
            "block_size": {"baseline": b["block_size"], "mod": m["block_size"]},
            "warps_per_sm": {
                "baseline": warps_per_sm(b["registers"], b["block_size"]),
                "mod": warps_per_sm(m["registers"], m["block_size"]),
            },
            "total_ms": {"baseline": round(b["total_ms"], 1),
                         "mod": round(m["total_ms"], 1)},
            "change_pct": round(100 * (m["total_ms"] / b["total_ms"] - 1), 2),
            "pct_of_device": round(100 * b["total_ms"] / btotal, 3),
        }
        kernels[label]["crossed_occupancy_step"] = (
            kernels[label]["warps_per_sm"]["mod"]
            > kernels[label]["warps_per_sm"]["baseline"]
        )

    # The controls bound run-to-run noise at the same scale as the treatment:
    # same block, same stack, untouched source.
    control = []
    for line, mline in sorted(CONTROL.items()):
        for b in bpop.get(line, []):
            try:
                m = pick(mpop, mline, b["launches"])
            except SystemExit:
                continue
            control.append({
                "baseline_line": line, "launches": b["launches"],
                "baseline_ms": round(b["total_ms"], 1),
                "mod_ms": round(m["total_ms"], 1),
                "delta_ms": round(m["total_ms"] - b["total_ms"], 1),
            })
    worst_control = max(abs(c["delta_ms"]) for c in control)

    # Whether the arm total moved, and what else moved in it. The treatment is
    # resolved against its own controls; the arm total need not be.
    ba, ma = arm_totals(BASE), arm_totals(MOD)
    treated_key = next(k for k in ba if "microphysics_cache" in k)
    drift = sum(ma.get(k, 0) - ba.get(k, 0) for k in set(ba) | set(ma)
                if k != treated_key)

    # The unprofiled AMIP pair, against the floor between identical arms. This
    # arrived by accident -- `calkit run summarize` re-ran the stale amip-mod
    # stage as a dependency -- so it is a test of this experiment's own
    # prediction rather than a measurement it set out to take.
    summary = json.load(open(SUMMARY))
    census = json.load(open(CENSUS))
    floor = census["current_window_mean_pct"]
    spread = census["current_window_spread_points"]
    margin = summary["speedup_pct"] - floor
    end_to_end = {
        "sypd": {"baseline": summary["baseline"]["sypd"],
                 "mod": summary["mod"]["sypd"]},
        "n_steps_timed": summary["mod"]["n_steps_timed"],
        "speedup_pct": round(summary["speedup_pct"], 3),
        "floor_pct": floor,
        "floor_spread_points": spread,
        "floor_n_nulls": census["n_current_window"],
        "margin_over_floor_points": round(margin, 3),
        # The margin has to beat the uncertainty in the floor it cleared, not
        # merely be positive: the floor is a mean of nulls that disagree by
        # more than this whole treatment is worth.
        "wall_clock_resolved": margin > spread,
        # Even a perfect conversion of the device saving could not clear it,
        # which is the part that does not depend on this pair at all.
        "max_points_if_fully_converted": round(
            100 * (treated_b - treated_m) / btotal, 3
        ),
    }

    out = {
        "provenance": {
            "treatment": "Val-fold the terminal-velocity parameters at both "
                         "set_precipitation_velocities! call sites",
            "baseline_ref": git("-C", "ClimaAtmos.jl", "rev-parse", "HEAD"),
            "mod_ref": git("-C", "ClimaAtmos.jl-mod", "rev-parse", "HEAD"),
            "mod_branch": git("-C", "ClimaAtmos.jl-mod", "rev-parse",
                              "--abbrev-ref", "HEAD"),
            "superproject_commit": git("rev-parse", "HEAD"),
            "baseline_sqlite_sha256_16": digest(BASE),
            "mod_sqlite_sha256_16": digest(MOD),
            "baseline_amip_log_sha256_16": digest(AMIP_LOGS["baseline"]),
            "mod_amip_log_sha256_16": digest(AMIP_LOGS["mod"]),
            "profiled_steps": 120,
        },
        "end_to_end": end_to_end,
        "kernels": kernels,
        "treated_total_ms": {"baseline": round(treated_b, 1),
                             "mod": round(treated_m, 1)},
        "treated_saved_ms": round(treated_b - treated_m, 1),
        "treated_change_pct": round(100 * (treated_m / treated_b - 1), 2),
        "treated_pct_of_device": round(100 * treated_b / btotal, 3),
        "saved_pct_of_device": round(100 * (treated_b - treated_m) / btotal, 3),
        "device_total_ms": {"baseline": round(btotal, 1),
                            "mod": round(mtotal, 1)},
        "device_total_change_pct": round(100 * (mtotal / btotal - 1), 3),
        "control_kernels": control,
        "worst_control_delta_ms": worst_control,
        "untreated_drift_ms": round(drift, 1),
        # The saving is larger than any control, so it is a measurement. It is
        # smaller than the drift across the untreated kernels, so the ARM total
        # is not: these are different claims and the file keeps them apart.
        "saving_resolved_against_controls": (treated_b - treated_m) > worst_control,
        "arm_total_resolved": abs(treated_b - treated_m) > abs(drift),
        "note": (
            "Registers and warps per SM come from the nsys launch records, not "
            "a model: each population is split by register count because one "
            "short name covers four tracers here and only the Chen 2022 snow "
            "one is register-heavy. The treated saving is resolved against the "
            "untreated broadcasts of the same block (worst_control_delta_ms), "
            "which bound run-to-run noise at the same scale. It is NOT resolved "
            "in the arm total: untreated_drift_ms is the sum of every other "
            "kernel's change, and it is the larger number, which is why "
            "arm_total_resolved is false and no end-to-end claim is made here. "
            "end_to_end prices the same treatment on the unprofiled AMIP "
            "pair. It does not resolve: the margin over the floor between "
            "identical arms is a fraction of the spread between the nulls that "
            "set the floor, and max_points_if_fully_converted shows that even a "
            "perfect conversion of the device saving could not have cleared it. "
            "Read the per-kernel figures, not this one."
        ),
    }
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    for label, k in kernels.items():
        print(
            f"{label}: {k['registers']['baseline']}->{k['registers']['mod']} regs "
            f"({k['registers_saved']} saved), "
            f"{k['warps_per_sm']['baseline']}->{k['warps_per_sm']['mod']} warps/SM, "
            f"{k['total_ms']['baseline']}->{k['total_ms']['mod']} ms "
            f"({k['change_pct']:+.2f}%), {k['pct_of_device']}% of device"
        )
    print(
        f"treated saved {out['treated_saved_ms']:.1f} ms "
        f"({out['treated_change_pct']:+.2f}%), worst control "
        f"{worst_control:.1f} ms -> saving "
        f"{'resolved' if out['saving_resolved_against_controls'] else 'NOT resolved'}"
    )
    print(
        f"arm total {out['device_total_change_pct']:+.3f}%, untreated drift "
        f"{out['untreated_drift_ms']:+.1f} ms -> arm total "
        f"{'resolved' if out['arm_total_resolved'] else 'NOT resolved'}"
    )
    e = out["end_to_end"]
    print(
        f"AMIP pair {e['speedup_pct']:+.3f}% against a {e['floor_pct']:+.3f}% floor "
        f"over {e['floor_n_nulls']} null(s): margin "
        f"{e['margin_over_floor_points']:+.3f} points on a "
        f"{e['floor_spread_points']:.3f}-point spread -> wall clock "
        f"{'resolved' if e['wall_clock_resolved'] else 'NOT resolved'}"
    )


main()
