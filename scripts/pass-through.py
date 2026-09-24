"""How much of a GPU kernel-time saving reaches SYPD, per tagged experiment.

Both numbers are read out of each tag with `git show`, so the table is a
calculation over the recorded experiments rather than figures copied forward:
kernel time from the profiled nsys stages, SYPD from the unprofiled AMIP ones.
"""

import json
import subprocess

# Every tag that recorded both a kernel population and an end-to-end SYPD.
# `subsystem` is where the treatment did its work, which is what the ratios
# turn out to separate.
EXPERIMENTS = [
    ("exp/2026-09-01-cm-fuse", "microphysics", "1M source terms fused into the linearization"),
    ("exp/2026-09-02-deps-and-fuse", "microphysics", "the same fusion on the updated stack"),
    ("exp/2026-09-12-rrtmgp-optics", "radiation", "aerosol-optics binary search plus 64-thread blocks"),
    ("meas/2026-09-13-binary-search-alone", "radiation", "the binary search alone"),
    ("exp/2026-09-22-clearsky-bound", "radiation", "the clear-sky solves removed"),
    ("exp/2026-09-23-fused-lw-optics", "radiation", "the longwave solves fused"),
    ("exp/2026-09-24-fused-radiation", "radiation", "both bands fused"),
]


def at(tag, path):
    return json.loads(subprocess.check_output(["git", "show", f"{tag}:{path}"]))


runs = []
for tag, subsystem, description in EXPERIMENTS:
    summary = at(tag, "results/summary.json")
    population = at(tag, "results/kernel-population.json")
    base = population["baseline"]["kernel_time_ms"]
    kernel_pct = 100 * (population["mod"]["kernel_time_ms"] - base) / base
    sypd_pct = summary["speedup_pct"]
    runs.append({
        "tag": tag,
        "subsystem": subsystem,
        "description": description,
        "sypd_pct": round(sypd_pct, 2),
        "kernel_time_pct": round(kernel_pct, 2),
        # Undefined unless the treatment actually removed kernel time
        "pass_through_pct": round(100 * sypd_pct / -kernel_pct, 1) if kernel_pct < 0 else None,
    })

by_subsystem = {}
for subsystem in sorted({r["subsystem"] for r in runs}):
    ratios = [r["pass_through_pct"] for r in runs if r["subsystem"] == subsystem and r["pass_through_pct"]]
    by_subsystem[subsystem] = {
        "n": len(ratios),
        "min_pct": min(ratios),
        "max_pct": max(ratios),
        "mean_pct": round(sum(ratios) / len(ratios), 1),
    }

json.dump(
    {"runs": runs, "by_subsystem": by_subsystem},
    open("results/pass-through.json", "w"),
    indent=2,
)
