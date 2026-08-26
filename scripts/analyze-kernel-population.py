"""Characterise the GPU kernel population from the Nsight Systems SQLite exports.

The per-kernel tables answer "which kernel is slowest". They cannot answer "are
there too many kernels", "how much of the time is the GPU idle", or "how much of
the work sits in kernels too small to be worth launching" -- and those turned out
to matter more than kernel time for this model. This derives all three from the
nsys timeline, for both arms, so the answer is reproducible rather than a
one-off query.

Idle is computed from the union of kernel intervals, not from summed durations:
summing double-counts nothing but silently omits any kernel a name-based summary
misses, which is how an earlier hand analysis concluded the GPU was ~50% idle
when the timeline says ~31%.
"""

import json
import re
import sqlite3
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Coupler steps captured per profiling run; see scripts/run.jl.
N_STEPS = 10
# Buckets chosen around the ~7.5 us host cost of issuing a launch: anything in
# the first two is cheaper to run than to launch.
BUCKETS = [(0, 2), (2, 5), (5, 10), (10, 25), (25, 100), (100, 1000), (1000, None)]
SUBSYSTEMS = {
    "spectral_element": r"spectral|divergence|gradient|curl|hyperdiffusion|tracer_advection|Interpolate|Restrict",
    "dss": r"dss",
    "matrix_field_solve": r"field_matrix_solver|single_field_solve|multiple_field_solve",
    "microphysics_cache": r"microphysics_cache",
    "generic_broadcast": r"gpu_broadcast_kernel|copyto_foreach|copyto__",
}


def analyse(db_path):
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    names = {r[0]: r[1] for r in cur.execute("SELECT id, value FROM StringIds")}
    rows = cur.execute(
        "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ).fetchall()
    con.close()
    if not rows:
        raise RuntimeError(f"no kernel rows in {db_path}")

    span = rows[-1][1] - rows[0][0]
    durs_us = [(e - s) / 1e3 for s, e, _ in rows]
    total_ns = sum(e - s for s, e, _ in rows)

    # Union of intervals -> genuine busy time, robust to any overlap.
    busy = 0
    cur_s, cur_e = rows[0][0], rows[0][1]
    for s, e, _ in rows[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s

    hist = []
    for lo, hi in BUCKETS:
        sel = [d for d in durs_us if d >= lo and (hi is None or d < hi)]
        if not sel:
            continue
        hist.append(
            {
                "min_us": lo,
                "max_us": hi,
                "launches": len(sel),
                "pct_of_launches": 100 * len(sel) / len(rows),
                "total_ms": sum(sel) / 1e3,
                "pct_of_kernel_time": 100 * sum(sel) / sum(durs_us),
            }
        )

    count, time = Counter(), Counter()
    for s, e, sn in rows:
        k = names.get(sn, "?")
        count[k] += 1
        time[k] += e - s

    claimed, subsystems = set(), {}
    for label, pattern in SUBSYSTEMS.items():
        ks = [k for k in count if re.search(pattern, k, re.I) and k not in claimed]
        claimed |= set(ks)
        subsystems[label] = {
            "launches": sum(count[k] for k in ks),
            "total_ms": sum(time[k] for k in ks) / 1e6,
            "pct_of_kernel_time": 100 * sum(time[k] for k in ks) / total_ns,
        }
    rest = [k for k in count if k not in claimed]
    subsystems["other"] = {
        "launches": sum(count[k] for k in rest),
        "total_ms": sum(time[k] for k in rest) / 1e6,
        "pct_of_kernel_time": 100 * sum(time[k] for k in rest) / total_ns,
    }

    return {
        "launches": len(rows),
        "launches_per_step": len(rows) / N_STEPS,
        "distinct_kernels": len(count),
        "span_s": span / 1e9,
        "gpu_busy_s": busy / 1e9,
        "gpu_utilisation_pct": 100 * busy / span,
        "gpu_idle_pct": 100 * (1 - busy / span),
        "kernel_time_ms": total_ns / 1e6,
        "duration_histogram": hist,
        "subsystems": subsystems,
        "top_by_launch_count": [
            {"kernel": k, "launches": c, "total_ms": time[k] / 1e6,
             "mean_us": time[k] / c / 1e3}
            for k, c in count.most_common(10)
        ],
    }


def main():
    out = {}
    for arm in ("baseline", "mod"):
        db = ROOT / "results" / "nsys" / f"{arm}.sqlite"
        if db.exists():
            out[arm] = analyse(db)
    dest = ROOT / "results" / "kernel-population.json"
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    for arm, d in out.items():
        small = sum(
            b["launches"] for b in d["duration_histogram"] if b["max_us"] and b["max_us"] <= 25
        )
        small_t = sum(
            b["pct_of_kernel_time"] for b in d["duration_histogram"]
            if b["max_us"] and b["max_us"] <= 25
        )
        print(
            f"{arm}: {d['launches']:,} launches ({d['launches_per_step']:,.0f}/step), "
            f"{d['distinct_kernels']} distinct, GPU idle {d['gpu_idle_pct']:.1f}%; "
            f"{small:,} launches under 25us = {small_t:.1f}% of kernel time"
        )
    print(f"wrote {dest.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
