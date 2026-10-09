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
    # First, so it claims the rte_* kernels before any other pattern can.
    "radiation": r"^rte_",
    "spectral_element": r"spectral|divergence|gradient|curl|hyperdiffusion|tracer_advection|Interpolate|Restrict",
    "dss": r"dss",
    # MatrixFields covers field_name_dict, which is the implicit solve's
    # largest kernel and was landing in `other`.
    "matrix_field_solve": r"field_matrix_solver|single_field_solve|multiple_field_solve|MatrixFields",
    "edmf": r"edmf",
    "microphysics_cache": r"microphysics_cache",
    "generic_broadcast": r"gpu_broadcast_kernel|copyto_foreach|copyto__",
}


def analyze(db_path):
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    names = {r[0]: r[1] for r in cur.execute("SELECT id, value FROM StringIds")}
    rows = cur.execute(
        "SELECT start, end, shortName FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ).fetchall()
    # Radiation is split by the demangled signature rather than by cost: with
    # `rad: allskywithclear` the solver runs twice, and the clear-sky pass is
    # the specialization with no cloud lookup, i.e. no LookUpCld argument.
    rad_rows = cur.execute(
        """SELECT s.value, COUNT(*), SUM(k.end - k.start)
           FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON k.demangledName = s.id
           WHERE s.value LIKE 'rte\\_%' ESCAPE '\\'
           GROUP BY s.value"""
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

    # `other` is a residual, so a large one hides the answer to "what is the
    # bottleneck" rather than reporting it. Name its biggest members.
    other_top = [
        {
            "kernel": k,
            "launches": count[k],
            "total_ms": round(time[k] / 1e6, 1),
            "mean_us": round(time[k] / count[k] / 1e3, 1),
            "pct_of_kernel_time": round(100 * time[k] / total_ns, 2),
        }
        for k in sorted(rest, key=lambda k: -time[k])[:15]
    ]

    # How few kernels carry the device: a single bottleneck and a broad
    # population give very different numbers here.
    def concentration():
        ranked = sorted(time.values(), reverse=True)
        out, run = {}, 0
        for i, ns in enumerate(ranked, 1):
            run += ns
            for frac in (50, 80, 90):
                key = f"kernels_for_{frac}pct"
                if key not in out and 100 * run / total_ns >= frac:
                    out[key] = i
        out["distinct_kernels"] = len(ranked)
        out["top1_pct"] = round(100 * ranked[0] / total_ns, 2)
        out["top2_pct"] = round(100 * sum(ranked[:2]) / total_ns, 2)
        out["top10_pct"] = round(100 * sum(ranked[:10]) / total_ns, 2)
        return out

    def radiation_split(ns_total):
        groups = {"clear_sky": [], "all_sky": []}
        for name, n, ns in rad_rows:
            groups["all_sky" if "LookUpCld" in name else "clear_sky"].append((n, ns))
        out = {}
        for label, items in groups.items():
            out[label] = {
                "kernels": len(items),
                "launches": sum(n for n, _ in items),
                "total_ms": round(sum(ns for _, ns in items) / 1e6, 1),
                "pct_of_kernel_time": round(
                    100 * sum(ns for _, ns in items) / ns_total, 2
                ),
            }
        out["pct_of_kernel_time"] = round(
            out["clear_sky"]["pct_of_kernel_time"]
            + out["all_sky"]["pct_of_kernel_time"],
            2,
        )
        return out

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
        "other_top": other_top,
        "concentration": concentration(),
        "radiation": radiation_split(total_ns),
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
            out[arm] = analyze(db)
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
