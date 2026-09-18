"""Extract host-side launch overhead from an nsys profile.

Every number this project has published about launch overhead was computed in a
throwaway script, which is exactly the habit that put unverifiable literals into
docs/learnings.md. These belong in the pipeline, keyed, so prose can cite them
and `scripts/verify-evidence.py` can check the citation.

Reports, per profile: launch count, GPU busy time, host time in sub-millisecond
gaps between kernels, and the CUDA driver calls issued per kernel launch. With
two profiles whose launch counts differ, also reports the marginal host cost per
launch as a slope, which is the only defensible way to price a launch --- the
mean gap includes Julia work between broadcasts and overstates what removing a
launch would return.
"""

import argparse
import json
import sqlite3


def profile_stats(db: str) -> dict:
    con = sqlite3.connect(db)
    rows = con.execute(
        "SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start"
    ).fetchall()
    launches = len(rows)
    busy_ms = sum(e - s for s, e in rows) / 1e6
    gap_ms = 0.0
    gap_count = 0
    prev = rows[0][1]
    for s, e in rows[1:]:
        if s > prev and (s - prev) < 1e6:  # sub-millisecond: not a phase change
            gap_ms += (s - prev) / 1e6
            gap_count += 1
        prev = max(prev, e)
    api = {}
    for name, n, ms in con.execute(
        "SELECT s.value, COUNT(*), SUM(r.end - r.start) / 1e6 "
        "FROM CUPTI_ACTIVITY_KIND_RUNTIME r "
        "JOIN StringIds s ON r.nameId = s.id GROUP BY s.value"
    ):
        api[name] = {
            "calls": n,
            "ms": round(ms, 1),
            "calls_per_launch": round(n / launches, 2),
            "us_per_launch": round(ms * 1000 / launches, 2),
        }
    small = con.execute(
        "SELECT COUNT(*) FROM (SELECT s.value, COUNT(*) n, "
        "SUM(k.end - k.start) ns FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName = s.id GROUP BY s.value "
        "HAVING ns / n < 25000)"
    ).fetchone()[0]
    small_launches = con.execute(
        "SELECT SUM(n) FROM (SELECT s.value, COUNT(*) n, "
        "SUM(k.end - k.start) ns FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.demangledName = s.id GROUP BY s.value "
        "HAVING ns / n < 25000)"
    ).fetchone()[0]
    return {
        "small_kernel_threshold_us": 25,
        "launches": launches,
        "gpu_busy_ms": round(busy_ms, 1),
        "host_gap_ms": round(gap_ms, 1),
        "gap_count": gap_count,
        "mean_gap_us": round(gap_ms * 1000 / gap_count, 2) if gap_count else None,
        "sub25us_kernel_kinds": small,
        "sub25us_launches": small_launches,
        "api": {
            k: v
            for k, v in sorted(api.items(), key=lambda kv: -kv[1]["ms"])[:8]
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--probe", help="profile with deliberately added launches")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    out = {"steps": a.steps, "baseline": profile_stats(a.baseline)}
    b = out["baseline"]
    out["baseline"]["launches_per_step"] = round(b["launches"] / a.steps, 1)
    out["baseline"]["gpu_busy_ms_per_step"] = round(b["gpu_busy_ms"] / a.steps, 1)
    out["baseline"]["host_gap_ms_per_step"] = round(b["host_gap_ms"] / a.steps, 1)

    if a.probe:
        pr = profile_stats(a.probe)
        out["probe"] = pr
        d_n = pr["launches"] - b["launches"]
        d_gap = pr["host_gap_ms"] - b["host_gap_ms"]
        d_busy = pr["gpu_busy_ms"] - b["gpu_busy_ms"]
        if d_n > 0:
            out["marginal"] = {
                "added_launches": d_n,
                "added_host_gap_ms": round(d_gap, 1),
                "added_gpu_busy_ms": round(d_busy, 1),
                "host_us_per_launch": round(d_gap * 1000 / d_n, 2),
                "wall_us_per_launch": round((d_gap + d_busy) * 1000 / d_n, 2),
            }
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
