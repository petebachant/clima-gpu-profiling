"""Which kernels are register-limited, across the whole population?

CliMA/ClimaCore.jl#2676 asks whether putting parameters in `Val` improves
pointwise kernels across the board. Val can only pay where two things hold: the
fold reduces registers, and registers are what caps occupancy. The second is a
property of the kernels as they run, measurable without touching any code, and
it bounds the answer on its own -- a kernel whose occupancy is capped by block
size or shared memory gains nothing from cheaper registers, however well they
fold.

`results/kernel-resources.csv` cannot answer this: it profiles six kernels of
one file with `--set full`. This reads the wide static-metric export instead,
where ncu reports each occupancy limit in blocks per SM and the binding one is
simply the smallest.
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "results/ncu/pointwise-details.csv")
OUT_CSV = Path("results/pointwise-registers.csv")
OUT_JSON = Path("results/pointwise-registers.json")

# ncu's long-format export: one row per (kernel, metric).
METRICS = {
    "launch__registers_per_thread": "registers",
    "launch__occupancy_limit_registers": "limit_registers",
    "launch__occupancy_limit_warps": "limit_warps",
    "launch__occupancy_limit_shared_mem": "limit_shared_mem",
    "launch__occupancy_limit_blocks": "limit_blocks",
    "sm__maximum_warps_per_active_cycle_pct": "max_warps_pct",
    "launch__block_size": "block_size",
    "launch__thread_count": "thread_count",
}
LIMITS = ("limit_registers", "limit_warps", "limit_shared_mem", "limit_blocks")


def _num(s):
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


def read_rows(path):
    """Collapse ncu's long format into one record per kernel."""
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        # ncu prepends banner lines before the real header.
        lines = f.readlines()
    start = next(
        (i for i, l in enumerate(lines) if l.startswith('"ID"') or l.startswith("ID,")),
        0,
    )
    reader = csv.DictReader(lines[start:])
    kernels = {}
    for row in reader:
        name = row.get("Kernel Name") or row.get("Kernel")
        metric = row.get("Metric Name")
        if not name or metric not in METRICS:
            continue
        rec = kernels.setdefault(name, {"kernel": name, "launches": 0})
        field = METRICS[metric]
        value = _num(row.get("Metric Value"))
        if value is None:
            continue
        # A kernel appears once per profiled launch; these are static, so the
        # values repeat and the first is as good as any.
        rec.setdefault(field, value)
        if field == "registers":
            rec["launches"] += 1
    return list(kernels.values())


def classify(rec):
    """Name the occupancy limit that binds, which is the smallest of them."""
    present = {k: rec[k] for k in LIMITS if rec.get(k) is not None}
    if not present:
        return None, None
    binding = min(present, key=present.get)
    lowest = present[binding]
    # A tie means registers are not the sole cap, so cutting them alone moves
    # nothing; that distinction is the whole question here.
    tied = [k for k, v in present.items() if v == lowest]
    return binding.removeprefix("limit_"), len(tied) > 1


def main():
    rows = read_rows(SRC)
    if not rows:
        sys.exit(f"no kernels parsed from {SRC}")
    for rec in rows:
        limiter, tied = classify(rec)
        rec["limiter"] = limiter
        rec["limiter_tied"] = tied
        rec["register_limited"] = limiter == "registers"

    rows.sort(key=lambda r: (-(r.get("registers") or 0), r["kernel"]))
    fields = [
        "kernel", "registers", "limiter", "limiter_tied", "register_limited",
        "max_warps_pct", "block_size", "thread_count",
        "limit_registers", "limit_warps", "limit_shared_mem", "limit_blocks",
        "launches",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    reg_limited = [r for r in rows if r["register_limited"]]
    # Registers can only be the lever where they bind ALONE; a tie means
    # something else caps occupancy at the same point.
    sole = [r for r in reg_limited if not r["limiter_tied"]]
    at_cap = [r for r in rows if (r.get("registers") or 0) >= 255]
    full_occ = [r for r in rows if (r.get("max_warps_pct") or 0) >= 99]
    pointwise = [r for r in rows if re.search(r"cache|broadcast|copyto", r["kernel"], re.I)]
    pw_reg = [r for r in pointwise if r["register_limited"]]

    summary = {
        "source": str(SRC),
        "kernels": len(rows),
        "register_limited": len(reg_limited),
        "register_limited_pct": round(100 * len(reg_limited) / len(rows), 2),
        "register_limited_solely": len(sole),
        "register_limited_solely_pct": round(100 * len(sole) / len(rows), 2),
        "at_register_cap": len(at_cap),
        "at_full_occupancy": len(full_occ),
        "at_full_occupancy_pct": round(100 * len(full_occ) / len(rows), 2),
        "pointwise_kernels": len(pointwise),
        "pointwise_register_limited": len(pw_reg),
        "pointwise_register_limited_pct": (
            round(100 * len(pw_reg) / len(pointwise), 2) if pointwise else None
        ),
        "limiter_counts": dict(Counter(r["limiter"] for r in rows)),
        "max_registers": max((r.get("registers") or 0) for r in rows),
        "note": (
            "A kernel gains from cheaper registers only where registers are "
            "what caps its occupancy, and only where they cap it alone. "
            "register_limited_solely_pct is therefore the ceiling on how much "
            "of the population folding parameters into types could help."
        ),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    print(f"wrote {OUT_CSV} ({len(rows)} kernels) and {OUT_JSON}")
    print(f"  register-limited:        {len(reg_limited)}/{len(rows)} "
          f"({summary['register_limited_pct']}%)")
    print(f"  register-limited alone:  {len(sole)}/{len(rows)} "
          f"({summary['register_limited_solely_pct']}%)")
    print(f"  already at full occupancy: {len(full_occ)}")
    print(f"  limiters: {summary['limiter_counts']}")


if __name__ == "__main__":
    main()
