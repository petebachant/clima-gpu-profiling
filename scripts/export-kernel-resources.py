"""Export per-kernel register/spill/occupancy stats from the Nsight Compute reports.

The nsys-derived table (results/top-kernels.csv) carries timings and registers
per thread, but not spilling -- nsys does not collect it. That omission is
actively misleading for this project's headline result: the hot microphysics
kernel sits at the 255-register cap in BOTH cases, so the nsys table shows an
unchanged number while the kernel gets 54.8% faster. The mechanism is only
visible here: the baseline spills 24.5% of its memory traffic, the mod spills
none.

Reads the `*-details.csv` files Nsight Compute exports, so it needs no GPU and
no ncu install.
"""

import csv
import sys
from pathlib import Path

RESULTS = Path("results/ncu")
OUT = Path("results/kernel-resources.csv")

# Metric name in the ncu export -> column name here.
METRICS = {
    "Registers Per Thread": "registers",
    "Local Memory Spilling Requests": "spill_requests",
    "Local Memory Spilling Request Overhead": "spill_overhead_pct",
    "Stack Size": "stack_bytes",
    "Block Limit Registers": "block_limit_registers",
    "Achieved Occupancy": "achieved_occupancy_pct",
    "Theoretical Occupancy": "theoretical_occupancy_pct",
}


def load(tag):
    """Collapse the long-format ncu export into one record per kernel."""
    path = RESULTS / f"{tag}-details.csv"
    if not path.exists():
        sys.exit(f"missing {path}; run the ncu stages first")
    out = {}
    for row in csv.DictReader(path.open()):
        metric = METRICS.get(row["Metric Name"])
        if metric is None:
            continue
        # Kernel names embed the source path, which differs between the two
        # cases only by the `-mod` suffix on the package directory. Strip it so
        # the same kernel lines up across cases.
        name = row["Kernel Name"].replace("ClimaAtmos_jl_mod", "ClimaAtmos_jl")
        out.setdefault(name, {})[metric] = row["Metric Value"]
    return out


def num(x):
    """Parse a metric value, tolerating thousands separators.

    Nsight Compute is not consistent about these: the same metric exported as
    `25484544` from one run came back as `25,075,872` from the next. Without the
    strip, float() raises, the spill summary silently reports zero spilling
    kernels, and the one number this table exists to surface goes missing.
    """
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def main():
    base, mod = load("baseline"), load("mod")
    cols = list(METRICS.values())

    rows = []
    for name in sorted(set(base) | set(mod)):
        b, m = base.get(name, {}), mod.get(name, {})
        rec = {"kernel": name}
        for c in cols:
            rec[f"{c}_baseline"] = b.get(c, "")
            rec[f"{c}_mod"] = m.get(c, "")
        rows.append(rec)

    # Rank by baseline register pressure: the kernels at the cap are the ones
    # where spilling is even possible, so they belong at the top.
    rows.sort(key=lambda r: num(r["registers_baseline"]) or -1, reverse=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kernel"] + [
            f"{c}_{case}" for c in cols for case in ("baseline", "mod")
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {OUT} ({len(rows)} kernels)")
    spillers = [
        r for r in rows
        if (num(r["spill_requests_baseline"]) or 0) > 0
        or (num(r["spill_requests_mod"]) or 0) > 0
    ]
    print(f"kernels spilling in either case: {len(spillers)}")
    for r in spillers:
        short = r["kernel"].split("__FILE_")[0][:48]
        print(
            f"  {short:<50}"
            f" regs {r['registers_baseline']:>4} -> {r['registers_mod']:<4}"
            f" spill {r['spill_overhead_pct_baseline']:>6}%"
            f" -> {r['spill_overhead_pct_mod']:>6}%"
        )


if __name__ == "__main__":
    main()
