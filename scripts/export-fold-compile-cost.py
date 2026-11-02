"""What CliMA/ClimaAtmos.jl#4947's parameter fold costs in compile time.

The fold's benefit is registers, measured elsewhere. Its price is that a runtime
value in a type parameter makes a specialization no pkgimage can hold, so its
inference and codegen are paid at every start. `measure-fold-compile-cost.jl`
times exactly that, one arm per process so every compile is cold.

Two repeats per arm in reversed order, because a single pair of compile times is
one draw. The fold's delta is reported against the larger within-arm spread, and
`resolved` is false when it does not clear it -- the same rule this project
applies to a SYPD headline against its null census: a difference smaller than
the noise of the thing measuring it is not a measurement.

Only the modes that read the parameters are measured; the `Fixed` modes ignore
them by signature. `fold_relevant_in_benchmark` records which of those the
benchmark window actually selects, which is the figure that decides whether the
cost buys anything here at all.
"""

import argparse
import json
import statistics
import tomllib
from pathlib import Path

ARMS = ("unfolded", "folded")


def load(in_dir):
    runs = {}
    for arm in ARMS:
        for path in sorted(Path(in_dir).glob(f"{arm}-*.toml")):
            with open(path, "rb") as f:
                runs.setdefault(arm, []).append(tomllib.load(f))
    for arm in ARMS:
        if arm not in runs:
            raise SystemExit(f"no runs found for arm {arm!r} in {in_dir}")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runs = load(args.in_dir)
    ref = runs["unfolded"][0]
    paths = ref["paths_measured"]

    # A compile time is only comparable against one taken the same way, so the
    # two arms must have measured the same paths in the same float type.
    for arm, rs in runs.items():
        for r in rs:
            if r["paths_measured"] != paths:
                raise SystemExit(f"{arm} measured different paths: {r['paths_measured']}")
            if r["float_type"] != ref["float_type"]:
                raise SystemExit(f"{arm} measured a different float type")

    out = {
        "float_type": ref["float_type"],
        "gpu": ref["gpu"],
        "julia": ref["julia"],
        "repeats_per_arm": {a: len(rs) for a, rs in runs.items()},
        "benchmark_modes": ref["benchmark_modes"],
        "fold_relevant_in_benchmark": ref["fold_relevant_in_benchmark"],
        "paths": {},
        "note": ref["note"],
    }

    spreads = []
    for arm in ARMS:
        totals = [r["total_compile_s"] for r in runs[arm]]
        spread = max(totals) - min(totals)
        spreads.append(spread)
        out[f"{arm}_total_compile_s"] = round(statistics.fmean(totals), 3)
        out[f"{arm}_total_spread_s"] = round(spread, 3)

    # The floor is the WORSE of the two arms' own repeat spreads: the quantity
    # has to clear the noise of whichever arm is noisier, not the average.
    floor = max(spreads)
    delta = out["folded_total_compile_s"] - out["unfolded_total_compile_s"]
    out["fold_total_cost_s"] = round(delta, 3)
    out["fold_total_cost_pct"] = round(
        100.0 * delta / out["unfolded_total_compile_s"], 2
    )
    out["repeat_spread_floor_s"] = round(floor, 3)
    out["resolved"] = abs(delta) > floor

    for i, name in enumerate(paths):
        row = {}
        for arm in ARMS:
            ts = [r[name]["compile_s"] for r in runs[arm]]
            row[f"{arm}_compile_s"] = round(statistics.fmean(ts), 3)
            row[f"{arm}_spread_s"] = round(max(ts) - min(ts), 3)
            row[f"{arm}_registers"] = runs[arm][0][name]["registers"]
            row[f"{arm}_warps_per_sm"] = runs[arm][0][name]["warps_per_sm"]
        row["compile_cost_s"] = round(
            row["folded_compile_s"] - row["unfolded_compile_s"], 3
        )
        row["registers_saved"] = (
            row["unfolded_registers"] - row["folded_registers"]
        )
        row["gains_occupancy_step"] = (
            row["folded_warps_per_sm"] > row["unfolded_warps_per_sm"]
        )
        row["in_benchmark"] = name in out["fold_relevant_in_benchmark"]
        # The first path measured in a process pays whatever the warmup kernel
        # did not, so it is flagged rather than silently averaged in.
        row["measured_first"] = i == 0
        out["paths"][name] = row

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"unfolded {out['unfolded_total_compile_s']:.3f} s "
          f"-> folded {out['folded_total_compile_s']:.3f} s "
          f"({out['fold_total_cost_s']:+.3f} s, "
          f"{out['fold_total_cost_pct']:+.2f}%)")
    print(f"repeat spread floor {out['repeat_spread_floor_s']:.3f} s; "
          f"resolved={out['resolved']}")
    for name, row in out["paths"].items():
        print(f"  {name:<16} {row['compile_cost_s']:+7.3f} s  "
              f"regs {row['unfolded_registers']:>3} -> "
              f"{row['folded_registers']:<3} "
              f"{'(in benchmark)' if row['in_benchmark'] else ''}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
