"""Export per-kernel register/spill/occupancy stats from the Nsight Compute reports.

The nsys-derived table (results/top-kernels.csv) carries timings and registers
per thread, but not spilling -- nsys does not collect it. That omission can hide
the mechanism behind a result: a kernel pinned at the 255-register cap in BOTH
arms shows an unchanged number in the nsys table while its spill traffic, and so
its speed, moves a lot.

Do NOT read a spill delta out of this file without checking the provenance
header it now writes. The two ncu stages are frozen independently, so the
`_baseline` and `_mod` columns can come from different revisions -- they did
between 2026-08-20 and 2026-08-23 -- in which case the side-by-side comparison
is two unrelated runs and means nothing.

Reads the `*-details.csv` files Nsight Compute exports, so it needs no GPU and
no ncu install.
"""

import csv
import json
import subprocess
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


def provenance(tag):
    """Which project state does this arm's ncu export actually describe?

    The two ncu stages are frozen independently and are expensive, so in
    practice they get refreshed at different times -- on 2026-08-20 and
    2026-08-23 respectively, three days and one experiment apart. The table then
    puts `_baseline` and `_mod` side by side as though they were one comparison
    when they were two unrelated runs, and a copy shared outside this repo
    carries no way to notice. So stamp each arm with the commit that last wrote
    its export and the treatment recorded there.
    """
    path = RESULTS / f"{tag}-details.csv"

    def git(*args):
        return subprocess.run(
            ["git", *args], capture_output=True, text=True
        ).stdout.strip()

    # A freshly regenerated export is not committed yet, so `git log` on it
    # reports whichever OLD commit last touched it -- stamping new data with a
    # stale revision, which is worse than no stamp at all. Check dirtiness
    # first and read the live treatment.json in that case.
    dirty = bool(git("status", "--porcelain", "--", str(path)))
    if dirty:
        sha = "UNCOMMITTED (working tree)"
        treat = Path("results/treatment.json").read_text() if Path(
            "results/treatment.json"
        ).exists() else ""
    else:
        sha = git("log", "-1", "--format=%h %cs", "--", str(path)) or "unknown"
        treat = git("show", f"{sha.split()[0]}:results/treatment.json")

    differs, subs = "unknown", {}
    try:
        t = json.loads(treat)
        differs = t.get("arms_differ_in")
        differs = (
            "null test (arms identical)" if differs == []
            else ", ".join(differs or []) or "unrecorded"
        )
        subs = {k: v.get("sha", "")[:9] for k, v in (t.get("submodules") or {}).items()}
    except Exception:
        pass
    return sha, differs, subs


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
    pb, pm = provenance("baseline"), provenance("mod")
    with OUT.open("w", newline="") as f:
        # Provenance first, as comment lines, so a copy of this file carries its
        # own answer to "which revision is this?" without needing the repo.
        f.write(f"# baseline arm: written by {pb[0]}, treatment: {pb[1]}\n")
        f.write(f"# mod arm:      written by {pm[0]}, treatment: {pm[1]}\n")
        if pb[0] != pm[0] and "UNCOMMITTED" not in pb[0] + pm[0]:
            f.write("# WARNING: the two arms were exported by DIFFERENT commits;"
                    " the _baseline vs _mod comparison is NOT a single run.\n")
        for arm, prov in (("baseline", pb), ("mod", pm)):
            for k, v in sorted(prov[2].items()):
                f.write(f"# {arm} {k}: {v}\n")
        w = csv.DictWriter(f, fieldnames=["kernel"] + [
            f"{c}_{case}" for c in cols for case in ("baseline", "mod")
        ])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {OUT} ({len(rows)} kernels)")
    print(f"  baseline arm from {pb[0]}  [{pb[1]}]")
    print(f"  mod arm      from {pm[0]}  [{pm[1]}]")
    if pb[0] != pm[0] and "UNCOMMITTED" not in pb[0] + pm[0]:
        print("  WARNING: arms exported by different commits; the side-by-side"
              " comparison in this file is not a single run")
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
