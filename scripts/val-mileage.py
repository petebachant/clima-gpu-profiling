"""How much device time can the Val parameter fold reach at all?

CliMA/ClimaCore.jl#2676 asks whether folding parameters helps "across the
board". results/pointwise-registers.json already bounds the POPULATION: of the
kernels profiled, some fraction is capped by registers alone and below full
occupancy, and the rest can gain nothing however well its parameters fold. But a
count of kernels is not a bound on time, and time is what SYPD is made of: 93
addressable kernels that together run for 2% of the device are worth less than
one unaddressable kernel that runs for 20%.

So this joins the two halves. Expense comes from the nsys profile, which times
every kernel in the window; addressability comes from the ncu static pass, which
records each kernel's occupancy limiter. The product is the ceiling: the share of
device time held by kernels where a register saving COULD raise occupancy.

That ceiling is an upper bound and nothing more. Being addressable means a
register saving could buy a block per SM, not that folding parameters saves
enough registers, nor that a block per SM makes the kernel proportionally
faster. One measured point relates the two -- the quadrature kernel crossed one
step and its time fell about half -- and one point is not a model, so this script
deliberately reports no predicted SYPD. Converting a device-time share into wall
clock goes through the pass-through ratio and the step distribution, which the
pass-through and step-breakdown stages measure separately.

Both inputs must describe the same arm AND the same revision, and a name match
does not establish either. The ncu pass profiles the mod arm; a frozen copy of it
once joined cleanly onto a fresh BASELINE profile because the hot kernel happened
to sit at the same source line in both, welding one arm's register count to the
other's time with nothing in the row to show it. So the two sources are made to
check each other: nsys records registersPerThread per kernel, ncu records
registers. One kernel NAME can cover several compiled specializations at
different register counts, so the test is not equality against a single number:
ncu's count must appear SOMEWHERE among the counts nsys recorded for that name.

The test is also weighted rather than absolute. A mismatched pair of builds
disagrees about nearly everything -- the case that prompted this guard disagreed
on 89 of 98 shared kernels, including the hottest -- while two legitimate
profiles of one build can still differ on a kernel or two, because the tools do
not report register allocation identically (the ClimaCore DSS kernels come back
as 32 from ncu and 30 from nsys). So the refusal triggers on the SHARE OF DEVICE
TIME held by disagreeing kernels, not their count: above the threshold nothing is
written, and below it they are dropped from the analysis and named in the output,
never silently kept.
"""

import csv
import json
import sqlite3
import sys
from pathlib import Path

NSYS = Path("results/nsys/mod.sqlite")
POP = Path("results/pointwise-registers.csv")
OUT = Path("results/val-mileage.json")
TOP_N = 25
# Below this share of device time accounted for, nothing can be bounded.
MIN_COVERAGE_PCT = 25.0

# The kernels that already carry the fold, by the same name patterns
# scripts/export-param-fold.py selects on. The ceiling counts them, because they
# are still register-limited and below full occupancy after the fold -- so the
# interesting figure is the ceiling with them taken out, which is what is left
# for anyone applying the pattern somewhere new.
TREATED = (
    "set_microphysics_tendency_cache",
    "set_cloud_fraction",
    "set_sgs_moments_and_cloud_fraction",
)

# A100 sm_80, per SM. Same constants as scripts/export-param-fold.py.
REGS_PER_SM = 65536
MAX_WARPS_PER_SM = 64
MAX_BLOCKS_PER_SM = 32

Q_PER_KERNEL = """
SELECT s.value, k.registersPerThread, k.blockX * k.blockY * k.blockZ,
       SUM(k.end - k.start), COUNT(*)
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds s ON k.demangledName = s.id
GROUP BY s.value, k.registersPerThread, k.blockX * k.blockY * k.blockZ
"""


def occupancy(registers, block_size):
    """Warps resident per SM, as ncu's occupancy limits compute it."""
    if not registers or not block_size:
        return None
    blocks = min(REGS_PER_SM // (block_size * registers), MAX_BLOCKS_PER_SM)
    return min(blocks * (block_size // 32), MAX_WARPS_PER_SM)


def next_step_registers(registers, block_size):
    """The register count the kernel must reach for one more block per SM."""
    if not registers or not block_size:
        return None
    blocks = REGS_PER_SM // (block_size * registers)
    if blocks < 1:
        return None
    target = REGS_PER_SM // (block_size * (blocks + 1))
    return target if target >= 1 else None


def norm(name):
    """A key that joins the two arms' spellings of one kernel.

    The `-mod` submodules appear in the mangled name as `ClimaAtmos_jl_mod_src`.
    The trailing `_L<line>` is KEPT: several distinct kernels share one function
    and file and differ only by line, so stripping it would collapse them and
    average away whichever one a treatment touched.
    """
    return name.replace("_mod_", "_")


def truthy(x):
    return str(x).strip().lower() in ("true", "1", "yes")


def main():
    for p in (NSYS, POP):
        if not p.exists():
            sys.exit(f"missing {p}")

    conn = sqlite3.connect(NSYS)
    timed = {}
    for name, regs, block, total_ns, launches in conn.execute(Q_PER_KERNEL):
        key = (norm(name), regs or 0, block or 0)
        rec = timed.setdefault(key, {"total_ns": 0, "launches": 0})
        rec["total_ns"] += total_ns or 0
        rec["launches"] += launches or 0
    conn.close()
    device_ns = sum(r["total_ns"] for r in timed.values())
    if device_ns == 0:
        sys.exit(f"{NSYS}: no kernel time recorded")

    pop = {}
    for row in csv.DictReader(POP.open()):
        regs = int(float(row["registers"]))
        block = int(float(row["block_size"]))
        warps = occupancy(regs, block)
        pop[(norm(row["kernel"]), regs, block)] = {
            "registers": regs,
            "block_size": block,
            "limiter": row["limiter"],
            "warps_per_sm": warps,
            "max_warps_pct": float(row["max_warps_pct"]),
            # Addressable exactly as results/pointwise-registers.json defines
            # it: capped by registers ALONE and not already at full occupancy.
            "addressable": (
                row["limiter"] == "registers"
                and not truthy(row["limiter_tied"])
                and float(row["max_warps_pct"]) < 100
            ),
            "registers_for_next_step": next_step_registers(regs, block),
        }

    # The ncu pass and the nsys window profile the same configuration but not
    # the same launches, so some names appear in only one. Coverage is reported
    # rather than assumed: a ceiling computed over half the device would be
    # wrong in a direction nothing in the number would reveal.
    matched = {k: v for k, v in pop.items() if k in timed}

    # The provenance guard. A kernel's register allocation is a property of the
    # build, so the two profiles must report the same number for it.
    # Coverage IS the provenance check now. Registers and block are part of the
    # key, so two profiles of different builds cannot match on the kernels whose
    # code moved: the match rate collapses instead of producing confident rows.
    # Low coverage also follows from ncu sampling only its launch budget, and this
    # cannot tell the two apart -- so it reports the share and refuses only when
    # too little of the device is accounted for to bound anything.

    matched_ns = sum(timed[k]["total_ns"] for k in matched)
    if 100 * matched_ns / device_ns < MIN_COVERAGE_PCT:
        sys.exit(
            f"refusing to write {OUT}: only "
            f"{100 * matched_ns / device_ns:.1f}% of device time is covered by "
            f"{POP} (need {MIN_COVERAGE_PCT}%); the inputs may describe "
            "different builds, or ncu's launch budget missed too much"
        )
    addressable_ns = sum(
        timed[k]["total_ns"] for k, v in matched.items() if v["addressable"]
    )
    full_occ_ns = sum(
        timed[k]["total_ns"]
        for k, v in matched.items()
        if v["max_warps_pct"] >= 100
    )

    rows = []
    for key, v in matched.items():
        if not v["addressable"]:
            continue
        t = timed[key]
        rows.append({
            "kernel": key[0][:120],
            "total_ms": round(t["total_ns"] / 1e6, 3),
            "pct_of_device": round(100 * t["total_ns"] / device_ns, 3),
            "launches": t["launches"],
            "registers": v["registers"],
            "block_size": v["block_size"],
            "warps_per_sm": v["warps_per_sm"],
            "registers_for_next_step": v["registers_for_next_step"],
            "registers_to_shed": (
                v["registers"] - v["registers_for_next_step"]
                if v["registers_for_next_step"] else None
            ),
        })
    rows.sort(key=lambda r: r["total_ms"], reverse=True)

    for r in rows:
        r["already_folded"] = any(t in r["kernel"] for t in TREATED)
    treated_ms = sum(r["total_ms"] for r in rows if r["already_folded"])
    untreated = [r for r in rows if not r["already_folded"]]

    out = {
        "source_nsys": str(NSYS),
        "source_population": str(POP),
        "device_total_ms": round(device_ns / 1e6, 1),
        "kernels_timed": len(timed),
        "kernels_in_population": len(pop),
        "kernels_matched": len(matched),
        "matched_pct_of_device": round(100 * matched_ns / device_ns, 2),
        "ceiling_pct_of_device": round(100 * addressable_ns / device_ns, 2),
        "ceiling_is_lower_bound": matched_ns < device_ns,
        "addressable_ms": round(addressable_ns / 1e6, 1),
        "addressable_kernels": sum(1 for v in matched.values() if v["addressable"]),
        "unmatched_pct_of_device": round(100 - 100 * matched_ns / device_ns, 2),
        "treated_pct_of_device": round(100 * treated_ms * 1e6 / device_ns, 2),
        "untreated_pct_of_device": round(
            sum(r["pct_of_device"] for r in untreated), 2),
        "untreated_kernels": len(untreated),
        "largest_untreated_pct_of_device": (
            untreated[0]["pct_of_device"] if untreated else 0.0),
        "largest_untreated_kernel": (
            untreated[0]["kernel"].split("__FILE_")[0] if untreated else None),
        "largest_untreated_registers_to_shed": (
            untreated[0]["registers_to_shed"] if untreated else None),
        "at_full_occupancy_pct_of_device": round(100 * full_occ_ns / device_ns, 2),
        "note": (
            "ceiling_pct_of_device is the share of device time held by kernels "
            "capped by registers alone and below full occupancy. It is an upper "
            "bound on what cheaper registers could reach, NOT a predicted saving: "
            "it assumes nothing about how many registers a fold sheds or what a "
            "block per SM is worth. No SYPD figure is derived here."
        ),
        "candidates": rows[:TOP_N],
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as f:
        json.dump(out, f, indent=2)

    print(f"wrote {OUT}")
    print(f"  device total            {out['device_total_ms']:.1f} ms")
    print(f"  kernels timed/pop/match {out['kernels_timed']}/"
          f"{out['kernels_in_population']}/{out['kernels_matched']}"
          f"  ({out['matched_pct_of_device']:.1f}% of device time matched)")
    print(f"  addressable             {out['addressable_kernels']} kernels, "
          f"{out['addressable_ms']:.1f} ms")
    print(f"  CEILING                 {out['ceiling_pct_of_device']:.2f}% of device time")
    print(f"    of which already folded {out['treated_pct_of_device']:.2f}%, "
          f"leaving {out['untreated_pct_of_device']:.2f}% over "
          f"{out['untreated_kernels']} kernels")
    print(f"  already full occupancy  {out['at_full_occupancy_pct_of_device']:.2f}% of device time")
    print()
    print(f"  {'ms':>9} {'%dev':>6} {'regs':>5} {'blk':>5} {'warps':>5} "
          f"{'next':>5} {'shed':>5}  kernel")
    for r in rows[:TOP_N]:
        nxt = r["registers_for_next_step"]
        shed = r["registers_to_shed"]
        print(f"  {r['total_ms']:>9.1f} {r['pct_of_device']:>6.2f} "
              f"{r['registers']:>5} {r['block_size']:>5} {r['warps_per_sm']:>5} "
              f"{nxt if nxt else '-':>5} {shed if shed else '-':>5}  "
              f"{r['kernel'].split('__FILE_')[0][:44]}")


if __name__ == "__main__":
    main()
