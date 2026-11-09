#!/usr/bin/env python3
"""What the 2026-11-03 kernel fusion actually cost, separated from the physics
that landed in the same pull.

`results/regression-record/attribution-1103.json` records every function's time
on both sides of that pull. Its `fusion_trade.all_other_growth_per_ms_saved`
charges the fusion every other change in the window -- including the brand-new
implicit 1M refresh -- and so reports a ratio nearly four times the defensible
one. This recomputes the ledger from the same committed function list, which is
why it needs no profile: the snapshot is the durable record and the 650 MB
sqlite it came from has been garbage-collected.

The aggregates matter because a fusion CHANGES THE KERNEL INVENTORY, which is
the one case where name-paired deltas mislead (AGENTS.md). Fusion moves work out
of the broadcast machinery into compute kernels, so a receiver's time rises while
a removed kernel's vanishes; only the net over survivors plus removals means
anything. New NAMED functions are new capability. New functions named for a FILE
AND LINE are ambiguous, because a pull that moves source lines renames exactly
those -- so they bound the answer rather than decide it.
"""
import json
import sys

GENERIC = (
    "copyto_by_slices",
    "gpu_broadcast_kernel",
    "partial_mapreduce",
    "knl_copyto_flat",
    "broadcast_kernel",
)


def bucket(r):
    f = r["function"]
    if f.startswith(GENERIC):
        return "broadcast"
    if "rte_" in f:
        # Byte-identical code, identical launches and registers: a cloud-state
        # difference, not something the pull did.
        return "radiation"
    if r["pre_ms"] is None:
        return "new_anonymous" if f.startswith("_FILE_") else "new_named"
    return "existing"


def main(src, out):
    d = json.load(open(src))
    rows = d["functions"]
    grew = d["device"]["grew_ms"]

    tally, counts = {}, {}
    for r in rows:
        b = bucket(r)
        tally[b] = tally.get(b, 0.0) + r["delta_ms"]
        counts[b] = counts.get(b, 0) + 1

    saved = -tally.get("broadcast", 0.0)
    bill_lo = tally.get("existing", 0.0)
    bill_hi = bill_lo + tally.get("new_anonymous", 0.0)
    named = tally.get("new_named", 0.0)

    doc = {
        "source": src,
        "source_commit": d["provenance"]["source_commit"],
        "window_grew_ms": grew,
        "window_grew_pct": d["device"]["grew_pct"],
        "launches_change_pct": d["device"]["launches_change_pct"],
        "broadcast_machinery_saved_ms": round(saved, 1),
        "receivers_net_ms": round(bill_lo, 1),
        "new_anonymous_ms": round(tally.get("new_anonymous", 0.0), 1),
        "new_named_ms": round(named, 1),
        "radiation_unexplained_ms": round(tally.get("radiation", 0.0), 1),
        "counts": counts,
        "paid_per_ms_saved_low": round(bill_lo / max(saved, 1e-9), 2),
        "paid_per_ms_saved_high": round(bill_hi / max(saved, 1e-9), 2),
        "recorded_all_other_growth_per_ms_saved": d["fusion_trade"].get(
            "all_other_growth_per_ms_saved",
            d["fusion_trade"].get("paid_per_ms_saved"),
        ),
        "new_named_share_of_growth_pct": round(100.0 * named / grew, 1),
        "fusion_about_break_even": bill_lo / max(saved, 1e-9) < 1.25,
        "largest_new_named": [
            {"function": r["function"], "delta_ms": r["delta_ms"],
             "registers": r["now_registers"]}
            for r in sorted(rows, key=lambda x: -x["delta_ms"])
            if bucket(r) == "new_named"
        ][:5],
        "largest_receivers": [
            {"function": r["function"], "delta_ms": r["delta_ms"],
             "pre_launches": r["pre_launches"], "now_launches": r["now_launches"],
             "pre_registers": r["pre_registers"], "now_registers": r["now_registers"]}
            for r in sorted(rows, key=lambda x: -x["delta_ms"])
            if bucket(r) == "existing"
        ][:5],
    }
    # The buckets must account for the whole window, or one is mis-classified.
    total = sum(tally.values())
    assert abs(total - grew) < 2.0, (total, grew)
    doc["buckets_account_for_window"] = True

    json.dump(doc, open(out, "w"), indent=2)
    print(f"saved {saved:.1f} ms of broadcast machinery; receivers net "
          f"{bill_lo:+.1f} ms ({doc['paid_per_ms_saved_low']}-"
          f"{doc['paid_per_ms_saved_high']} per ms saved); new named work "
          f"{named:+.1f} ms = {doc['new_named_share_of_growth_pct']}% of the window")
    print(f"wrote {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
