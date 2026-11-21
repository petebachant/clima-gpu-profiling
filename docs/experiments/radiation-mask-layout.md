# What the grouped cloud mask's memory layout is worth

Every wide workspace a grouped radiation round writes is column-fast.
The optics arrays come from `_coalesced_3d_zeros(DA, FT, ncol, nlay, 3)` and are sliced `view(op.τ, wcol, :)`; the partial accumulators are `(ncolw, nlay + 1, planes)`.
The cloud mask was the one exception, `(nlay, ncolw)`, so a warp's threads stood `nlay` bytes apart and one one-byte access touched 32 sectors where it could touch one.

The cost that predicts is wavefronts, not bandwidth.
A warp walking 80 layers issues 80 one-byte instructions; layer-fast each spans 32 distinct sectors, so the walk asks for 2560 sector-requests to reach 80 distinct sectors, while column-fast asks for 80.
The tile is the same either way and L1 absorbs it, which is why this is a request-count treatment and not a DRAM one --- and why it was ranked second among radiation's levers off a sector census rather than off a clock.

## Why the cap state is an axis and not a constant

`radiation-memory` found that raising occupancy made these kernels **more** latency-exposed, not less: `long_scoreboard` went from 59.2% of measured stalls to 80.4%, and the stall itself from 4.23 cycles to 10.2.
A treatment that removes memory requests should therefore be worth more with more warps resident, so measuring it only at the shipped register caps would conflate the layout with the caps, and measuring it only uncapped would price it in a regime that no longer exists.

So this is a 2x2 --- layer-fast and column-fast, uncapped and at the shipped caps --- and `layout_worth_more_at_the_caps` reports that prediction's outcome rather than assuming it.
Both layouts are live in the same process because the layout is a type, which is what lets the comparison be paired inside one job rather than taken across two: `radiation-register-cap` shows this machine's drift between runs exceeds what is being measured here.

```python calkit stage name=values environment=py outputs=[{path: results/docs/radiation-mask-layout.json, storage: git}] inputs=[results/radiation-mask-layout.toml]
import json
import tomllib

d = tomllib.load(open("results/radiation-mask-layout.toml", "rb"))
BANDS = ("lw", "sw")


def band(name):
    b = d["bands"][name]
    pts = b["points"]

    def point(cfg):
        p = pts[cfg]
        return {
            "rounds_ms": round(p["rounds_ms"], 1),
            "change_pct": round(p["change_pct"], 2),
            "halves_spread_pts": round(
                abs(p["change_pct_first_half"] - p["change_pct_second_half"]), 2
            ),
            "bit_for_bit": p["bit_for_bit"],
            "rounds_faster": p["rounds_negative"],
            "rounds_slower": p["rounds_positive"],
            "sign_test_p": p["sign_test_p"],
        }

    out = {
        "band": name,
        "rounds": b["rounds"],
        "values_compared": b["values_compared"],
        "all_bit_for_bit": b["all_bit_for_bit"],
        "shipped_optics": b["shipped_cap_optics"],
        "shipped_sweeps": b["shipped_cap_sweeps"],
        "layout_pct_uncapped": round(b["layout_gain_pct_uncapped"], 2),
        "layout_pct_capped": round(b["layout_gain_pct_capped"], 2),
        "layout_resolved_at_the_caps": b["layout_is_resolved_at_the_caps"],
        "layout_worth_more_at_the_caps": b["layout_worth_more_at_the_caps"],
        "cap_only_pct": round(b["cap_only"]["change_pct"], 2),
        "both_pct": round(b["both_treatments"]["change_pct"], 2),
        "uncapped_rounds_ms": round(b["cap_only"]["reference_rounds_ms"], 1),
        "capped_rounds_ms": round(b["cap_only"]["rounds_ms"], 1),
        "points": {c: point(c) for c in b["configurations"]},
    }
    # The layout's own cost in registers and spill, which is the other half of
    # why it is not the trade it looked like: it is cheaper in both and no
    # faster.
    ks = d["kernels"]
    for state in ("uncapped", "capped"):
        for layout in ("layer_fast", "column_fast"):
            k = ks[f"{name}_sweeps_{state}_{layout}"]
            out[f"sweeps_{state}_{layout}"] = {
                "registers": k["registers"],
                "local_bytes": k["local_bytes"],
                "warps_per_sm": k["warps_per_sm"],
            }
    return out


data = {b: band(b) for b in BANDS}

# The two round loops added, because a reader deciding whether to ship the
# layout is deciding about radiation and not about one band.
def combined(key):
    return sum(data[b][key] for b in BANDS)


cap = sum(d["bands"][b]["cap_only"]["rounds_ms"] for b in BANDS)
both = sum(d["bands"][b]["both_treatments"]["rounds_ms"] for b in BANDS)
ref = sum(d["bands"][b]["cap_only"]["reference_rounds_ms"] for b in BANDS)
data["combined"] = {
    "uncapped_ms": round(ref, 1),
    "caps_only_ms": round(cap, 1),
    "caps_and_layout_ms": round(both, 1),
    "caps_pct": round(100 * (cap - ref) / ref, 2),
    "caps_and_layout_pct": round(100 * (both - ref) / ref, 2),
    "layout_on_top_of_caps_pct": round(100 * (both - cap) / cap, 2),
    "layout_on_top_of_caps_ms": round(both - cap, 2),
}

data["all_bit_for_bit"] = all(data[b]["all_bit_for_bit"] for b in BANDS)
data["bands_where_column_fast_wins"] = (
    ", ".join(b for b in BANDS if data[b]["layout_pct_capped"] < 0) or "none"
)
data["bands_where_it_is_resolved"] = (
    ", ".join(b for b in BANDS if data[b]["layout_resolved_at_the_caps"]) or "none"
)
data["worth_more_at_the_caps_everywhere"] = all(
    data[b]["layout_worth_more_at_the_caps"] for b in BANDS
)
data["column_fast_is_worth_shipping"] = d["verdict"][
    "column_fast_is_worth_shipping"
]
data["verdict"] = d["verdict"]

json.dump(data, open("results/docs/radiation-mask-layout.json", "w"), indent=2)
```

## Method

One reservation, one process, one settled AMIP state, four configurations per band differing only in the mask store's shape and the `maxregs` the two halves of a round compile to.
Timing is interleaved and paired: each round visits every configuration once in a rotating order, and an effect is the median of its per-round differences against the same round's reference.
The sign is tested over the rounds with an exact two-sided sign test, and the rounds are split in half so an effect that does not hold its magnitude across the run says so.

A layout cannot change the answer --- the McICA draw is keyed on `(key, gcol, igpt, ilay)`, not on memory order --- so the mask is compared in a canonical orientation, where a transposition error cannot read as agreement.

## Result

Every configuration is bit-for-bit, over <!-- calkit value key=lw.values_compared path=results/docs/radiation-mask-layout.json -->159252480<!-- /calkit value --> longwave and <!-- calkit value key=sw.values_compared path=results/docs/radiation-mask-layout.json -->222953472<!-- /calkit value --> shortwave values.

| band | uncapped | at the shipped caps | resolved there? | worth more at the caps? |
| --- | --- | --- | --- | --- |
| longwave | <!-- calkit value key=lw.layout_pct_uncapped path=results/docs/radiation-mask-layout.json -->1.29<!-- /calkit value -->% | <!-- calkit value key=lw.layout_pct_capped path=results/docs/radiation-mask-layout.json -->0.03<!-- /calkit value -->% | <!-- calkit value key=lw.layout_resolved_at_the_caps path=results/docs/radiation-mask-layout.json -->false<!-- /calkit value --> | <!-- calkit value key=lw.layout_worth_more_at_the_caps path=results/docs/radiation-mask-layout.json -->true<!-- /calkit value --> |
| shortwave | <!-- calkit value key=sw.layout_pct_uncapped path=results/docs/radiation-mask-layout.json -->0.46<!-- /calkit value -->% | <!-- calkit value key=sw.layout_pct_capped path=results/docs/radiation-mask-layout.json -->-0.48<!-- /calkit value -->% | <!-- calkit value key=sw.layout_resolved_at_the_caps path=results/docs/radiation-mask-layout.json -->true<!-- /calkit value --> | <!-- calkit value key=sw.layout_worth_more_at_the_caps path=results/docs/radiation-mask-layout.json -->true<!-- /calkit value --> |

**The prediction about occupancy was right and the level was not there.**
Column-fast is worth more at the caps than uncapped in both bands, which is what the stall reading said it should be.
But it starts from the wrong side: uncapped it is **slower** in both bands, by <!-- calkit value key=lw.layout_pct_uncapped path=results/docs/radiation-mask-layout.json -->1.29<!-- /calkit value -->% and <!-- calkit value key=sw.layout_pct_uncapped path=results/docs/radiation-mask-layout.json -->0.46<!-- /calkit value -->%, and the caps only move it to <!-- calkit value key=lw.layout_pct_capped path=results/docs/radiation-mask-layout.json -->0.03<!-- /calkit value -->% and <!-- calkit value key=sw.layout_pct_capped path=results/docs/radiation-mask-layout.json -->-0.48<!-- /calkit value -->% --- resolved in the shortwave, not resolved in the longwave.

Across both round loops the layout is worth <!-- calkit value key=combined.layout_on_top_of_caps_pct path=results/docs/radiation-mask-layout.json -->-0.17<!-- /calkit value -->%, which is <!-- calkit value key=combined.layout_on_top_of_caps_ms path=results/docs/radiation-mask-layout.json -->-1.06<!-- /calkit value --> ms on <!-- calkit value key=combined.caps_only_ms path=results/docs/radiation-mask-layout.json -->625.6<!-- /calkit value --> ms.
The register caps, measured in the same job against the same reference, are worth <!-- calkit value key=combined.caps_pct path=results/docs/radiation-mask-layout.json -->-15.56<!-- /calkit value -->%, taking the two loops from <!-- calkit value key=combined.uncapped_ms path=results/docs/radiation-mask-layout.json -->740.8<!-- /calkit value --> ms to <!-- calkit value key=combined.caps_only_ms path=results/docs/radiation-mask-layout.json -->625.6<!-- /calkit value --> ms.

**It is not that the transposition cost something elsewhere.**
Column-fast compiles to fewer registers in the longwave sweeps (<!-- calkit value key=lw.sweeps_uncapped_column_fast.registers path=results/docs/radiation-mask-layout.json -->137<!-- /calkit value --> against <!-- calkit value key=lw.sweeps_uncapped_layer_fast.registers path=results/docs/radiation-mask-layout.json -->140<!-- /calkit value -->) and spills less at the cap in both bands (<!-- calkit value key=lw.sweeps_capped_column_fast.local_bytes path=results/docs/radiation-mask-layout.json -->208<!-- /calkit value --> bytes against <!-- calkit value key=lw.sweeps_capped_layer_fast.local_bytes path=results/docs/radiation-mask-layout.json -->216<!-- /calkit value --> in the longwave, <!-- calkit value key=sw.sweeps_capped_column_fast.local_bytes path=results/docs/radiation-mask-layout.json -->176<!-- /calkit value --> against <!-- calkit value key=sw.sweeps_capped_layer_fast.local_bytes path=results/docs/radiation-mask-layout.json -->192<!-- /calkit value --> in the shortwave), at identical warps per multiprocessor.
So it is cheaper in registers, cheaper in spill, asks for 32 times fewer sector-requests, and is not faster.

## Verdict

**Do not ship it.**
The layer-fast mask stays, and the one-exception-to-the-convention reading of `gpt_groups.jl` is a cosmetic complaint rather than a performance one.

What this closes is bigger than the change itself.
The mask was ranked second among radiation's levers because it was **75.5%** of the longwave sweeps kernel's excess sectors, and that share was read off a sector census.
This measurement says a share of excess sectors is not a share of anything that costs time: the requests were real, removing them was real, and the clock did not move.
The same reasoning had ranked the ClimaAtmos-side `cld_mask_lw`, `cld_mask_sw` and `aero_mask` arrays, which are the same shape for the same reason --- they now carry a measured prior of about zero, which is worth knowing before spending a cross-repository pull request on them.

The lever that remains in this family is the gather, and it is a different quantity: `kmajor`'s layout costs **45.6% of every sector radiation moves** rather than a share of one kernel's excess, and it is absorbed by L1 at an 86% hit rate.
Measure the per-warp temperature and pressure bin spread before permuting that table, for the reason this experiment just demonstrated.
