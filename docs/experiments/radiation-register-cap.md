# Shedding registers off the grouped radiation kernels

The g-point and phase split took the two radiation solvers off the 255-register cap and left four kernels well below it.
Each of them already collects every warp its own register count permits, and `results/group-block-size.toml` closed the launch-configuration side of that independently: the shipped 128-thread block is the best of ten sizes in both bands.
So what remains between these kernels and more warps is the register count itself, and this asks what `ptxas` charges to lower it.

The question is not whether occupancy would rise.
That is arithmetic, and the driver confirms it here before anything is timed.
The question is what the allocator does to comply.

## Why this is not the treatment that lost twice

Two caps on this family have been measured and rejected, both at 128 registers: two blocks per multiprocessor cost +3.93% and a 128-thread block at 168 registers cost +2.91%.
Both were caps on the **fused** solvers, which compiled to 255 registers with 248 and 288 bytes of local memory, and forcing those to 128 took the local frame to 576 and 600 bytes.
That is the trade those rejections priced: roughly 2.3x the local memory, on kernels whose stalls were already dominated by memory dependency with nothing to hide the extra behind.

The kernels here are a different proposition.
They carry <!-- calkit value key=kernels.lw_optics.local_bytes path=results/docs/radiation-register-cap.json -->32<!-- /calkit value --> bytes of local memory each, and the cheapest step down is a dozen registers rather than 127 --- though this sweep goes four steps deep per kernel, because a two-step search cannot tell a floor from the bottom of its own range.
`register-cap-filter` found that `ptxas` meets a cut that small almost entirely without spilling --- and that is also the trap this experiment is built around, because the allocator can meet a tighter budget by rematerializing values instead, which costs instructions and no local memory at all.
A clean local frame therefore reads the same whether the cap was free or expensive, so the frame is a diagnostic here and the decision rests on the clock.

## Method

Caps are computed rather than chosen.
For each kernel the four highest multiples of eight below its own register count that the allocation granularity says each reach more warps than the step above are taken, so a cap that would spend registers and buy nothing is never asked for.
Every cap is then put to the driver's own `cuOccupancyMaxActiveBlocksPerMultiprocessor` before it is timed, and both the model's prediction and the driver's answer are recorded: a step the model gets wrong reads as a disagreement rather than as a silent premise.
That matters because dividing the register file by the thread count gets several A100 steps wrong, which this project has done twice.

Every configuration runs in one process against one settled AMIP state, with the same workspaces and lookups, and differs only in the `maxregs` passed to one or both halves of a round.
Each cap is applied to the sweeps kernel alone, to the optics kernel alone, and to both, because the two halves carry different shares of the event and a gain on one would otherwise be hidden inside a pair.
The matched pairs are what check the single-kernel points rather than what competes with them: if a pair's effect is the sum of its two halves', then each kernel's budget can be read off its own column and the best pair need never have been timed together.

Timing is interleaved and paired rather than blocked.
Timing every repeat of one configuration and then every repeat of the next attributes to the cap whatever else moved on the device between the two blocks, and on this machine that drift has been larger than the effect being measured.
Each round visits every configuration once in an order that rotates by one per round, and a configuration's effect is the median of its per-round differences against the uncapped point of the same round.
The sign of that effect is tested over the rounds with an exact two-sided sign test, which is the bar used here: the spread of one point's own repeats is set by the drift and is not an error bar for the difference.
The rounds are split in half and both halves reported, because an effect that holds its magnitude across the run is worth more than one that does not.

A register budget cannot change arithmetic --- the allocator spills and rematerializes, it does not reassociate --- so every point should be bit-for-bit with the uncapped one, and that is recorded rather than assumed.

```python calkit stage name=values environment=py outputs=[{path: results/docs/radiation-register-cap.json, storage: git}] inputs=[results/radiation-register-cap.toml, results/radiation-mask-layout.toml]
import json
import tomllib

d = tomllib.load(open("results/radiation-register-cap.toml", "rb"))
ks = d["kernels"]
block = d["block_threads"]

HALVES = ("optics", "sweeps")
BANDS = ("lw", "sw")


def say(items):
    """A list as prose, since a marker renders a value and not a list."""
    items = [str(i) for i in items]
    if not items:
        return "none"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def kernel(band, half):
    """The uncapped kernel, and every cap that was compiled for it."""
    base = ks[f"{band}_{half}"]
    caps = {}
    for key, rec in ks.items():
        if not key.startswith(f"{band}_{half}_cap"):
            continue
        cap = rec["requested_cap"]
        if not rec["compiled"]:
            caps[str(cap)] = {"cap": cap, "compiled": False}
            continue
        caps[str(cap)] = {
            "cap": cap,
            "compiled": True,
            "registers": rec["registers"],
            "registers_shed": rec["registers_shed"],
            "met_cap": rec["met_cap"],
            "local_bytes": rec["local_bytes"],
            "local_bytes_added": rec["local_bytes_added"],
            "warps_per_sm": rec["warps_per_sm"],
            "warps_gained": rec["warps_gained"],
            "predicted_warps": rec["predicted_warps"],
            "model_agrees": rec["model_agrees_with_driver"],
            # The share of the register file a thread's registers occupy is
            # what the cap moves, so the local frame is read as a multiple of
            # what the kernel carried unforced rather than as a raw byte count.
            "local_growth_x": (
                round(rec["local_bytes"] / base["local_bytes"], 2)
                if base["local_bytes"]
                else None
            ),
        }
    return {
        "registers": base["registers"],
        "local_bytes": base["local_bytes"],
        "max_threads": base["max_threads"],
        "warps_per_sm": base["warps_per_sm"],
        "predicted_warps": base["predicted_warps"],
        "model_agrees": base["model_agrees_with_driver"],
        "caps_compiled": say(sorted((c["cap"] for c in caps.values()), reverse=True)),
        "caps": caps,
    }


kernels = {f"{b}_{h}": kernel(b, h) for b in BANDS for h in HALVES}


def band(name):
    b = d["bands"][name]
    if not b.get("any_cap_timed", False):
        return {"band": name, "any_cap_timed": False, "points": {}}
    pts = b["points"]

    def point(cfg):
        p = pts[cfg]
        return {
            "config": cfg,
            "rounds_ms": round(p["rounds_ms"], 1),
            "change_pct": round(p["change_pct"], 2),
            "change_pct_first_half": round(p["change_pct_first_half"], 2),
            "change_pct_second_half": round(p["change_pct_second_half"], 2),
            "rounds_faster": p["rounds_negative"],
            "rounds_slower": p["rounds_positive"],
            "sign_test_p": p["sign_test_p"],
            # Resolved means the rounds agree on the sign, not that the number
            # cleared a spread: the spread here is the device's drift.
            "resolved": p["sign_test_p"] < 0.05,
            # And a magnitude is only quoted if its two halves put it the same
            # way, since a half-to-half reversal is drift inside the run.
            "halves_agree": (
                p["change_pct_first_half"] * p["change_pct_second_half"] > 0
            ),
            "bit_for_bit": p["bit_for_bit_with_uncapped"],
        }

    points = {cfg: point(cfg) for cfg in b["configurations"]}
    timed = [c for c in b["configurations"] if c != "uncapped"]
    faster = [c for c in timed if points[c]["change_pct"] < 0 and points[c]["resolved"]]
    slower = [c for c in timed if points[c]["change_pct"] > 0 and points[c]["resolved"]]
    return {
        "band": name,
        "any_cap_timed": True,
        "n_gpt": b["n_gpt"],
        "rounds": b["rounds"],
        "uncapped_rounds_ms": round(b["uncapped_rounds_ms"], 1),
        "n_configs_timed": len(timed),
        "configs_timed": say(timed),
        "best_config": b["best_configuration"],
        "best_change_pct": round(b["best_change_pct"], 2),
        "best_is_resolved": b["best_is_resolved"],
        "n_faster": len(faster),
        "n_slower": len(slower),
        "configs_slower": say(slower),
        "all_bit_for_bit": b["all_bit_for_bit"],
        # The worst thing a cap did, which is the figure a verdict rests on
        # when nothing got faster.
        "worst_change_pct": round(
            max(points[c]["change_pct"] for c in timed), 2
        ),
        "points": points,
    }


bands = {name: band(name) for name in BANDS}


def composition(band_name):
    """A matched pair against the sum of its two single-kernel points.

    This is the check that licenses reading each kernel's own optimum off its
    own column, and shipping a pair whose two halves were never timed
    together: if the halves compose, the grid's matched points are redundant
    with the single-kernel ones and a mixed budget needs no separate round.
    """
    b = d["bands"][band_name]
    if not b.get("any_cap_timed", False):
        return {"checked": 0, "rows": {}, "worst_residual_pts": None}
    pts = b["points"]
    rows = {}
    for cfg in b["configurations"]:
        if not cfg.startswith("both_"):
            continue
        cap = cfg.split("_", 1)[1]
        predicted = (
            pts[f"sweeps_{cap}"]["change_pct"] + pts[f"optics_{cap}"]["change_pct"]
        )
        rows[cap] = {
            "cap": int(cap),
            "predicted_pct": round(predicted, 2),
            "measured_pct": round(pts[cfg]["change_pct"], 2),
            "residual_pts": round(pts[cfg]["change_pct"] - predicted, 3),
        }
    return {
        "checked": len(rows),
        "rows": rows,
        "worst_residual_pts": max(
            (abs(r["residual_pts"]) for r in rows.values()), default=None
        ),
    }


compositions = {name: composition(name) for name in BANDS}


def best_per_kernel(band_name):
    """Each half's own fastest cap, and what the two together predict.

    The grid times matched caps, so a band whose two halves bottom out at
    different counts has its best budget OUTSIDE the configurations timed.
    That pair is quoted as a prediction and labeled as one.
    """
    b = d["bands"][band_name]
    if not b.get("any_cap_timed", False):
        return {"available": False}
    pts = b["points"]
    chosen = {}
    for half in HALVES:
        candidates = {
            int(cfg.split("_", 1)[1]): pts[cfg]["change_pct"]
            for cfg in b["configurations"]
            if cfg.startswith(f"{half}_")
        }
        cap = min(candidates, key=candidates.get)
        chosen[half] = {"cap": cap, "change_pct": round(candidates[cap], 2)}
    predicted = sum(c["change_pct"] for c in chosen.values())
    matched = chosen["optics"]["cap"] == chosen["sweeps"]["cap"]
    return {
        "available": True,
        "optics_cap": chosen["optics"]["cap"],
        "sweeps_cap": chosen["sweeps"]["cap"],
        "optics_change_pct": chosen["optics"]["change_pct"],
        "sweeps_change_pct": chosen["sweeps"]["change_pct"],
        "halves_agree_on_a_cap": matched,
        "predicted_change_pct": round(predicted, 2),
        # A matched pair was timed, so its figure is measured rather than
        # summed; a mixed one was not, and the difference is worth naming.
        "is_measured": matched,
        "gain_over_best_timed_pts": round(
            predicted - round(b["best_change_pct"], 2), 2
        ),
    }


per_kernel = {name: best_per_kernel(name) for name in BANDS}

# Did the occupancy arrive and the time not? That is the specific outcome this
# experiment exists to distinguish from a cap the allocator could not meet, and
# it is a boolean rather than a reading of two tables side by side.
def warps_arrived(band_name, half, cap):
    c = kernels[f"{band_name}_{half}"]["caps"].get(str(cap))
    return bool(c and c["compiled"] and c["warps_gained"] > 0)


occupancy_bought_nothing = []
for name, b in bands.items():
    if not b["any_cap_timed"]:
        continue
    for cfg, p in b["points"].items():
        if cfg == "uncapped" or p["change_pct"] < 0:
            continue
        half, _, cap = cfg.partition("_")
        halves = HALVES if half == "both" else (half,)
        if all(warps_arrived(name, h, int(cap)) for h in halves):
            occupancy_bought_nothing.append(f"{name} {cfg}")

data = {
    "block_threads": block,
    "rounds_interleaved": d["rounds_interleaved"],
    "timings_per_round": d["timings_per_round"],
    "ncol": d["ncol"],
    "nlay": d["nlay"],
    "ngroups": d["ngroups"],
    "kernels": kernels,
    "bands": bands,
    "verdict": d["verdict"],
    "any_cap_is_worth_shipping": d["verdict"]["any_cap_is_worth_shipping"],
    "bands_with_a_faster_cap": say(d["verdict"]["bands_with_a_faster_cap"]),
    "model_agrees_everywhere": d["verdict"]["model_agrees_with_driver_everywhere"],
    "all_bit_for_bit": d["verdict"]["all_bit_for_bit"],
    "n_occupancy_bought_nothing": len(occupancy_bought_nothing),
    "occupancy_bought_nothing": say(sorted(occupancy_bought_nothing)),
    "compositions": compositions,
    "per_kernel": per_kernel,
    "n_composition_checks": sum(c["checked"] for c in compositions.values()),
    "worst_composition_residual_pts": max(
        (
            c["worst_residual_pts"]
            for c in compositions.values()
            if c["worst_residual_pts"] is not None
        ),
        default=None,
    ),
    "bands_whose_halves_want_different_caps": say(
        sorted(
            name
            for name, k in per_kernel.items()
            if k["available"] and not k["halves_agree_on_a_cap"]
        )
    ),
}

# Every cap that compiled, so the prose can say what the allocator charged
# without a reader pairing two tables by eye.
data["worst_local_growth_x"] = max(
    (
        c["local_growth_x"]
        for k in kernels.values()
        for c in k["caps"].values()
        if c["compiled"] and c["local_growth_x"] is not None
    ),
    default=None,
)
data["all_caps_met"] = all(
    c["met_cap"]
    for k in kernels.values()
    for c in k["caps"].values()
    if c["compiled"]
)

# The shipped pairs are composed from single-kernel columns rather than timed
# here, so the one measurement that times them directly is read back in. Its
# `cap_only` arm is the shipped budgets against the same uncapped kernels, in
# its own job on its own day, which is what makes it a test of the composition
# rather than a restatement of it.
layout = tomllib.load(open("results/radiation-mask-layout.toml", "rb"))
confirmation = {}
for b in BANDS:
    pred = data["per_kernel"][b].get("predicted_change_pct")
    if pred is None:
        continue
    cap_only = layout["bands"][b]["cap_only"]
    measured = round(cap_only["change_pct"], 2)
    confirmation[b] = {
        "predicted_pct": pred,
        "measured_pct": measured,
        "residual_pts": round(measured - pred, 2),
        "shipped_optics": layout["bands"][b]["shipped_cap_optics"],
        "shipped_sweeps": layout["bands"][b]["shipped_cap_sweeps"],
        "rounds": len(cap_only["per_round_diff_ms"]),
        "rounds_faster": cap_only["rounds_negative"],
        "sign_test_p": cap_only["sign_test_p"],
        "halves_agree": round(
            abs(
                cap_only["change_pct_first_half"]
                - cap_only["change_pct_second_half"]
            ),
            2,
        ),
    }
data["confirmation"] = confirmation
data["worst_confirmation_residual_pts"] = (
    max((abs(c["residual_pts"]) for c in confirmation.values()), default=None)
)
data["shipped_pairs_confirmed"] = bool(confirmation) and all(
    c["measured_pct"] < 0 and c["rounds_faster"] == c["rounds"]
    for c in confirmation.values()
)

json.dump(data, open("results/docs/radiation-register-cap.json", "w"), indent=2)
```

## What the compiles say

Every cap asked for was met, and the driver's occupancy query agreed with the granularity model at every step of every kernel, so what follows rests on measured warps rather than on a register file divided by a thread count.
The four kernels start at <!-- calkit value key=kernels.lw_optics.registers path=results/docs/radiation-register-cap.json -->156<!-- /calkit value -->, <!-- calkit value key=kernels.lw_sweeps.registers path=results/docs/radiation-register-cap.json -->140<!-- /calkit value -->, <!-- calkit value key=kernels.sw_optics.registers path=results/docs/radiation-register-cap.json -->124<!-- /calkit value --> and <!-- calkit value key=kernels.sw_sweeps.registers path=results/docs/radiation-register-cap.json -->117<!-- /calkit value --> registers, carrying <!-- calkit value key=kernels.lw_optics.local_bytes path=results/docs/radiation-register-cap.json -->32<!-- /calkit value --> bytes of local memory each, and each cap below buys exactly four more warps per multiprocessor than the step above it.

The allocator paid for those warps in spill, and it charged the two halves of a round very differently.
Against the same budget of <!-- calkit value key=kernels.lw_optics.caps.96.cap path=results/docs/radiation-register-cap.json -->96<!-- /calkit value --> registers the longwave optics frame grows to <!-- calkit value key=kernels.lw_optics.caps.96.local_bytes path=results/docs/radiation-register-cap.json -->272<!-- /calkit value --> bytes while the longwave sweeps frame grows only to <!-- calkit value key=kernels.lw_sweeps.caps.96.local_bytes path=results/docs/radiation-register-cap.json -->144<!-- /calkit value -->, and the worst frame anywhere in the sweep is <!-- calkit value key=worst_local_growth_x path=results/docs/radiation-register-cap.json -->11.5<!-- /calkit value --> times what its kernel carried unforced.
That asymmetry is the mechanism behind the result below: a kernel whose live state fits the budget keeps gaining from the next step down, and one whose does not stops gaining well before it runs out of steps.

| cap | optics: regs, local, warps | sweeps: regs, local, warps | sweeps only | optics only | both | halves summed |
| --- | --- | --- | --- | --- | --- | --- |
| none | <!-- calkit value key=kernels.lw_optics.registers path=results/docs/radiation-register-cap.json -->156<!-- /calkit value -->, <!-- calkit value key=kernels.lw_optics.local_bytes path=results/docs/radiation-register-cap.json -->32<!-- /calkit value --> B, <!-- calkit value key=kernels.lw_optics.warps_per_sm path=results/docs/radiation-register-cap.json -->12<!-- /calkit value --> | <!-- calkit value key=kernels.lw_sweeps.registers path=results/docs/radiation-register-cap.json -->140<!-- /calkit value -->, <!-- calkit value key=kernels.lw_sweeps.local_bytes path=results/docs/radiation-register-cap.json -->32<!-- /calkit value --> B, <!-- calkit value key=kernels.lw_sweeps.warps_per_sm path=results/docs/radiation-register-cap.json -->12<!-- /calkit value --> | --- | --- | <!-- calkit value key=bands.lw.uncapped_rounds_ms path=results/docs/radiation-register-cap.json -->435.4<!-- /calkit value --> ms | --- |
| 128 | <!-- calkit value key=kernels.lw_optics.caps.128.registers path=results/docs/radiation-register-cap.json -->128<!-- /calkit value -->, <!-- calkit value key=kernels.lw_optics.caps.128.local_bytes path=results/docs/radiation-register-cap.json -->144<!-- /calkit value --> B, <!-- calkit value key=kernels.lw_optics.caps.128.warps_per_sm path=results/docs/radiation-register-cap.json -->16<!-- /calkit value --> | <!-- calkit value key=kernels.lw_sweeps.caps.128.registers path=results/docs/radiation-register-cap.json -->128<!-- /calkit value -->, <!-- calkit value key=kernels.lw_sweeps.caps.128.local_bytes path=results/docs/radiation-register-cap.json -->40<!-- /calkit value --> B, <!-- calkit value key=kernels.lw_sweeps.caps.128.warps_per_sm path=results/docs/radiation-register-cap.json -->16<!-- /calkit value --> | <!-- calkit value key=bands.lw.points.sweeps_128.change_pct path=results/docs/radiation-register-cap.json -->-4.32<!-- /calkit value -->% | <!-- calkit value key=bands.lw.points.optics_128.change_pct path=results/docs/radiation-register-cap.json -->-7.81<!-- /calkit value -->% | <!-- calkit value key=bands.lw.points.both_128.change_pct path=results/docs/radiation-register-cap.json -->-12.1<!-- /calkit value -->% | <!-- calkit value key=compositions.lw.rows.128.predicted_pct path=results/docs/radiation-register-cap.json -->-12.13<!-- /calkit value -->% |
| 96 | <!-- calkit value key=kernels.lw_optics.caps.96.registers path=results/docs/radiation-register-cap.json -->96<!-- /calkit value -->, <!-- calkit value key=kernels.lw_optics.caps.96.local_bytes path=results/docs/radiation-register-cap.json -->272<!-- /calkit value --> B, <!-- calkit value key=kernels.lw_optics.caps.96.warps_per_sm path=results/docs/radiation-register-cap.json -->20<!-- /calkit value --> | <!-- calkit value key=kernels.lw_sweeps.caps.96.registers path=results/docs/radiation-register-cap.json -->96<!-- /calkit value -->, <!-- calkit value key=kernels.lw_sweeps.caps.96.local_bytes path=results/docs/radiation-register-cap.json -->144<!-- /calkit value --> B, <!-- calkit value key=kernels.lw_sweeps.caps.96.warps_per_sm path=results/docs/radiation-register-cap.json -->20<!-- /calkit value --> | <!-- calkit value key=bands.lw.points.sweeps_96.change_pct path=results/docs/radiation-register-cap.json -->-6.2<!-- /calkit value -->% | <!-- calkit value key=bands.lw.points.optics_96.change_pct path=results/docs/radiation-register-cap.json -->-11.41<!-- /calkit value -->% | <!-- calkit value key=bands.lw.points.both_96.change_pct path=results/docs/radiation-register-cap.json -->-17.61<!-- /calkit value -->% | <!-- calkit value key=compositions.lw.rows.96.predicted_pct path=results/docs/radiation-register-cap.json -->-17.61<!-- /calkit value -->% |
| 80 | <!-- calkit value key=kernels.lw_optics.caps.80.registers path=results/docs/radiation-register-cap.json -->80<!-- /calkit value -->, <!-- calkit value key=kernels.lw_optics.caps.80.local_bytes path=results/docs/radiation-register-cap.json -->336<!-- /calkit value --> B, <!-- calkit value key=kernels.lw_optics.caps.80.warps_per_sm path=results/docs/radiation-register-cap.json -->24<!-- /calkit value --> | <!-- calkit value key=kernels.lw_sweeps.caps.80.registers path=results/docs/radiation-register-cap.json -->80<!-- /calkit value -->, <!-- calkit value key=kernels.lw_sweeps.caps.80.local_bytes path=results/docs/radiation-register-cap.json -->184<!-- /calkit value --> B, <!-- calkit value key=kernels.lw_sweeps.caps.80.warps_per_sm path=results/docs/radiation-register-cap.json -->24<!-- /calkit value --> | <!-- calkit value key=bands.lw.points.sweeps_80.change_pct path=results/docs/radiation-register-cap.json -->-7.75<!-- /calkit value -->% | <!-- calkit value key=bands.lw.points.optics_80.change_pct path=results/docs/radiation-register-cap.json -->-9.27<!-- /calkit value -->% | <!-- calkit value key=bands.lw.points.both_80.change_pct path=results/docs/radiation-register-cap.json -->-17.0<!-- /calkit value -->% | <!-- calkit value key=compositions.lw.rows.80.predicted_pct path=results/docs/radiation-register-cap.json -->-17.02<!-- /calkit value -->% |
| 72 | <!-- calkit value key=kernels.lw_optics.caps.72.registers path=results/docs/radiation-register-cap.json -->72<!-- /calkit value -->, <!-- calkit value key=kernels.lw_optics.caps.72.local_bytes path=results/docs/radiation-register-cap.json -->368<!-- /calkit value --> B, <!-- calkit value key=kernels.lw_optics.caps.72.warps_per_sm path=results/docs/radiation-register-cap.json -->28<!-- /calkit value --> | <!-- calkit value key=kernels.lw_sweeps.caps.72.registers path=results/docs/radiation-register-cap.json -->72<!-- /calkit value -->, <!-- calkit value key=kernels.lw_sweeps.caps.72.local_bytes path=results/docs/radiation-register-cap.json -->216<!-- /calkit value --> B, <!-- calkit value key=kernels.lw_sweeps.caps.72.warps_per_sm path=results/docs/radiation-register-cap.json -->28<!-- /calkit value --> | <!-- calkit value key=bands.lw.points.sweeps_72.change_pct path=results/docs/radiation-register-cap.json -->-8.02<!-- /calkit value -->% | <!-- calkit value key=bands.lw.points.optics_72.change_pct path=results/docs/radiation-register-cap.json -->-0.73<!-- /calkit value -->% | <!-- calkit value key=bands.lw.points.both_72.change_pct path=results/docs/radiation-register-cap.json -->-8.75<!-- /calkit value -->% | <!-- calkit value key=compositions.lw.rows.72.predicted_pct path=results/docs/radiation-register-cap.json -->-8.76<!-- /calkit value -->% |

| cap | optics: regs, local, warps | sweeps: regs, local, warps | sweeps only | optics only | both | halves summed |
| --- | --- | --- | --- | --- | --- | --- |
| none | <!-- calkit value key=kernels.sw_optics.registers path=results/docs/radiation-register-cap.json -->124<!-- /calkit value -->, <!-- calkit value key=kernels.sw_optics.local_bytes path=results/docs/radiation-register-cap.json -->32<!-- /calkit value --> B, <!-- calkit value key=kernels.sw_optics.warps_per_sm path=results/docs/radiation-register-cap.json -->16<!-- /calkit value --> | <!-- calkit value key=kernels.sw_sweeps.registers path=results/docs/radiation-register-cap.json -->117<!-- /calkit value -->, <!-- calkit value key=kernels.sw_sweeps.local_bytes path=results/docs/radiation-register-cap.json -->32<!-- /calkit value --> B, <!-- calkit value key=kernels.sw_sweeps.warps_per_sm path=results/docs/radiation-register-cap.json -->16<!-- /calkit value --> | --- | --- | <!-- calkit value key=bands.sw.uncapped_rounds_ms path=results/docs/radiation-register-cap.json -->305.6<!-- /calkit value --> ms | --- |
| 96 | <!-- calkit value key=kernels.sw_optics.caps.96.registers path=results/docs/radiation-register-cap.json -->96<!-- /calkit value -->, <!-- calkit value key=kernels.sw_optics.caps.96.local_bytes path=results/docs/radiation-register-cap.json -->120<!-- /calkit value --> B, <!-- calkit value key=kernels.sw_optics.caps.96.warps_per_sm path=results/docs/radiation-register-cap.json -->20<!-- /calkit value --> | <!-- calkit value key=kernels.sw_sweeps.caps.96.registers path=results/docs/radiation-register-cap.json -->96<!-- /calkit value -->, <!-- calkit value key=kernels.sw_sweeps.caps.96.local_bytes path=results/docs/radiation-register-cap.json -->56<!-- /calkit value --> B, <!-- calkit value key=kernels.sw_sweeps.caps.96.warps_per_sm path=results/docs/radiation-register-cap.json -->20<!-- /calkit value --> | <!-- calkit value key=bands.sw.points.sweeps_96.change_pct path=results/docs/radiation-register-cap.json -->-2.61<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.optics_96.change_pct path=results/docs/radiation-register-cap.json -->-4.65<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.both_96.change_pct path=results/docs/radiation-register-cap.json -->-7.25<!-- /calkit value -->% | <!-- calkit value key=compositions.sw.rows.96.predicted_pct path=results/docs/radiation-register-cap.json -->-7.26<!-- /calkit value -->% |
| 80 | <!-- calkit value key=kernels.sw_optics.caps.80.registers path=results/docs/radiation-register-cap.json -->80<!-- /calkit value -->, <!-- calkit value key=kernels.sw_optics.caps.80.local_bytes path=results/docs/radiation-register-cap.json -->184<!-- /calkit value --> B, <!-- calkit value key=kernels.sw_optics.caps.80.warps_per_sm path=results/docs/radiation-register-cap.json -->24<!-- /calkit value --> | <!-- calkit value key=kernels.sw_sweeps.caps.80.registers path=results/docs/radiation-register-cap.json -->80<!-- /calkit value -->, <!-- calkit value key=kernels.sw_sweeps.caps.80.local_bytes path=results/docs/radiation-register-cap.json -->112<!-- /calkit value --> B, <!-- calkit value key=kernels.sw_sweeps.caps.80.warps_per_sm path=results/docs/radiation-register-cap.json -->24<!-- /calkit value --> | <!-- calkit value key=bands.sw.points.sweeps_80.change_pct path=results/docs/radiation-register-cap.json -->-5.08<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.optics_80.change_pct path=results/docs/radiation-register-cap.json -->-2.03<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.both_80.change_pct path=results/docs/radiation-register-cap.json -->-7.07<!-- /calkit value -->% | <!-- calkit value key=compositions.sw.rows.80.predicted_pct path=results/docs/radiation-register-cap.json -->-7.11<!-- /calkit value -->% |
| 72 | <!-- calkit value key=kernels.sw_optics.caps.72.registers path=results/docs/radiation-register-cap.json -->72<!-- /calkit value -->, <!-- calkit value key=kernels.sw_optics.caps.72.local_bytes path=results/docs/radiation-register-cap.json -->216<!-- /calkit value --> B, <!-- calkit value key=kernels.sw_optics.caps.72.warps_per_sm path=results/docs/radiation-register-cap.json -->28<!-- /calkit value --> | <!-- calkit value key=kernels.sw_sweeps.caps.72.registers path=results/docs/radiation-register-cap.json -->72<!-- /calkit value -->, <!-- calkit value key=kernels.sw_sweeps.caps.72.local_bytes path=results/docs/radiation-register-cap.json -->144<!-- /calkit value --> B, <!-- calkit value key=kernels.sw_sweeps.caps.72.warps_per_sm path=results/docs/radiation-register-cap.json -->28<!-- /calkit value --> | <!-- calkit value key=bands.sw.points.sweeps_72.change_pct path=results/docs/radiation-register-cap.json -->-4.91<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.optics_72.change_pct path=results/docs/radiation-register-cap.json -->6.29<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.both_72.change_pct path=results/docs/radiation-register-cap.json -->1.35<!-- /calkit value -->% | <!-- calkit value key=compositions.sw.rows.72.predicted_pct path=results/docs/radiation-register-cap.json -->1.39<!-- /calkit value -->% |
| 64 | <!-- calkit value key=kernels.sw_optics.caps.64.registers path=results/docs/radiation-register-cap.json -->64<!-- /calkit value -->, <!-- calkit value key=kernels.sw_optics.caps.64.local_bytes path=results/docs/radiation-register-cap.json -->248<!-- /calkit value --> B, <!-- calkit value key=kernels.sw_optics.caps.64.warps_per_sm path=results/docs/radiation-register-cap.json -->32<!-- /calkit value --> | <!-- calkit value key=kernels.sw_sweeps.caps.64.registers path=results/docs/radiation-register-cap.json -->64<!-- /calkit value -->, <!-- calkit value key=kernels.sw_sweeps.caps.64.local_bytes path=results/docs/radiation-register-cap.json -->192<!-- /calkit value --> B, <!-- calkit value key=kernels.sw_sweeps.caps.64.warps_per_sm path=results/docs/radiation-register-cap.json -->32<!-- /calkit value --> | <!-- calkit value key=bands.sw.points.sweeps_64.change_pct path=results/docs/radiation-register-cap.json -->-5.51<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.optics_64.change_pct path=results/docs/radiation-register-cap.json -->4.37<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.both_64.change_pct path=results/docs/radiation-register-cap.json -->-1.23<!-- /calkit value -->% | <!-- calkit value key=compositions.sw.rows.64.predicted_pct path=results/docs/radiation-register-cap.json -->-1.14<!-- /calkit value -->% |

The last column is not a measurement.
It is the sum of the two single-kernel effects in the same row, printed beside the pair that was actually timed so the two can be read against each other.

## Result

Every one of the <!-- calkit value key=bands.lw.n_configs_timed path=results/docs/radiation-register-cap.json -->12<!-- /calkit value --> longwave configurations and <!-- calkit value key=bands.sw.n_configs_timed path=results/docs/radiation-register-cap.json -->12<!-- /calkit value --> shortwave configurations is bit-for-bit with its uncapped reference, over <!-- calkit value key=bands.lw.rounds path=results/docs/radiation-register-cap.json -->16<!-- /calkit value --> interleaved rounds each, which is what a register budget should do and is recorded rather than assumed.
A cap is faster in both bands: the best pair timed is <!-- calkit value key=bands.lw.best_config path=results/docs/radiation-register-cap.json -->both_96<!-- /calkit value --> at <!-- calkit value key=bands.lw.best_change_pct path=results/docs/radiation-register-cap.json -->-17.61<!-- /calkit value -->% in the longwave and <!-- calkit value key=bands.sw.best_config path=results/docs/radiation-register-cap.json -->both_96<!-- /calkit value --> at <!-- calkit value key=bands.sw.best_change_pct path=results/docs/radiation-register-cap.json -->-7.25<!-- /calkit value -->% in the shortwave, both resolved in each half of the run.

**The two halves of a round compose.**
Over <!-- calkit value key=n_composition_checks path=results/docs/radiation-register-cap.json -->8<!-- /calkit value --> checks --- every cap in both bands --- the pair's measured effect landed within <!-- calkit value key=worst_composition_residual_pts path=results/docs/radiation-register-cap.json -->0.088<!-- /calkit value --> points of the sum of its two single-kernel effects, against effects as large as <!-- calkit value key=bands.lw.best_change_pct path=results/docs/radiation-register-cap.json -->-17.61<!-- /calkit value -->%.
Four of those checks straddle the turnover, including the shortwave rows where one half gains and the other loses, so the additivity is not an artifact of both halves pointing the same way.
That is the finding that makes the rest of the experiment separable: each kernel's budget can be read off its own column, and the best configuration need never have been timed as a pair.

**Each band's two halves want different budgets, in both bands.**
The longwave optics bottoms out at <!-- calkit value key=per_kernel.lw.optics_cap path=results/docs/radiation-register-cap.json -->96<!-- /calkit value --> registers for <!-- calkit value key=per_kernel.lw.optics_change_pct path=results/docs/radiation-register-cap.json -->-11.41<!-- /calkit value -->% while the longwave sweeps is still improving at <!-- calkit value key=per_kernel.lw.sweeps_cap path=results/docs/radiation-register-cap.json -->72<!-- /calkit value --> for <!-- calkit value key=per_kernel.lw.sweeps_change_pct path=results/docs/radiation-register-cap.json -->-8.02<!-- /calkit value -->%, and the shortwave optics bottoms out at <!-- calkit value key=per_kernel.sw.optics_cap path=results/docs/radiation-register-cap.json -->96<!-- /calkit value --> for <!-- calkit value key=per_kernel.sw.optics_change_pct path=results/docs/radiation-register-cap.json -->-4.65<!-- /calkit value -->% while the shortwave sweeps keeps going to <!-- calkit value key=per_kernel.sw.sweeps_cap path=results/docs/radiation-register-cap.json -->64<!-- /calkit value --> for <!-- calkit value key=per_kernel.sw.sweeps_change_pct path=results/docs/radiation-register-cap.json -->-5.51<!-- /calkit value -->%.
Composed, those mixed pairs predict <!-- calkit value key=per_kernel.lw.predicted_change_pct path=results/docs/radiation-register-cap.json -->-19.43<!-- /calkit value -->% and <!-- calkit value key=per_kernel.sw.predicted_change_pct path=results/docs/radiation-register-cap.json -->-10.16<!-- /calkit value -->%, against the <!-- calkit value key=bands.lw.best_change_pct path=results/docs/radiation-register-cap.json -->-17.61<!-- /calkit value -->% and <!-- calkit value key=bands.sw.best_change_pct path=results/docs/radiation-register-cap.json -->-7.25<!-- /calkit value -->% of the best pair a matched grid could express, which is the whole reason the caps are chosen per kernel.
Neither was timed here, which is the honest statement of what this sweep on its own establishes: the composition property is what carries them.

**The prediction has since been timed, in a separate job on a separate day.**
`radiation-mask-layout` compiles the shipped budgets --- optics <!-- calkit value key=confirmation.lw.shipped_optics path=results/docs/radiation-register-cap.json -->96<!-- /calkit value --> and sweeps <!-- calkit value key=confirmation.lw.shipped_sweeps path=results/docs/radiation-register-cap.json -->72<!-- /calkit value --> in the longwave, optics <!-- calkit value key=confirmation.sw.shipped_optics path=results/docs/radiation-register-cap.json -->96<!-- /calkit value --> and sweeps <!-- calkit value key=confirmation.sw.shipped_sweeps path=results/docs/radiation-register-cap.json -->64<!-- /calkit value --> in the shortwave --- against the same uncapped kernels, and measures <!-- calkit value key=confirmation.lw.measured_pct path=results/docs/radiation-register-cap.json -->-19.38<!-- /calkit value -->% and <!-- calkit value key=confirmation.sw.measured_pct path=results/docs/radiation-register-cap.json -->-10.07<!-- /calkit value -->% against the <!-- calkit value key=confirmation.lw.predicted_pct path=results/docs/radiation-register-cap.json -->-19.43<!-- /calkit value -->% and <!-- calkit value key=confirmation.sw.predicted_pct path=results/docs/radiation-register-cap.json -->-10.16<!-- /calkit value -->% composed from the columns above.
That is a worst residual of <!-- calkit value key=worst_confirmation_residual_pts path=results/docs/radiation-register-cap.json -->0.09<!-- /calkit value --> points, over <!-- calkit value key=confirmation.lw.rounds path=results/docs/radiation-register-cap.json -->20<!-- /calkit value --> rounds in which every single one was faster, with the two halves of each run agreeing to within <!-- calkit value key=confirmation.sw.halves_agree path=results/docs/radiation-register-cap.json -->0.04<!-- /calkit value --> points.
So the mixed pairs are measured, not inferred, and the composition property that produced them is confirmed by a configuration it predicted before anything timed it.

**Three configurations bought warps and lost time**, all in the shortwave: <!-- calkit value key=occupancy_bought_nothing path=results/docs/radiation-register-cap.json -->sw both_72, sw optics_64 and sw optics_72<!-- /calkit value -->.
They are the reason the local frame is a diagnostic here and not the decision.
The shortwave optics at <!-- calkit value key=kernels.sw_optics.caps.72.cap path=results/docs/radiation-register-cap.json -->72<!-- /calkit value --> registers reaches <!-- calkit value key=kernels.sw_optics.caps.72.warps_per_sm path=results/docs/radiation-register-cap.json -->28<!-- /calkit value --> warps per multiprocessor, up from <!-- calkit value key=kernels.sw_optics.warps_per_sm path=results/docs/radiation-register-cap.json -->16<!-- /calkit value -->, and runs <!-- calkit value key=bands.sw.points.optics_72.change_pct path=results/docs/radiation-register-cap.json -->6.29<!-- /calkit value -->% slower for it --- the occupancy arrived and the time did not.
So the turnover is a real feature of each kernel rather than a cap the allocator failed to meet, and reading it off the clock is what separates the two.

## Verdict

**Shed registers off all four kernels, at a budget chosen per kernel rather than per band.**
The caps that ship are the argmin of each kernel's own column --- longwave optics <!-- calkit value key=per_kernel.lw.optics_cap path=results/docs/radiation-register-cap.json -->96<!-- /calkit value --> and sweeps <!-- calkit value key=per_kernel.lw.sweeps_cap path=results/docs/radiation-register-cap.json -->72<!-- /calkit value -->, shortwave optics <!-- calkit value key=per_kernel.sw.optics_cap path=results/docs/radiation-register-cap.json -->96<!-- /calkit value --> and sweeps <!-- calkit value key=per_kernel.sw.sweeps_cap path=results/docs/radiation-register-cap.json -->64<!-- /calkit value --> --- and composed they take <!-- calkit value key=confirmation.lw.measured_pct path=results/docs/radiation-register-cap.json -->-19.38<!-- /calkit value -->% off the longwave round loop and <!-- calkit value key=confirmation.sw.measured_pct path=results/docs/radiation-register-cap.json -->-10.07<!-- /calkit value -->% off the shortwave, measured directly rather than summed.

This inverts two earlier rejections on this same family, and the regime is what changed rather than the arithmetic.
Both of those were caps on the *fused* 255-register solvers, where forcing 128 registers roughly doubled a local frame that was already hundreds of bytes, and both lost.
Here the kernels start at <!-- calkit value key=kernels.lw_optics.local_bytes path=results/docs/radiation-register-cap.json -->32<!-- /calkit value --> bytes of local memory and sit at <!-- calkit value key=kernels.lw_optics.warps_per_sm path=results/docs/radiation-register-cap.json -->12<!-- /calkit value --> to <!-- calkit value key=kernels.sw_optics.warps_per_sm path=results/docs/radiation-register-cap.json -->16<!-- /calkit value --> warps against stalls that are overwhelmingly waits on memory, so the warps a cap buys hide latency worth more than the spill traffic they cost.
Spill is not free here either --- it is up to <!-- calkit value key=worst_local_growth_x path=results/docs/radiation-register-cap.json -->11.5<!-- /calkit value --> times the unforced frame --- and it is paid for anyway.

What this does not establish is a wall-clock figure.
The effect is measured on the radiation round loop in isolation, against one settled AMIP state, which resolves far finer than the AMIP stages but says nothing about radiation's share of a step.
The AMIP pair is what converts it, and it is the measurement to quote for SYPD.
