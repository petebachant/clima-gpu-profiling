# The kernels this device cannot fill

The longwave g-point split came out of a single observation, recorded in section 4l of `docs/learnings.md` on 2026-09-11: at one thread per column the radiation solve's grid is 96 blocks against 108 SMs, so the whole problem is resident at once, nothing is queued behind it, and an SM that finishes its column early idles until the slowest column is done.
That document also wrote down the fix -- parallelize over g-points -- and called it a substantial restructure rather than a tuning change.
CliMA/RRTMGP.jl#637 is that restructure, and it is worth <!-- calkit value key=conversion.lw_sypd_pct format="{:.3f}" path=results/docs/wave-census.json -->0.695<!-- /calkit value -->% of SYPD.

This experiment asks the same question of every kernel in the profile, so the next member of the family is chosen from a measurement instead of guessed.
It finds <!-- calkit value key=family.configs path=results/docs/wave-census.json -->4<!-- /calkit value --> kernels with the shape the longwave solve had, confirms that the shortwave solve is its closest twin, and predicts and then measures what treating it is worth.
It also finds that the single largest geometric loss in the profile sits in kernels this lever cannot reach, which is the more useful half of the result.

## The ceiling every per-column kernel runs into

At <!-- calkit value key=ceiling.registers_at_cap path=results/docs/wave-census.json -->255<!-- /calkit value --> registers per thread an A100 SM holds <!-- calkit value key=ceiling.resident_threads_per_sm_at_cap path=results/docs/wave-census.json -->256<!-- /calkit value --> threads, so the device holds <!-- calkit value key=ceiling.resident_threads_at_cap format="{:,}" path=results/docs/wave-census.json -->27,648<!-- /calkit value --> of them, and the benchmark's <!-- calkit value key=ceiling.columns format="{:,}" path=results/docs/wave-census.json -->24,576<!-- /calkit value --> columns at one thread each come to <!-- calkit value key=ceiling.waves_at_cap format="{:.2f}" path=results/docs/wave-census.json -->0.89<!-- /calkit value --> of a wave.
That is why all <!-- calkit value key=family.configs path=results/docs/wave-census.json -->4<!-- /calkit value --> kernels at the register cap have the *same* launch geometry whatever their block size: <!-- calkit value key=family.waves format="{:.2f}" path=results/docs/wave-census.json -->0.89<!-- /calkit value --> waves and <!-- calkit value key=family.warp_capacity_pct format="{:.1f}" path=results/docs/wave-census.json -->11.1<!-- /calkit value -->% of the SM's warp capacity.
Block size cannot move it, because the resident thread count at a given register count is fixed and the problem's thread count is fixed with it.

The weaker version of the same ceiling applies to every per-column kernel regardless of registers.
Ignoring register limits entirely, an A100 holds <!-- calkit value key=machine.resident_thread_slots format="{:,}" path=results/docs/wave-census.json -->221,184<!-- /calkit value --> resident threads, and one thread per column occupies <!-- calkit value key=ceiling.column_threads_as_pct_of_slots format="{:.1f}" path=results/docs/wave-census.json -->11.1<!-- /calkit value -->% of them.
This model at this resolution is simply too small for this card along the column axis, and every kernel that parallelizes over columns alone inherits that.
Escaping it means finding a second axis, which is exactly what splitting the g-point loop does.

## What the census costs and what it refuses to cost

```python calkit stage name=values environment=py outputs=[{path: results/docs/wave-census.json, storage: git}] inputs=[results/wave-census.json, results/gpoint-groups.json, results/docs/gpoint-group-count.json, results/sw-gpoint-sweep.toml, results/sw-gpoint-wallclock.toml]
import json
import os
import tomllib

census = json.load(open("results/wave-census.json"))
nsys = json.load(open("results/gpoint-groups.json"))
lwdoc = json.load(open("results/docs/gpoint-group-count.json"))
swsweep = tomllib.load(open("results/sw-gpoint-sweep.toml", "rb"))
swwall = tomllib.load(open("results/sw-gpoint-wallclock.toml", "rb"))

kernels = census["kernels"]
mach = census["machine"]
cap = mach["register_cap"]

# A kernel at the cap gets one block of `threads` per SM only if its registers
# allow it; at 255 registers that is 256 threads however the block is shaped.
resident_per_sm = mach["registers_per_sm"] // (cap + 1)
capped = [k for k in kernels if k["at_register_cap"]]
assert capped, "no kernel is at the register cap"
# Every member must share the geometry, which is the section's claim.
assert len({k["waves"] for k in capped}) == 1, "capped kernels differ in waves"


def member(frag):
    """One capped kernel by name fragment, asserted unique.

    Matching kernels by name is how three other stages here break on an
    upstream rename. The assert makes that a loud failure rather than a
    silently mislabeled row.
    """
    hits = [k for k in capped if frag in k["kernel"]]
    assert len(hits) == 1, f"{frag!r} matched {len(hits)} capped kernels"
    return hits[0]


lw, sw = member("rte_lw_2stream"), member("rte_sw_2stream")

# Device time to SYPD, anchored on the one case where both ends are measured:
# the longwave saving in the profile, and the longwave SYPD gain timed on real
# coupler steps. Anchoring this way cancels the window length rather than
# requiring it to be declared.
lw_saved_ms = nsys["longwave"]["saved_ms"]
lw_sypd_pct = lwdoc["wallclock"]["arms"]["16"]["gain_steady_pct"]
per_ms = lw_sypd_pct / lw_saved_ms


def sypd_of(ms):
    return ms * per_ms


# How much of the geometric floor the longwave treatment actually collected.
# Used to turn another member's floor into a prediction rather than a ceiling.
collected = lw_saved_ms / lw["quantization_loss_ms"]

cats = {
    name: {
        "configs": v["configs"],
        "ms": v["ms"],
        "pct_of_censused": v["pct_of_censused"],
        "loss_ms": v["quantization_loss_ms"],
        "loss_sypd_pct": round(sypd_of(v["quantization_loss_ms"]), 3),
    }
    for name, v in census["categories"].items()
}

# The partial-wave kernels that are NOT at the register cap: low register
# counts, one thread per column, and a grid far below what their registers
# would allow. Every one of them is a vertical scan -- a column accumulate, a
# column reduce, or an implicit tridiagonal solve -- so the serial axis is the
# vertical and there is no second axis to split.
vertical = [
    k
    for k in kernels
    if k["category"] == "partial_wave" and not k["at_register_cap"]
]
vertical.sort(key=lambda k: -k["quantization_loss_ms"])
vertical_loss = sum(k["quantization_loss_ms"] for k in vertical)

# The prediction: the shortwave floor, scaled by what the longwave collected.
sw_predicted_ms = sw["quantization_loss_ms"] * collected
sw_predicted_sypd = sypd_of(sw_predicted_ms)
sw_measured_sypd = swwall["arms"][str(swwall["default_ngpt_groups"])][
    "sypd_gain_pct"
]
sw_pts = swsweep["points"]

treated_kernels = {lw["kernel"], sw["kernel"]}
untreated = [k for k in capped if k["kernel"] not in treated_kernels]
remaining_sypd = sum(sypd_of(k["quantization_loss_ms"]) * collected for k in untreated)
collected_sypd = lw_sypd_pct + sw_measured_sypd

out = {
    "machine": {
        "sms": mach["sms"],
        "resident_thread_slots": mach["resident_thread_slots"],
    },
    "device": {
        "elapsed_ms": census["device"]["elapsed_ms"],
        "censused_ms": census["device"]["censused_ms"],
        "censused_pct_of_device": census["device"]["censused_pct_of_device"],
        "total_loss_ms": round(sum(c["loss_ms"] for c in cats.values()), 1),
        "total_loss_pct_of_device": round(
            100.0
            * sum(c["loss_ms"] for c in cats.values())
            / census["device"]["elapsed_ms"],
            2,
        ),
        "total_loss_sypd_pct": round(
            sypd_of(sum(c["loss_ms"] for c in cats.values())), 3
        ),
    },
    "ceiling": {
        "columns": census["problem"]["columns"],
        "column_threads_as_pct_of_slots": census["problem"][
            "column_threads_as_pct_of_slots"
        ],
        "registers_at_cap": cap,
        "resident_threads_per_sm_at_cap": resident_per_sm,
        "resident_threads_at_cap": mach["sms"] * resident_per_sm,
        "waves_at_cap": capped[0]["waves"],
    },
    "conversion": {
        "lw_saved_ms": lw_saved_ms,
        "lw_sypd_pct": lw_sypd_pct,
        "sypd_pct_per_device_ms": round(per_ms, 6),
        "lw_geometric_loss_ms": lw["quantization_loss_ms"],
        "lw_collected_fraction": round(collected, 3),
    },
    "family": {
        "configs": len(capped),
        "ms": round(sum(k["ms"] for k in capped), 1),
        "pct_of_device": census["at_register_cap"]["pct_of_device"],
        "waves": capped[0]["waves"],
        "warp_capacity_pct": capped[0]["warp_capacity_pct"],
        "loss_ms": round(sum(k["quantization_loss_ms"] for k in capped), 1),
        "treated": len(treated_kernels),
        "untreated": len(untreated),
        "remaining_sypd_pct": round(remaining_sypd, 3),
        "collected_sypd_pct": round(collected_sypd, 3),
        "largest_untreated": (
            max(untreated, key=lambda k: k["quantization_loss_ms"])["kernel"]
            if untreated
            else None
        ),
        "lw": {"ms": lw["ms"], "loss_ms": lw["quantization_loss_ms"]},
        "sw": {
            "ms": sw["ms"],
            "loss_ms": sw["quantization_loss_ms"],
            "blocks": sw["blocks"],
            "threads": sw["threads_per_block"],
        },
    },
    "prediction": {
        "sw_predicted_saving_ms": round(sw_predicted_ms, 1),
        "sw_predicted_sypd_pct": round(sw_predicted_sypd, 3),
        "sw_measured_sypd_pct": sw_measured_sypd,
        "error_pts": round(sw_measured_sypd - sw_predicted_sypd, 3),
        "measured_beat_prediction": sw_measured_sypd > sw_predicted_sypd,
        # Directly comparable to lw_collected_fraction: what share of its own
        # geometric floor each band's split actually collected.
        "sw_collected_fraction": round(
            sw_measured_sypd / sypd_of(sw["quantization_loss_ms"]), 3
        ),
        # The longwave figure moved 0.96 points between two simulation states,
        # which is the method's own reproducibility and the right yardstick.
        "state_shift_pts": abs(lwdoc["prediction"]["state_shift_pts"]),
    },
    "shortwave": {
        "n_gpt": swsweep["n_gpt_sw"],
        "best_change_pct": swsweep["best_change_pct"],
        "sufficient_ngroups": swsweep["sufficient_ngroups"],
        "sufficient_is_sweep_maximum": swsweep["sufficient_is_sweep_maximum"],
        "repeat_spread_pct": swsweep["repeat_spread_pct"],
        "default_change_pct": sw_pts[str(swsweep["default_ngpt_groups"])][
            "change_pct"
        ],
        "default_workspace_gib": sw_pts[str(swsweep["default_ngpt_groups"])][
            "workspace_gib"
        ],
        "worst_flux_deviation": max(
            sw_pts[str(n)]["worst_flux_deviation"] for n in swsweep["group_counts"]
        ),
        "aod_bit_identical": swsweep["aod_bit_identical_at_every_count"],
        # A reordering bug would scale with the number of partials; roundoff
        # does not. Eight times the partials, and this is the ratio.
        "deviation_ratio_32_over_4": round(
            sw_pts["32"]["worst_flux_deviation"]
            / sw_pts["4"]["worst_flux_deviation"],
            3,
        ),
        "sypd_grouped": swwall["arms"][str(swwall["default_ngpt_groups"])][
            "sypd_grouped"
        ],
        "sypd_ungrouped": swwall["arms"][str(swwall["default_ngpt_groups"])][
            "sypd_ungrouped"
        ],
        "step_ms": swwall["step_ms"],
    },
    "categories": cats,
    "vertical": {
        "configs": len(vertical),
        "ms": round(sum(k["ms"] for k in vertical), 1),
        "loss_ms": round(vertical_loss, 1),
        "loss_sypd_pct": round(sypd_of(vertical_loss), 3),
        "worst_waves": min(k["waves"] for k in vertical),
        "largest_loss_ms": vertical[0]["quantization_loss_ms"],
        "largest_ms": vertical[0]["ms"],
        "largest_blocks_resident_per_sm": vertical[0]["blocks_resident_per_sm"],
        "all_one_thread_per_column": all(
            k["one_thread_per_column"] for k in vertical
        ),
    },
}
os.makedirs("results/docs", exist_ok=True)
json.dump(out, open("results/docs/wave-census.json", "w"), indent=2, sort_keys=True)
```

The census classifies every kernel by launch geometry and charges each one the time its geometry wastes: blocks occupy `ceil(waves)` wave-times to do `waves` wave-times of work, so the waste is `1 - waves / ceil(waves)`.
Across <!-- calkit value key=device.censused_ms format="{:,.0f}" path=results/docs/wave-census.json -->33,493<!-- /calkit value --> ms of censused kernel time, which is <!-- calkit value key=device.censused_pct_of_device format="{:.0f}" path=results/docs/wave-census.json -->96<!-- /calkit value -->% of device time, that comes to <!-- calkit value key=device.total_loss_ms format="{:,.0f}" path=results/docs/wave-census.json -->1,752<!-- /calkit value --> ms, or <!-- calkit value key=device.total_loss_pct_of_device format="{:.2f}" path=results/docs/wave-census.json -->5.00<!-- /calkit value -->% of device time and <!-- calkit value key=device.total_loss_sypd_pct format="{:.2f}" path=results/docs/wave-census.json -->4.57<!-- /calkit value -->% of SYPD if all of it were somehow collectable.

What the census refuses to charge for is occupancy.
A kernel can have a block on every SM and still hold a fraction of the warps its registers would allow, and that is reported beside the loss rather than inside it.
The reason is in `docs/learnings.md`: ncu's top-ranked recommendation has been occupancy four times in this project and has been the wrong lever every time, and on these very radiation kernels doubling occupancy measured +3.93%.
So the loss column is a floor on what launch geometry alone wastes, not an estimate of what any treatment returns.

The conversion from device milliseconds to SYPD is anchored on the one case where both ends are measured: the longwave saving of <!-- calkit value key=conversion.lw_saved_ms format="{:.1f}" path=results/docs/wave-census.json -->266.2<!-- /calkit value --> ms in the profile against the <!-- calkit value key=conversion.lw_sypd_pct format="{:.3f}" path=results/docs/wave-census.json -->0.695<!-- /calkit value -->% of SYPD timed on real coupler steps.
Anchoring it that way cancels the profiling window's length instead of requiring it to be declared, and it makes every SYPD figure below traceable to a measurement rather than to an assumed conversion.

## How the longwave split scored against its own geometry

The longwave kernel's geometry wastes <!-- calkit value key=family.lw.loss_ms format="{:.1f}" path=results/docs/wave-census.json -->299.9<!-- /calkit value --> ms of its <!-- calkit value key=family.lw.ms format="{:,.1f}" path=results/docs/wave-census.json -->2,698.8<!-- /calkit value --> ms, and the split collected <!-- calkit value key=conversion.lw_saved_ms format="{:.1f}" path=results/docs/wave-census.json -->266.2<!-- /calkit value --> ms of that, or <!-- calkit value key=conversion.lw_collected_fraction format="{:.0%}" path=results/docs/wave-census.json -->89%<!-- /calkit value -->.
That fraction is the useful number, because it turns another member's floor into a prediction rather than a ceiling.
It is below 1 for reasons the treatment does not remove: the grouped path pays for its own reduction and hoisted aerosol mask, and the blocks are not perfectly equal in cost.

## The shortwave prediction, and what it measured

The shortwave two-stream solve is the closest member of the family.
It has the same <!-- calkit value key=family.sw.blocks path=results/docs/wave-census.json -->96<!-- /calkit value --> blocks of <!-- calkit value key=family.sw.threads path=results/docs/wave-census.json -->256<!-- /calkit value --> threads, the same register cap, the same geometry, and the same independent g-point loop with flux accumulation -- so porting the split to it is a port and not a design.
Its geometry wastes <!-- calkit value key=family.sw.loss_ms format="{:.1f}" path=results/docs/wave-census.json -->282.0<!-- /calkit value --> ms, which at the longwave's collected fraction predicts <!-- calkit value key=prediction.sw_predicted_saving_ms format="{:.1f}" path=results/docs/wave-census.json -->250.3<!-- /calkit value --> ms and <!-- calkit value key=prediction.sw_predicted_sypd_pct format="{:.3f}" path=results/docs/wave-census.json -->0.654<!-- /calkit value -->% of SYPD.

Measured, timing real coupler steps in the same process the way the longwave sweep did, it is <!-- calkit value key=prediction.sw_measured_sypd_pct format="{:.3f}" path=results/docs/wave-census.json -->0.605<!-- /calkit value -->% of SYPD, taking SYPD from <!-- calkit value key=shortwave.sypd_ungrouped format="{:.4f}" path=results/docs/wave-census.json -->1.0979<!-- /calkit value --> to <!-- calkit value key=shortwave.sypd_grouped format="{:.4f}" path=results/docs/wave-census.json -->1.1045<!-- /calkit value -->.
The prediction erred by <!-- calkit value key=prediction.error_pts format="{:+.3f}" path=results/docs/wave-census.json -->-0.048<!-- /calkit value --> percentage points, against the <!-- calkit value key=prediction.state_shift_pts format="{:.2f}" path=results/docs/wave-census.json -->0.96<!-- /calkit value --> points by which the longwave figure moved between two simulation states -- which is the method's own reproducibility and so the right yardstick for it.

The shortwave has one source of saving the longwave does not, and it is worth stating because it is the reason the prediction above could be too low rather than too high.
Half the columns are dark, and section 4l measured that: the night criterion covers 50% of points and 45.96% of warps, so it is genuinely warp-coherent.
Those columns skip both sweeps, so they finish early -- and in the ungrouped solve that buys nothing, because the grid is one resident wave and a block that retires frees no capacity for work that is not queued.
Splitting the g-point loop is what queues the work, so the same imbalance the night skip could not collect in 2026-09-11 becomes absorbable.
Geometry alone does not predict that, so whether the measurement shows it is recorded rather than argued: `measured_beat_prediction` is <!-- calkit value key=prediction.measured_beat_prediction path=results/docs/wave-census.json -->false<!-- /calkit value -->.

It did not, and the two bands' collected fractions are the clean way to see it: the longwave took <!-- calkit value key=conversion.lw_collected_fraction format="{:.0%}" path=results/docs/wave-census.json -->89%<!-- /calkit value --> of its geometric floor and the shortwave <!-- calkit value key=prediction.sw_collected_fraction format="{:.0%}" path=results/docs/wave-census.json -->82%<!-- /calkit value -->.
So the shortwave collected slightly *less* of its floor rather than more, and the night-column imbalance section 4l could not reach is still not collected -- splitting the g-point loop made it absorbable in principle without the solve going looking for it, because the optics are still computed for a dark column and only the two sweeps are skipped.
Skipping the optics as well is a separate change, and this is the measurement that says what is left on that table.

In the solve itself the split is worth <!-- calkit value key=shortwave.default_change_pct format="{:.2f}" path=results/docs/wave-census.json -->-8.47<!-- /calkit value -->% at the shipped default of 16 groups, for <!-- calkit value key=shortwave.default_workspace_gib format="{:.2f}" path=results/docs/wave-census.json -->2.05<!-- /calkit value --> GiB of workspace, against a repeat spread of <!-- calkit value key=shortwave.repeat_spread_pct format="{:.3f}" path=results/docs/wave-census.json -->0.258<!-- /calkit value -->%.
The best saving the sweep reached is <!-- calkit value key=shortwave.best_change_pct format="{:.2f}" path=results/docs/wave-census.json -->-10.56<!-- /calkit value -->% at <!-- calkit value key=shortwave.sufficient_ngroups path=results/docs/wave-census.json -->32<!-- /calkit value --> groups.
Whether that is a knee or just the end of the sweep is recorded rather than inferred: `sufficient_is_sweep_maximum` is <!-- calkit value key=shortwave.sufficient_is_sweep_maximum path=results/docs/wave-census.json -->true<!-- /calkit value -->.

Correctness is checked in the same pass.
The worst flux deviation from the ungrouped solve across every group count is <!-- calkit value key=shortwave.worst_flux_deviation format="{:.2e}" path=results/docs/wave-census.json -->1.46e-03<!-- /calkit value --> W/m2, against fluxes of order a few hundred.
What separates a different summation order from a dropped g-point is whether that deviation scales with the number of partials, and it does not: going from 4 groups to 32 multiplies the partials by eight and the deviation by <!-- calkit value key=shortwave.deviation_ratio_32_over_4 format="{:.2f}" path=results/docs/wave-census.json -->1.04<!-- /calkit value -->.
The 550 nm aerosol optical depth is asserted *exactly* rather than to a tolerance, and it holds at every count: <!-- calkit value key=shortwave.aod_bit_identical path=results/docs/wave-census.json -->true<!-- /calkit value -->.
It needs asserting because it is the one quantity the split could corrupt without touching a flux -- it is zeroed and re-accumulated per g-point, so it is a read-modify-write on one per-column address, and concurrent groups would interleave their partial sums rather than agree.
The implementation nominates the single g-point the ungrouped loop leaves standing as the only writer.

## The largest geometric loss is in kernels this lever cannot reach

This is the half of the census worth more than the shortwave port.
The biggest loss category is not the register-capped family at all; it is <!-- calkit value key=vertical.configs path=results/docs/wave-census.json -->8<!-- /calkit value --> kernels wasting <!-- calkit value key=vertical.loss_ms format="{:,.0f}" path=results/docs/wave-census.json -->974<!-- /calkit value --> ms, which is <!-- calkit value key=vertical.loss_sypd_pct format="{:.2f}" path=results/docs/wave-census.json -->2.54<!-- /calkit value -->% of SYPD, running as low as <!-- calkit value key=vertical.worst_waves format="{:.2f}" path=results/docs/wave-census.json -->0.11<!-- /calkit value --> of a wave.
Every one of them is one thread per column: <!-- calkit value key=vertical.all_one_thread_per_column path=results/docs/wave-census.json -->true<!-- /calkit value -->.
Unlike the capped family they have registers to spare, so their SMs could hold <!-- calkit value key=vertical.largest_blocks_resident_per_sm path=results/docs/wave-census.json -->5<!-- /calkit value --> blocks each and are given one.

The largest is `ClimaAtmos.jl/src/prognostic_equations/advection.jl:798`, wasting <!-- calkit value key=vertical.largest_loss_ms format="{:.0f}" path=results/docs/wave-census.json -->373<!-- /calkit value --> ms of <!-- calkit value key=vertical.largest_ms format="{:.0f}" path=results/docs/wave-census.json -->469<!-- /calkit value --> ms, and that line is `Operators.column_accumulate!`.
The others are the same shape: `ClimaCore.jl/src/Operators/integrals.jl:150` is `column_reduce_device!`, and the rest are implicit tridiagonal solves in `prognostic_equations/implicit/`.
They are vertical scans, and section 4l already ruled on them -- the vertical sweeps are recursive and cannot be parallelized -- so the g-point lever does not apply and no block-size change helps either, because the waste comes from the thread count and not from how the threads are grouped.

Collecting it would need a parallel scan over levels in ClimaCore, which is a real project rather than a tuning change.
The census's contribution is to price it: <!-- calkit value key=vertical.loss_sypd_pct format="{:.2f}" path=results/docs/wave-census.json -->2.54<!-- /calkit value -->% of SYPD, concentrated in <!-- calkit value key=vertical.configs path=results/docs/wave-census.json -->8<!-- /calkit value --> kernels, which is a better-founded case for that work than anything previously available.

## What the family is worth

Of the register-capped family, <!-- calkit value key=family.treated path=results/docs/wave-census.json -->2<!-- /calkit value --> of <!-- calkit value key=family.configs path=results/docs/wave-census.json -->4<!-- /calkit value --> are now treated, and they were worth <!-- calkit value key=family.collected_sypd_pct format="{:.2f}" path=results/docs/wave-census.json -->1.30<!-- /calkit value -->% of SYPD between them.
The <!-- calkit value key=family.untreated path=results/docs/wave-census.json -->2<!-- /calkit value --> that remain are worth <!-- calkit value key=family.remaining_sypd_pct format="{:.3f}" path=results/docs/wave-census.json -->0.048<!-- /calkit value -->% of SYPD between them at the longwave's collected fraction, dominated by `non_orographic_gravity_wave_forcing`, which is not obviously worth the code.
So the lever is close to exhausted and the two radiation solves were nearly all of it.

The census's lasting value is therefore not the shortwave port.
It is the <!-- calkit value key=vertical.loss_sypd_pct format="{:.2f}" path=results/docs/wave-census.json -->2.54<!-- /calkit value -->% of SYPD sitting in the vertical scans, and the statement that no amount of launch tuning will reach it.

## What it costs

The shortwave workspace is <!-- calkit value key=shortwave.default_workspace_gib format="{:.2f}" path=results/docs/wave-census.json -->2.05<!-- /calkit value --> GiB at the shipped default, on top of the longwave's, and both scale with `ncol * ngroups`.
At this resolution the AMIP run uses about 11 GiB of an 80 GiB card, so that is affordable; at ten times the columns it would not be, and the group count is the dial that trades it back.
The split is not bit-for-bit, which is a real cost and not an accounting one: a 256-term `Float32` sum taken in N partials is a different summation order, so neither band's fluxes reproduce the serial path exactly.
