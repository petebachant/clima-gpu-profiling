"""Every parameter-fold attempt in the record, against the cheap predictors.

The practical question is whether a candidate kernel can be priced before the
fold is written, and in particular whether the size of the parameter set
predicts anything. This collects the attempts the project has measured -- the
compile-only probes and the four kernels folded in the model -- and tests three
candidate predictors against their outcomes.

No GPU and no profiling: every input is a file an earlier stage produced.
"""

import json
import sys
import tomllib
from pathlib import Path

PROBES = Path("results/val-params-registers.toml")
IN_MODEL = Path("results/fold-record/param-fold.json")
POPULATION = Path("results/pointwise-registers.json")
MILEAGE = Path("results/val-mileage.json")
OUT = Path("results/fold-decision.json")

# A register is 4 bytes, so an isbits parameter set of S bytes can occupy at
# most S/4 registers. Call that the payload: the most a fold could free if the
# kernel were holding the whole struct and nothing else changed.
BYTES_PER_REGISTER = 4

# Which probe variants wrap which parameter sets, read off
# scripts/measure-val-params-registers.jl. Every microphysics variant wraps
# both Val(MP) and Val(TPS); the Thermodynamics variants wrap Val(TPS) alone.
MP_AND_TPS = ("mp_sizeof_bytes", "tps_sizeof_bytes")
TPS_ONLY = ("tps_sizeof_bytes",)

# `boxed` is the code-shape fact the probes were built to isolate: whether
# anything stores the parameters into a struct FIELD between the wrap and the
# arithmetic, which reads the constant back at run time and discards the fold.
PROBE_SITES = [
    {
        "site": "one microphysics evaluation",
        "params": MP_AND_TPS,
        "before": "runtime_arguments",
        "after": "val_type_parameters",
        "ceiling": "const_globals",
        "boxed": False,
    },
    {
        "site": "quadrature path, Val unwrapped at the boundary",
        "params": MP_AND_TPS,
        "before": "quad_runtime_arguments",
        "after": "quad_val_type_parameters",
        "ceiling": "quad_const_globals",
        "boxed": True,
    },
    {
        "site": "quadrature path, Val carried through",
        "params": MP_AND_TPS,
        "before": "quad_runtime_arguments",
        "after": "quad_val_carried_in",
        "ceiling": None,
        "boxed": False,
    },
    {
        "site": "evaluator, parameters moved from fields into the type",
        "params": MP_AND_TPS,
        "before": "evaluator_params_as_fields",
        "after": "evaluator_params_in_type",
        "ceiling": None,
        "boxed": False,
    },
    {
        "site": "Thermodynamics q_vap_saturation",
        "params": TPS_ONLY,
        "before": "td_runtime_arguments",
        "after": "td_val_type_parameters",
        "ceiling": "td_const_globals",
        "boxed": False,
    },
]

# The two in-model kernels whose fold wraps exactly the pair the probes wrap
# (`vcmp, vthp` at the ClimaAtmos call sites). The other two folded kernels
# carry a payload this file cannot size, so they are measured for magnitude
# but excluded from the payload test.
IN_MODEL_PAYLOAD = {
    "microphysics_quadrature": MP_AND_TPS,
    "microphysics_updraft": MP_AND_TPS,
}


def payload_registers(probes, keys):
    total = sum(probes[k] for k in keys)
    assert total > 0, f"no parameter size recorded for {keys}"
    return total // BYTES_PER_REGISTER, total


def pct(part, whole):
    return round(100.0 * part / whole, 1) if whole else None


def probe_rows(probes):
    rows = []
    for site in PROBE_SITES:
        before, after = probes[site["before"]], probes[site["after"]]
        regs, total_bytes = payload_registers(probes, site["params"])
        saved = before["registers"] - after["registers"]
        row = {
            "site": site["site"],
            "measured": "compile-only",
            "payload_bytes": total_bytes,
            "payload_registers": regs,
            "registers_before": before["registers"],
            "registers_after": after["registers"],
            "registers_saved": saved,
            "payload_recovered_pct": pct(saved, regs),
            "warps_before": before["warps_per_sm"],
            "warps_after": after["warps_per_sm"],
            "crossed_step": after["warps_per_sm"] > before["warps_per_sm"],
            "boxed_into_a_struct_field": site["boxed"],
            "kernel_larger_than_payload": before["registers"] > regs,
        }
        if site["ceiling"] is not None:
            ceiling = probes[site["ceiling"]]
            row["const_global_registers"] = ceiling["registers"]
            row["val_matches_const_globals"] = (
                after["registers"] == ceiling["registers"]
            )
        rows.append(row)
    return rows


def in_model_rows(record, probes):
    rows = []
    for name, k in sorted(record["kernels"].items()):
        base, mod = k["baseline"], k["mod"]
        row = {
            "site": name,
            "measured": "in the model",
            "registers_before": base["registers"],
            "registers_after": mod["registers"],
            "registers_saved": k["registers_saved"],
            "warps_before": base["warps_per_sm"],
            "warps_after": mod["warps_per_sm"],
            "crossed_step": k["crossed_occupancy_step"],
            "residency_gain_pct": pct(
                mod["warps_per_sm"] - base["warps_per_sm"],
                base["warps_per_sm"],
            ),
            "kernel_time_change_pct": k["mean_change_pct"],
            "baseline_ms": base["total_ms"],
            "pct_of_device_total": pct(
                base["total_ms"], record["device_total"]["baseline_ms"]
            ),
            "boxed_into_a_struct_field": False,
        }
        if name in IN_MODEL_PAYLOAD:
            regs, total_bytes = payload_registers(
                probes, IN_MODEL_PAYLOAD[name]
            )
            row["payload_bytes"] = total_bytes
            row["payload_registers"] = regs
            row["payload_recovered_pct"] = pct(k["registers_saved"], regs)
            row["kernel_larger_than_payload"] = base["registers"] > regs
        rows.append(row)
    return rows


def payload_test(rows):
    """Does the size of the parameter set predict the saving?"""
    sized = [r for r in rows if "payload_registers" in r]
    assert sized, "no row carries a payload size"
    by_payload: dict[int, list[int]] = {}
    for r in sized:
        by_payload.setdefault(r["payload_registers"], []).append(
            r["registers_saved"]
        )
        # A payload the fold recovers more than all of is a payload that
        # cannot be an upper bound; the fold also lets the compiler constant
        # fold downstream of the parameters.
    largest = max(by_payload, key=lambda p: len(by_payload[p]))
    savings = sorted(by_payload[largest])
    over = [r for r in sized if r["registers_saved"] > r["payload_registers"]]
    # Where the fold lands at all: nothing boxes the parameters, and the
    # kernel is bigger than the payload it is being asked to shed.
    landing = [
        r
        for r in sized
        if not r["boxed_into_a_struct_field"]
        and r["kernel_larger_than_payload"]
        and r["registers_saved"] > 0
    ]
    recovered = sorted(r["payload_recovered_pct"] for r in landing)
    others = [r for r in sized if r not in landing]
    return {
        "note": (
            "payload_registers is sizeof(params)/4, the most a fold could free"
            " if the kernel held the whole struct. The test is whether that"
            " number orders the outcomes."
        ),
        "largest_payload_registers": largest,
        "sites_sharing_that_payload": len(savings),
        "savings_at_that_one_payload": savings,
        "saving_spread_registers": savings[-1] - savings[0],
        "size_predicts_the_saving": savings[-1] - savings[0] == 0,
        "savings_exceeding_their_payload": len(over),
        "largest_excess_registers": max(
            (r["registers_saved"] - r["payload_registers"] for r in over),
            default=0,
        ),
        "size_is_an_upper_bound": not over,
        "sites_where_the_fold_lands": len(landing),
        "payload_recovered_pct_when_it_lands": recovered,
        "recovered_min_pct": recovered[0] if recovered else None,
        "recovered_max_pct": recovered[-1] if recovered else None,
        "sites_where_it_does_not": [
            {
                "site": r["site"],
                "registers_before": r["registers_before"],
                "payload_registers": r["payload_registers"],
                "registers_saved": r["registers_saved"],
                "boxed_into_a_struct_field": r["boxed_into_a_struct_field"],
                "kernel_larger_than_payload": r["kernel_larger_than_payload"],
            }
            for r in others
        ],
    }


def no_has_two_causes(rows, probes):
    """What a "no" from the ceiling probe actually rules out.

    Const globals returning nothing says the body AS WRITTEN cannot be folded,
    not that no version of it can be. Two different things produce that answer
    -- a kernel that never spent registers on its parameters, and a kernel
    where something boxes the constant back into a struct field -- and they
    have different remedies. Separating them is another compile-only probe,
    not a profile, which is what makes the distinction worth drawing here.
    """
    by_site = {r["site"]: r for r in rows}
    boxed = by_site["quadrature path, Val unwrapped at the boundary"]
    unboxed = by_site["evaluator, parameters moved from fields into the type"]
    nothing = by_site["Thermodynamics q_vap_saturation"]
    return {
        "nothing_to_fold": {
            "site": nothing["site"],
            "registers": nothing["registers_before"],
            "payload_registers": nothing["payload_registers"],
            "remedy": "none; the kernel was not paying for its parameters",
        },
        "constant_boxed_into_a_struct_field": {
            "site": boxed["site"],
            "registers": boxed["registers_before"],
            "const_global_registers": boxed["const_global_registers"],
            "registers_after_removing_the_boundary": unboxed[
                "registers_after"
            ],
            "registers_saved_after_removing_the_boundary": unboxed[
                "registers_saved"
            ],
            "remedy": (
                "carry the wrapper through to the arithmetic instead of"
                " storing the parameters in a struct field"
            ),
            "settled_by": "another compile-only probe, not a profile",
        },
        "note": (
            "Both of this project's `no` results came from the same probe"
            " reading the same number, and only one of them was final. So a"
            " `no` is a verdict on the body as written: it closes the fold as"
            " written and asks which cause it is, and both answers are"
            " compile-only."
        ),
    }


def ceiling_test(rows, payload):
    """Does the const-global probe predict what Val does?"""
    compared = [r for r in rows if "val_matches_const_globals" in r]
    assert compared, "no probe recorded a const-global comparison"
    agree = [r for r in compared if r["val_matches_const_globals"]]
    # The cross-check that makes this a test rather than a correlation: the
    # sites where a fold returned nothing are the sites where const globals
    # also returned nothing, so the probe would have refused them in advance.
    failed = {r["site"] for r in payload["sites_where_it_does_not"]}
    flagged = {
        r["site"]
        for r in compared
        if r["const_global_registers"] == r["registers_before"]
    }
    return {
        "non_landing_sites": sorted(failed),
        "sites_the_probe_flagged": sorted(flagged),
        "flagged_every_non_landing_site": failed <= flagged,
        "flagged_a_site_that_would_have_paid": sorted(flagged - failed),
        "a_no_has_two_causes": no_has_two_causes(rows, None),
        "note": (
            "Const globals are the ceiling of any constant-folding scheme, so"
            " a body that does not move with them cannot be folded by any"
            " means. Compile-only with `CUDA.@cuda launch=false`: no"
            " simulation, no coupled model, no profile."
        ),
        "sites_compared": len(compared),
        "sites_where_val_matched_const_globals": len(agree),
        "predicts_val_everywhere_measured": len(agree) == len(compared),
        "largest_disagreement_registers": max(
            (
                abs(r["registers_after"] - r["const_global_registers"])
                for r in compared
            ),
            default=0,
        ),
        "sites": [
            {
                "site": r["site"],
                "runtime_arguments": r["registers_before"],
                "const_globals": r["const_global_registers"],
                "val": r["registers_after"],
            }
            for r in compared
        ],
    }


def magnitude_test(rows):
    """Given that a fold lands, what decides how much time it returns?"""
    timed = [r for r in rows if r.get("kernel_time_change_pct") is not None]
    assert timed, "no in-model row carries a time change"
    control = min(timed, key=lambda r: abs(r["registers_saved"]))
    assert control["registers_saved"] == 0, (
        "the control must be a kernel the fold left at the same register "
        f"count; {control['site']} saved {control['registers_saved']}"
    )
    band = abs(control["kernel_time_change_pct"])
    paid = [r for r in timed if abs(r["kernel_time_change_pct"]) > band]
    flat = [r for r in timed if abs(r["kernel_time_change_pct"]) <= band]
    return {
        "note": (
            "A fold returns time by letting another block onto the SM, so the"
            " quantity is warps per SM rather than registers. The control is"
            " the folded kernel whose register count did not move, and its"
            " change is the per-kernel noise this measurement can resolve."
        ),
        "control_site": control["site"],
        "control_change_pct": control["kernel_time_change_pct"],
        "noise_band_pct": band,
        "sites_above_the_band": [
            {
                "site": r["site"],
                "warps_before": r["warps_before"],
                "warps_after": r["warps_after"],
                "residency_gain_pct": r["residency_gain_pct"],
                "kernel_time_change_pct": r["kernel_time_change_pct"],
                "pct_of_device_total": r["pct_of_device_total"],
            }
            for r in sorted(paid, key=lambda r: r["kernel_time_change_pct"])
        ],
        "sites_within_the_band": [
            {
                "site": r["site"],
                "warps_before": r["warps_before"],
                "warps_after": r["warps_after"],
                "crossed_step": r["crossed_step"],
                "registers_saved": r["registers_saved"],
                "kernel_time_change_pct": r["kernel_time_change_pct"],
                "pct_of_device_total": r["pct_of_device_total"],
            }
            for r in sorted(flat, key=lambda r: r["warps_before"])
        ],
        # The record's payoffs all start from the same occupancy, so it bounds
        # the threshold rather than locating it.
        "paid_at_warps_per_sm": sorted({r["warps_before"] for r in paid}),
        "flat_at_warps_per_sm": sorted({r["warps_before"] for r in flat}),
        "crossed_a_step_and_stayed_flat": [
            r["site"] for r in flat if r["crossed_step"]
        ],
        # The two gates are confounded in this record, and the file should say
        # so rather than let the occupancy reading take credit for both: every
        # flat site is also a small one.
        "largest_flat_pct_of_device": max(
            r["pct_of_device_total"] for r in flat
        ),
        "smallest_paying_pct_of_device": min(
            r["pct_of_device_total"] for r in paid
        ),
        "occupancy_and_device_share_confounded": max(
            r["pct_of_device_total"] for r in flat
        )
        < min(r["pct_of_device_total"] for r in paid),
    }


def screen(population, mileage):
    """How many kernels get past the free part of the test at all."""
    return {
        "note": (
            "The free screen, from one static ncu pass over a whole run and"
            " one nsys profile the pipeline already has. A kernel gains from"
            " cheaper registers only where registers cap its occupancy alone"
            " and it is below full occupancy."
        ),
        "kernels_profiled": population["kernels"],
        "register_limited": population["register_limited"],
        "addressable": population["addressable"],
        "addressable_pct": population["addressable_pct"],
        "at_full_occupancy": population["at_full_occupancy"],
        "at_the_register_cap": population["at_register_cap"],
        "ceiling_pct_of_device": mileage["ceiling_pct_of_device"],
        "treated_pct_of_device": mileage["treated_pct_of_device"],
        "untreated_pct_of_device": mileage["untreated_pct_of_device"],
        "untreated_kernels": mileage["untreated_kernels"],
        "largest_untreated_kernel": mileage["largest_untreated_kernel"],
        "largest_untreated_pct_of_device": mileage[
            "largest_untreated_pct_of_device"
        ],
        "largest_untreated_registers_to_shed": mileage[
            "largest_untreated_registers_to_shed"
        ],
    }


def summarize(rows, payload, ceiling, magnitude, scr):
    """Injection-ready scalars, so no figure in the prose is typed."""
    by_site = {r["site"]: r for r in rows}
    td = by_site["Thermodynamics q_vap_saturation"]
    paid = magnitude["sites_above_the_band"]
    flat = magnitude["sites_within_the_band"]
    cf = next(r for r in flat if r["crossed_step"])
    best = min(paid, key=lambda r: r["kernel_time_change_pct"])
    biggest = max(paid, key=lambda r: r["pct_of_device_total"])
    return {
        "payload_registers": payload["largest_payload_registers"],
        "payload_bytes": payload["largest_payload_registers"]
        * BYTES_PER_REGISTER,
        "payload_sites": payload["sites_sharing_that_payload"],
        "savings_list": ", ".join(
            str(v) for v in payload["savings_at_that_one_payload"]
        ),
        "saving_spread": payload["saving_spread_registers"],
        "excess_registers": payload["largest_excess_registers"],
        "landing_sites": payload["sites_where_the_fold_lands"],
        "recovered_min_pct": payload["recovered_min_pct"],
        "recovered_max_pct": payload["recovered_max_pct"],
        "recovered_spread_pct": round(
            payload["recovered_max_pct"] - payload["recovered_min_pct"], 1
        ),
        "ceiling_sites": ceiling["sites_compared"],
        "ceiling_agree": ceiling["sites_where_val_matched_const_globals"],
        "ceiling_flagged": len(ceiling["sites_the_probe_flagged"]),
        "ceiling_false_alarms": len(
            ceiling["flagged_a_site_that_would_have_paid"]
        ),
        "ceiling_disagreement": ceiling["largest_disagreement_registers"],
        "boxed_registers": ceiling["a_no_has_two_causes"][
            "constant_boxed_into_a_struct_field"
        ]["registers"],
        "unboxed_registers": ceiling["a_no_has_two_causes"][
            "constant_boxed_into_a_struct_field"
        ]["registers_after_removing_the_boundary"],
        "unboxed_saving": ceiling["a_no_has_two_causes"][
            "constant_boxed_into_a_struct_field"
        ]["registers_saved_after_removing_the_boundary"],
        "td_payload_registers": td["payload_registers"],
        "td_registers": td["registers_before"],
        "noise_band_pct": magnitude["noise_band_pct"],
        "paid_warps": ", ".join(
            str(v) for v in magnitude["paid_at_warps_per_sm"]
        ),
        "flat_warps": " and ".join(
            str(v) for v in magnitude["flat_at_warps_per_sm"]
        ),
        "best_change_pct": best["kernel_time_change_pct"],
        "best_warps_before": best["warps_before"],
        "best_warps_after": best["warps_after"],
        "biggest_change_pct": biggest["kernel_time_change_pct"],
        "biggest_warps_before": biggest["warps_before"],
        "biggest_warps_after": biggest["warps_after"],
        "biggest_pct_of_device": biggest["pct_of_device_total"],
        "crossed_flat_change_pct": cf["kernel_time_change_pct"],
        "crossed_flat_warps_before": cf["warps_before"],
        "crossed_flat_warps_after": cf["warps_after"],
        "crossed_flat_registers_saved": cf["registers_saved"],
        "largest_flat_pct_of_device": magnitude["largest_flat_pct_of_device"],
        "smallest_paying_pct_of_device": magnitude[
            "smallest_paying_pct_of_device"
        ],
        "kernels_profiled": scr["kernels_profiled"],
        "addressable": scr["addressable"],
        "addressable_pct": scr["addressable_pct"],
        "at_full_occupancy": scr["at_full_occupancy"],
        "ceiling_pct_of_device": scr["ceiling_pct_of_device"],
        "untreated_pct_of_device": scr["untreated_pct_of_device"],
        "untreated_kernels": scr["untreated_kernels"],
        "largest_untreated_pct": scr["largest_untreated_pct_of_device"],
        "largest_untreated_shed": scr["largest_untreated_registers_to_shed"],
    }


def main():
    for path in (PROBES, IN_MODEL, POPULATION, MILEAGE):
        if not path.exists():
            sys.exit(f"{path} is missing; run the stage that writes it first")
    with open(PROBES, "rb") as f:
        probes = tomllib.load(f)
    record = json.loads(IN_MODEL.read_text())
    population = json.loads(POPULATION.read_text())
    mileage = json.loads(MILEAGE.read_text())
    assert probes["mp_isbits"] and probes["tps_isbits"], (
        "a parameter set that is not isbits cannot be a type parameter, so "
        "the fold is not available at all"
    )

    rows = probe_rows(probes) + in_model_rows(record, probes)
    payload = payload_test(rows)
    ceiling = ceiling_test(rows, payload)
    magnitude = magnitude_test(rows)
    scr = screen(population, mileage)
    out = {
        "note": (
            "Every parameter-fold attempt this project has measured, against"
            " three candidate predictors. Assembled from earlier stages'"
            " outputs, so it needs no GPU: the point of the file is that the"
            " decision does not need a profile."
        ),
        "sites": len(rows),
        "attempts": rows,
        "payload_test": payload,
        "ceiling_test": ceiling,
        "magnitude_test": magnitude,
        "screen": scr,
        "summary": summarize(rows, payload, ceiling, magnitude, scr),
        "sources": {
            "probes": str(PROBES),
            "in_model": str(IN_MODEL),
            "population": str(POPULATION),
            "mileage": str(MILEAGE),
            "in_model_prior_commit": record["vs_prior_stack"]["prior_commit"],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    p, c, m = out["payload_test"], out["ceiling_test"], out["magnitude_test"]
    print(f"{out['sites']} fold attempts in the record")
    print(
        f"  payload:  one {p['largest_payload_registers']}-register payload "
        f"saved {p['savings_at_that_one_payload']} registers at "
        f"{p['sites_sharing_that_payload']} sites "
        f"(spread {p['saving_spread_registers']}); "
        f"{p['savings_exceeding_their_payload']} exceeded the payload"
    )
    print(
        f"  ceiling:  const globals predicted Val at "
        f"{c['sites_where_val_matched_const_globals']}/"
        f"{c['sites_compared']} sites, largest disagreement "
        f"{c['largest_disagreement_registers']} registers; flagged "
        f"{len(c['sites_the_probe_flagged'])} site(s) in advance, covering "
        f"the non-landing ones: {c['flagged_every_non_landing_site']}"
    )
    print(
        f"  magnitude: payoffs at {m['paid_at_warps_per_sm']} warps/SM, "
        f"flat at {m['flat_at_warps_per_sm']}, noise band "
        f"±{m['noise_band_pct']}%"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
