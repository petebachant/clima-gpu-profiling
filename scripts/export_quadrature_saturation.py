"""Price the removal of the SGS quadrature kernel's redundant saturation work.

Two saturation specific humidities per node were computed and thrown away: the
conditional node carried a total water that nothing read, and the node function
recomputed the saturation `_point_value` had just evaluated at the same
temperature and density. Removing either is bit-for-bit -- one deletes a field
no production path reads, the other reuses a float already in a register -- and
the compile-only probe checks it that way, by hashing every output bit.

The arithmetic saving is real and the occupancy arithmetic decides whether any
of it arrives. `ClimaCore` sizes a slice loop's grid from its register count, and
the environment quadrature broadcast compiles to exactly the top of its occupancy
band, so it has no register headroom at all: the variant that adds two registers
loses a quarter of its resident warps and runs slower while doing less work. The
same removal moves the instantaneous sibling the other way across a band.

The profiles are transient, so the figures come from
`results/quadrature-profiles.json`, which accumulates one entry per revision
(`scripts/snapshot-quadrature-profile.py`), and every comparison is within the
mod arm -- against this arm before the treatment, not against the baseline arm,
which also differs by the sedimentation fold.
"""

import json
import sys
import tomllib
from pathlib import Path

PROFILES = "results/quadrature-profiles.json"
OCCUPANCY = "results/experiments/occupancy-launch-model/stats.json"
PROBES = {
    "untreated": "results/quadrature-registers-work-baseline.toml",
    "saturation_reuse": "results/quadrature-registers-work-mod-satreuse-only.toml",
    "dead_field": "results/quadrature-registers-work-mod-deadfield-only.toml",
    "both_removals": "results/quadrature-registers-work-mod.toml",
}
CAP = "results/quadrature-registers-cap-installed.toml"
CAP_REGISTERS = 128
CAP_TARGET = "quadrature broadcast (L1297)"
CAP_THROUGH_API = "quadrature broadcast, ClimaCore cap"
OUT = Path("results/experiments/quadrature-saturation/stats.json")
BEFORE = "pre_treatment"
KERNELS = ("quadrature_cache", "implicit_tendency")
PROBE_TARGET = "quadrature broadcast (L1297)"
PROBE_CONTROL = "non-quadrature sibling"


def probe_case(path, case):
    d = tomllib.load(open(path, "rb"))
    if d["climaatmos_dirty"]:
        sys.exit(f"{path}: probe ran on a dirty ClimaAtmos tree")
    return d["cases"][case], d["climaatmos_commit"], d["layout"]["points"]


def pct(after, before):
    return round(100 * (after - before) / before, 2)


def main(out=OUT):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    occ = json.load(open(OCCUPANCY))
    profiles = json.load(open(PROFILES))["variants"]
    if BEFORE not in profiles:
        sys.exit(f"{PROFILES}: no '{BEFORE}' entry to compare against")

    # Only variants measured against the SAME arms are comparable. An upstream
    # pull changes the physics under both kernels -- the 2026-11-09 one moved a
    # 1M default onto the precipitation shaft -- so a variant from before it is
    # not a control for one after, however identical its own diff. The group is
    # the one the reference entry belongs to; the rest are reported as dropped
    # rather than silently averaged in.
    stack = profiles[BEFORE].get("stack_digest", "unknown")
    other_stacks = sorted(
        n for n, v in profiles.items()
        if v.get("stack_digest", "unknown") != stack
    )
    profiles = {
        n: v for n, v in profiles.items()
        if v.get("stack_digest", "unknown") == stack
    }
    base = profiles[BEFORE]

    # Every profiled variant against this arm before the treatment. Device time
    # per kernel is what nsys measures reliably; the register count and the block
    # size are compile-time facts, so the occupancy move is exact even where the
    # milliseconds carry run-to-run noise.
    pct_per_ms = occ["prediction"]["sypd_pct_per_device_ms"]
    profiled = {}
    for name, v in profiles.items():
        if name == BEFORE:
            continue
        row = {"label": v["label"], "climaatmos_commit": v["climaatmos_commit"][:9]}
        treated_ms = 0.0
        for k in KERNELS:
            b, a = base[k], v[k]
            row[k] = {
                "before_ms": b["total_ms"], "after_ms": a["total_ms"],
                "change_ms": round(a["total_ms"] - b["total_ms"], 1),
                "change_pct": pct(a["total_ms"], b["total_ms"]),
                "registers_before": b["registers"], "registers_after": a["registers"],
                "registers_change": a["registers"] - b["registers"],
                "block_before": b["block_size"], "block_after": a["block_size"],
                "warps_before": b["warps_per_sm"], "warps_after": a["warps_per_sm"],
                "warps_change": a["warps_per_sm"] - b["warps_per_sm"],
                "band_crossed": a["warps_per_sm"] != b["warps_per_sm"],
            }
            treated_ms += a["total_ms"] - b["total_ms"]
        row["treated_change_ms"] = round(treated_ms, 1)
        row["device_before_ms"] = base["device_total_ms"]
        row["device_after_ms"] = v["device_total_ms"]
        row["device_change_ms"] = round(
            v["device_total_ms"] - base["device_total_ms"], 1)
        row["device_change_pct"] = pct(v["device_total_ms"],
                                       base["device_total_ms"])
        row["sypd_points"] = round(-row["device_change_ms"] * pct_per_ms, 2)
        row["faster"] = row["device_change_ms"] < 0
        # A variant is only unambiguously an improvement if nothing it touches
        # lost occupancy. Net device time can be positive while the heaviest
        # kernel in the model is slower, which is a cost the total hides.
        row["kernels_slower"] = [k for k in KERNELS if row[k]["change_ms"] > 0]
        row["kernels_losing_warps"] = [k for k in KERNELS
                                       if row[k]["warps_change"] < 0]
        row["no_regression"] = not row["kernels_losing_warps"]
        profiled[name] = row

    # The two ways to choose between the variants: the largest net device saving,
    # and the largest saving that costs no kernel any occupancy. They disagree
    # here, which is the decision this experiment hands back.
    headline = None
    if profiled:
        best = min(profiled.items(), key=lambda kv: kv[1]["device_change_ms"])
        clean = [kv for kv in profiled.items() if kv[1]["no_regression"]]
        best_clean = (min(clean, key=lambda kv: kv[1]["device_change_ms"])
                      if clean else None)
        headline = {
            "best_variant": best[0],
            "best_device_change_ms": best[1]["device_change_ms"],
            "best_sypd_points": best[1]["sypd_points"],
            "best_no_regression": best[1]["no_regression"],
            "best_clean_variant": best_clean[0] if best_clean else None,
            "best_clean_device_change_ms": (best_clean[1]["device_change_ms"]
                                            if best_clean else None),
            "best_clean_sypd_points": (best_clean[1]["sypd_points"]
                                       if best_clean else None),
            "best_is_clean": bool(best_clean) and best[0] == best_clean[0],
        }

    # Which half moved the registers, and what the other half is worth at a
    # geometry the two share. The expectation was the opposite of the result:
    # deleting a field removes eight registers from a table live across the whole
    # integration, and it is the half that made `ptxas` allocate more.
    attribution = None
    if {"dead_field", "both_removals"} <= set(profiles):
        pre, dead, both = (profiles[k]["quadrature_cache"]
                           for k in (BEFORE, "dead_field", "both_removals"))
        attribution = {
            "pre_registers": pre["registers"],
            # Measured directly rather than inferred: the reuse applied on its
            # own leaves the count where it was, so the field accounts for all
            # of the move and the reuse for none of it.
            "reuse_alone_registers": (
                profiles["saturation_reuse"]["quadrature_cache"]["registers"]
                if "saturation_reuse" in profiles else None),
            "reuse_alone_warps": (
                profiles["saturation_reuse"]["quadrature_cache"]["warps_per_sm"]
                if "saturation_reuse" in profiles else None),
            "dead_field_registers": dead["registers"],
            "both_registers": both["registers"],
            "dead_field_moved_registers": dead["registers"] - pre["registers"],
            "reuse_moved_registers": both["registers"] - dead["registers"],
            "reuse_is_register_neutral": both["registers"] == dead["registers"],
            "same_geometry": (both["registers"] == dead["registers"]
                              and both["block_size"] == dead["block_size"]),
            # What the saturation reuse is worth once the geometry is settled.
            "reuse_quadrature_change_ms": round(
                both["total_ms"] - dead["total_ms"], 1),
            "reuse_device_change_ms": round(
                profiles["both_removals"]["device_total_ms"]
                - profiles["dead_field"]["device_total_ms"], 1),
            "table_registers_removed": 8,
        }

    # The cliff. The band the quadrature kernel sits in ends at its own register
    # count, so one extra register costs a quarter of its resident warps -- which
    # is why a treatment that removes work can still lose.
    cliff = {
        "kernel_registers": base["quadrature_cache"]["registers"],
        "band_max_registers": occ["model"]["current_band"]["max_registers"],
        "warps_in_band": base["quadrature_cache"]["warps_per_sm"],
        "next_band_down_warps": occ["model"]["steps"][
            [s["max_registers"] for s in occ["model"]["steps"]].index(
                occ["model"]["current_band"]["max_registers"]) + 1
        ]["warps_per_sm"] if occ["model"]["steps"] else None,
    }
    cliff["headroom_registers"] = (
        cliff["band_max_registers"] - cliff["kernel_registers"])
    cliff["at_band_edge"] = cliff["headroom_registers"] == 0

    # The probe, which prices the arithmetic at fixed launch geometry and carries
    # the accuracy claim. It compiles the same register count in every variant,
    # which is exactly why it could not see the cliff: its reading is two above
    # the profile's, and two registers is the whole margin.
    probe, commits = {}, {}
    points = None
    for tag, path in PROBES.items():
        case, ref, points = probe_case(path, PROBE_TARGET)
        ctrl, _, _ = probe_case(path, PROBE_CONTROL)
        probe[tag] = {
            "device_ms": case["device_ms"], "registers": case["registers"],
            "spill_bytes": case["local_bytes"], "block_size": case["maxthreads"],
            "kernels_launched": case["kernels_launched"],
            "out_bits_hash": case["out_bits_hash"],
            "control_device_ms": ctrl["device_ms"],
            "control_out_bits_hash": ctrl["out_bits_hash"],
        }
        commits[tag] = ref[:9]
    u = probe["untreated"]
    probe_summary = {
        "points": points,
        "saved_pct": {t: round(100 * (u["device_ms"] - p["device_ms"])
                               / u["device_ms"], 2)
                      for t, p in probe.items() if t != "untreated"},
        "control_change_pct": {
            t: pct(p["control_device_ms"], u["control_device_ms"])
            for t, p in probe.items() if t != "untreated"},
        "spill_change_bytes": {t: p["spill_bytes"] - u["spill_bytes"]
                               for t, p in probe.items() if t != "untreated"},
        # The point of the treatment: identical output, not nearly identical.
        "bit_for_bit": len({p["out_bits_hash"] for p in probe.values()}) == 1,
        "control_bit_for_bit": len(
            {p["control_out_bits_hash"] for p in probe.values()}) == 1,
        # Identical registers and block in every variant, which is what makes the
        # probe's figure a pure arithmetic measurement -- and what blinded it.
        "registers_unchanged": len({p["registers"] for p in probe.values()}) == 1,
        "block_size_unchanged": len({p["block_size"] for p in probe.values()}) == 1,
        "one_kernel_each": {p["kernels_launched"] for p in probe.values()} == {1},
        "registers": u["registers"],
        "block_size": u["block_size"],
        "profiled_registers": base["quadrature_cache"]["registers"],
    }
    probe_summary["error_registers"] = (
        probe_summary["registers"] - probe_summary["profiled_registers"])
    # The probe is blind to exactly what decided the outcome: its error is at
    # least the headroom the kernel had.
    probe_summary["error_exceeds_headroom"] = (
        probe_summary["error_registers"] > cliff["headroom_registers"])

    # The saturation evaluations one cloudy cell performs in this kernel, by
    # where they sit: one per node to fill the node's unread total water, one per
    # temperature point in `_point_value` to build the state the evaluator
    # receives, and one per temperature point inside the node function to recover
    # the saturation excess from that state. The first and third are what the
    # treatment removes; the second is the only one that has to happen, because
    # each point is at its own temperature.
    nodes, points_max = 8, 16
    work = {
        "nodes": nodes, "points_max": points_max,
        "calls_node_table": nodes, "calls_point_state": points_max,
        "calls_node_function": points_max,
        "calls_before": nodes + 2 * points_max, "calls_after": points_max,
    }
    work["calls_removed"] = work["calls_before"] - work["calls_after"]

    # The register cap, measured through the API the treatment uses rather than
    # by handing `cufunction` a budget directly: a dropped keyword would look
    # exactly like no cap and would only surface as a wasted profiling pair.
    cap_d = tomllib.load(open(CAP, "rb"))
    if cap_d["climaatmos_dirty"]:
        sys.exit(f"{CAP}: the cap probe ran on a dirty ClimaAtmos tree")
    uncapped = cap_d["cases"][CAP_TARGET]
    capped = cap_d["cases"][CAP_THROUGH_API]
    cap = {
        "climaatmos_commit": cap_d["climaatmos_commit"][:9],
        "target_registers": CAP_REGISTERS,
        "uncapped_registers": uncapped["registers"],
        "capped_registers": capped["registers"],
        "uncapped_warps_per_sm": uncapped["warps_per_sm"],
        "capped_warps_per_sm": capped["warps_per_sm"],
        "uncapped_local_bytes": uncapped["local_bytes"],
        "capped_local_bytes": capped["local_bytes"],
        "local_bytes_added": capped["local_bytes"] - uncapped["local_bytes"],
        "uncapped_probe_ms": uncapped["device_ms"],
        "capped_probe_ms": capped["device_ms"],
        "probe_change_pct": pct(capped["device_ms"], uncapped["device_ms"]),
        # The request survived every layer between the broadcast and `ptxas`.
        "request_reaches_the_compile": capped["registers"] <= CAP_REGISTERS
        and capped["registers"] < uncapped["registers"],
        # A cap moves where values live and cannot move the answer; asserted,
        # not assumed.
        "bit_for_bit": capped["out_bits_hash"] == uncapped["out_bits_hash"],
        "buys_a_band": capped["warps_per_sm"] > uncapped["warps_per_sm"],
        # What a harder cap would cost, for the next person who asks.
        "at_96_registers": {
            "registers": uncapped["capped"]["96"]["registers"],
            "local_bytes": uncapped["capped"]["96"]["local_bytes"],
            "warps_per_sm": uncapped["capped"]["96"]["warps_per_sm"],
        },
    }

    doc = {
        "note": __doc__.strip().splitlines()[0],
        "provenance": {
            "profiles": PROFILES,
            "compared_against": BEFORE,
            "pre_treatment_commit": base["climaatmos_commit"][:9],
            "pre_treatment_transcribed_from": base.get("transcribed_from"),
            "probe_commits": commits,
            "profiled_steps": 120,
        },
        "cliff": cliff,
        "headline": headline,
        "attribution": attribution,
        "profiled": profiled,
        "stack": {
            "digest": stack,
            "arms": base.get("stack"),
            "variants_compared": sorted(profiled),
            "variants_on_other_stacks": other_stacks,
            # A reference entry with no stack recorded would compare anything
            # with anything, so say so rather than proceed quietly.
            "reference_stack_known": stack != "unknown",
        },
        "probe": probe,
        "probe_summary": probe_summary,
        "work": work,
        "frames": occ["frames"],
        "cap": cap,
    }
    assert cap["request_reaches_the_compile"], cap
    assert cap["bit_for_bit"], cap
    assert cap["buys_a_band"], cap
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {out}")
    for name, r in profiled.items():
        print(f"  {name}: quadrature {r['quadrature_cache']['change_pct']}% "
              f"({r['quadrature_cache']['warps_before']}->"
              f"{r['quadrature_cache']['warps_after']} warps), implicit "
              f"{r['implicit_tendency']['change_pct']}% "
              f"({r['implicit_tendency']['warps_before']}->"
              f"{r['implicit_tendency']['warps_after']}), device "
              f"{r['device_change_ms']} ms = {r['sypd_points']} points")
    print(f"  probe {probe_summary['saved_pct']}, bit-for-bit "
          f"{probe_summary['bit_for_bit']}")
    print(f"  stack {stack}: compared {sorted(profiled)}")
    if other_stacks:
        print(f"  dropped (other stacks): {other_stacks}")
    return doc


if __name__ == "__main__":
    main()
