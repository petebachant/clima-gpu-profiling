"""Shares of GPU kernel time by block, from results/top-kernels.csv and the
kernel population, as JSON so question answers can cite them as value evidence."""

import csv
import json

rows = list(csv.DictReader(open("results/top-kernels.csv")))
population = json.load(open("results/kernel-population.json"))


def arm(suffix):
    # A kernel absent from one arm has empty columns there, e.g. the clear-sky
    # radiation solves under `rad: allsky`, and contributed nothing to it
    pct = lambda r: float(r[f"pct_of_gpu_time_{suffix}"] or 0.0)
    rad = [r for r in rows if r["kernel"].startswith("rte_")]
    mp = [r for r in rows if r["kernel"].startswith("set_microphysics_tendency_cache")]
    return {
        "radiation_pct": round(sum(map(pct, rad)), 2),
        "radiation_kernels": len(rad),
        "microphysics_hot_kernel_pct": round(max(map(pct, mp)), 2),
        "microphysics_all_pct": round(sum(map(pct, mp)), 2),
    }


def small_kernels(suffix):
    # What the short-launch band costs on the device, which is what fusing it
    # could recover by not materializing intermediates -- a different quantity
    # from the host-side launch overhead priced in the launch-cost experiment.
    bins = [
        b
        for b in population[suffix]["duration_histogram"]
        if b["max_us"] is not None and b["max_us"] <= 25
    ]
    return {
        "threshold_us": 25,
        "launches": sum(b["launches"] for b in bins),
        "pct_of_launches": round(sum(b["pct_of_launches"] for b in bins), 2),
        "pct_of_kernel_time": round(sum(b["pct_of_kernel_time"] for b in bins), 2),
        "total_ms": round(sum(b["total_ms"] for b in bins), 1),
    }


json.dump(
    {
        "baseline": arm("baseline") | {"small_kernels": small_kernels("baseline")},
        "mod": arm("mod") | {"small_kernels": small_kernels("mod")},
    },
    open("results/kernel-shares.json", "w"),
    indent=2,
)
