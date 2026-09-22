"""Shares of GPU kernel time by block, from results/top-kernels.csv, as JSON so
question answers can cite them as value evidence."""

import csv
import json

rows = list(csv.DictReader(open("results/top-kernels.csv")))


def arm(suffix):
    pct = lambda r: float(r[f"pct_of_gpu_time_{suffix}"])
    rad = [r for r in rows if r["kernel"].startswith("rte_")]
    mp = [r for r in rows if r["kernel"].startswith("set_microphysics_tendency_cache")]
    return {
        "radiation_pct": round(sum(map(pct, rad)), 2),
        "radiation_kernels": len(rad),
        "microphysics_hot_kernel_pct": round(max(map(pct, mp)), 2),
        "microphysics_all_pct": round(sum(map(pct, mp)), 2),
    }


json.dump(
    {"baseline": arm("baseline"), "mod": arm("mod")},
    open("results/kernel-shares.json", "w"),
    indent=2,
)
