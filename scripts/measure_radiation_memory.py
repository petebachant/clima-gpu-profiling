"""Extract the radiation solvers' memory behavior from an ncu report.

The two `rte_*_2stream_solve_*` kernels are about 15% of device time between
them and sit at the 255-register cap, which pins them to one block per SM and
eight resident warps. At that occupancy there is nothing to hide a memory stall
behind, so the question is not how much traffic they move but how much of it
they move badly: a request that fetches more sectors than its warp asked for
pays full latency with no other warp ready to issue in the meantime.

`--set full` already collects every counter needed, so these stages reuse
`scripts/run-ncu.sh` unchanged rather than adding a script with its own metric
list. Two pages are read, because neither alone is enough:

  * the **raw** page carries the counters under their metric IDs, which is the
    only place sectors and requests appear separately -- `--set full` does not
    collect the `average_t_sectors_per_request` ratios, so coalescing is
    computed here from the two counters rather than read off.
  * the **details** page carries ncu's own rule output and the derived spill
    figures. The recommendations are the part of an ncu report that says what to
    do, and they are lost if only the counters are kept.

The raw page's first data row holds the units rather than a kernel, so it is
read as units and skipped; the units are reported rather than assumed, since
ncu scales a counter's value to Gbyte or ms without asking.
"""

import collections
import csv

# Counters from the raw page. Sectors AND requests for each space, because the
# ratio is the coalescing figure and neither number means much alone.
RAW = {
    "dram_read": "dram__bytes_read.sum",
    "dram_write": "dram__bytes_write.sum",
    "dram_pct_peak": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "compute_pct_peak": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "duration": "gpu__time_duration.sum",
    "l1_hit_pct": "l1tex__t_sector_hit_rate.pct",
    "l2_hit_pct": "lts__t_sector_hit_rate.pct",
    "achieved_occupancy_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "registers": "launch__registers_per_thread",
    "block_size": "launch__block_size",
    "grid_size": "launch__grid_size",
    "global_ld_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "global_st_sectors": "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum",
    "global_ld_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "global_st_requests": "l1tex__t_requests_pipe_lsu_mem_global_op_st.sum",
    "local_ld_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum",
    "local_st_sectors": "l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum",
    "local_ld_requests": "l1tex__t_requests_pipe_lsu_mem_local_op_ld.sum",
    "local_st_requests": "l1tex__t_requests_pipe_lsu_mem_local_op_st.sum",
}
STALLS = {
    "long_scoreboard": "long_scoreboard",   # waiting on a global/local load
    "lg_throttle": "lg_throttle",           # local/global pipe saturated
    "mio_throttle": "mio_throttle",         # memory IO queue full
    "short_scoreboard": "short_scoreboard",  # waiting on shared/MIO
    "wait": "wait",                         # fixed-latency dependency
    "no_instruction": "no_instruction",     # instruction cache miss
    "math_pipe_throttle": "math_pipe_throttle",
    "not_selected": "not_selected",         # ready, another warp went first
}
STALL_FMT = "smsp__average_warps_issue_stalled_{}_per_issue_active.ratio"

# Derived figures the details page computes and the raw counters do not state.
DETAILS = {
    "spill_requests": "Local Memory Spilling Requests",
    "spill_overhead_pct": "Local Memory Spilling Request Overhead",
    "stack_size": "Stack Size",
    "theoretical_occupancy_pct": "Theoretical Occupancy",
    "mem_throughput_pct": "Max Bandwidth",
    "avg_active_threads_per_warp": "Avg. Active Threads Per Warp",
}

# A metric that disappears upstream would otherwise drop a field silently, and
# a document citing a dropped field renders nothing rather than failing.
# Columns from the source page, which is the only place a per-instruction ideal
# appears. A warp-averaged sectors-per-request is diluted by every uniform load
# that needs one sector, so it understates the waste; ncu compares each
# instruction against the sectors its own bytes would need. Note its
# `derived__..._excessive` column loses scale in CSV export -- it prints 18 for
# 17.7 billion -- so excess is computed here as actual minus ideal.
SOURCE = {
    "sectors": "L2 Theoretical Sectors Global",
    "ideal": "L2 Theoretical Sectors Global Ideal",
    "size_bits": "Access Size",
    "op": "Access Operation",
}

REQUIRED = ("dram_read", "global_ld_sectors", "global_ld_requests",
            "local_ld_sectors", "achieved_occupancy_pct", "duration")


def short(name):
    """The solver a kernel name belongs to, without the signature."""
    return name.split("_CUDA_")[0].split("(")[0].strip()


def num(s):
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


def load_raw(path):
    """Per-kernel counters from `ncu --csv --page raw`, averaged per launch."""
    want = dict(RAW)
    want.update({f"stall_{k}": STALL_FMT.format(v) for k, v in STALLS.items()})
    vals = collections.defaultdict(lambda: collections.defaultdict(list))
    units, seen = {}, set()
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"kernels": {}, "units": {}, "missing": sorted(REQUIRED)}
    unit_row = rows[0]
    for key, mid in want.items():
        if mid in unit_row:
            units[key] = unit_row[mid]
            seen.add(key)
    for r in rows[1:]:
        k = short(r.get("Kernel Name", ""))
        if not k:
            continue
        for key, mid in want.items():
            v = num(r.get(mid))
            if v is not None:
                vals[k][key].append(v)
    out = {}
    for k, mv in vals.items():
        kernel = {"launches_profiled": max(len(v) for v in mv.values())}
        for key, v in mv.items():
            kernel[key] = sum(v) / len(v)
        out[k] = kernel
    return {"kernels": out, "units": units, "missing": sorted(set(REQUIRED) - seen)}


def load_details(path):
    """Derived figures and ncu's own recommendations, per kernel."""
    vals = collections.defaultdict(dict)
    rules = collections.defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            k = short(r.get("Kernel Name", ""))
            if not k:
                continue
            m = r.get("Metric Name", "")
            for key, disp in DETAILS.items():
                if m == disp:
                    v = num(r.get("Metric Value"))
                    if v is not None:
                        vals[k][key] = v
            rule = r.get("Rule Name", "")
            if rule and r.get("Rule Description"):
                # ncu repeats a rule once per metric row in this export.
                rules[k][rule] = {
                    "section": r.get("Section Name", ""),
                    "description": " ".join(r["Rule Description"].split()),
                    "estimated_speedup_pct": num(r.get("Estimated Speedup")),
                    "speedup_type": r.get("Estimated Speedup Type", ""),
                }
    return {
        k: {
            **vals[k],
            "rules": sorted(
                ({"rule": n, **b} for n, b in rules[k].items()),
                key=lambda x: -(x["estimated_speedup_pct"] or 0),
            ),
        }
        for k in set(vals) | set(rules)
    }


def derive(k):
    """What the counters imply but do not say.

    `sectors_per_request` is the coalescing figure: a warp of 32 lanes reading
    contiguous 4-byte floats asks for 4 sectors, so 4 is ideal and 32 means
    every lane fetched its own 32-byte sector and 28/32 of the traffic was
    thrown away. `local_share_pct` is register spill as a fraction of all L1
    traffic, which is what 255 registers per thread costs.
    """
    def ratio(sec, req):
        s, q = k.get(sec), k.get(req)
        return round(s / q, 2) if s and q else None

    local = k.get("local_ld_sectors", 0) + k.get("local_st_sectors", 0)
    glob = k.get("global_ld_sectors", 0) + k.get("global_st_sectors", 0)
    total = local + glob
    ld = ratio("global_ld_sectors", "global_ld_requests")
    total_stall = sum(k.get(f"stall_{n}", 0) or 0 for n in STALLS)
    return {
        "global_ld_sectors_per_request": ld,
        "global_st_sectors_per_request": ratio(
            "global_st_sectors", "global_st_requests"
        ),
        "local_ld_sectors_per_request": ratio(
            "local_ld_sectors", "local_ld_requests"
        ),
        "local_sectors": round(local),
        "global_sectors": round(glob),
        "local_share_pct": round(100.0 * local / total, 2) if total else None,
        # The warp-average, kept only because it is what a reader computes
        # first and it is misleading: uniform loads needing one sector dilute
        # it, so it reads 1.3x where the per-instruction ideal says 2.3x. Use
        # `coalescing.amplification_x` from the source page instead.
        "ld_waste_x_warp_average_diluted": round(ld / 4.0, 2) if ld else None,
        # Share of the stall reasons collected here, NOT of all stall cycles:
        # the `STALLS` list is a subset, so this says which of the measured
        # reasons dominates and must not be quoted as a fraction of runtime.
        #
        # It is also the only way to connect the sector waste to time, because
        # per-instruction stall sampling cannot: a long-scoreboard stall is
        # charged to the instruction that CONSUMES the loaded register, not to
        # the load, so attributing samples to the loads finds ~0% on them and
        # ~98% on instructions with no memory access at all. That is an artifact
        # of where the sample lands, not evidence the loads are free.
        "stall_long_scoreboard_share_of_measured_pct": (
            round(100.0 * k["stall_long_scoreboard"] / total_stall, 1)
            if total_stall else None
        ),
        "occupancy_shortfall_pts": (
            round(k["theoretical_occupancy_pct"] - k["achieved_occupancy_pct"], 2)
            if k.get("theoretical_occupancy_pct") and k.get("achieved_occupancy_pct")
            else None
        ),
    }


def arm(raw_path, details_path, source_path=None):
    """One arm's radiation kernels, counters and rules merged."""
    raw = load_raw(raw_path)
    det = load_details(details_path)
    src = load_source(source_path) if source_path else {}
    out = {"units": raw["units"], "missing_metrics": raw["missing"], "kernels": {}}
    for name, k in raw["kernels"].items():
        merged = dict(k)
        merged.update({a: b for a, b in det.get(name, {}).items() if a != "rules"})
        merged["derived"] = derive(merged)
        merged["rules"] = det.get(name, {}).get("rules", [])
        if name in src:
            merged["coalescing"] = src[name]
        out["kernels"][name] = {
            a: (round(b, 4) if isinstance(b, float) else b)
            for a, b in merged.items()
        }
    return out


def classify(size_bits, op, amp):
    """Name the access pattern a (width, amplification) pair describes.

    The widths are what separates the causes. A 4-byte load at 8x has every
    lane in its own 32-byte sector, which is the gas-optics lookup gather: the
    table index depends on the lane's own column, so the addresses are
    unrelated. A 1-byte access at 32x is a `Bool` mask stored with the level
    index fastest, so consecutive lanes sit a column apart -- the same
    transposition the flux buffers already avoid.
    """
    if amp <= 1.05:
        return "coalesced"
    if size_bits == 8:
        return "mask_byte_strided"
    if size_bits == 32 and amp >= 4.0:
        return "lookup_gather"
    return "partially_coalesced"


def load_source(path):
    """Per-kernel sector totals and the excess grouped by access pattern.

    The page repeats a kernel once per profiled launch and reports absolute
    counts, so shares and ratios are quoted rather than totals: both are
    independent of how many launches were profiled.
    """
    kernels = collections.defaultdict(list)
    cur, hdr = None, None
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0] == "Kernel Name":
                cur, hdr = short(row[1]), None
            elif row[0] == "Address":
                hdr = row
            elif hdr and cur:
                kernels[cur].append(dict(zip(hdr, row)))

    out = {}
    for name, rows in kernels.items():
        groups = collections.defaultdict(lambda: {"instructions": 0, "sectors": 0,
                                                  "ideal": 0, "excessive": 0})
        tot = ide = 0
        for r in rows:
            a, i = num(r[SOURCE["sectors"]]), num(r[SOURCE["ideal"]])
            if not a:
                continue
            tot, ide = tot + a, ide + i
            amp = a / i if i else 0.0
            g = groups[classify(num(r[SOURCE["size_bits"]]), r[SOURCE["op"]], amp)]
            g["instructions"] += 1
            g["sectors"] += a
            g["ideal"] += i
            g["excessive"] += a - i
        excess = tot - ide
        for g in groups.values():
            g["amplification_x"] = round(g["sectors"] / g["ideal"], 2) if g["ideal"] else None
            g["pct_of_excess"] = round(100.0 * g["excessive"] / excess, 1) if excess else None
            for key in ("sectors", "ideal", "excessive"):
                g[key] = round(g[key])
        out[name] = {
            "amplification_x": round(tot / ide, 2) if ide else None,
            "excessive_pct": round(100.0 * excess / tot, 1) if tot else None,
            "groups": dict(groups),
        }
    return out
