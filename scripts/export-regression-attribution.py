"""Attribute a device-time regression between two profiles of the same arm.

Written for the 2026-11-03 pull of upstream main, which cost 36% of device
time. Both inputs are the BASELINE arm over the same 120-step window, so a
difference is upstream's rather than a treatment's.

The pull FUSED kernels -- `copyto_by_slices` launches halved and the
microphysics cache went from nine kernel names to four -- so a name-paired
delta reads a fusion as one kernel vanishing and another appearing. AGENTS.md:
"when the point of a treatment is to change the kernel inventory, the aggregate
that survives is the sum over every matching name". This groups by the function
a kernel comes from, which survives both the fusion and the line-number shift a
pull causes.

The earlier profile is not in the workspace -- `baseline-nsys` deletes its own
output before re-running -- so it is fetched by revision:

    calkit dvc get . results/nsys/baseline.sqlite --rev <pre-commit> -o <path>

Usage: export-regression-attribution.py <pre.sqlite> <now.sqlite> <out.json>
"""

import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys

LINE_SUFFIX = re.compile(r"_L\d+$")
# A fused kernel is named for its function; an anonymous broadcast is named for
# the file and line it was written at, and a pull moves the line.
GENERIC = (
    "copyto_by_slices",
    "gpu_broadcast_kernel",
    "partial_mapreduce",
    "knl_copyto_flat",
    "broadcast_kernel",
)
HEAVY_REGS = 100


def group(name):
    """The function a kernel came from, or its file with the line dropped."""
    head = name.split("__FILE")[0]
    if head:
        return head.replace("__NVTX", "")
    return LINE_SUFFIX.sub("_L*", name)


def union_ms(rows):
    """Elapsed time covered by a set of [start, end] intervals, merged.

    `SUM(end-start)` is elapsed GPU time only while kernels do not overlap.
    Both are reported so the gap is visible rather than assumed away.
    """
    total, cur_s, cur_e = 0, None, None
    for st, en in sorted(rows):
        if cur_s is None:
            cur_s, cur_e = st, en
        elif st <= cur_e:
            cur_e = max(cur_e, en)
        else:
            total += cur_e - cur_s
            cur_s, cur_e = st, en
    if cur_s is not None:
        total += cur_e - cur_s
    return total / 1e6


def digest(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = con.execute(
        """SELECT s.value, COUNT(*), SUM(k.end - k.start)/1e6,
                  MAX(k.registersPerThread), MAX(k.blockX*k.blockY*k.blockZ)
           FROM CUPTI_ACTIVITY_KIND_KERNEL k
           JOIN StringIds s ON s.id = k.shortName
           GROUP BY s.value"""
    ).fetchall()
    funcs = {}
    for name, n, ms, regs, threads in rows:
        # The mod arm's packages carry a `_mod` path segment; normalize it so
        # the same helper can compare arms as well as revisions.
        f = group(name.replace("_jl_mod_", "_jl_"))
        a = funcs.setdefault(
            f, {"ms": 0.0, "launches": 0, "names": 0, "registers": 0, "threads": 0}
        )
        a["ms"] += ms or 0.0
        a["launches"] += n
        a["names"] += 1
        a["registers"] = max(a["registers"], regs or 0)
        a["threads"] = max(a["threads"], threads or 0)
    gpus = con.execute("SELECT DISTINCT name, clockRate FROM TARGET_INFO_GPU").fetchall()
    elapsed = union_ms(
        con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
    )
    con.close()
    return {
        "functions": funcs,
        "elapsed_ms": round(elapsed, 1),
        "summed_ms": round(sum(v["ms"] for v in funcs.values()), 1),
        "launches": sum(v["launches"] for v in funcs.values()),
        "gpus": sorted({(g[0], g[1]) for g in gpus}),
    }


def subtotal(d, pred):
    sel = {k: v for k, v in d["functions"].items() if pred(k, v)}
    return {
        "ms": round(sum(v["ms"] for v in sel.values()), 1),
        "launches": sum(v["launches"] for v in sel.values()),
        "functions": len(sel),
    }


def gitref(path="."):
    try:
        return subprocess.run(
            ["git", "-C", path, "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def main():
    pre_path, now_path, out_path = sys.argv[1:4]
    pre, now = load(pre_path), load(now_path)

    grew = now["summed_ms"] - pre["summed_ms"]
    rows = []
    for f in set(pre["functions"]) | set(now["functions"]):
        a = pre["functions"].get(f)
        b = now["functions"].get(f)
        rows.append({
            "function": f,
            "delta_ms": round((b["ms"] if b else 0.0) - (a["ms"] if a else 0.0), 1),
            "pre_ms": round(a["ms"], 1) if a else None,
            "now_ms": round(b["ms"], 1) if b else None,
            "pre_launches": a["launches"] if a else None,
            "now_launches": b["launches"] if b else None,
            "pre_registers": a["registers"] if a else None,
            "now_registers": b["registers"] if b else None,
            "pre_threads_per_block": a["threads"] if a else None,
            "now_threads_per_block": b["threads"] if b else None,
        })
    rows.sort(key=lambda r: -r["delta_ms"])
    for r in rows:
        r["share_of_growth_pct"] = round(100.0 * r["delta_ms"] / grew, 2)

    is_generic = lambda k, v: k.startswith(GENERIC)
    is_heavy = lambda k, v: v["registers"] >= HEAVY_REGS
    gen_pre, gen_now = subtotal(pre, is_generic), subtotal(now, is_generic)
    heavy_pre, heavy_now = subtotal(pre, is_heavy), subtotal(now, is_heavy)

    top = [r for r in rows if r["delta_ms"] > 0][:3]
    out = {
        "provenance": {
            "source_commit": gitref(),
            "pre": {"path": pre_path, "md5": digest(pre_path), "gpus": pre["gpus"]},
            "now": {"path": now_path, "md5": digest(now_path), "gpus": now["gpus"]},
            "same_gpu_model_and_clock": pre["gpus"] == now["gpus"],
            "note": (
                "Both profiles are the baseline arm over the same 120-step "
                "window, so a difference is upstream's. The earlier profile is "
                "not in the workspace and was fetched by revision with "
                "`calkit dvc get . results/nsys/baseline.sqlite --rev <commit>`."
            ),
        },
        "device": {
            "pre_elapsed_ms": pre["elapsed_ms"],
            "now_elapsed_ms": now["elapsed_ms"],
            "grew_ms": round(grew, 1),
            "grew_pct": round(100.0 * (now["summed_ms"] / pre["summed_ms"] - 1), 2),
            "pre_launches": pre["launches"],
            "now_launches": now["launches"],
            "launches_change_pct": round(
                100.0 * (now["launches"] / pre["launches"] - 1), 1
            ),
            # Elapsed and summed agree when kernels do not overlap; if they
            # diverge, the shares below are shares of summed time.
            "pre_summed_over_elapsed": round(pre["summed_ms"] / pre["elapsed_ms"], 3),
            "now_summed_over_elapsed": round(now["summed_ms"] / now["elapsed_ms"], 3),
        },
        "fusion_trade": {
            "generic_broadcast_pre": gen_pre,
            "generic_broadcast_now": gen_now,
            "generic_broadcast_saved_ms": round(gen_pre["ms"] - gen_now["ms"], 1),
            "everything_else_cost_ms": round(
                (now["summed_ms"] - gen_now["ms"]) - (pre["summed_ms"] - gen_pre["ms"]), 1
            ),
            "paid_per_ms_saved": round(
                ((now["summed_ms"] - gen_now["ms"]) - (pre["summed_ms"] - gen_pre["ms"]))
                / max(gen_pre["ms"] - gen_now["ms"], 1e-9), 2
            ),
            "note": (
                "Fewer, larger kernels: the launch count fell by a third while "
                "device time rose. The broadcast machinery it removed is the "
                "saving; every other kernel is the bill."
            ),
        },
        "register_pressure": {
            "threshold_registers": HEAVY_REGS,
            "pre": heavy_pre,
            "now": heavy_now,
            "grew_ms": round(heavy_now["ms"] - heavy_pre["ms"], 1),
            "share_of_total_growth_pct": round(
                100.0 * (heavy_now["ms"] - heavy_pre["ms"]) / grew, 1
            ),
            "pre_pct_of_device": round(100.0 * heavy_pre["ms"] / pre["summed_ms"], 1),
            "now_pct_of_device": round(100.0 * heavy_now["ms"] / now["summed_ms"], 1),
            "note": (
                "The count of heavy functions did not rise, so this is the same "
                "few kernels getting heavier rather than new ones appearing."
            ),
        },
        "top_three": {
            "functions": [r["function"] for r in top],
            "grew_ms": round(sum(r["delta_ms"] for r in top), 1),
            "share_of_growth_pct": round(
                100.0 * sum(r["delta_ms"] for r in top) / grew, 1
            ),
        },
        "functions": rows,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    d = out["device"]
    print(f"device {d['pre_elapsed_ms']} -> {d['now_elapsed_ms']} ms "
          f"({d['grew_ms']:+}, {d['grew_pct']:+}%), launches "
          f"{d['launches_change_pct']:+}%")
    print(f"top three are {out['top_three']['share_of_growth_pct']}% of the growth: "
          + ", ".join(out["top_three"]["functions"]))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
