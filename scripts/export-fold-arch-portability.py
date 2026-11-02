"""Does the parameter fold still pay on a GPU that is not an A100?

The fold's cost is architecture-independent: a runtime value in a type
parameter makes a specialization no pkgimage can hold, measured at
`results/fold-compile-cost.json`. Its benefit is not. The benefit is fewer
registers per thread buying another resident block per SM, and how many
registers that takes depends on three per-architecture numbers.

Two things are computed here, and they answer different halves.

The occupancy half is exact arithmetic, so it is checked rather than
trusted: the model must reproduce every warps/SM this project has actually
measured on sm_80 before any prediction is emitted. It reproduces six of
six, which is what makes the sm_86/sm_89/sm_90 columns worth reading.

The fragility half is the one that matters. `ptxas` allocates registers per
target, so the same PTX does not get the same count on every architecture,
and a fold that lands exactly on a step boundary loses its step to a drift
of one register. `headroom_registers` is the distance to the next boundary;
`drift_registers` is how far allocation actually moves between targets,
measured by compiling one body for each of them. A fold whose headroom is
below the drift is a fold that may not survive the move.

What this cannot say is whether the kernel is still latency-bound on the
other device, which is what makes occupancy worth anything. That needs the
hardware. An occupancy argument predicts where work is placed, never how
long it takes --- see section 8m of docs/learnings.md, which is the same
error in the opposite direction.
"""

import argparse
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

# Registers per SM and the register allocation granule are the same across
# these targets; threads/SM and blocks/SM are not. blocks/SM is read off
# NVIDIA's own cuda_occupancy.h (cudaOccMaxBlocksPerMultiprocessor), which
# is the authority available locally.
ARCHS = {
    "sm_80": dict(name="A100", threads_sm=2048, max_blocks=32),
    "sm_86": dict(name="GA10x", threads_sm=1536, max_blocks=16),
    "sm_89": dict(name="Ada/L40S", threads_sm=1536, max_blocks=24),
    "sm_90": dict(name="H100", threads_sm=2048, max_blocks=32),
}
REGS_PER_SM = 65536
REG_GRANULE = 256  # per warp, CC 7.0+


def warps_per_sm(regs, block, threads_sm, max_blocks):
    """Resident warps per SM, by CUDA's own occupancy rules."""
    warps_per_block = block // 32
    if warps_per_block == 0 or regs <= 0:
        return 0
    regs_per_warp = math.ceil(regs * 32 / REG_GRANULE) * REG_GRANULE
    by_regs = (REGS_PER_SM // regs_per_warp) // warps_per_block
    by_threads = threads_sm // block
    return min(by_regs, by_threads, max_blocks) * warps_per_block


def headroom(regs, block, arch):
    """How many more registers this kernel could use before losing a step,
    and how many fewer it would need to gain one."""
    base = warps_per_sm(regs, block, **_lim(arch))
    up = 0
    while up < 256 and warps_per_sm(regs + up + 1, block, **_lim(arch)) == base:
        up += 1
    down = 0
    while (
        regs - down - 1 > 0
        and down < 256
        and warps_per_sm(regs - down - 1, block, **_lim(arch)) == base
    ):
        down += 1
    return up, down


def _lim(arch):
    a = ARCHS[arch]
    return dict(threads_sm=a["threads_sm"], max_blocks=a["max_blocks"])


# A body with enough live state to land in the range these kernels occupy,
# compiled once to PTX and then assembled for each target. Synthetic, so it
# bounds the size of the drift rather than predicting any one kernel's.
_PROBE_CU = r"""
__global__ void probe(float* o, const float* a, int n) {
    float v[NLIVE];
#pragma unroll
    for (int i = 0; i < NLIVE; ++i) v[i] = a[i] * (float)(i + 1);
    float s = 0.f;
    for (int k = 0; k < n; ++k) {
        float x = (float)k;
#pragma unroll
        for (int i = 0; i < NLIVE; ++i) {
            v[i] = __fmaf_rn(v[i], 1.000001f, __sinf(x + v[(i + 1) % NLIVE]));
            s += v[i];
        }
    }
    o[threadIdx.x] = s;
}
"""
_REGS_RE = re.compile(r"Used (\d+) registers")


def measure_drift(nlives=(8, 16, 24, 32)):
    """How far ptxas's register allocation moves between targets, for the
    same PTX. Returns per-arch deltas against sm_80, or None when the CUDA
    toolchain is unavailable."""
    if not (shutil.which("nvcc") and shutil.which("ptxas")):
        return None
    rows = []
    with tempfile.TemporaryDirectory() as d:
        src = Path(d) / "probe.cu"
        src.write_text(_PROBE_CU)
        for nlive in nlives:
            ptx = Path(d) / f"probe{nlive}.ptx"
            r = subprocess.run(
                ["nvcc", f"-DNLIVE={nlive}", "-arch=sm_80", "-ptx",
                 str(src), "-o", str(ptx)],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                continue
            counts = {}
            for arch in ARCHS:
                p = subprocess.run(
                    ["ptxas", f"-arch={arch}", "-v", str(ptx), "-o", "/dev/null"],
                    capture_output=True, text=True,
                )
                m = _REGS_RE.search(p.stderr + p.stdout)
                if m:
                    counts[arch] = int(m.group(1))
            if "sm_80" in counts and len(counts) == len(ARCHS):
                rows.append({
                    "live_values": nlive,
                    "registers": counts,
                    "delta_vs_sm_80": {
                        a: counts[a] - counts["sm_80"] for a in counts
                    },
                })
    if not rows:
        return None
    # Only an upward drift can cost a step: landing on fewer registers
    # keeps the block count or improves it. So the figure a headroom is
    # compared against is the largest positive move, per target.
    worst_up = {
        a: max(
            [row["delta_vs_sm_80"].get(a, 0) for row in rows] + [0]
        )
        for a in ARCHS
    }
    return {
        "probes": rows,
        "max_positive_delta_registers": worst_up,
        "max_abs_delta_registers": max(
            abs(v) for row in rows for v in row["delta_vs_sm_80"].values()
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sedimentation", default="results/sedimentation-fold.json")
    ap.add_argument("--population", default="results/val-mileage.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sed = json.load(open(args.sedimentation))
    pop = json.load(open(args.population))

    # Every warps/SM this project has measured on sm_80, which the model has
    # to reproduce before its other columns mean anything.
    checks = []
    for key, label in [
        ("env_terminal_velocity", "env"),
        ("updraft_terminal_velocity", "updraft"),
    ]:
        k = sed["kernels"][key]
        for arm in ("baseline", "mod"):
            checks.append({
                "kernel": f"{label}_{arm}",
                "registers": k["registers"][arm],
                "block": k["block_size"][arm],
                "measured_warps_per_sm": k["warps_per_sm"][arm],
            })
    for cand in pop["candidates"][:2]:
        checks.append({
            "kernel": cand["kernel"].split("__FILE__")[0][:48],
            "registers": cand["registers"],
            "block": cand["block_size"],
            "measured_warps_per_sm": cand["warps_per_sm"],
        })
    for c in checks:
        c["model_warps_per_sm"] = warps_per_sm(
            c["registers"], c["block"], **_lim("sm_80")
        )
        c["agrees"] = c["model_warps_per_sm"] == c["measured_warps_per_sm"]
    if not all(c["agrees"] for c in checks):
        bad = [c["kernel"] for c in checks if not c["agrees"]]
        raise SystemExit(
            "the occupancy model does not reproduce measured sm_80 values "
            f"for {bad}; refusing to emit per-architecture predictions"
        )

    drift = measure_drift()
    out = {
        "architectures": {
            a: {**v, "regs_per_sm": REGS_PER_SM} for a, v in ARCHS.items()
        },
        "model_validation": {
            "points": checks,
            "all_agree": True,
            "note": "Resident warps/SM from registers, block size and the "
                    "per-architecture limits, checked against every value "
                    "measured on sm_80 in this project.",
        },
        "ptxas_drift": drift,
        "folds": {},
        "note": __doc__.strip().splitlines()[0],
    }

    for key, label in [
        ("env_terminal_velocity", "env_terminal_velocity"),
        ("updraft_terminal_velocity", "updraft_terminal_velocity"),
    ]:
        k = sed["kernels"][key]
        rb, bb = k["registers"]["baseline"], k["block_size"]["baseline"]
        rm, bm = k["registers"]["mod"], k["block_size"]["mod"]
        per_arch = {}
        for arch in ARCHS:
            b = warps_per_sm(rb, bb, **_lim(arch))
            m = warps_per_sm(rm, bm, **_lim(arch))
            up, down = headroom(rm, bm, arch)
            drift_up = (
                None if drift is None
                else drift["max_positive_delta_registers"].get(arch, 0)
            )
            per_arch[arch] = {
                "unfolded_warps_per_sm": b,
                "folded_warps_per_sm": m,
                "warps_gained": m - b,
                "gains_a_step": m > b,
                "headroom_registers": up,
                "registers_to_next_step": down,
                "measured_upward_drift_registers": drift_up,
                "survives_drift": (
                    None if drift_up is None else up >= drift_up
                ),
            }
        gains = {a: v["warps_gained"] for a, v in per_arch.items()}
        hr = min(v["headroom_registers"] for v in per_arch.values())
        out["folds"][label] = {
            "registers": {"unfolded": rb, "folded": rm},
            "block_size": {"unfolded": bb, "folded": bm},
            "pct_of_device_sm_80": k["pct_of_device"],
            "per_architecture": per_arch,
            "gain_is_architecture_independent": len(set(gains.values())) == 1,
            "min_headroom_registers": hr,
            "survives_drift_by_architecture": {
                a: v["survives_drift"] for a, v in per_arch.items()
            },
            "at_risk_architectures": [
                a for a, v in per_arch.items() if v["survives_drift"] is False
            ],
        }

    # Scalars the write-up quotes, so the prose needs no arithmetic of its own.
    all_gains = {
        f["per_architecture"][a]["warps_gained"]
        for f in out["folds"].values()
        for a in ARCHS
    }
    at_risk = sorted(
        {a for f in out["folds"].values() for a in f["at_risk_architectures"]}
    )
    out["summary"] = {
        "n_validation_points": len(checks),
        "architectures_considered": len(ARCHS),
        "warps_gained_everywhere": (
            all_gains.pop() if len(all_gains) == 1 else None
        ),
        "gain_identical_across_architectures": len(all_gains) == 0,
        "min_headroom_registers": min(
            f["min_headroom_registers"] for f in out["folds"].values()
        ),
        "max_upward_drift_registers": (
            None if drift is None
            else max(drift["max_positive_delta_registers"].values())
        ),
        "drift_on_sm_90_registers": (
            None if drift is None
            else drift["max_positive_delta_registers"].get("sm_90")
        ),
        "at_risk_architectures": at_risk,
        "n_at_risk": len(at_risk),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")

    print(f"occupancy model reproduces {len(checks)}/{len(checks)} "
          "measured sm_80 points")
    if drift:
        print("ptxas register drift vs sm_80, same PTX: largest upward "
              f"move {drift['max_positive_delta_registers']}")
        for row in drift["probes"]:
            print(f"  {row['live_values']:3} live: {row['delta_vs_sm_80']}")
    for name, f in out["folds"].items():
        print(f"\n{name}: {f['registers']['unfolded']} -> "
              f"{f['registers']['folded']} registers")
        for arch, v in f["per_architecture"].items():
            print(f"  {arch} ({ARCHS[arch]['name']:9}) "
                  f"{v['unfolded_warps_per_sm']:2} -> "
                  f"{v['folded_warps_per_sm']:2} warps/SM  "
                  f"({v['warps_gained']:+d}), headroom "
                  f"+{v['headroom_registers']} regs")
        print(f"  architecture-independent gain: "
              f"{f['gain_is_architecture_independent']}; "
              f"at risk from drift: {f['at_risk_architectures'] or 'none'}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
