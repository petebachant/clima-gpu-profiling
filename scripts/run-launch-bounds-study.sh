#!/bin/bash
#SBATCH --gpus=1
#
# Launch-bounds study: does asking ptxas for an occupancy target via
# __launch_bounds__ make the register-bound microphysics kernels faster, and is
# the spill-growth guard that gates it doing any work?
#
# Three phases, in ONE job on ONE GPU so the comparison is not confounded by node
# or device variance, each in its own process so the configuration is set at load
# time exactly as a real run would set it:
#
#   1. off       -- no launch bounds; the baseline arm's behaviour
#   2. guarded   -- 12 warps/SM target with the shipped 256-byte spill budget
#   3. unguarded -- same target, but a budget loose enough that nothing is
#                   rejected. This is the counterfactual: it is the only way to
#                   observe what the guard prevents, since a rejected
#                   configuration never appears in an ordinary profiling run.
#
# On sm_80 the register occupancy steps are 255 -> 8 warps/SM, 168 -> 12,
# 128 -> 16, so 12 warps is the cheapest useful step above where a
# 255-register kernel is pinned.

set -u

# Requeue if we landed on a known-bad GPU. NB: sbatch copies this script to
# /var/spool/slurmd/..., so BASH_SOURCE is not a usable anchor; SLURM_SUBMIT_DIR
# and PWD are both the repo root.
__guard="${SLURM_SUBMIT_DIR:-$PWD}/scripts/gpu-guard.sh"
[ -f "$__guard" ] || __guard="$(dirname "${BASH_SOURCE[0]}")/gpu-guard.sh"
if [ -f "$__guard" ]; then
    source "$__guard"
else
    echo "ERROR: gpu-guard.sh not found; refusing to run unguarded on a GPU" >&2
    exit 1
fi

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <project_dir> <config_file> <out_toml>" >&2
    exit 1
fi
PROJECT_DIR=$1
CONFIG_FILE=$2
OUT_TOML=$3

module purge
module load climacommon/2025_05_15
export CLIMACOMMS_DEVICE=CUDA
export JULIA_LOAD_PATH=@:@stdlib

run_phase() {
    local label=$1
    local target=$2
    local budget=$3
    echo
    echo "============ phase: $label (target=$target warps/SM, budget=$budget B) ============"
    date -u +"start %Y-%m-%dT%H:%M:%SZ"
    local t0=$SECONDS
    julia --project="$PROJECT_DIR" scripts/measure-launch-bounds.jl \
        --config_file "$CONFIG_FILE" --label "$label" \
        --target "$target" --budget "$budget" --out "$OUT_TOML"
    local rc=$?
    echo "phase $label wall: $((SECONDS - t0)) s (exit $rc)"
    return $rc
}

mkdir -p "$(dirname "$OUT_TOML")"
rm -f "$OUT_TOML"

echo "=============== instantiate ==============="
julia --project="$PROJECT_DIR" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using CUDA; CUDA.precompile_runtime()'

# `off` runs first so that if the job is cut short, the reference the other two
# phases are read against is the one that survives.
run_phase off       0  256  || exit 1
run_phase guarded   12 256  || exit 1
run_phase unguarded 12 8192 || exit 1

echo
echo "==================== summary ===================="
cat "$OUT_TOML"
