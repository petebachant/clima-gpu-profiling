#!/bin/bash
#SBATCH --gpus=1
#
# Compile-time study: how much of a profiling job is compilation, and how much of
# it can be cached across jobs?
#
# Two phases, in ONE job on ONE GPU so the comparison is not confounded by node
# or device variance:
#
#   1. warmup_off -- status quo
#   2. warmup_on  -- identical code, but `import AMIPWarmup` first, making its
#                    precompiled ClimaCore specializations available
#
# (1 - 2) is the wall time recoverable per job by loading the warmup package.
#
# HISTORY -- this study previously A/B'd GPUCompiler's `disk_cache` preference.
# That was measured and abandoned: the preference is honoured
# (`disk_cache_enabled()` flips true) but GPUCompiler writes nothing, leaving 0
# files in `disk_cache_path()`. It is a no-op in this stack (Julia 1.11.5 /
# CUDA.jl Wfi8S / GPUCompiler Yuvf5), so the three-phase cache experiment could
# not answer anything. What survived it: ~88% of a job is host-side Julia JIT,
# which is what AMIPWarmup targets.
#
# Isolated measurement of the warmup, before wiring it in: first call to the
# warmed operator set 10.2 s -> 1.8 s, grid construction 45.4 s -> 32.4 s.

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
    local warmup=$2
    echo
    echo "==================== phase: $label (warmup=$warmup) ===================="
    date -u +"start %Y-%m-%dT%H:%M:%SZ"
    local t0=$SECONDS
    julia --project="$PROJECT_DIR" scripts/measure-compile-time.jl \
        --config_file "$CONFIG_FILE" --label "$label" --warmup "$warmup" \
        --out "$OUT_TOML"
    local rc=$?
    echo "phase $label wall: $((SECONDS - t0)) s (exit $rc)"
    return $rc
}

mkdir -p "$(dirname "$OUT_TOML")"
rm -f "$OUT_TOML"

echo "=============== instantiate ==============="
# AMIPWarmup's precompilation runs its workload here, once, and is then reused by
# both phases -- so this cost is paid regardless and does not bias the A/B.
julia --project="$PROJECT_DIR" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using CUDA; CUDA.precompile_runtime()'

run_phase warmup_off no || exit 1
run_phase warmup_on yes || exit 1

echo
echo "==================== summary ===================="
cat "$OUT_TOML"
