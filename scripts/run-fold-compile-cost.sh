#!/bin/bash
#SBATCH --gpus=1
#
# What #4947's parameter fold costs in compile time. Cheap (~10 min): four
# compile-only processes, no simulation and no coupled model.
#
# One arm per process, because a compile time is only a compile time the first
# time -- two arms in one session would let the second reuse the first's
# inference of every shared callee. The order is reversed between repeats so a
# drift in machine state does not land on one arm.

set -eu
__guard="${SLURM_SUBMIT_DIR:-$PWD}/scripts/gpu-guard.sh"
[ -f "$__guard" ] || __guard="$(dirname "${BASH_SOURCE[0]}")/gpu-guard.sh"
if [ -f "$__guard" ]; then
    source "$__guard"
else
    echo "ERROR: gpu-guard.sh not found; refusing to run unguarded on a GPU" >&2
    exit 1
fi

PROJECT_DIR=${1:?usage: $0 <project_dir> <out_dir>}
OUT_DIR=${2:?usage: $0 <project_dir> <out_dir>}
module purge
module load climacommon/2026_11_06-1.13.3
export CLIMACOMMS_DEVICE=CUDA
export JULIA_LOAD_PATH=@:@stdlib
mkdir -p "$OUT_DIR"

for spec in unfolded:1 folded:1 folded:2 unfolded:2; do
    variant=${spec%%:*}
    rep=${spec##*:}
    echo "=== ${variant} rep ${rep} ==="
    julia --project="$PROJECT_DIR" scripts/measure-fold-compile-cost.jl \
        --variant "$variant" --rep "$rep" \
        --out "$OUT_DIR/${variant}-${rep}.toml"
done
