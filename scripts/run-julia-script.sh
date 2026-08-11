#!/bin/bash
#SBATCH --gpus=1

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

# Load modules
module purge
module load climacommon/2025_05_15

export CLIMACOMMS_DEVICE=CUDA

# Parse --project, which is required
PROJECT_ARG=""
for arg in "$@"; do
    if [[ $arg == --project=* ]]; then
        PROJECT_ARG=$arg
        break
    fi
done
if [[ -z $PROJECT_ARG ]]; then
    echo "Error: --project is required."
    exit 1
fi

julia $PROJECT_ARG -e 'using Pkg; Pkg.instantiate(;verbose=true); Pkg.precompile(;strict=true); using CUDA; CUDA.precompile_runtime(); Pkg.status()'

# Pass all args to julia
julia "$@"
