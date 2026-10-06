#!/bin/bash
#SBATCH --gpus=1
#
# Does Val-wrapping the microphysics parameters fold them the way const globals
# do? Cheap (~5 min): compiles three variants of one body and reads their
# register counts. No simulation, no coupled model.

set -u
__guard="${SLURM_SUBMIT_DIR:-$PWD}/scripts/gpu-guard.sh"
[ -f "$__guard" ] || __guard="$(dirname "${BASH_SOURCE[0]}")/gpu-guard.sh"
if [ -f "$__guard" ]; then
    source "$__guard"
else
    echo "ERROR: gpu-guard.sh not found; refusing to run unguarded on a GPU" >&2
    exit 1
fi

PROJECT_DIR=${1:?usage: $0 <project_dir>}
module purge
module load climacommon/2025_05_15
export CLIMACOMMS_DEVICE=CUDA
export JULIA_LOAD_PATH=@:@stdlib
mkdir -p results
exec julia --project="$PROJECT_DIR" scripts/measure-val-params-registers.jl
