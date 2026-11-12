#!/bin/bash
#SBATCH --gpus=1
#
# Register census of the SGS-quadrature microphysics broadcast. Cheap (~5 min):
# it compiles the broadcast and reads the register count ClimaCore's launch
# recorded, it does not run a simulation.

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
module load climacommon/2026_11_06-1.13.3
export CLIMACOMMS_DEVICE=CUDA
export JULIA_LOAD_PATH=@:@stdlib
mkdir -p results
exec julia --project="$PROJECT_DIR" scripts/measure-quadrature-registers.jl
