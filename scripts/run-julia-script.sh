#!/bin/bash
#SBATCH --gpus=1

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
