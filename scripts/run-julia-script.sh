#!/bin/bash
#SBATCH --gpus=1

# Load modules
module purge
module load climacommon/2025_05_15

export CLIMACOMMS_DEVICE=CUDA

julia --project=. -e 'using Pkg; Pkg.instantiate(;verbose=true); Pkg.precompile(;strict=true); using CUDA; CUDA.precompile_runtime(); Pkg.status()'

# Pass all args to julia
julia --project=. "$@"
