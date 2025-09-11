#!/bin/bash

# First command line argument is the nsys output file prefix
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <nsys_output_prefix> [additional_config_files...]"
  exit 1
fi
NSYS_OUTPUT_PREFIX=$1
shift

# Parse additional configs from command line
if [ "$#" -gt 0 ]; then
  EXTRA_CONFIGS=""
  for arg in "$@"; do
    EXTRA_CONFIGS="$EXTRA_CONFIGS --config $arg"
  done
fi

mkdir -p nsys-outputs

# Load modules
module purge
module load climacommon/2025_05_15

# Set environment variables for GPU usage
export CLIMACOMMS_DEVICE=CUDA

# Instantiate julia environment, precompile, and build CUDA
julia --project=. -e 'using Pkg; Pkg.instantiate(;verbose=true); Pkg.precompile(;strict=true); using CUDA; CUDA.precompile_runtime(); Pkg.status()'

# Run nsys
nsys profile \
    --start-later=true \
    --capture-range=cudaProfilerApi \
    --kill=none \
    --trace=nvtx,mpi,cuda,osrt \
    --output=$NSYS_OUTPUT_PREFIX \
    julia --project=. \
    ClimaAtmos.jl/perf/benchmark_step_gpu.jl \
    --config ClimaAtmos.jl/config/default_configs/default_config.yml \
    $EXTRA_CONFIGS
