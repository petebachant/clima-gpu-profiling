#!/bin/bash
#SBATCH --gpus=1

# First command line argument is the nsys output file prefix
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <nsys_output_prefix> <project_directory> [additional_config_files...]"
  exit 1
fi
NSYS_OUTPUT_PREFIX=$1
shift

# Second command line argument is the project directory
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <nsys_output_prefix> <project_directory> [additional_config_files...]"
  exit 1
fi
PROJECT_DIR=$1
shift

# Parse additional configs from command line
if [ "$#" -gt 0 ]; then
  EXTRA_CONFIGS=""
  for arg in "$@"; do
    EXTRA_CONFIGS="$EXTRA_CONFIGS --config_file $arg"
  done
fi

# Ensure the output prefix parent directory exists
OUTPUT_DIR=$(dirname "$NSYS_OUTPUT_PREFIX")
mkdir -p "$OUTPUT_DIR"

# Load modules
module purge
module load climacommon/2025_05_15

# Set environment variables for GPU usage
export CLIMACOMMS_DEVICE=CUDA
export CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE=true

# Set environmental variable for julia to not use global packages for
# reproducibility
export JULIA_LOAD_PATH=@:@stdlib

# Instantiate julia environment, precompile, and build CUDA
julia --project=$PROJECT_DIR -e 'using Pkg; Pkg.instantiate(;verbose=true); Pkg.precompile(;strict=true); using CUDA; CUDA.precompile_runtime(); Pkg.status()'

# Run nsys
nsys profile \
    --start-later=true \
    --capture-range=cudaProfilerApi \
    --kill=none \
    --trace=nvtx,mpi,cuda,osrt \
    --output=$NSYS_OUTPUT_PREFIX \
    julia --project=$PROJECT_DIR \
    scripts/run.jl \
    $EXTRA_CONFIGS
