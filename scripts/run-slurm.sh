#!/bin/bash
#SBATCH --gpus=1

# First command line argument is nsys or ncu
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <nsys|ncu> <output_prefix> [additional_config_files...] [-- ncu_args...]"
  exit 1
fi
NSIGHT_APP=$1

# Second command line argument is the output file prefix
if [ "$#" -lt 2 ]; then
  echo "Usage: $0 <nsys|ncu> <output_prefix> [additional_config_files...] [-- ncu_args...]"
  exit 1
fi
OUTPUT_PREFIX=$2
shift 2

# Parse configs and ncu args separated by --
EXTRA_CONFIGS=""
NCU_ARGS=""
PARSING_NCU_ARGS=false

for arg in "$@"; do
  if [ "$arg" == "--" ]; then
    PARSING_NCU_ARGS=true
  elif [ "$PARSING_NCU_ARGS" == true ]; then
    NCU_ARGS="$NCU_ARGS $arg"
  else
    EXTRA_CONFIGS="$EXTRA_CONFIGS --config $arg"
  fi
done

# Ensure the output prefix parent directory exists
OUTPUT_DIR=$(dirname "$OUTPUT_PREFIX")
mkdir -p "$OUTPUT_DIR"

# Load modules
module purge
module load climacommon/2025_05_15

# Set environment variables for GPU usage
export CLIMACOMMS_DEVICE=CUDA

# Set environmental variable for julia to not use global packages for
# reproducibility
export JULIA_LOAD_PATH=@:@stdlib

# Instantiate julia environment, precompile, and build CUDA
julia --project=. -e 'using Pkg; Pkg.instantiate(;verbose=true); Pkg.precompile(;strict=true); using CUDA; CUDA.precompile_runtime(); Pkg.status()'

if [ "$NSIGHT_APP" == "ncu" ]; then
  ncu $NCU_ARGS \
    -o $OUTPUT_PREFIX \
    julia --project=.buildkite perf/benchmark_step_gpu.jl \
    --config ClimaAtmos.jl/config/default_configs/default_config.yml \
    $EXTRA_CONFIGS
elif [ "$NSIGHT_APP" == "nsys" ]; then
  # Run nsys
  nsys profile \
    --start-later=true \
    --capture-range=cudaProfilerApi \
    --kill=none \
    --trace=nvtx,mpi,cuda,osrt \
    --output=$OUTPUT_PREFIX \
    julia --project=. \
    ClimaAtmos.jl/perf/benchmark_step_gpu.jl \
    --config ClimaAtmos.jl/config/default_configs/default_config.yml \
    $EXTRA_CONFIGS
else
  echo "Invalid first argument. Use 'nsys' or 'ncu'."
  exit 1
fi
