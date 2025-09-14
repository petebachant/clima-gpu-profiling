#!/bin/bash
#SBATCH --gpus=1

# First command line argument is the ncu output file prefix
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <ncu_output_prefix> [additional_config_files...]"
  exit 1
fi
NCU_OUTPUT_PREFIX=$1
shift

# Parse additional configs from command line
if [ "$#" -gt 0 ]; then
  EXTRA_CONFIGS=""
  for arg in "$@"; do
    EXTRA_CONFIGS="$EXTRA_CONFIGS --config $arg"
  done
fi

# Parse --kernel-name, --launch-skip, and --launch-count from command line
# (if provided) and remove them from EXTRA_CONFIGS
KERNEL_NAME=""
LAUNCH_SKIP=""
LAUNCH_COUNT=""
NEW_EXTRA_CONFIGS=""
for arg in $EXTRA_CONFIGS; do
  if [[ $arg == --kernel-name* ]]; then
    KERNEL_NAME="${arg#*=}"
  elif [[ $arg == --launch-skip* ]]; then
    LAUNCH_SKIP="${arg#*=}"
  elif [[ $arg == --launch-count* ]]; then
    LAUNCH_COUNT="${arg#*=}"
  else
    NEW_EXTRA_CONFIGS="$NEW_EXTRA_CONFIGS $arg"
  fi
done
EXTRA_CONFIGS="$NEW_EXTRA_CONFIGS"

# Ensure the output prefix parent directory exists
OUTPUT_DIR=$(dirname "$NSYS_OUTPUT_PREFIX")
mkdir -p "$OUTPUT_DIR"

export CLIMACOMMS_DEVICE=CUDA

export NSIGHT_COMPUTE_TMP=$HOME/tmp_nsight_compute
export TMPDIR=$NSIGHT_COMPUTE_TMP
mkdir -p "$NSIGHT_COMPUTE_TMP"

module purge
module load climacommon/2025_05_15

# Set environmental variable for julia to not use global packages for
# reproducibility
export JULIA_LOAD_PATH=@:@stdlib

# Instantiate julia environment, precompile, and build CUDA
julia --project=. -e 'using Pkg; Pkg.instantiate(;verbose=true); Pkg.precompile(;strict=true); using CUDA; CUDA.precompile_runtime(); Pkg.status()'

ncu --kernel-name $KERNEL_NAME \
    --launch-skip $LAUNCH_SKIP \
    --launch-count $LAUNCH_COUNT \
    -o $NCU_OUTPUT_PREFIX \
    --import-source 1 \
    --set full \
    julia --project=. \
    ClimaAtmos.jl/perf/benchmark_step_gpu.jl \
    --config ClimaAtmos.jl/config/default_configs/default_config.yml \
    $EXTRA_CONFIGS
