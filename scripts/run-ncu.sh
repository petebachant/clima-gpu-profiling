#!/bin/bash
#SBATCH --gpus=1

# Default values
NCU_OUTPUT_PREFIX=""
EXTRA_CONFIGS=""
KERNEL_NAME=""
LAUNCH_SKIP=""
LAUNCH_COUNT=""
PROJECT="."

# Function to show usage
show_usage() {
  echo "Usage: $0 <ncu_output_prefix> [options]"
  echo "Options:"
  echo "  --config <file>        Add additional config file (can be used multiple times)"
  echo "  --kernel-name <name>   NCU kernel name filter"
  echo "  --launch-skip <n>      NCU launch skip count"
  echo "  --launch-count <n>     NCU launch count"
  echo "  --project <path>       Julia project directory (default: .)"
  echo "  -h, --help            Show this help message"
}

# Parse command line arguments
if [ "$#" -lt 1 ]; then
  show_usage
  exit 1
fi

# First argument is always the output prefix
NCU_OUTPUT_PREFIX=$1
shift

# Parse remaining arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --config)
      if [ -z "$2" ]; then
        echo "Error: --config requires a value"
        exit 1
      fi
      EXTRA_CONFIGS="$EXTRA_CONFIGS --config $2"
      shift 2
      ;;
    --kernel-name)
      if [ -z "$2" ]; then
        echo "Error: --kernel-name requires a value"
        exit 1
      fi
      KERNEL_NAME="$2"
      shift 2
      ;;
    --launch-skip)
      if [ -z "$2" ]; then
        echo "Error: --launch-skip requires a value"
        exit 1
      fi
      LAUNCH_SKIP="$2"
      shift 2
      ;;
    --launch-count)
      if [ -z "$2" ]; then
        echo "Error: --launch-count requires a value"
        exit 1
      fi
      LAUNCH_COUNT="$2"
      shift 2
      ;;
    --project)
      if [ -z "$2" ]; then
        echo "Error: --project requires a value"
        exit 1
      fi
      PROJECT="$2"
      shift 2
      ;;
    -h|--help)
      show_usage
      exit 0
      ;;
    *)
      echo "Error: Unknown option $1"
      show_usage
      exit 1
      ;;
  esac
done

# Ensure the output prefix parent directory exists
OUTPUT_DIR=$(dirname "$NCU_OUTPUT_PREFIX")
mkdir -p "$OUTPUT_DIR"

export CLIMACOMMS_DEVICE=CUDA
export CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE=true

export NSIGHT_COMPUTE_TMP=$HOME/tmp_nsight_compute
export TMPDIR=$NSIGHT_COMPUTE_TMP
mkdir -p "$NSIGHT_COMPUTE_TMP"

module purge
module load climacommon/2025_05_15

# Set environmental variable for julia to not use global packages for
# reproducibility
export JULIA_LOAD_PATH=@:@stdlib

# Instantiate julia environment, precompile, and build CUDA
julia --project=$PROJECT -e 'using Pkg; Pkg.instantiate(;verbose=true); Pkg.precompile(;strict=true); using CUDA; CUDA.precompile_runtime(); Pkg.status()'

# Build NCU command with optional arguments
NCU_CMD="ncu"
if [ -n "$KERNEL_NAME" ]; then
  NCU_CMD="$NCU_CMD --kernel-name $KERNEL_NAME"
fi
if [ -n "$LAUNCH_SKIP" ]; then
  NCU_CMD="$NCU_CMD --launch-skip $LAUNCH_SKIP"
fi
if [ -n "$LAUNCH_COUNT" ]; then
  NCU_CMD="$NCU_CMD --launch-count $LAUNCH_COUNT"
fi

$NCU_CMD \
    -o $NCU_OUTPUT_PREFIX \
    --import-source 1 \
    --set full \
    julia --project=$PROJECT \
    scripts/run.jl \
    $EXTRA_CONFIGS
