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

# Default values
NCU_OUTPUT_PREFIX=""
EXTRA_CONFIGS=()
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
      EXTRA_CONFIGS+=(--config "$2")
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

# Build NCU command with optional arguments. Use an array so values
# containing regex metacharacters (e.g. --kernel-name "regex:..._L[0-9]+")
# survive intact instead of being word-split / glob-expanded.
NCU_METRICS=sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__issue_active.avg.pct_of_peak_sustained_active,\
smsp__warps_eligible.avg.per_cycle_active,\
smsp__inst_executed.avg.per_cycle_active,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__registers_per_thread,\
launch__occupancy_limit_registers,\
sm__maximum_warps_per_active_cycle_pct,\
l1tex__throughput.avg.pct_of_peak_sustained_elapsed

# ClimaCore builds a kernel's name from its source file and line, so any
# upstream insertion above renumbers it and a hardcoded name silently stops
# matching -- which is how this stage failed on 2026-10-07, still asking for
# L1013 after the fold moved the kernel to L1024. `auto:<target>[,<target>]`
# resolves the names from results/param-fold.json, which records the line each
# target actually compiled to, so the filter follows upstream instead of
# rotting. The JSON is small, committed and derived, which is what a stage may
# depend on; the nsys database it came from is transient and may not be.
if [[ "$KERNEL_NAME" == auto:* ]]; then
  PARAM_FOLD="${SLURM_SUBMIT_DIR:-$PWD}/results/param-fold.json"
  if [ ! -f "$PARAM_FOLD" ]; then
    echo "ERROR: --kernel-name auto: needs $PARAM_FOLD; run export-param-fold first" >&2
    exit 1
  fi
  KERNEL_NAME=$(python3 - "$PARAM_FOLD" "${KERNEL_NAME#auto:}" <<'PYEOF'
import json, sys
path, targets = sys.argv[1], sys.argv[2].split(",")
kernels = json.load(open(path))["kernels"]
names = []
for t in targets:
    if t not in kernels:
        sys.exit(f"ERROR: no target {t!r} in {path}; have {sorted(kernels)}")
    n = kernels[t]["mod"]["kernel"]
    if not n:
        sys.exit(f"ERROR: target {t!r} has no kernel name in {path}")
    names.append(n)
print("regex:^(" + "|".join(names) + ")$")
PYEOF
  ) || exit 1
  echo "resolved --kernel-name to: $KERNEL_NAME"
fi

NCU_ARGS=()
if [ -n "$KERNEL_NAME" ]; then
  NCU_ARGS+=(--kernel-name "$KERNEL_NAME")
fi
if [ -n "$LAUNCH_SKIP" ]; then
  NCU_ARGS+=(--launch-skip "$LAUNCH_SKIP")
fi
if [ -n "$LAUNCH_COUNT" ]; then
  NCU_ARGS+=(--launch-count "$LAUNCH_COUNT")
fi

ncu "${NCU_ARGS[@]}" \
    -o "$NCU_OUTPUT_PREFIX" \
    --metrics "$NCU_METRICS" \
    julia --project="$PROJECT" \
    scripts/run.jl \
    "${EXTRA_CONFIGS[@]}"

# Export per-kernel details to CSV alongside the .ncu-rep
ncu --csv -i "$NCU_OUTPUT_PREFIX.ncu-rep" --page details \
    --log-file "$NCU_OUTPUT_PREFIX-details.csv"
