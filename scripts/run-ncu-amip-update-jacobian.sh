#!/bin/bash
#SBATCH --gpus=1

# Ensure the output prefix parent directory exists
OUTPUT_DIR=ncu-outputs
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

ncu --nvtx --call-stack --set full \
  --kernel-name regex:"update_jacobian.*60.*" \
  --profile-from-start no \
  --export $OUTPUT_DIR/amip-update-jacobian-kernel \
  julia --color=yes --project=. \
  ClimaAtmos.jl/perf/benchmark_step_gpu.jl \
  --config ClimaAtmos.jl/config/default_configs/default_config.yml \
  --config ClimaAtmos.jl/config/common_configs/numerics_sphere_he30ze63.yml \
  --config config/amip_target_edonly.yml
