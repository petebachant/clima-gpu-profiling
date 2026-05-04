#!/bin/bash
#SBATCH --gpus=1

# Drop into a Julia REPL with Revise + scripts/run.jl loaded, using the same
# project and config as the mod-nsys stage.

PROJECT_DIR=ClimaCoupler.jl-mod/experiments/AMIP
CONFIG_FILE=ClimaCoupler.jl-mod/config/benchmark_configs/amip_progedmf_1m_land_he16.yml

module purge
module load climacommon/2025_05_15

export CLIMACOMMS_DEVICE=CUDA
export CLIMA_NAME_CUDA_KERNELS_FROM_STACK_TRACE=true

julia --project=$PROJECT_DIR -e 'using Pkg; Pkg.instantiate(;verbose=true); Pkg.precompile(;strict=true); using CUDA; CUDA.precompile_runtime(); Pkg.status()'

exec julia --project=$PROJECT_DIR -i -e 'using Revise; try; include("scripts/run.jl"); catch e; @error "scripts/run.jl failed; dropping into REPL" exception=(e, catch_backtrace()); end' -- --config_file $CONFIG_FILE
