#!/bin/bash
#SBATCH --gpus=1

# Runs the baseline AMIP benchmark with every diagnostic disabled. Differencing
# its wall clock against `amip-baseline` gives what diagnostics cost in total,
# which the per-step breakdown cannot separate because computation is spread
# across every step while only the writes stand out.

__guard="${SLURM_SUBMIT_DIR:-$PWD}/scripts/gpu-guard.sh"
[ -f "$__guard" ] || __guard="$(dirname "${BASH_SOURCE[0]}")/gpu-guard.sh"
if [ -f "$__guard" ]; then
    source "$__guard"
else
    echo "ERROR: gpu-guard.sh not found; refusing to run unguarded on a GPU" >&2
    exit 1
fi

SRC=ClimaCoupler.jl/config/benchmark_configs/amip_progedmf_1m_land_he16.yml
DST=ClimaCoupler.jl/config/benchmark_configs/amip_progedmf_1m_land_he16_nodiag.yml

# Generated rather than committed, and removed on exit: an untracked file in the
# submodule would make `record-treatment` report a dirty tree.
trap 'rm -f "$DST"' EXIT

awk '
    /^extra_atmos_diagnostics:/ { skip = 1; next }
    skip && /^[^ \t]/           { skip = 0 }
    skip                        { next }
    /^output_default_diagnostics:/ { print "output_default_diagnostics: false"; found = 1; next }
    { print }
    END { if (!found) { print "ERROR: output_default_diagnostics not found" > "/dev/stderr"; exit 1 } }
' "$SRC" > "$DST" || exit 1

if grep -q 'extra_atmos_diagnostics\|output_default_diagnostics: true' "$DST"; then
    echo "ERROR: diagnostics not fully disabled in $DST" >&2
    exit 1
fi
echo "--- generated $DST ---"
diff "$SRC" "$DST"

module purge
module load climacommon/2025_05_15
export CLIMACOMMS_DEVICE=CUDA

julia --project=ClimaCoupler.jl/experiments/AMIP -e 'using Pkg; Pkg.instantiate(;verbose=true); Pkg.precompile(;strict=true); using CUDA; CUDA.precompile_runtime(); Pkg.status()'

julia --project=ClimaCoupler.jl/experiments/AMIP \
    ClimaCoupler.jl/experiments/AMIP/run_simulation.jl \
    --config_file "$DST" \
    --job_id gpu_amip_progedmf_1M_land_he16_nodiag
