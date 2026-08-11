#!/bin/bash
#SBATCH --gpus=1
#
# Compile-time study: how much of a profiling job is compilation, and how much of
# that is recoverable by GPUCompiler's on-disk kernel cache?
#
# Three phases, all in ONE job on ONE GPU so the comparison is not confounded by
# node or device variance:
#
#   1. cache_off   -- status quo (disk_cache unset, GPUCompiler's default)
#   2. cache_cold  -- disk cache enabled, empty: populates it
#   3. cache_warm  -- disk cache enabled, populated: the payoff
#
# (1 - 3) is the wall time recoverable per job by enabling the cache.
#
# Note on instrumentation: GPUCompiler's `compile_hook` cannot be used to count
# kernels here. Setting it forces recompilation and skips the disk-cache lookup
# entirely (see `actual_compilation` in GPUCompiler/src/execution.jl), so it
# would destroy the very effect being measured. Hence wall-clock deltas.

set -u

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

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <project_dir> <config_file> <out_toml>" >&2
    exit 1
fi
PROJECT_DIR=$1
CONFIG_FILE=$2
OUT_TOML=$3

module purge
module load climacommon/2025_05_15
export CLIMACOMMS_DEVICE=CUDA
export JULIA_LOAD_PATH=@:@stdlib

PREFS="$PROJECT_DIR/LocalPreferences.toml"
PREFS_BACKUP="${PREFS}.compile-study-backup"

# The disk cache is toggled by writing GPUCompiler's preference into the AMIP
# environment. That environment is shared with the profiling stages, so the
# original state is restored on exit -- otherwise this study would silently
# change how every later nsys run compiles.
#
# Snapshot ONCE, before anything is modified. Backing up inside the toggle would
# capture an already-modified file (or, on the first call, delete a pre-existing
# file before it was ever saved).
if [ -f "$PREFS" ]; then
    HAD_PREFS=1
    cp -f "$PREFS" "$PREFS_BACKUP"
else
    HAD_PREFS=0
fi

restore_prefs() {
    if [ "$HAD_PREFS" = "1" ]; then
        cp -f "$PREFS_BACKUP" "$PREFS"
    else
        rm -f "$PREFS"
    fi
}
trap 'restore_prefs; rm -f "$PREFS_BACKUP"' EXIT

set_disk_cache() {
    local state=$1
    # Always rebuild from the pristine snapshot so repeated toggles cannot drift.
    restore_prefs
    # GPUCompiler UUID 61eb1bfa-7361-4325-ad38-22787b887f55
    python3 - "$PREFS" "$state" <<'PY'
import sys, os
path, state = sys.argv[1], sys.argv[2]
lines = []
if os.path.exists(path):
    keep, skip = [], False
    for ln in open(path):
        if ln.strip().startswith("["):
            skip = ln.strip() == "[GPUCompiler]"
        if not skip:
            keep.append(ln)
    lines = keep
lines.append(f'\n[GPUCompiler]\ndisk_cache = "{state}"\n')
open(path, "w").write("".join(lines))
PY
    echo "--- LocalPreferences.toml now:"
    cat "$PREFS"
}

clear_disk_cache() {
    # Scratch space, not ~/.julia/compiled: GPUCompiler stores kernel objects via
    # @get_scratch!("disk_cache").
    find "$HOME/.julia/scratchspaces" -maxdepth 3 -type d -name disk_cache \
        -exec rm -rf {} + 2>/dev/null
    echo "--- cleared GPUCompiler disk cache"
}

run_phase() {
    local label=$1
    echo
    echo "==================== phase: $label ===================="
    date -u +"start %Y-%m-%dT%H:%M:%SZ"
    local t0=$SECONDS
    julia --project="$PROJECT_DIR" scripts/measure-compile-time.jl \
        --config_file "$CONFIG_FILE" --label "$label" --out "$OUT_TOML"
    local rc=$?
    echo "phase $label wall: $((SECONDS - t0)) s (exit $rc)"
    return $rc
}

mkdir -p "$(dirname "$OUT_TOML")"
rm -f "$OUT_TOML"

echo "=============== instantiate ==============="
julia --project="$PROJECT_DIR" -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(); using CUDA; CUDA.precompile_runtime()'

# Phase 1: status quo. Cache explicitly off and cleared, so a cache left behind
# by earlier work cannot flatter the baseline.
clear_disk_cache
set_disk_cache false
run_phase cache_off || exit 1

# Phase 2: enable and populate.
set_disk_cache true
clear_disk_cache
run_phase cache_cold || exit 1

# Phase 3: same settings, cache now warm. The delta against phase 1 is the
# recoverable compilation time.
run_phase cache_warm || exit 1

echo
echo "==================== summary ===================="
cat "$OUT_TOML"
