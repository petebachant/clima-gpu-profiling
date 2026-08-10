#!/bin/bash
# Bail out of (and requeue) a SLURM job that landed on a known-bad GPU.
#
# Source this near the top of any batch script that uses a GPU:
#
#     source "$(dirname "$0")/gpu-guard.sh"
#
# Background: `clima` is a single node with 8 A100s exposed to SLURM via
# gres.conf `File=/dev/nvidia[0-7]`. SLURM has no "allocate any GPU except N"
# flag, so we cannot ask the scheduler to skip a bad device. What we *can* do is
# notice which device we were handed -- SLURM sets CUDA_VISIBLE_DEVICES /
# SLURM_STEP_GPUS to the global physical index, not a remapped 0 -- and requeue
# ourselves so the scheduler deals us a different one.
#
# The real fix is for a cluster admin to drop the bad device from gres.conf:
#     Name=gpu Type=A100 File=/dev/nvidia[0-4,6-7]
#     scontrol reconfigure
# Once that lands, BAD_GPUS below can go back to empty.

# Space- or comma-separated list of bad global GPU indices.
BAD_GPUS="${BAD_GPUS:-5}"

# Give up after this many requeues so a fully bad node can't loop forever.
GPU_GUARD_MAX_RETRIES="${GPU_GUARD_MAX_RETRIES:-8}"

__gpu_guard() {
    # Not in a SLURM allocation: nothing to guard.
    [ -n "${SLURM_JOB_ID:-}" ] || return 0

    local assigned
    assigned="${SLURM_STEP_GPUS:-${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-}}}"
    if [ -z "$assigned" ]; then
        echo "gpu-guard: no GPU assignment visible; skipping check" >&2
        return 0
    fi

    local bad_list
    bad_list=$(echo "$BAD_GPUS" | tr ',' ' ')

    local got bad hit=""
    for got in $(echo "$assigned" | tr ',' ' '); do
        for bad in $bad_list; do
            [ "$got" = "$bad" ] && hit="$got"
        done
    done

    if [ -z "$hit" ]; then
        echo "gpu-guard: running on GPU(s) [$assigned] -- OK (avoiding [$BAD_GPUS])"
        return 0
    fi

    local restarts="${SLURM_RESTART_COUNT:-0}"
    if [ "$restarts" -ge "$GPU_GUARD_MAX_RETRIES" ]; then
        echo "gpu-guard: FATAL -- landed on bad GPU $hit again after" \
             "$restarts requeues; giving up so this doesn't spin." >&2
        exit 1
    fi

    echo "gpu-guard: landed on bad GPU $hit (attempt $((restarts + 1)));" \
         "requeuing job $SLURM_JOB_ID to get a different device." >&2
    scontrol requeue "$SLURM_JOB_ID"
    # Requeue is asynchronous; stop doing work while we wait to be killed.
    sleep 60
    exit 1
}

__gpu_guard
