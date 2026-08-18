# Agent instructions

In this project we are trying to optimize an AMIP simulation.
We have submodules for relevant packages, with and without a `-mod` suffix,
which indicates our modified version used to compare against baseline.

NVIDIA Nsight Systems profiling is set up to run as part of a Calkit pipeline,
submitting SLURM jobs on the `clima` cluster one at a time
(so we don't take up too many resources).

The logs in `.calkit/scheduler/logs` print important information like estimated
SYPD and a table of kernel resource consumption.

The general process we follow is:

1. Make some modifications to suboptimal kernels.
2. Run `calkit run mod-nsys` to profile and wait for the SLURM job to finish.
   You can use `calkit slurm queue` to view the jobs associated with this
   project.
3. If the `estimated_sypd` is significantly higher than baseline, run the
   rest of the pipeline with `calkit run`.
   If not, try some more modifications in the `-mod` suffix submodules.
4. If we have significant speedup from the longer AMIP simulation, e.g.,
   greater than 5%, stop and let the user make commits to the submodules,
   push, and run `calkit save` on the repo to push the Nsight reports to the
   cloud for archival.

## Never touch a GPU without reserving one

`clima` is a shared machine. Never run anything that uses a GPU on the login
node or outside a scheduler reservation. Reserve one first with `srun`:

```sh
srun-gpu   # alias for: srun --gpus=1 --mpi=none --time=180 --pty bash
```

This applies to everything, not just the pipeline: quick kernel benchmarks,
`CUDA.jl` sanity checks, package test suites, and one-off scripts all need a
reservation. The Calkit pipeline stages already reserve their own GPUs through
the `clima` SLURM environment, so `calkit run` is fine as-is.

## `clima` is a shared single node cluster

We should usually only have one job in the queue during working hours,
when other typically need it, so we don't back things up behind us.

### GPU 5 was bad — guard now disabled (2026-08-13)

`clima` is one node with 8 A100s. GPU 5 was faulty; as of 2026-08-13 it reports
0 uncorrected ECC errors and is in normal use, so `BAD_GPUS` is now empty and
the guard is inert. Note SLURM never stopped offering it: `gres.conf` still
reads `File=/dev/nvidia[0-7]`. To re-arm, list the bad indices in
`scripts/gpu-guard.sh` — nothing else needs to change.

Historical detail follows. SLURM has no "allocate any GPU except N" flag,
so `scripts/gpu-guard.sh`
handles it from inside the job: SLURM sets `CUDA_VISIBLE_DEVICES` /
`SLURM_STEP_GPUS` to the *global* device index, so the guard checks that index
and calls `scontrol requeue` if it landed on a bad one. It is sourced at the top
of `run-nsys.sh`, `run-ncu.sh`, and `run-julia-script.sh`.

For interactive work, check what you were given and drop the allocation if it is
GPU 5:

```sh
srun-gpu
echo $CUDA_VISIBLE_DEVICES   # exit and re-run srun-gpu if this is 5
```

The proper fix needs a cluster admin, since `/etc/slurm/gres.conf` is root-owned:
change `File=/dev/nvidia[0-7]` to `File=/dev/nvidia[0-4,6-7]` and run
`scontrol reconfigure`.

Once the hardware is fixed, turn the guard off by setting `BAD_GPUS=""` in
`scripts/gpu-guard.sh` — and **do not delete the `source` block from the run
scripts**. `gpu-guard.sh` is deliberately not a stage dependency, so editing the
bad-GPU list invalidates nothing. But `run-nsys.sh`, `run-ncu.sh`, and
`run-julia-script.sh` *are* dependencies of the profiling stages, so removing the
block from them would invalidate every stage and force hours of re-profiling for
no scientific reason.

The `make-diffs` stage collects up all changes across all packages and
puts them in the `diffs` folder, so it can be archived along with the results.

This history of this repo serves as a record of numerical experiments.
If we see something interesting we should save it in the history by
running the pipeline so we ensure it's reproducible.

## Updating ClimaCoupler to reflect the latest `main`

We are running on branches that should be identical to `main` except for the
changes to their `experiments/AMIP/Manifest-v1.11.toml` files,
which dev in the local packages in the submodules.
So we can `cd` into each Coupler submodule and run

```sh
bash scripts/update-coupler-to-main.sh
```
