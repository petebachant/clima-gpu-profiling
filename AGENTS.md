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
