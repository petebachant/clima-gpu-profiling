# Agent instructions

In this project we are trying to optimize an AMIP simulation.
We have submodules for relevant packages, with and without a `-mod` suffix,
which indicates our modified version used to compare against baseline.

NVIDIA Nsight Systems profiling is set up to run as part of a Calkit pipeline,
submitting SLURM jobs on the `clima` cluster one at a time
(so we don't take up too many resources).

The logs in `.calkit/slurm/logs` print important information like estimated
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
