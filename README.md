# CliMA GPU profiling

Profiling GPU performance for CliMA.

## Getting started

To run this project, first install Calkit on the `clima` machine
(this will install in your home directory):

```sh
curl -LsSf install.calkit.org | sh
```

Next,
[authenticate with with calkit.io](https://docs.calkit.org/cloud-integration/)
(where we store version-controlled Nsight reports to avoid bloating the Git
repo):

```sh
calkit cloud login
```

If you don't already have an SSH key added to GitHub,
either follow their
[documentation](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
or run:

```sh
calkit config github-ssh
```

Then clone the project:

```sh
calkit clone --ssh petebachant/clima-gpu-profiling
```

## Usage

To see if any of the results are "stale," i.e., the code has changed since
they were last generated, call:

```sh
calkit status
```

To run the pipeline (skipping up-to-date stages), call:

```sh
calkit run
```

Stages will be executed in the order they're defined in
`calkit.yaml`, and `sbatch` stages are robust to disconnects, i.e.,
if the pipeline is interrupted and run again, it will pick up where it left
off without submitting new jobs.
If you'd like to run a single stage (in a reproducible way),
you can use its name as the first positional argument to `calkit run`.
For example:

```sh
calkit run mod-nsys
```

However, by default, only stages whose outputs are now invalid
(since their inputs have changed since the last run)
will run.

To view the Nsight reports, commit and push them with:

```sh
calkit save -am "Run with <code changes description>"
```

Then, run `calkit pull` inside a clone of the repo
on a machine with a display and open with Nsight.
Note that if Nsight modifies the report after opening, you may need to use
`calkit pull -f` to force checking out the newest version in the workspace.

## Use this project as a template

To create a new project using this one as a template, run:

```sh
calkit new project --cloud --public \
    --title "CliMA GPU profiling" \
    --template petebachant/clima-gpu-profiling \
    your-project-folder-location
```

This will create a new project both on GitHub and calkit.io and link them
together.

## Running a Jupyter notebook with a GPU reserved on `clima`

Use the [Calkit VS Code extension](https://marketplace.visualstudio.com/items?itemName=Calkit.calkit-vscode) to define a `slurm` and `julia` environment
and attach it to a notebook.

## Submodule branches

Run `bash scripts/show-submodule-status.sh` to see current statuses.

## Experiment results

The complete, machine-generated record lives in
[`experiments.md`](experiments.md) (and `experiments.csv`, which is the evidence
table shown on calkit.io). It is built from `exp/*` git tags:

```sh
python3 scripts/build-experiment-index.py
```

Every row is read out of git at the tagged commit, so the index is derived
rather than maintained and cannot drift from what was actually recorded.

### Recording an experiment

1. Run the pipeline (`calkit run`). The `record-treatment` stage writes
   `results/treatment.json`, capturing what actually reached the run: submodule
   SHAs and refs, which packages each arm `dev`s, whether any tree was dirty,
   the `CLIMA_*` variables the stages export, and a digest of `diffs/`. This is
   the difference between "which commit" and "what ran" -- a submodule can be
   checked out and have no effect at all if it is not `dev`'d.
2. `calkit save -am "<description, including the hashes under test>"`.
3. Tag it, in the super-repo **and in each submodule whose commit the experiment
   uses**:

   ```sh
   git tag -a exp/<slug> -m "<description>"
   git -C <submodule> tag -a exp/<slug> -m "<description>" && git -C <submodule> push origin exp/<slug>
   ```

   Tagging the submodules is not bookkeeping. The super-repo pins submodule
   commits that live on `pb/*` branches; if such a branch is force-pushed or
   deleted, the pinned commit becomes unreachable and the experiment stops being
   reproducible even though git still "records" it. A tag keeps it alive.
4. Regenerate the index and commit it.

### Reproducing an experiment

```sh
git checkout <commit>
git submodule update --init --recursive
calkit run
```

### Interpreting a result

`speedup_pct` is `(mod - baseline) / mod` in SYPD. The floor for calling a
result real is set by the *null tests* in the index -- runs where both arms were
byte-identical. Those have come in between -0.43% and +0.21%, so differences of
a few tenths of a percent are measurable and anything above ~1% is unambiguous.
Note that the baseline itself drifts substantially with upstream (SYPD has
ranged 0.26 to 0.32 over this project), so `speedup_pct` is only meaningful
against its own run's baseline, never across rows.

### Earlier qualitative notes

Hand-maintained, predating the generated index; kept because the change
summaries are not recoverable from the pipeline outputs.

| Commit (super-repo) | Change summary | Result |
|---------------------|----------------|--------|
| `130baab` |  | Occupancy for `run_field_matrix_solver` increased and reduced registers per thread but slowed down overall. |
| `e5845b7` | | Similar as above, but not quite as slow. |
| `ff26f4b` | Use PCR for tri-diagonal matrix solve. | Seems to be 3% faster, but higher error. May not have isolated changed properly though. |
| `e6099c2` | Try capping all threads to 256 | 1% slower on flagship. |
| `23c9104` | Attempt to coalesce memory access in solvers. | 5% slowdown. |
| `7614ca6` | [Thread block restructuring and LocalGeometry caching](https://github.com/CliMA/ClimaCore.jl/pull/2425). | No significant change. |
| `f9eb67a` | [Tr/mem access patterns](https://github.com/CliMA/ClimaCore.jl/pull/2396) | 9% speedup. |
