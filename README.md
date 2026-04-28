# CliMA GPU profiling

Profiling GPU performance for CliMA.

To run this project, first install Calkit on the `clima` machine
(this will install in your home directory):

```sh
curl -LsSf install.calkit.org | sh
```

Next,
[authenticate with with calkit.io](https://docs.calkit.org/cloud-integration/)
(where we store version-controlled Nsight reports):

```sh
calkit cloud login
```

If you don't already have an SSH key added to GitHub,
either follow their documentation or run:

```sh
calkit config github-ssh
```

Then clone the project:

```sh
calkit clone --ssh petebachant/clima-gpu-profiling
```

Lastly, call:

```sh
calkit run
```

This will run all pipeline stages in the order they're defined in
`calkit.yaml`.
If you'd like to run a single stage (in a reproducible way),
you can use its name as the first positional argument to `calkit run`.
For example:

```sh
calkit run mod-nsys
```

However, by default, only stages whose Nsight reports are now invalid
(since their inputs have changed since the last run)
will run.

## Running a Jupyter notebook with a GPU reserved on `clima`

Use the [Calkit VS Code extension](https://marketplace.visualstudio.com/items?itemName=Calkit.calkit-vscode) to define a `slurm` and `julia` environment
and attach it to a notebook.

## Submodule branches

Run `bash scripts/show-submodule-status.sh` to see current statuses.

## Experiment results

| Commit (super-repo) | Change summary | Result |
|---------------------|----------------|--------|
| `130baab` |  | Occupancy for `run_field_matrix_solver` increased and reduced registers per thread but slowed down overall. |
| `e5845b7` | | Similar as above, but not quite as slow. |
| `ff26f4b` | Use PCR for tri-diagonal matrix solve. | Seems to be 3% faster, but higher error. May not have isolated changed properly though. |
| `e6099c2` | Try capping all threads to 256 | 1% slower on flagship. |
| `23c9104` | Attempt to coalesce memory access in solvers. | 5% slowdown. |
| `7614ca6` | [Thread block restructuring and LocalGeometry caching](https://github.com/CliMA/ClimaCore.jl/pull/2425). | No significant change. |
| `f9eb67a` | [Tr/mem access patterns](https://github.com/CliMA/ClimaCore.jl/pull/2396) | 9% speedup. |
