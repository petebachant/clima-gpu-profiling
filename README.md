# CliMA GPU profiling

Profiling GPU performance for CliMA.

To run this project, first install Calkit on the `clima` machine:

```sh
curl -LsSf https://github.com/calkit/calkit/raw/refs/heads/main/scripts/install.sh | sh
```

Next,
[configure a token for interacting with calkit.io](https://docs.calkit.org/cloud-integration/)
(where we store version-controlled Nsight reports).

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
calkit run amip-clima-nsys
```

However, by default, only stages whose Nsight reports are now invalid
(since their inputs have changed since the last run)
will run.

## Running a Jupyter server on `clima` with `srun`

```sh
srun --gpus=1 --mpi=none --pty bash
```

```sh
calkit jupyter lab --ip=0.0.0.0 --no-browser
```

Then, copy the server URL, which starts with `http://127.0.0.1`,
and in VS Code, use that when selecting a kernel for the notebook.

## Submodule branches

- `ClimaCore.jl` --> `pb/rm-nvtx`
- `ClimaCore.jl-mod` --> `pb/perf`
- `ClimaCoupler.jl` --> `pb/rm-nvtx`
- `ClimaCoupler.jl-mod` --> `pb/perf`
- `ClimaAtmos.jl-mod` --> `pb/perf`

## Experiment results

| Commit (super-repo) | Change summary | Result |
|---------------------|----------------|--------|
| `130baab` |  | Occupancy for `run_field_matrix_solver` increased and reduced registers per thread but slowed down overall. |
| `e5845b7` | | Similar as above, but not quite as slow. |
| `ff26f4b1` | Use PCR for tri-diagonal matrix solve. | Seems to be 3% faster, but higher error. May not have isolated changed properly though. |
| `e6099c2` | Try capping all threads to 256 | 1% slower on flagship. |
