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
