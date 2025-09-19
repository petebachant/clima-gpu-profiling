env:
	julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Start an IJulia kernel for the notebook as a SLURM job
srun-nb:
	srun --gpus=1 --mpi=none --pty calkit jupyter lab --ip=0.0.0.0 --no-browser
