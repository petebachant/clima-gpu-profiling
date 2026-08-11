# Run the benchmark
import CUDA

# Figure out which project is currently activated and include the setup
# script
project_dir = dirname(Base.active_project())
@info "Active project: $project_dir"
include(joinpath(project_dir, "code_loading.jl"))

# Get the configuration file from the command line (or manually set it here)
# For the integrated land model, use:
# longrun_configs/amip_edonly_integrated_land.yml
# For the bucket model, use:
# longrun_configs/amip_edonly.yml
config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]

# Set up and run the coupled simulation
cs = CoupledSimulation(config_file)

# Run a few warmup steps so JIT compilation, kernel caches, and any
# step-dependent code paths (e.g. variable Newton iteration counts) settle
# before profiling. Earlier we used a single warmup step, but the captured
# window then reflected first-call costs that don't recur in steady-state
# runs and skewed per-kernel times relative to long simulations.
n_warmup_steps = 3
for i in 1:n_warmup_steps
    @info "Warmup step $i / $n_warmup_steps"
    step!(cs)
end

# Now profile a window large enough that one-shot per-step variation
# averages out and steady-state kernel times dominate.
n_steps = 10
use_external_profiler = CUDA.Profile.detect_cupti()
if use_external_profiler
    @info "Using external CUDA profiler"
    CUDA.@profile external = true begin
        for i in 1:n_steps
            @info "Step $i / $n_steps"
            step!(cs)
        end
    end
else
    @info "Using internal CUDA profiler"
    res = CUDA.@profile external = false begin
        for i in 1:n_steps
            @info "Step $i / $n_steps"
            step!(cs)
        end
    end
    show(IOContext(stdout, :limit => false), res)
end
