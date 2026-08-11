# Run the benchmark
import CUDA

# Loading this package makes its precompiled ClimaCore specializations available,
# which is the whole point of it -- the cache is only consulted for code that is
# actually loaded. Measured in isolation: first call to the warmed operator set
# drops 10.2 s -> 1.8 s, grid construction 45.4 s -> 32.4 s.
#
# It deliberately does not depend on ClimaAtmos or CloudMicrophysics, so editing
# those does not invalidate it. Set AMIPWARMUP_SKIP=1 to build it without the
# workload. Guarded so a broken warmup degrades to "slow", never "job fails".
try
    @eval import AMIPWarmup
catch err
    @warn "AMIPWarmup unavailable; continuing without warm caches" err
end

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
