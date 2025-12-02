# Run the benchmark
import CUDA

# Figure out which project is currently activated and include the setup
# script
project_dir = dirname(Base.active_project())
@info "Active project: $project_dir"
include(joinpath(project_dir, "setup_run.jl"))

# Get the configuration file from the command line (or manually set it here)
# For the integrated land model, use:
# longrun_configs/amip_edonly_integrated_land.yml
# For the bucket model, use:
# longrun_configs/amip_edonly.yml
config_file = parse_commandline(argparse_settings())["config_file"]

# Set up and run the coupled simulation
cs = CoupledSimulation(config_file)

# Run a single step to compile
step!(cs)

# Now profile
n_steps = 1
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
