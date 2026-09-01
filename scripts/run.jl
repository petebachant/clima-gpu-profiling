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

# Per-step host accounting, to attribute the GPU idle that dominates the
# profiled window. Measured on 2026-09-01: 52% of all idle sits in ~38 gaps
# longer than 500 us, those gaps contain essentially no CUDA API activity
# (1.3 ms of 326.9 ms), and they cluster on particular steps rather than
# decaying like first-call cost. That leaves Julia GC and late host-side
# compilation as the candidates, and `Base.GC_Diff` separates them directly:
# if the spikes are GC, `gc_ms` accounts for them; if they are compilation,
# `gc_ms` stays near zero while `wall_ms` spikes.
function timed_step!(cs, label)
    gc0 = Base.gc_num()
    t0 = time_ns()
    step!(cs)
    wall_ms = (time_ns() - t0) / 1e6
    d = Base.GC_Diff(Base.gc_num(), gc0)
    @info label wall_ms = round(wall_ms; digits = 2) gc_ms =
        round(d.total_time / 1e6; digits = 2) gc_pauses = d.pause full_sweeps =
        d.full_sweep alloc_MB = round(d.allocd / 2^20; digits = 1)
    return nothing
end

# Run a few warmup steps so JIT compilation, kernel caches, and any
# step-dependent code paths (e.g. variable Newton iteration counts) settle
# before profiling. Earlier we used a single warmup step, but the captured
# window then reflected first-call costs that don't recur in steady-state
# runs and skewed per-kernel times relative to long simulations.
n_warmup_steps = 3
for i in 1:n_warmup_steps
    timed_step!(cs, "Warmup step $i / $n_warmup_steps")
end

# Now profile a window large enough that one-shot per-step variation
# averages out and steady-state kernel times dominate.
n_steps = 10
use_external_profiler = CUDA.Profile.detect_cupti()
if use_external_profiler
    @info "Using external CUDA profiler"
    CUDA.@profile external = true begin
        for i in 1:n_steps
            timed_step!(cs, "Step $i / $n_steps")
        end
    end
else
    @info "Using internal CUDA profiler"
    res = CUDA.@profile external = false begin
        for i in 1:n_steps
            timed_step!(cs, "Step $i / $n_steps")
        end
    end
    show(IOContext(stdout, :limit => false), res)
end
