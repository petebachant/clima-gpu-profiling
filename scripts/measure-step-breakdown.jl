# How much of a coupler step is GPU work, measured WITHOUT a profiler?
#
# The question this answers: SYPD implies ~277 ms per coupler step, while nsys
# reports ~136 ms of GPU kernel time per step. If that gap is real, roughly half
# the wall clock is not GPU kernels, and no amount of kernel optimisation can
# reach it -- measured pass-through is 0.43, so even zeroing all kernel time
# caps out near +43% SYPD.
#
# But the gap may not be real. docs/learnings.md 2b-i established that nsys's own
# instrumentation dominates the GPU-idle measurement in this model: the profiled
# mod arm reports 48.3% idle against baseline's 30.5% while being FASTER in SYPD,
# which is incoherent. So the idle figure cannot be trusted and neither can a
# non-GPU fraction derived from it.
#
# This measures the same quantity with CUDA events instead. Events are recorded
# on the stream around each step and read back after synchronising, so the only
# overhead is two event records per step -- nanoseconds, against a ~277 ms step.
#
# Reports per step:
#   wall_ms      host wall time for step!(cs), including any GPU wait
#   gpu_span_ms  first-to-last GPU activity on the default stream
#   gap_ms       wall - gpu_span: host time outside the GPU timeline
#
# gpu_span is a SPAN, not busy time: it includes gaps between kernels. So it is
# an UPPER bound on GPU busy, which makes `gap_ms` a LOWER bound on non-GPU time.
# If gap_ms is large, the structural headroom is real regardless.

import ClimaComms
ClimaComms.@import_required_backends
import CUDA
import Statistics
import TOML
using Printf

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

out_path = "results/step-breakdown.toml"
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i)
        out_path = ARGS[i + 1]
        deleteat!(ARGS, i:(i + 1))
    end
end

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)

n_warmup = 3
n_steps = 12
let i = findfirst(==("--steps"), ARGS)
    if !isnothing(i)
        n_steps = parse(Int, ARGS[i + 1])
        deleteat!(ARGS, i:(i + 1))
    end
end
for i in 1:n_warmup
    @info "warmup step $i / $n_warmup"
    step!(cs)
end
CUDA.synchronize()

wall = Float64[]
span = Float64[]
med_so_far = Inf
for i in 1:n_steps
    e0, e1 = CUDA.CuEvent(), CUDA.CuEvent()
    CUDA.record(e0)
    t0 = time_ns()
    step!(cs)
    CUDA.record(e1)
    CUDA.synchronize()
    w = (time_ns() - t0) / 1e6
    g = CUDA.elapsed(e0, e1) * 1000
    push!(wall, w)
    push!(span, g)
    # Print sparsely on long runs; the per-step series goes to the TOML anyway.
    if n_steps <= 20 || i % 60 == 0 || w > 1.5 * (isempty(wall) ? w : med_so_far)
        @printf("step %4d  wall %8.1f ms   gpu_span %8.1f ms   gap %6.1f ms (%.1f%%)\n",
                i, w, g, w - g, 100 * (w - g) / w)
    end
    global med_so_far = Statistics.median(wall)
end

# Duty cycle: on a long window the mean is what SYPD reflects, and the gap
# between mean and median is exactly the periodic work a short window misses.
function duty_report(wall)
    m, md = Statistics.mean(wall), Statistics.median(wall)
    over = [w for w in wall if w > 1.25 * md]
    @printf("\n  mean %.1f ms   median %.1f ms   mean/median %.2f\n", m, md, m / md)
    @printf("  %d of %d steps exceed 1.25x median; they carry %.1f%% of total wall time\n",
            length(over), length(wall), 100 * sum(over) / sum(wall))
    return m, md, length(over), (isempty(over) ? 0.0 : sum(over) / sum(wall))
end

med(x) = Statistics.median(x)
mw, mg = med(wall), med(span)
@printf("\nmedian over %d steps\n", n_steps)
@printf("  wall      %8.1f ms\n", mw)
@printf("  gpu_span  %8.1f ms  (%.1f%% of wall)\n", mg, 100mg / mw)
@printf("  gap       %8.1f ms  (%.1f%% of wall)  <- host time outside the GPU timeline\n",
        mw - mg, 100 * (mw - mg) / mw)
@printf("\n  nsys reports ~136 ms/step of GPU KERNEL time; gpu_span here is a span\n")
@printf("  (includes inter-kernel gaps) so it bounds busy time from above.\n")

mean_ms, median_ms, n_expensive, expensive_frac = duty_report(wall)

TOML.print(open(out_path, "w"), Dict(
    "steps" => n_steps,
    "wall_ms_mean" => mean_ms,
    "expensive_steps" => n_expensive,
    "expensive_wall_frac" => expensive_frac,
    "wall_ms_median" => mw,
    "gpu_span_ms_median" => mg,
    "gap_ms_median" => mw - mg,
    "gap_frac_of_wall" => (mw - mg) / mw,
    "wall_ms" => wall,
    "gpu_span_ms" => span,
    "note" => "Measured without a profiler, using CUDA events around step!(). " *
        "gpu_span is first-to-last GPU activity on the default stream and " *
        "includes inter-kernel gaps, so it is an upper bound on GPU busy time " *
        "and gap_ms is a lower bound on host time outside the GPU timeline.",
))
@info "wrote $out_path"
