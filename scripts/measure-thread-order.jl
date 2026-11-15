# What a level-aligned thread order is worth on the SGS 1M closure.
#
# `cloud-mask-census` measured the opportunity: `_sum_1m_over_sgs` returns a
# zero tendency at 22.91% of environment cells, and under the order the launch
# uses -- the parent layout's own, vertical fastest, so a warp is 32 levels of
# one column -- those cells fill a warp 0.31% of the time. Reordered so that a
# warp is two spectral elements at one level, they fill 10.26% of warps and
# 7.80% of 128-thread blocks. The census could only bracket what that is worth,
# between 0.89% and 1.17% of device on the explicit path, because a warp count
# cannot say how much of a skipped point body was CloudMicrophysics and how
# much was the node rule around it.
#
# This measures it. `DataLayouts.LevelAlignedOrder` marks every point active,
# like `NoMask`, and only changes which points consecutive threads take, so the
# two orders evaluate the same function at the same cells with no reduction
# between them and must agree to the bit. That is checked here rather than
# argued: each order writes its own destination field and the two are compared
# as integers.
#
# Both paths are timed. The explicit call runs twice per step and the implicit
# one six times, and the guard is live on both -- `mode` binds the positional
# argument ahead of `args...`, so `isempty(args)` holds whichever mode is
# passed -- so the implicit path is not a control here but the larger half of
# the prize, and `sgs-substep-cost` could not price it because `Instantaneous`
# has no substep loop to fit.
#
# Every point runs against one settled AMIP state in one process, which is the
# only way to resolve a change of this size on a machine where one arm wandered
# 7.73% between two runs of identical code.

import CUDA
import TOML
import ClimaAtmos as CA
import ClimaCore: DataLayouts, Fields

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

# `let` is a hard scope, so assigning `out_path` inside one makes a local and
# leaves the global at its default: the flag parses and is then discarded.
function arg_value(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) && return default
    value = ARGS[i + 1]
    deleteat!(ARGS, i:(i + 1))
    return value
end

out_path = arg_value("--out", "results/thread-order.toml")

const REPEATS = 7
const WARMUP_STEPS = 3
const STEPS = 120
const EXPLICIT_CALLS_PER_STEP = 2
const IMPLICIT_CALLS_PER_STEP = 6

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]

cs = CoupledSimulation(config_file)

for i in 1:WARMUP_STEPS
    @info "warmup step $i / $WARMUP_STEPS"
    step!(cs)
end

integrator = cs.model_sims.atmos_sim.integrator
Y = integrator.u
p = integrator.p
atmos = p.atmos
mp1m = atmos.microphysics_model
tm = atmos.turbconv_model

mp1m isa CA.NonEquilibriumMicrophysics1M ||
    error("expected NonEquilibriumMicrophysics1M, got $(typeof(mp1m))")
tm isa CA.PrognosticEDMFX ||
    error("expected PrognosticEDMFX, got $(typeof(tm))")
CA.not_quadrature(atmos.sgs_quadrature) &&
    error("the SGS quadrature is off, so the kernel this orders is not launched")

thp = CA.Parameters.thermodynamics_params(p.params)
cmp = CA.Parameters.microphysics_1m_params(p.params)
cmp_i = CA.Parameters.microphysics_1m_implicit_params(p.params)
has_implicit = !isnothing(cmp_i) &&
               hasproperty(p.precomputed, :ᶜmp_tendency_implicit⁰)
nsubs = mp1m.n_substeps
nsubs_quad = mp1m.n_substeps_quad
dt_explicit = CA.mp_averaging_dt(p, p.mp_averaging)
dt_implicit = p.dtγ

# The orders, as the `mask` argument `copyto_capped!` already takes. `NoMask` is
# the vertical order the launch uses without being asked.
const ORDERS = (
    "vertical" => DataLayouts.NoMask(),
    "level_aligned" => DataLayouts.LevelAlignedOrder(),
)

explicit!(dest, order) = CA.set_environment_1m_tendency!(
    dest, Y, p, tm, cmp, thp, dt_explicit, nsubs, nsubs_quad, (), order,
)
implicit!(dest, order) = CA.set_environment_1m_tendency!(
    dest, Y, p, tm, cmp_i, thp, dt_implicit, nsubs, nsubs_quad,
    (CA.BMT.Instantaneous(),), order,
)

# Median of repeats: one timing on this machine is worth little, and the mean is
# dragged by whatever else lands on the GPU.
function timed_ms(f; repeats = REPEATS)
    f()
    CUDA.synchronize()
    ts = Float64[]
    for _ in 1:repeats
        CUDA.synchronize()
        push!(ts, CUDA.@elapsed(f()) * 1e3)
    end
    sort!(ts)
    return (median = ts[(length(ts) + 1) ÷ 2], min = first(ts), max = last(ts))
end

# Bitwise, not approximate: a reorder that changes a value has a bug, and a
# float comparison would hide a NaN moving. The integers are the field's own
# storage reinterpreted, so padding is compared too.
function agreement(a, b)
    FT = eltype(a)
    FT === Float32 || error("expected Float32 fields, got $FT")
    ia = reinterpret(UInt32, vec(a))
    ib = reinterpret(UInt32, vec(b))
    differing = count(!=(0), ia .⊻ ib)
    diff = abs.(a .- b)
    worst = differing == 0 ? 0.0 :
            Float64(maximum(x -> isfinite(x) ? x : zero(x), diff))
    return (; values = length(ia), differing, worst)
end

function arm(call!, template, calls_per_step)
    dests = Dict(name => similar(template) for (name, _) in ORDERS)
    points = Dict{String, Any}()
    for (name, order) in ORDERS
        @info "order $name"
        t = timed_ms(() -> call!(dests[name], order))
        points[name] = Dict{String, Any}(
            "ms" => t.median,
            "ms_min" => t.min,
            "ms_max" => t.max,
        )
        @info "  $(t.median) ms"
    end
    base = points["vertical"]
    best = points["level_aligned"]
    saved = base["ms"] - best["ms"]
    spread = max(
        base["ms_max"] - base["ms_min"],
        best["ms_max"] - best["ms_min"],
    )
    (; values, differing, worst) = agreement(
        Array(parent(dests["vertical"])),
        Array(parent(dests["level_aligned"])),
    )
    return Dict{String, Any}(
        "points" => points,
        "calls_per_step" => calls_per_step,
        "saved_ms_per_call" => saved,
        "saved_pct" => 100.0 * saved / base["ms"],
        "window_saved_ms" => saved * calls_per_step * STEPS,
        "repeat_spread_ms" => spread,
        # The saving has to exceed the scatter of the two timings it is a
        # difference of, or it is not a measurement.
        "resolved" => saved > spread,
        "values_compared" => values,
        "values_differing" => differing,
        "bitwise_identical" => differing == 0,
        "worst_abs_difference" => worst,
    )
end

results = Dict{String, Any}(
    "config_file" => config_file,
    "float_type" => string(eltype(Y.c.ρ)),
    "repeats" => REPEATS,
    "warmup_steps" => WARMUP_STEPS,
    "steps" => STEPS,
    "n_substeps_quad" => nsubs_quad,
    "orders" => [name for (name, _) in ORDERS],
    "has_implicit_path" => has_implicit,
    "explicit" => arm(
        explicit!, p.precomputed.ᶜmp_tendency⁰, EXPLICIT_CALLS_PER_STEP,
    ),
    "note" => "Each point calls the live `set_environment_1m_tendency!` on one " *
              "simulation state with a different thread order and nothing " *
              "else changed, so the points differ by the order and no two " *
              "runs are differenced. `vertical` is `DataLayouts.NoMask()`, " *
              "the order the launch uses unasked: the parent layout's own, " *
              "with the vertical fastest, so a warp is 32 levels of one " *
              "column. `level_aligned` is " *
              "`DataLayouts.LevelAlignedOrder()`, which gives a warp the " *
              "cells of one level. Both mark every point active and evaluate " *
              "the same function at the same cells with no reduction between " *
              "them, so they must agree to the bit; bitwise_identical is " *
              "that check, taken on the fields' own storage reinterpreted as " *
              "integers so a NaN cannot pass as equal. The saving is a per " *
              "call figure and window_saved_ms scales it by the calls a step " *
              "makes and the 120 steps the profile covers. The guard the " *
              "order collects is live on both paths, since `mode` binds the " *
              "positional argument ahead of `args...`.",
)
if has_implicit
    results["implicit"] = arm(
        implicit!, p.precomputed.ᶜmp_tendency_implicit⁰, IMPLICIT_CALLS_PER_STEP,
    )
end

mkpath(dirname(out_path))
open(out_path, "w") do io
    TOML.print(io, results; sorted = true)
end
@info "wrote $out_path" results["explicit"]["saved_pct"] results["explicit"]["bitwise_identical"]
