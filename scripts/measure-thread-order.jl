# What a level-aligned thread order is worth on the SGS 1M closure.
#
# `cloud-mask-census` measured the opportunity: `_sum_1m_over_sgs` returns a
# zero tendency at 22.91% of environment cells, and under the order the launch
# uses -- the parent layout's own, vertical fastest, so a warp is 32 levels of
# one column -- those cells fill a warp 0.31% of the time. Reordered so that a
# warp is two spectral elements at one level, they fill 10.26% of warps and
# 7.80% of 128-thread blocks. The census could only bracket what that is worth,
# because a warp count cannot say how much of a skipped point body was
# CloudMicrophysics and how much was the node rule around it.
#
# `DataLayouts.LevelAlignedOrder` marks every point active, like `NoMask`, and
# only changes which points consecutive threads take, so the two orders evaluate
# the same function at the same cells with no reduction between them and must
# agree to the bit. That is checked here rather than argued: each order writes
# the production destination in turn and the results are compared as integers.
#
# Both paths are timed. The explicit call runs twice per step and the implicit
# one six times, and the guard is live on both -- `mode` binds the positional
# argument ahead of `args...`, so `isempty(args)` holds whichever mode is
# passed -- so the implicit calls are a second site to decide rather than a
# control.
#
# THE DEVICE DRIFTS DURING A RUN, AND THAT IS WHAT THE DESIGN IS FOR.
# Timing each configuration in its own block of repeats and comparing the
# blocks' medians reads the drift as a difference between the configurations:
# measured that way the explicit call fell monotonically with execution order,
# 20.04 -> 16.90 -> 16.23 -> 13.89 ms across four blocks, with a 3.3 ms spread
# inside a single block, while the same four configurations put the two
# destinations 19% apart in one direction on the explicit path and 22% the other
# way on the implicit one. So the configurations are interleaved instead: one
# round visits all of them, the visiting order rotates by one each round, and
# the effect is the median of the per-round differences. A drift that is slow
# compared with a round then falls on both arms of every difference, and the
# claim that an effect is resolved is that every round agrees on its sign --
# which is a statement about the differences rather than about a spread that the
# drift itself sets.
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

const ROUNDS = 15
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
# the vertical order, the one the launch uses without being asked.
const ORDERS = (
    "vertical" => DataLayouts.NoMask(),
    "level_aligned" => DataLayouts.LevelAlignedOrder(),
)

# The precomputed field the model writes, and a fresh allocation beside it. The
# second is carried because it is what a probe reaches for, so the interleaved
# rounds can say whether the choice matters at all.
const DESTS = ("production", "fresh")

config_key(dest, order) = "$dest/$order"

explicit!(dest, order) = CA.set_environment_1m_tendency!(
    dest, Y, p, tm, cmp, thp, dt_explicit, nsubs, nsubs_quad, (), order,
)
implicit!(dest, order) = CA.set_environment_1m_tendency!(
    dest, Y, p, tm, cmp_i, thp, dt_implicit, nsubs, nsubs_quad,
    (CA.BMT.Instantaneous(),), order,
)

median(xs) = (s = sort(xs); s[(length(s) + 1) ÷ 2])

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

function arm(call!, production, calls_per_step)
    dests = Dict("production" => production, "fresh" => similar(production))
    calls = Pair{String, Any}[]
    for dest in DESTS, (name, order) in ORDERS
        f = let d = dests[dest], o = order
            () -> call!(d, o)
        end
        push!(calls, config_key(dest, name) => f)
    end

    for (_, f) in calls
        f()
    end
    CUDA.synchronize()

    times = Dict(key => Float64[] for (key, _) in calls)
    rounds = Dict{String, Float64}[]
    for r in 1:ROUNDS
        round = Dict{String, Float64}()
        for (key, f) in circshift(calls, r)
            CUDA.synchronize()
            ms = CUDA.@elapsed(f()) * 1e3
            round[key] = ms
            push!(times[key], ms)
        end
        push!(rounds, round)
    end

    configs = Dict{String, Any}()
    for (key, _) in calls
        ts = times[key]
        configs[key] = Dict{String, Any}(
            "ms" => median(ts),
            "ms_min" => minimum(ts),
            "ms_max" => maximum(ts),
        )
    end

    # One difference per round, so a drift slow compared with a round cancels.
    function effect(a, b)
        ds = [round[a] - round[b] for round in rounds]
        med = median(ds)
        base = median([round[a] for round in rounds])
        return Dict{String, Any}(
            "median_ms" => med,
            "min_ms" => minimum(ds),
            "max_ms" => maximum(ds),
            "pct" => 100.0 * med / base,
            # Every round agreeing on the sign is the claim; a spread the drift
            # sets is not an error bar.
            "sign_consistent" => minimum(ds) > 0 || maximum(ds) < 0,
        )
    end

    order_effect = Dict(
        dest => effect(
            config_key(dest, "vertical"), config_key(dest, "level_aligned"),
        ) for dest in DESTS
    )
    destination_effect = Dict(
        name => effect(config_key("fresh", name), config_key("production", name))
        for (name, _) in ORDERS
    )

    # Agreement on the production field: the two orders run into it in turn and
    # the result is copied out between them, so the values compared are the ones
    # the model would go on to use.
    snapshots = Dict{String, Any}()
    for (name, order) in ORDERS
        call!(production, order)
        CUDA.synchronize()
        snapshots[name] = Array(parent(production))
    end
    (; values, differing, worst) =
        agreement(snapshots["vertical"], snapshots["level_aligned"])

    # The treatment is read off the destination the model writes.
    headline = order_effect["production"]
    saved = headline["median_ms"]
    return Dict{String, Any}(
        "configs" => configs,
        "order_effect" => order_effect,
        "destination_effect" => destination_effect,
        "calls_per_step" => calls_per_step,
        "vertical_ms" => configs[config_key("production", "vertical")]["ms"],
        "level_aligned_ms" =>
            configs[config_key("production", "level_aligned")]["ms"],
        "saved_ms_per_call" => saved,
        "saved_pct" => headline["pct"],
        "window_saved_ms" => saved * calls_per_step * STEPS,
        "resolved" => headline["sign_consistent"],
        "values_compared" => values,
        "values_differing" => differing,
        "bitwise_identical" => differing == 0,
        "worst_abs_difference" => worst,
    )
end

results = Dict{String, Any}(
    "config_file" => config_file,
    "float_type" => string(eltype(Y.c.ρ)),
    "rounds" => ROUNDS,
    "warmup_steps" => WARMUP_STEPS,
    "steps" => STEPS,
    "n_substeps_quad" => nsubs_quad,
    "orders" => [name for (name, _) in ORDERS],
    "destinations" => [d for d in DESTS],
    "has_implicit_path" => has_implicit,
    "explicit" => arm(
        explicit!, p.precomputed.ᶜmp_tendency⁰, EXPLICIT_CALLS_PER_STEP,
    ),
    "note" => "Each configuration calls the live " *
              "`set_environment_1m_tendency!` on one simulation state, so the " *
              "configurations differ by the thread order and the destination " *
              "field and no two runs are differenced. `vertical` is " *
              "`DataLayouts.NoMask()`, the order the launch uses unasked: the " *
              "parent layout's own, with the vertical fastest, so a warp is " *
              "32 levels of one column. `level_aligned` is " *
              "`DataLayouts.LevelAlignedOrder()`, which gives a warp the " *
              "cells of one level. Both mark every point active and evaluate " *
              "the same function at the same cells with no reduction between " *
              "them, so they must agree to the bit; bitwise_identical is that " *
              "check, taken on the production field's own storage " *
              "reinterpreted as integers so a NaN cannot pass as equal. The " *
              "device drifts during a run, so the configurations are " *
              "interleaved: one round visits all four, the visiting order " *
              "rotates each round, and an effect is the median of the " *
              "per-round differences with sign_consistent saying whether " *
              "every round agreed on its direction. A per-block spread is not " *
              "quoted as an error bar because the drift sets it. Figures " *
              "outside `configs` are read off the production destination, and " *
              "destination_effect is what the probe's own choice of field is " *
              "worth. window_saved_ms scales the per-call figure by the calls " *
              "a step makes and the 120 steps the profile covers. The guard " *
              "the order collects is live on both paths, since `mode` binds " *
              "the positional argument ahead of `args...`.",
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
@info "wrote $out_path" results["explicit"]["saved_pct"] results["explicit"]["resolved"] results["explicit"]["bitwise_identical"]
