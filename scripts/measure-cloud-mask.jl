# Which cells could the SGS 1M closure skip, and does any thread order put them
# in the same warp?
#
# `sgs-substep-cost` measured the two halves of the problem: the CM calls are
# 75.9% of the explicit closure call, and 43.6% of environment cells carry no
# condensate while only 2.48% of 32-cell warps are clear in every lane. A
# treatment has to move the second number, and there are two ways to try: skip
# the clear cells where they lie (which needs them to share a warp) or gather
# them out of the way (which needs a compacted index list).
#
# Two things have to be measured before either is built.
#
# The first is the hit rate of the guard that already exists.
# `_sum_1m_over_sgs` returns `zero_1m_tendency` for a condensate-free cell with
# no equilibrium cloud, and it tests "no equilibrium cloud" as
# `!isfinite(λ_eq)`. `λ_eq` is `-Inf` only where a truncated displacement makes
# the equilibrium condensate vanish; with `sgs_displacement_truncation = Inf`
# and `sgs_variance_curvature = false`, both defaults here, `λ_eq` is the mean
# cloud excess and is always finite. So the guard should never fire, and this
# counts how often it does rather than inferring it from the code.
#
# The second is how a repaired criterion would distribute over warps. A dry
# cell's node rule puts both cloud edges far out in the tail -- the actual edge
# at `t_a = -λ_lagrange/α` and the equilibrium edge at `t_e = -λ_eq` -- and the
# outermost interval always keeps two nodes with weight `Q(|edge|/σ_S)`. That
# weight is not zero, so skipping the cell is not bit-for-bit; it is below
# rounding once the edges are `k` standard deviations out, and `k` is swept
# rather than chosen.
#
# The warp census is then taken under two thread orders, because the order is a
# property of the launch and not of the field: the current one (the parent
# layout's, vertical fastest) and a level-aligned one (horizontal fastest, so a
# warp is two spectral elements at one level). Perfect compaction is reported
# beside them as the per-cell rate, which no order can beat.

import CUDA
import TOML
import ClimaAtmos as CA

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

out_path = arg_value("--out", "results/cloud-mask.toml")

const EDGE_SIGMAS = [2, 4, 6, 8]
const GROUP_SIZES = [32, 128, 256]
const REPEATS = 7
const WARMUP_STEPS = 3

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
    error("the SGS quadrature is off, so the closure has no node rule to skip")

thp = CA.Parameters.thermodynamics_params(p.params)
cmp = CA.Parameters.microphysics_1m_params(p.params)
nsubs = mp1m.n_substeps
nsubs_quad = mp1m.n_substeps_quad
dt_explicit = CA.mp_averaging_dt(p, p.mp_averaging)

# The explicit environment call from `microphysics_cache.jl`. Running it is what
# materializes the environment condensates in the scratch fields the census
# reads, so the mask describes the state the kernel just ran on.
explicit!() = CA.set_environment_1m_tendency!(
    p.precomputed.ᶜmp_tendency⁰, Y, p, tm, cmp, thp,
    dt_explicit, nsubs, nsubs_quad, (),
)

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

call = timed_ms(explicit!)
@info "explicit call $(call.median) ms"

# The parent of a center field is (v, i, j, f, h) with one field component, so
# dropping that axis leaves the cell grid the thread order is defined on. The
# census is a claim about which cells share a warp, so a layout that does not
# have that shape has to fail here rather than be reshaped into agreement.
function cells(field)
    a = Array(parent(field))
    s = size(a)
    length(s) == 5 && s[4] == 1 ||
        error("expected a (v, i, j, 1, h) center layout, got $s")
    return reshape(a, s[1], s[2], s[3], s[5])
end

moments = p.precomputed.ᶜsgs_moments
q_lcl = cells(p.scratch.ᶜtemp_scalar_2)
q_icl = cells(p.scratch.ᶜtemp_scalar_3)
q_rai = cells(p.scratch.ᶜtemp_scalar_4)
q_sno = cells(p.scratch.ᶜtemp_scalar_5)
T = cells(p.precomputed.ᶜT⁰)
T′T′ = cells(p.precomputed.ᶜT′T′)
α = cells(p.precomputed.ᶜα)
λ_eq = cells(moments.λ_eq)
λ_lagrange = cells(moments.λ_lagrange)
sigma_S = cells(moments.sigma_S)

FT = eltype(q_lcl)
T_freeze = FT(CA.TD.Parameters.T_freeze(thp))
T_icenuc = FT(CA.TD.Parameters.T_icenuc(thp))

# The guard's three conditions, each as its own mask so the census says which
# one binds rather than only reporting the conjunction.
no_cond = @. (q_lcl <= 0) & (q_icl <= 0) & (q_rai <= 0) & (q_sno <= 0)
# `ΔT_max = 8√T′T′` is the guard's own bound on the node temperatures.
t_band = let ΔT = @. 8 * sqrt(max(T′T′, zero(FT)))
    @. (T - ΔT >= T_freeze) | (T + ΔT <= T_icenuc)
end
eq_nonfinite = @. !isfinite(λ_eq)
guard_live = @. no_cond & eq_nonfinite & t_band

# How far out the node rule's two cloud edges sit, in standard deviations of the
# saturation excess. `σ_S` and `α` are floored the way the kernel floors them,
# so a degenerate cell reports a finite distance rather than an infinity.
edge_sigmas = let σ = (@. max(sigma_S, eps(FT))), α_safe = (@. max(α, eps(FT)))
    t_a = @. -λ_lagrange / α_safe
    t_e = @. -λ_eq
    @. min(t_a, t_e) / σ
end

# A thread order is a permutation of the cell grid; a group of consecutive
# threads under that order is skipped only if every one of its lanes is.
const ORDERS = (
    # The parent layout's own order, which is what the pointwise kernel uses:
    # vertical fastest, so a warp is 32 levels of one column.
    "vertical_fastest" => (1, 2, 3, 4),
    # Horizontal fastest, so a warp is two spectral elements at one level.
    "level_aligned" => (2, 3, 4, 1),
)

function group_census(flat, width)
    ngroups = length(flat) ÷ width
    allskip = 0
    for g in 1:ngroups
        allskip += all(@view flat[(width * (g - 1) + 1):(width * g)]) ? 1 : 0
    end
    return (; ngroups, allskip)
end

function order_census(mask)
    out = Dict{String, Any}()
    for (name, perm) in ORDERS
        flat = vec(perm == (1, 2, 3, 4) ? mask : permutedims(mask, perm))
        groups = Dict{String, Any}()
        for width in GROUP_SIZES
            (; ngroups, allskip) = group_census(flat, width)
            groups[string(width)] = Dict{String, Any}(
                "groups" => ngroups,
                "all_skippable" => allskip,
                "all_skippable_fraction" => allskip / ngroups,
            )
        end
        out[name] = groups
    end
    return out
end

# What an order is worth: the work now is one warp per warp that is not entirely
# skippable, and the work after is the same count under the new order. Perfect
# compaction pays the per-cell rate, which no order reaches.
function savings(mask, orders)
    cell_frac = count(mask) / length(mask)
    base = orders["vertical_fastest"]["32"]["all_skippable_fraction"]
    out = Dict{String, Any}(
        "cell_fraction" => cell_frac,
        "compaction_saving_fraction" => base < 1 ?
            1 - (1 - cell_frac) / (1 - base) : 0.0,
    )
    for (name, _) in ORDERS
        f = orders[name]["32"]["all_skippable_fraction"]
        out[name * "_saving_fraction"] = base < 1 ? 1 - (1 - f) / (1 - base) : 0.0
    end
    return out
end

function census(mask)
    orders = order_census(mask)
    return merge(Dict{String, Any}("orders" => orders), savings(mask, orders))
end

masks = Dict{String, Any}(
    # The guard as it is written: this is the number that says whether it fires.
    "guard_live" => census(guard_live),
    # Its three conditions on their own.
    "no_condensate" => census(no_cond),
    "temperature_band" => census(t_band),
)
for k in EDGE_SIGMAS
    # The repaired criterion: condensate-free, outside the mixed-phase band, and
    # with both cloud edges at least `k` standard deviations out.
    masks["edges_beyond_$(k)_sigma"] = census(@. no_cond & t_band & (edge_sigmas >= k))
end

results = Dict{String, Any}(
    "config_file" => config_file,
    "float_type" => string(FT),
    "repeats" => REPEATS,
    "warmup_steps" => WARMUP_STEPS,
    "n_substeps_quad" => nsubs_quad,
    "cells" => length(no_cond),
    "nlev" => size(no_cond, 1),
    "group_sizes" => GROUP_SIZES,
    "edge_sigmas" => EDGE_SIGMAS,
    "orders" => [name for (name, _) in ORDERS],
    "explicit_call_ms" => call.median,
    "explicit_call_ms_min" => call.min,
    "explicit_call_ms_max" => call.max,
    "eq_nonfinite_cells" => count(eq_nonfinite),
    "sgs_displacement_truncation" =>
        Float64(CA.Parameters.sgs_displacement_truncation(p.params)),
    "sgs_variance_curvature" => atmos.sgs_variance_curvature,
    "masks" => masks,
    "note" => "Every mask is over environment cells of one settled AMIP " *
              "state, read from the scratch fields the explicit closure call " *
              "just materialized. guard_live is the condition " *
              "`_sum_1m_over_sgs` tests before returning a zero tendency, so " *
              "its cell_fraction is the rate at which the closure's own " *
              "early-out fires; eq_nonfinite_cells is the `!isfinite(λ_eq)` " *
              "term on its own. edges_beyond_K_sigma replaces that term with " *
              "the distance of the node rule's two cloud edges, " *
              "`min(-λ_lagrange/α, -λ_eq)/σ_S`, which is what decides whether " *
              "any node of a dry cell carries weight: skipping such a cell is " *
              "not bit-for-bit, since the outermost interval keeps two nodes " *
              "of weight Q(K), so K is swept rather than chosen. Each mask is " *
              "censused under two thread orders and three group sizes: a group " *
              "is skippable only when every lane is, and " *
              "all_skippable_fraction at 32 is the warp rate a treatment " *
              "would have to move. The saving fractions are relative to the " *
              "current order's warp rate, which is what is being paid now; " *
              "compaction_saving_fraction is the per-cell bound no order " *
              "reaches.",
)

mkpath(dirname(out_path))
open(out_path, "w") do io
    TOML.print(io, results; sorted = true)
end
@info "wrote $out_path" count(guard_live) count(no_cond) / length(no_cond)
