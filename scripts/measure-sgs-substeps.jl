# What does one CloudMicrophysics evaluation cost inside the SGS 1M closure?
#
# The closure is 21.1% of device time across 8 calls per step, and every
# proposal for cutting it rests on how that time splits between the quadrature
# machinery and the CM bulk-tendency calls inside it. `docs/learnings.md` S4r
# put the CM call at ~3.7% of the kernel, inferred from skipping it wherever
# the clear-air guard fires. That inference needs the per-point cost to be
# uniform, and it is not: the guard fires at clear points, where the call is
# cheap, and under SIMT a warp pays the cloudy path if any one of its 32 lanes
# is cloudy. So the figure bounds what SKIPPING buys, not what the call costs.
#
# `n_substeps_quad` is the handle that measures it directly. CM's
# `LinearizedAverage` mode runs `_substep_1m` exactly `nsub` times per
# quadrature node (`BulkMicrophysicsTendencies.jl`), and nothing else in the
# kernel depends on it, so timing the same call at several substep counts and
# fitting a line gives the marginal cost of one CM evaluation as the slope and
# everything else -- the node rule, the saturation calls, the data movement --
# as the intercept. No source is modified and no second run is differenced:
# every point runs against one simulation state, which is the only way to
# resolve this on a machine where one arm wandered 7.7% between identical runs.
#
# It also prices the knob itself, since `n_substeps_quad: 2 -> 1` is one config
# value, and it reports the divergence directly: the condensate-free fraction
# per point AND per 32 consecutive points, because a fraction of points is not
# a fraction of warps and the difference is what a compaction treatment would
# have to recover.
#
# The implicit path is measured beside it as a control. It runs
# `BMT.Instantaneous()`, which has no substep loop at all, so its slope against
# `nsub` must be flat -- and 6 of the 8 calls per step take that path.

import CUDA
import TOML
import ClimaAtmos as CA

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

out_path = "results/sgs-substeps.toml"
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i)
        out_path = ARGS[i + 1]
        deleteat!(ARGS, i:(i + 1))
    end
end

const SUBSTEPS = [1, 2, 3, 4]
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
    error("the SGS quadrature is off, so there is no substep loop to measure")

thp = CA.Parameters.thermodynamics_params(p.params)
cmp = CA.Parameters.microphysics_1m_params(p.params)
cmp_i = CA.Parameters.microphysics_1m_implicit_params(p.params)
has_implicit = !isnothing(cmp_i) &&
               hasproperty(p.precomputed, :ᶜmp_tendency_implicit⁰)
nsubs = mp1m.n_substeps
dt_explicit = CA.mp_averaging_dt(p, p.mp_averaging)
dt_implicit = p.dtγ

# The two environment calls from `microphysics_cache.jl`, with the substep
# count lifted out of the model so it can be varied. `set_environment_1m_tendency!`
# takes it as an argument, so these are the call sites' own arguments and not a
# re-implementation of them.
explicit!(n) = CA.set_environment_1m_tendency!(
    p.precomputed.ᶜmp_tendency⁰, Y, p, tm, cmp, thp,
    dt_explicit, nsubs, n, (),
)
implicit!(n) = CA.set_environment_1m_tendency!(
    p.precomputed.ᶜmp_tendency_implicit⁰, Y, p, tm, cmp_i, thp,
    dt_implicit, nsubs, n, (CA.BMT.Instantaneous(),),
)

# Median of repeats: one timing on this machine is worth little, and the mean
# is dragged by whatever else lands on the GPU.
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

# Ordinary least squares on (nsub, ms). The slope is one CM evaluation over the
# whole environment field; the intercept is the rest of the call.
function fit_line(xs, ys)
    n = length(xs)
    x̄ = sum(xs) / n
    ȳ = sum(ys) / n
    sxx = sum((x - x̄)^2 for x in xs)
    sxy = sum((xs[i] - x̄) * (ys[i] - ȳ) for i in 1:n)
    slope = sxy / sxx
    intercept = ȳ - slope * x̄
    ss_tot = sum((y - ȳ)^2 for y in ys)
    ss_res = sum((ys[i] - (intercept + slope * xs[i]))^2 for i in 1:n)
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : 1.0
    return (; slope, intercept, r2)
end

function sweep(call!)
    points = Dict{String, Any}()
    for n in SUBSTEPS
        @info "n_substeps_quad = $n"
        t = timed_ms(() -> call!(n))
        points[string(n)] = Dict{String, Any}(
            "n_substeps_quad" => n,
            "ms" => t.median,
            "ms_min" => t.min,
            "ms_max" => t.max,
        )
        @info "  $(t.median) ms"
    end
    return points
end

function summarize(points, at)
    xs = Float64.(SUBSTEPS)
    ys = [points[string(n)]["ms"] for n in SUBSTEPS]
    (; slope, intercept, r2) = fit_line(xs, ys)
    here = points[string(at)]["ms"]
    spread = 100.0 * (points[string(at)]["ms_max"] -
                      points[string(at)]["ms_min"]) / here
    return Dict{String, Any}(
        "points" => points,
        "n_substeps_quad_in_use" => at,
        "ms_in_use" => here,
        "repeat_spread_pct" => spread,
        "per_substep_ms" => slope,
        "fixed_ms" => intercept,
        "fit_r2" => r2,
        "substep_share_pct" => 100.0 * at * slope / here,
        "saving_one_fewer_substep_pct" => 100.0 * slope / here,
        "slope_resolved" => abs(slope) > (points[string(at)]["ms_max"] -
                                          points[string(at)]["ms_min"]),
    )
end

explicit_points = sweep(explicit!)
# Read the environment condensate the call just materialized, before anything
# else writes the scratch fields it used.
function divergence(p)
    mask = map(
        (a, b, c, d) -> (a <= 0) & (b <= 0) & (c <= 0) & (d <= 0),
        Array(parent(p.scratch.ᶜtemp_scalar_2)),
        Array(parent(p.scratch.ᶜtemp_scalar_3)),
        Array(parent(p.scratch.ᶜtemp_scalar_4)),
        Array(parent(p.scratch.ᶜtemp_scalar_5)),
    )
    flat = vec(mask)
    npts = length(flat)
    clear = count(flat)
    # A warp is 32 consecutive points of the layout the pointwise kernel
    # indexes, and it runs the cloudy path unless every one of its lanes is
    # clear. Partial trailing groups are dropped.
    nwarps = npts ÷ 32
    allclear = 0
    for w in 1:nwarps
        allclear += all(@view flat[(32 * (w - 1) + 1):(32 * w)]) ? 1 : 0
    end
    return (; npts, clear, nwarps, allclear)
end
(; npts, clear, nwarps, allclear) = divergence(p)

implicit_points = has_implicit ? sweep(implicit!) : nothing

clear_frac = clear / npts
warp_frac = allclear / nwarps
results = Dict{String, Any}(
    "config_file" => config_file,
    "repeats" => REPEATS,
    "warmup_steps" => WARMUP_STEPS,
    "substep_counts" => SUBSTEPS,
    "n_substeps" => nsubs,
    "float_type" => string(eltype(Y.c.ρ)),
    "points" => npts,
    "condensate_free_points" => clear,
    "condensate_free_fraction" => clear_frac,
    "warps" => nwarps,
    "fully_clear_warps" => allclear,
    "fully_clear_warp_fraction" => warp_frac,
    # What a perfect compaction of the cloudy points could remove from the CM
    # calls: they are paid by every warp that is not fully clear and would be
    # paid by the cloudy points alone.
    "compaction_ceiling_of_cm_pct" =>
        warp_frac < 1 ? 100.0 * (1 - (1 - clear_frac) / (1 - warp_frac)) : 0.0,
    "explicit" => summarize(explicit_points, mp1m.n_substeps_quad),
    "has_implicit_path" => has_implicit,
    "note" => "Each point calls the live `set_environment_1m_tendency!` on " *
              "one simulation state with a different `n_substeps_quad`, so " *
              "the points differ by the substep count and nothing else and " *
              "no two runs are differenced. per_substep_ms is the fitted " *
              "slope: one CM `_substep_1m` over the whole environment field, " *
              "under the warp divergence the real field has. fixed_ms is the " *
              "intercept -- the conditional-moment rule, the per-node " *
              "saturation calls, the column loop that materializes the " *
              "environment, and the data movement. The implicit arm runs " *
              "BMT.Instantaneous(), which has no substep loop, so its slope " *
              "is a control and should be flat. condensate_free_fraction is " *
              "per point and fully_clear_warp_fraction is per 32 consecutive " *
              "points of the layout the pointwise kernel indexes: the gap " *
              "between them is the divergence, and " *
              "compaction_ceiling_of_cm_pct is the share of the CM calls a " *
              "perfect gather could remove.",
)
if has_implicit
    results["implicit"] = summarize(implicit_points, mp1m.n_substeps_quad)
end

mkpath(dirname(out_path))
open(io -> TOML.print(io, results; sorted = true), out_path, "w")
@info "wrote $out_path" results["explicit"]["per_substep_ms"] results["explicit"]["substep_share_pct"] clear_frac warp_frac
