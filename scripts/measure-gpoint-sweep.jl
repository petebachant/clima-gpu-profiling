# Does the longwave g-point split need 16 groups?
#
# `DEFAULT_NGPT_GROUPS` is 16, which widens every longwave workspace the solve
# indexes by column to `ncol * 16`. That memory is the whole cost of the
# treatment, so the default should come from a measurement of where the saving
# stops rather than from the first group count that worked.
#
# Each point rebuilds ONLY the longwave RTE's group workspaces. Everything the
# solve reads -- atmospheric state, lookups, boundary conditions, and the
# narrow optics, sources and flux buffers the ungrouped path uses -- is the
# live simulation's own, so two points differ by the group count and nothing
# else. That makes this an in-block comparison: it never differences two runs,
# which is the only way to resolve a few percent on this machine.
#
# `ngroups = 1` passes `nothing` and takes the ungrouped specialization, so the
# baseline is a point in the same sweep rather than a separate measurement.

import CUDA
import RRTMGP
import TOML

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

const RTE = RRTMGP.RTE
const RTESolver = RRTMGP.RTESolver

out_path = "results/gpoint-sweep.toml"
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i)
        out_path = ARGS[i + 1]
        deleteat!(ARGS, i:(i + 1))
    end
end

const GROUPS = [1, 4, 8, 16]
const REPEATS = 7
const WARMUP_STEPS = 3

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)

# Radiation fires every dt_rad, so step until it has run and the state the
# solve reads is settled.
for i in 1:WARMUP_STEPS
    @info "warmup step $i / $WARMUP_STEPS"
    step!(cs)
end

solver = cs.model_sims.atmos_sim.integrator.p.radiation.rrtmgp_solver
lws = solver.lws
gp = solver.grid_params
as = solver.as
lk = solver.lookups
clear_acc = solver.clear_flux_acc_lw
ms = solver.deep_atmosphere_inverse_scaling
nlay, ncol = RRTMGP.AtmosphericStates.get_dims(as)
FT = eltype(gp)

lws isa RTE.TwoStreamLWRTE ||
    error("expected a TwoStreamLWRTE longwave solver, got $(typeof(lws))")
isnothing(lws.band_flux) ||
    error("band_flux is being kept, so the grouped path is unreachable")

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
    return (
        median = ts[(length(ts) + 1) ÷ 2],
        min = first(ts),
        max = last(ts),
    )
end

# Swap in one group workspace, leaving every other field of the live solver
# alone. The grouped path reads its wide optics, sources and mask out of the
# workspace, so the narrow ones shared here are only ever used by `ngroups = 1`.
function with_groups(lw::RTE.TwoStreamLWRTE, groups)
    return RTE.TwoStreamLWRTE(
        lw.context,
        lw.op,
        lw.src,
        lw.bcs,
        lw.fluxb,
        lw.flux,
        lw.band_flux,
        lw.state_cache,
        groups,
    )
end

lw_solve!(lw) = RTESolver.solve_lw_both_skies!(
    lw,
    clear_acc,
    as,
    lk.lookup_lw,
    lk.lookup_lw_cld,
    lk.lookup_lw_aero,
    ms,
)

# The flux a point produced, on the host, so later points cannot overwrite the
# comparison: every point writes into the same `flux` buffers by construction.
flux_snapshot(lw) = (
    up = Array(lw.flux.flux_up),
    dn = Array(lw.flux.flux_dn),
    net = Array(lw.flux.flux_net),
    clear_up = Array(clear_acc.flux_up),
)

function worst_deviation(a, b)
    worst = 0.0
    for f in (:up, :dn, :net, :clear_up)
        x, y = getfield(a, f), getfield(b, f)
        size(x) == size(y) || error("flux shape changed between group counts")
        worst = max(worst, maximum(abs.(Float64.(x) .- Float64.(y))))
    end
    return worst
end

# Everything that accumulates across the sweep lives in here. A `for` at top
# level makes its assignments local, so a reference captured in one iteration is
# not visible in the next -- the UndefVarError that has cost this project four
# reservations.
function sweep()
    points = Dict{String, Any}()
    reference = nothing
    reference_ms = 0.0

    for n in GROUPS
        @info "ngpt_groups = $n"
        CUDA.reclaim()
        free_before = CUDA.available_memory()
        groups = n > 1 ? RTE.LWGptGroups(gp, n; params = solver.params) : nothing
        CUDA.synchronize()
        bytes = free_before - CUDA.available_memory()

        lw = with_groups(lws, groups)
        t = timed_ms(() -> lw_solve!(lw))
        shot = flux_snapshot(lw)

        if isnothing(reference)
            reference = shot
            reference_ms = t.median
        end

        # The accumulators alone, from their shapes, as a floor under `bytes`:
        # two `(ncol * ngroups, nlev, 3)` arrays. The rest of the widening is
        # the optics, sources, band fluxes and cloud mask.
        acc_bytes = n > 1 ? 2 * (ncol * n) * (nlay + 1) * 3 * sizeof(FT) : 0

        points[string(n)] = Dict{String, Any}(
            "ngroups" => n,
            "solve_ms" => t.median,
            "solve_ms_min" => t.min,
            "solve_ms_max" => t.max,
            "change_pct" => 100.0 * (t.median - reference_ms) / reference_ms,
            "workspace_bytes" => bytes,
            "workspace_gib" => bytes / 2^30,
            "accumulator_bytes" => acc_bytes,
            "worst_flux_deviation" => worst_deviation(reference, shot),
        )

        GC.gc()
        CUDA.reclaim()
    end
    return points
end

points = sweep()

# Which group count is worth paying for: the smallest one within a repeat
# spread of the best saving any point reached. The spread is this sweep's own,
# taken from the reference point, so the threshold is measured rather than
# assumed.
spread_pct =
    100.0 * (points["1"]["solve_ms_max"] - points["1"]["solve_ms_min"]) /
    points["1"]["solve_ms"]
best = minimum(points[string(n)]["change_pct"] for n in GROUPS)
sufficient = minimum(
    n for n in GROUPS if points[string(n)]["change_pct"] <= best + spread_pct
)

results = Dict{String, Any}(
    "ncol" => ncol,
    "nlay" => nlay,
    "n_gpt_lw" => length(lk.lookup_lw.band_data.major_gpt2bnd),
    "float_type" => string(FT),
    "repeats" => REPEATS,
    "warmup_steps" => WARMUP_STEPS,
    "config_file" => config_file,
    "group_counts" => GROUPS,
    "repeat_spread_pct" => spread_pct,
    "best_change_pct" => best,
    "sufficient_ngroups" => sufficient,
    "default_ngpt_groups" => RTE.DEFAULT_NGPT_GROUPS,
    "default_is_sufficient_only" => RTE.DEFAULT_NGPT_GROUPS == sufficient,
    "points" => points,
    "note" => "solve_ms times the whole longwave both-skies solve, so it " *
              "charges the grouped path for its reduction and its state " *
              "refresh -- it is the net figure, not the kernel's. Every " *
              "point shares one simulation's state and buffers and differs " *
              "only in group count, so these are in-block comparisons and " *
              "not a difference of runs. worst_flux_deviation is against " *
              "ngroups = 1, which is the ungrouped specialization: the " *
              "split is not bit-for-bit, since a g-point sum taken in N " *
              "partials is a different summation order. workspace_bytes is " *
              "the device memory the group workspace took, measured as a " *
              "free-memory delta across its construction.",
)

mkpath(dirname(out_path))
open(io -> TOML.print(io, results; sorted = true), out_path, "w")
@info "wrote $out_path" sufficient best spread_pct
