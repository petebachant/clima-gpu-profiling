# Does splitting the g-point loop change the answer?
#
# The two sweeps measure what the split is worth. This checks that it is the
# same solve, and it is the only place that does so from the pipeline: the test
# runs quoted in CliMA/RRTMGP.jl#637 were taken by hand, and a correctness claim
# nothing regenerates is the kind of figure AGENTS.md exists to stop.
#
# It is separate from the sweeps on purpose. A sweep times blocks of real coupler
# steps and so has to own a long reservation; this needs no timing at all, so it
# can afford group counts a sweep cannot. In particular it uses counts that do
# NOT divide the g-point count. Both sweeps use 4, 8, 16 and 32 against 256
# longwave and 224 shortwave g-points, which divide exactly, so every group gets
# a stride of equal length and the ragged case never runs. A dropped or
# double-counted g-point at the end of an uneven stride is exactly the bug this
# shape of code has, and it is invisible to the sweeps.
#
# What is asserted, and why each one is the test it is:
#
#   * Fluxes, against the ungrouped solve, to a relative tolerance. The split is
#     NOT bit-for-bit -- a 256-term Float32 sum taken in N partials is a
#     different summation order -- so this is a tolerance, and the interesting
#     property is that it does not GROW with the group count. A reordering error
#     that accumulated with the number of partials would rise across the sweep.
#   * Cloud cover, exactly. It is a count of cloudy g-points divided by their
#     number, so the reduction either sums the groups' tallies correctly or it
#     does not; roundoff cannot enter.
#   * The 550 nm aerosol optical depth, exactly, shortwave only. It is zeroed and
#     re-accumulated per g-point, so it is a read-modify-write on one per-column
#     address that only one nominated g-point may write. If the nomination is
#     wrong, groups interleave their partial sums and this moves.

import CUDA
import RRTMGP
import TOML

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

const RTE = RRTMGP.RTE
const RTESolver = RRTMGP.RTESolver

out_path = "results/gpoint-equivalence.toml"
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i)
        out_path = ARGS[i + 1]
        deleteat!(ARGS, i:(i + 1))
    end
end

# 3, 5, 6, 7 and 24 divide neither 256 nor 224; 16 is the shipped default and is
# in for comparison. 24 is large enough to leave a short final stride in both
# bands rather than only in one.
const GROUPS = [1, 3, 5, 6, 7, 16, 24]

# Fluxes are O(100) W/m2, so this is a few parts in 10^6 of the signal. It is a
# ceiling on roundoff, not a fitted number: the measured deviations are
# reported beside it so a reader can see the margin.
const FLUX_TOLERANCE_REL = 1e-5

const WARMUP_STEPS = 3

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]

cs = CoupledSimulation(config_file)

# Radiation fires every dt_rad, so step until the state the solve reads has
# settled and the clouds are real rather than initial.
function warmup!(cs, n)
    for i in 1:n
        @info "warmup step $i / $n"
        step!(cs)
    end
    return nothing
end
warmup!(cs, WARMUP_STEPS)

solver = cs.model_sims.atmos_sim.integrator.p.radiation.rrtmgp_solver
lws = solver.lws
sws = solver.sws
gp = solver.grid_params
as = solver.as
lk = solver.lookups
ms = solver.deep_atmosphere_inverse_scaling
nlay, ncol = RRTMGP.AtmosphericStates.get_dims(as)
FT = eltype(gp)

lws isa RTE.TwoStreamLWRTE ||
    error("expected a TwoStreamLWRTE longwave solver, got $(typeof(lws))")
sws isa RTE.TwoStreamSWRTE ||
    error("expected a TwoStreamSWRTE shortwave solver, got $(typeof(sws))")
isnothing(lws.band_flux) && isnothing(sws.band_flux) ||
    error("band_flux is being kept, so the grouped path is unreachable")

with_lw_groups(lw, g) = RTE.TwoStreamLWRTE(
    lw.context,
    lw.op,
    lw.src,
    lw.bcs,
    lw.fluxb,
    lw.flux,
    lw.band_flux,
    lw.state_cache,
    g,
)

with_sw_groups(sw, g) = RTE.TwoStreamSWRTE(
    sw.context,
    sw.op,
    sw.src,
    sw.bcs,
    sw.fluxb,
    sw.flux,
    sw.band_flux,
    sw.state_cache,
    g,
)

lw_solve!(lw) = RTESolver.solve_lw_both_skies!(
    lw,
    solver.clear_flux_acc_lw,
    as,
    lk.lookup_lw,
    lk.lookup_lw_cld,
    lk.lookup_lw_aero,
    ms,
)

sw_solve!(sw) = RTESolver.solve_sw_both_skies!(
    sw,
    solver.clear_flux_acc_sw,
    as,
    lk.lookup_sw,
    lk.lookup_sw_cld,
    lk.lookup_sw_aero,
    ms,
)

# Snapshots go to the host, because every group count writes into the same flux
# buffers by construction: a reference held on the device would be overwritten
# by the point it is meant to be compared against.
function lw_snapshot()
    cover = as.cloud_state isa RRTMGP.AtmosphericStates.CloudState ?
        as.cloud_state.cld_cover_lw : nothing
    return (
        fluxes = Dict(
            "flux_up" => Array(lws.flux.flux_up),
            "flux_dn" => Array(lws.flux.flux_dn),
            "flux_net" => Array(lws.flux.flux_net),
            "clear_up" => Array(solver.clear_flux_acc_lw.flux_up),
            "clear_dn" => Array(solver.clear_flux_acc_lw.flux_dn),
        ),
        cover = isnothing(cover) ? nothing : Array(cover),
        aod = nothing,
    )
end

function sw_snapshot()
    cover = as.cloud_state isa RRTMGP.AtmosphericStates.CloudState ?
        as.cloud_state.cld_cover_sw : nothing
    aod = isnothing(lk.lookup_sw_aero) ? nothing :
        Dict(
            "aod_sw_ext" => Array(as.aerosol_state.aod_sw_ext),
            "aod_sw_sca" => Array(as.aerosol_state.aod_sw_sca),
        )
    return (
        fluxes = Dict(
            "flux_up" => Array(sws.flux.flux_up),
            "flux_dn" => Array(sws.flux.flux_dn),
            "flux_net" => Array(sws.flux.flux_net),
            "flux_dn_dir" => Array(sws.flux.flux_dn_dir),
            "clear_up" => Array(solver.clear_flux_acc_sw.flux_up),
            "clear_dn_dir" => Array(solver.clear_flux_acc_sw.flux_dn_dir),
        ),
        cover = isnothing(cover) ? nothing : Array(cover),
        aod = aod,
    )
end

# Worst absolute deviation per field, and the scale to read it against, so the
# relative figure is the field's own rather than the largest field's.
function compare_fluxes(ref, got)
    worst = Dict{String, Any}()
    for (k, a) in ref
        b = got[k]
        size(a) == size(b) || error("shape of $k changed between group counts")
        scale = maximum(abs.(Float64.(a)))
        dev = maximum(abs.(Float64.(a) .- Float64.(b)))
        worst[k] = Dict(
            "abs" => dev,
            "scale" => scale,
            "rel" => scale > 0 ? dev / scale : 0.0,
        )
    end
    return worst
end

exact_match(a, b) = isnothing(a) && isnothing(b) ? true :
    isnothing(a) || isnothing(b) ? false : a == b

function exact_dicts_match(a, b)
    (isnothing(a) || isnothing(b)) && return isnothing(a) && isnothing(b)
    return all(a[k] == b[k] for k in keys(a))
end

# Everything that accumulates lives in here. A `for` at top level makes its
# assignments local, so a reference captured in one iteration is invisible in
# the next -- the UndefVarError that has cost this project four reservations.
function check_band(name, n_gpt, make_groups, with_groups, live, solve!, snapshot)
    points = Dict{String, Any}()
    reference = nothing

    for n in GROUPS
        @info "$name ngpt_groups = $n"
        CUDA.reclaim()
        groups = n > 1 ? make_groups(n) : nothing
        rte = with_groups(live, groups)
        solve!(rte)
        CUDA.synchronize()
        shot = snapshot()

        if isnothing(reference)
            reference = shot
        end

        worst = compare_fluxes(reference.fluxes, shot.fluxes)
        worst_rel = maximum(v["rel"] for v in values(worst))
        stride_remainder = n_gpt % n
        points[string(n)] = Dict{String, Any}(
            "ngroups" => n,
            "divides_n_gpt" => stride_remainder == 0,
            "stride_remainder" => stride_remainder,
            "groups_with_short_stride" =>
                stride_remainder == 0 ? 0 : n - stride_remainder,
            "worst_relative_deviation" => worst_rel,
            "within_tolerance" => worst_rel <= FLUX_TOLERANCE_REL,
            "fields" => worst,
            "cover_exact" => exact_match(reference.cover, shot.cover),
            "aod_exact" => exact_dicts_match(reference.aod, shot.aod),
        )

        groups = nothing
        GC.gc()
        CUDA.reclaim()
    end

    # Restore the live solver's own answer: these solves wrote into the shared
    # flux buffers, and the simulation keeps running after this script.
    solve!(live)
    CUDA.synchronize()

    nondiv = [p for p in values(points) if !p["divides_n_gpt"]]
    return Dict{String, Any}(
        "n_gpt" => n_gpt,
        "group_counts" => GROUPS,
        "points" => points,
        "nondivisor_counts_checked" => length(nondiv),
        "all_within_tolerance" =>
            all(p["within_tolerance"] for p in values(points)),
        "all_cover_exact" => all(p["cover_exact"] for p in values(points)),
        "all_aod_exact" => all(p["aod_exact"] for p in values(points)),
        "worst_relative_deviation" =>
            maximum(p["worst_relative_deviation"] for p in values(points)),
        # The property that separates roundoff from a reordering bug: a sum
        # taken in more partials is not systematically further from the serial
        # one, so the deviation must not climb with the group count.
        "deviation_grows_with_groups" => let
            ns = sort(GROUPS)
            devs = [points[string(n)]["worst_relative_deviation"] for n in ns]
            last(devs) > 2 * maximum(devs[2:(end - 1)]; init = 0.0)
        end,
    )
end

n_gpt_lw = length(lk.lookup_lw.band_data.major_gpt2bnd)
n_gpt_sw = length(lk.lookup_sw.band_data.major_gpt2bnd)

lw = check_band(
    "longwave",
    n_gpt_lw,
    n -> RTE.LWGptGroups(gp, n; params = solver.params),
    with_lw_groups,
    lws,
    lw_solve!,
    lw_snapshot,
)
sw = check_band(
    "shortwave",
    n_gpt_sw,
    n -> RTE.SWGptGroups(gp, n),
    with_sw_groups,
    sws,
    sw_solve!,
    sw_snapshot,
)

results = Dict{String, Any}(
    "ncol" => ncol,
    "nlay" => nlay,
    "float_type" => string(FT),
    "config_file" => config_file,
    "warmup_steps" => WARMUP_STEPS,
    "flux_tolerance_rel" => FLUX_TOLERANCE_REL,
    "longwave" => lw,
    "shortwave" => sw,
    "passed" =>
        lw["all_within_tolerance"] &&
        lw["all_cover_exact"] &&
        sw["all_within_tolerance"] &&
        sw["all_cover_exact"] &&
        sw["all_aod_exact"],
    "note" => "Equivalence of the grouped g-point solve against the ungrouped " *
              "one, on the live AMIP state at full resolution. Group counts " *
              "include values that divide neither g-point count, which both " *
              "sweeps miss: 4, 8, 16 and 32 divide 256 and 224 exactly, so " *
              "every group gets an equal stride and the ragged final stride " *
              "never runs. Fluxes get a relative tolerance because the split " *
              "is not bit-for-bit -- a g-point sum taken in N partials is a " *
              "different summation order -- and the figure that matters is " *
              "that the deviation does not grow with the group count, which " *
              "is what separates roundoff from a dropped or double-counted " *
              "g-point. Cloud cover is a count and is asserted exactly. The " *
              "550 nm aerosol optical depth is asserted exactly and shortwave " *
              "only: it is a per-column read-modify-write with one nominated " *
              "writer, so any difference is groups interleaving. This does " *
              "NOT cover ngroups > n_gpt, where a group has no g-points at " *
              "all; that needs more groups than there are g-points and so " *
              "belongs in the unit tests, at a column count where the " *
              "workspace is affordable.",
)

mkpath(dirname(out_path))
open(io -> TOML.print(io, results; sorted = true), out_path, "w")
@info "wrote $out_path" passed = results["passed"] lw_worst =
    lw["worst_relative_deviation"] sw_worst = sw["worst_relative_deviation"]
results["passed"] || error("grouped solve did not match the ungrouped one")
