# Does the shortwave g-point split pay what the longwave one did?
#
# The longwave solve was the first member of a family: kernels at the 255-register
# cap whose grid is one resident wave, where the only lever left is to split the
# kernel so there are more blocks. `results/wave-census.json` names the family.
# The shortwave two-stream solve is its closest member -- the same 96 blocks of
# 256 threads, the same cap, the same independent g-point loop -- so its saving
# is a prediction this sweep tests rather than a new question.
#
# Method is the longwave sweep's, in `measure-gpoint-sweep.jl`: each point
# rebuilds ONLY the shortwave RTE's group workspaces, so two points differ by
# the group count and nothing else, and `ngroups = 1` takes the ungrouped
# specialization so the baseline is a point in the same sweep. That makes this
# an in-block comparison, which is the only way to resolve a few percent here.
#
# Two things the longwave sweep did not have to check. The shortwave carries a
# direct-beam profile, so it is in the deviation. And the 550 nm aerosol optical
# depth is a per-column read-modify-write that only one nominated g-point may
# write; it is asserted equal rather than close, because a race there would show
# up as a difference no tolerance should absorb.

import CUDA
import RRTMGP
import TOML

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

const RTE = RRTMGP.RTE
const RTESolver = RRTMGP.RTESolver

out_path = "results/sw-gpoint-sweep.toml"
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i)
        out_path = ARGS[i + 1]
        deleteat!(ARGS, i:(i + 1))
    end
end

wall_path = "results/sw-gpoint-wallclock.toml"
let i = findfirst(==("--out-wallclock"), ARGS)
    if !isnothing(i)
        wall_path = ARGS[i + 1]
        deleteat!(ARGS, i:(i + 1))
    end
end

const GROUPS = [1, 4, 8, 16, 32]
const REPEATS = 7
const WARMUP_STEPS = 3

# Wall clock, for the SYPD question. Blocks are a whole number of radiation
# periods so every block carries the same radiation work, and 12 of them keeps
# the timed window inside `t_end`.
const RAD_PERIODS_PER_BLOCK = 6
const WALLCLOCK_BLOCKS = 12

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]

# Read the cadences rather than assume them, and do it before the model is
# built: a bad parse should cost nothing.
const UNIT_SECONDS =
    Dict("secs" => 1.0, "mins" => 60.0, "hours" => 3600.0, "days" => 86400.0)

function duration_seconds(path, key)
    text = read(path, String)
    m = match(Regex("^$(key):\\s*\"?([0-9.]+)([a-z]+)\"?", "m"), text)
    isnothing(m) && error("could not find `$key` in $path")
    haskey(UNIT_SECONDS, m[2]) ||
        error("unknown time unit `$(m[2])` for `$key` in $path")
    return parse(Float64, m[1]) * UNIT_SECONDS[m[2]]
end

coupler_root = first(split(config_file, "/config/"))
atmos_config_file =
    let m = match(r"^atmos_config_file:\s*\"([^\"]+)\""m, read(config_file, String))
        isnothing(m) && error("could not find `atmos_config_file` in $config_file")
        joinpath(coupler_root, m[1])
    end

dt_cpl = duration_seconds(config_file, "dt_cpl")
dt_rad = duration_seconds(atmos_config_file, "dt_rad")
t_end = duration_seconds(config_file, "t_end")
rad_period_steps = dt_rad / dt_cpl
isinteger(rad_period_steps) ||
    error("dt_rad $(dt_rad)s is not a whole number of dt_cpl $(dt_cpl)s steps")
rad_period_steps = Int(rad_period_steps)
block_steps = RAD_PERIODS_PER_BLOCK * rad_period_steps
WARMUP_STEPS + WALLCLOCK_BLOCKS * block_steps <= t_end / dt_cpl ||
    error("the timed window runs past t_end")

cs = CoupledSimulation(config_file)

# Radiation fires every dt_rad, so step until it has run and the state the
# solve reads is settled.
for i in 1:WARMUP_STEPS
    @info "warmup step $i / $WARMUP_STEPS"
    step!(cs)
end

# Wall clock first, so the sweep cannot perturb a timed step. These blocks run
# the real model: the live solver, the real diagnostics, the real coupling.
function timed_blocks(cs, nblocks, steps)
    seconds = Float64[]
    for b in 1:nblocks
        CUDA.synchronize()
        t0 = time_ns()
        for _ in 1:steps
            step!(cs)
        end
        CUDA.synchronize()
        push!(seconds, (time_ns() - t0) / 1e9)
        @info "block $b / $nblocks" seconds = last(seconds)
    end
    return seconds
end

block_seconds = timed_blocks(cs, WALLCLOCK_BLOCKS, block_steps)

solver = cs.model_sims.atmos_sim.integrator.p.radiation.rrtmgp_solver
sws = solver.sws
gp = solver.grid_params
as = solver.as
lk = solver.lookups
clear_acc = solver.clear_flux_acc_sw
ms = solver.deep_atmosphere_inverse_scaling
nlay, ncol = RRTMGP.AtmosphericStates.get_dims(as)
FT = eltype(gp)

sws isa RTE.TwoStreamSWRTE ||
    error("expected a TwoStreamSWRTE shortwave solver, got $(typeof(sws))")
isnothing(sws.band_flux) ||
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
    return (median = ts[(length(ts) + 1) ÷ 2], min = first(ts), max = last(ts))
end

# Swap in one group workspace, leaving every other field of the live solver
# alone. The grouped path reads its wide optics, sources and mask out of the
# workspace, so the narrow ones shared here are only ever used by `ngroups = 1`.
function with_groups(sw::RTE.TwoStreamSWRTE, groups)
    return RTE.TwoStreamSWRTE(
        sw.context,
        sw.op,
        sw.src,
        sw.bcs,
        sw.fluxb,
        sw.flux,
        sw.band_flux,
        sw.state_cache,
        groups,
    )
end

sw_solve!(sw) = RTESolver.solve_sw_both_skies!(
    sw,
    clear_acc,
    as,
    lk.lookup_sw,
    lk.lookup_sw_cld,
    lk.lookup_sw_aero,
    ms,
)

# The flux a point produced, on the host, so later points cannot overwrite the
# comparison: every point writes into the same `flux` buffers by construction.
flux_snapshot(sw) = (
    up = Array(sw.flux.flux_up),
    dn = Array(sw.flux.flux_dn),
    net = Array(sw.flux.flux_net),
    dir = Array(sw.flux.flux_dn_dir),
    clear_up = Array(clear_acc.flux_up),
    clear_dir = Array(clear_acc.flux_dn_dir),
)

# The 550 nm optical depth is not a flux and gets no tolerance: it is written by
# one nominated g-point, so a group racing on it is a difference, not roundoff.
aod_snapshot() =
    isnothing(lk.lookup_sw_aero) ? nothing :
    (
        ext = Array(as.aerosol_state.aod_sw_ext),
        sca = Array(as.aerosol_state.aod_sw_sca),
    )

function worst_deviation(a, b)
    worst = 0.0
    for f in (:up, :dn, :net, :dir, :clear_up, :clear_dir)
        x, y = getfield(a, f), getfield(b, f)
        size(x) == size(y) || error("flux shape changed between group counts")
        worst = max(worst, maximum(abs.(Float64.(x) .- Float64.(y))))
    end
    return worst
end

function aod_matches(a, b)
    (isnothing(a) || isnothing(b)) && return true
    return a.ext == b.ext && a.sca == b.sca
end

# Everything that accumulates across the sweep lives in here. A `for` at top
# level makes its assignments local, so a reference captured in one iteration is
# not visible in the next -- the UndefVarError that has cost this project four
# reservations.
function sweep()
    points = Dict{String, Any}()
    reference = nothing
    reference_aod = nothing
    reference_ms = 0.0

    for n in GROUPS
        @info "ngpt_groups = $n"
        CUDA.reclaim()
        free_before = CUDA.available_memory()
        groups = n > 1 ? RTE.SWGptGroups(gp, n) : nothing
        CUDA.synchronize()
        bytes = free_before - CUDA.available_memory()

        sw = with_groups(sws, groups)
        t = timed_ms(() -> sw_solve!(sw))
        shot = flux_snapshot(sw)
        aod = aod_snapshot()

        if isnothing(reference)
            reference = shot
            reference_aod = aod
            reference_ms = t.median
        end

        # The accumulators alone, from their shapes, as a floor under `bytes`:
        # two `(ncol * ngroups, nlev, 4)` arrays. The rest of the widening is
        # the optics, sources, band fluxes and cloud mask.
        acc_bytes = n > 1 ? 2 * (ncol * n) * (nlay + 1) * 4 * sizeof(FT) : 0

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
            "aod_bit_identical" => aod_matches(reference_aod, aod),
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
sufficient =
    minimum(n for n in GROUPS if points[string(n)]["change_pct"] <= best + spread_pct)
all_aod_identical = all(points[string(n)]["aod_bit_identical"] for n in GROUPS)

results = Dict{String, Any}(
    "ncol" => ncol,
    "nlay" => nlay,
    "n_gpt_sw" => length(lk.lookup_sw.band_data.major_gpt2bnd),
    "float_type" => string(FT),
    "repeats" => REPEATS,
    "warmup_steps" => WARMUP_STEPS,
    "config_file" => config_file,
    "group_counts" => GROUPS,
    "repeat_spread_pct" => spread_pct,
    "best_change_pct" => best,
    "sufficient_ngroups" => sufficient,
    "sufficient_is_sweep_maximum" => sufficient == maximum(GROUPS),
    "default_ngpt_groups" => RTE.DEFAULT_NGPT_GROUPS,
    "default_is_sufficient_only" => RTE.DEFAULT_NGPT_GROUPS == sufficient,
    "aod_bit_identical_at_every_count" => all_aod_identical,
    "points" => points,
    "note" => "solve_ms times the whole shortwave both-skies solve, so it " *
              "charges the grouped path for its reduction, its hoisted " *
              "aerosol mask and its state refresh -- it is the net figure, " *
              "not the kernel's. Every point shares one simulation's state " *
              "and buffers and differs only in group count, so these are " *
              "in-block comparisons and not a difference of runs. " *
              "worst_flux_deviation is against ngroups = 1, the ungrouped " *
              "specialization, and covers the direct beam as well as up, " *
              "down and net in both skies: the split is not bit-for-bit, " *
              "since a g-point sum taken in N partials is a different " *
              "summation order. aod_bit_identical is equality rather than a " *
              "tolerance because the 550 nm optical depth is a per-column " *
              "read-modify-write with one nominated writer, so any " *
              "difference there is groups interleaving and not roundoff. " *
              "sufficient_is_sweep_maximum says whether the sweep found a " *
              "knee or only moved the bound to its own endpoint.",
)

mkpath(dirname(out_path))
open(io -> TOML.print(io, results; sorted = true), out_path, "w")
@info "wrote $out_path" sufficient best spread_pct all_aod_identical

# SYPD, by the coupler's own definition: simulated seconds per wall second over
# 365.25 days. The grouped figure is measured end to end on real steps. The
# ungrouped one adds the measured in-block saving back onto that measured
# block, which is the single arithmetic step here -- there is no second run to
# difference, and the solve is synchronous on the step's critical path, so a
# millisecond saved in it is a millisecond of wall clock.
function wallclock_results(seconds, points)
    sorted = sort(seconds)
    n = length(sorted)
    mid = sorted[(n + 1) ÷ 2]
    q1 = sorted[max(1, (n + 3) ÷ 4)]
    q3 = sorted[min(n, (3 * n + 1) ÷ 4)]
    mean = sum(sorted) / n
    sd = sqrt(sum((x - mean)^2 for x in sorted) / (n - 1))
    sim_seconds = block_steps * dt_cpl
    events = RAD_PERIODS_PER_BLOCK
    sypd(w) = sim_seconds / w / 365.25

    arms = Dict{String, Any}()
    for g in GROUPS
        g == 1 && continue
        per_event = points["1"]["solve_ms"] - points[string(g)]["solve_ms"]
        saved = events * per_event / 1e3
        arms[string(g)] = Dict{String, Any}(
            "ngroups" => g,
            "solve_saved_per_event_ms" => per_event,
            "saved_per_block_s" => saved,
            "sypd_grouped" => sypd(mid),
            "sypd_ungrouped" => sypd(mid + saved),
            "sypd_gain_pct" => 100.0 * saved / mid,
        )
    end

    default_gain = arms[string(RTE.DEFAULT_NGPT_GROUPS)]["sypd_gain_pct"]
    iqr_pct = 100.0 * (q3 - q1) / mid
    return Dict{String, Any}(
        "block_steps" => block_steps,
        "blocks" => n,
        "rad_period_steps" => rad_period_steps,
        "radiation_events_per_block" => events,
        "write_period_steps" => Int(86400 ÷ dt_cpl),
        "dt_cpl_s" => dt_cpl,
        # In time order, so which block carried the diagnostics write is a fact
        # rather than an inference from the cadence. The longwave run sorted
        # these and lost it.
        "block_seconds_in_order" => seconds,
        "block_seconds" => sorted,
        "block_median_s" => mid,
        "block_min_s" => first(sorted),
        "block_max_s" => last(sorted),
        "block_spread_pct" => 100.0 * (last(sorted) - first(sorted)) / mid,
        "block_iqr_pct" => iqr_pct,
        "block_rel_se_pct" => 100.0 * (sd / sqrt(n)) / mid,
        "step_ms" => 1e3 * mid / block_steps,
        "default_ngpt_groups" => RTE.DEFAULT_NGPT_GROUPS,
        "default_sypd_gain_pct" => default_gain,
        "arms" => arms,
        "note" => "sypd_grouped is measured: a block of real coupler steps " *
                  "with the live solver, timed with the wall clock the " *
                  "coupler's own SYPD is defined on. Note what the live " *
                  "solver now is: the longwave split is already in it, so " *
                  "these blocks are faster than the longwave sweep's and the " *
                  "two runs' block times are not comparable. Each sweep is " *
                  "in-block, so that does not affect either gain. " *
                  "sypd_ungrouped adds the in-block solve saving back rather " *
                  "than differencing a second run, because the AMIP pair " *
                  "cannot resolve a sub-percent effect: one arm wandered " *
                  "7.73% between two runs of identical code. It is a " *
                  "steady-state figure and is NOT comparable to the " *
                  "pipeline's end-to-end SYPD, which includes first-call " *
                  "compilation. Blocks are a whole number of radiation " *
                  "periods so each carries the same radiation work; the " *
                  "daily diagnostics write falls in one block in every " *
                  "write_period_steps / block_steps, which is why the median " *
                  "is quoted and the full spread reported beside it. " *
                  "gain_below_block_iqr is deliberately not reported here: " *
                  "it compares a ratio against a difference-of-blocks spread, " *
                  "which reads a resolved gain as unresolved.",
    )
end

wall = wallclock_results(block_seconds, points)
mkpath(dirname(wall_path))
open(io -> TOML.print(io, wall; sorted = true), wall_path, "w")
@info "wrote $wall_path" step_ms = wall["step_ms"] gain =
    wall["default_sypd_gain_pct"]
