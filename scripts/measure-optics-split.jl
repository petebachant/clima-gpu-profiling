# Does splitting the optics out of the grouped radiation solve pay?
#
# The grouped two-stream kernels need 255 registers, which pins them to one
# block per SM: 8 of the A100's 64 warps, and nothing on the card is saturated
# at that occupancy, so the solve waits on latency it has no warps to hide.
# `results/experiments/radiation-phase-split/phases.toml` priced the
# precondition: compiled on its own the gas and aerosol optics phase -- 99% of
# the solve's time -- needs 156 registers in the longwave and 128 in the
# shortwave with 32 bytes of stack, where forcing the fused kernel to the same
# budget spilled 672 and 712 bytes. So the phase fits a higher occupancy and
# the fused kernel does not.
#
# This measures what the split is worth, which the register count does not say.
# `op` and `src` are device arrays the fused kernel already addresses by wide
# column, so the two halves exchange nothing across the kernel boundary that
# was not already going to memory -- but the boundary is still a second launch
# per round and a second pass over the state each half reads.
#
# Three arms, all on the live simulation's own state, lookups and buffers:
#
#   ungrouped  no group workspace at all, the upstream serial g-point loop
#   fused      the grouped solve as one kernel per round (what the PR ships)
#   split      the grouped solve as optics then sweeps, two kernels per round
#
# `fused` and `split` share one workspace and differ only in the `split_optics`
# flag, so the comparison is the same buffers twice and not two allocations.
# The split is bit-for-bit with the fused path and this asserts it: both halves
# address the same wide column, the gas and aerosol optics never read the cloud
# mask the sweeps draw, and no arithmetic moved.
#
# It also reports what the occupancy actually is, three ways: the register count
# ptxas chose, the CUDA driver's own
# `cuOccupancyMaxActiveBlocksPerMultiprocessor`, and the allocator model
# ClimaCore's `max_active_blocks` implements. The model predicted this split was
# worth building; the driver is the authority on whether it was right.

import CUDA
import RRTMGP
import TOML

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

const RTE = RRTMGP.RTE
const RTESolver = RRTMGP.RTESolver

# `let` is a hard scope, so assigning `out_path` inside one makes a local and
# leaves the global at its default: the flag parses and is then discarded.
function arg_value(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) && return default
    value = ARGS[i + 1]
    deleteat!(ARGS, i:(i + 1))
    return value
end

out_path = arg_value("--out", "results/optics-split.toml")

const REPEATS = 7
const WARMUP_STEPS = 3

# Wall clock, for the SYPD question. Blocks are a whole number of radiation
# periods so every block carries the same radiation work, and 12 of them keeps
# the timed window inside `t_end`. Same shape as the g-point sweeps, so the
# figures are comparable with theirs.
const RAD_PERIODS_PER_BLOCK = 6
const WALLCLOCK_BLOCKS = 12

# ---------------------------------------------------------------------------
# The occupancy model, as ClimaCore's `max_active_blocks` implements it. It is
# not a division of the register file: registers are allocated per warp in
# 256-register units, a block's warps round up to the scheduler count, and the
# file is divided per scheduler rather than per multiprocessor. Dividing
# instead puts the 128-thread step at 170 registers where the real one is 168.
# ---------------------------------------------------------------------------
const A100_REGS_PER_SM = 65536
const A100_THREADS_PER_SM = 2048
const A100_MAX_BLOCKS_PER_SM = 32
const A100_REGS_PER_ALLOCATION = 256
const A100_MAX_REGS_PER_BLOCK = 65536
const A100_SCHEDULERS_PER_SM = 4
const THREADS_PER_WARP = 32

round_up(n, divisor) = cld(n, divisor) * divisor

function model_warps_per_sm(regs, block)
    warps_per_block = cld(block, THREADS_PER_WARP)
    max_warps = A100_THREADS_PER_SM ÷ THREADS_PER_WARP
    if regs > 0
        allocated_warps = round_up(warps_per_block, A100_SCHEDULERS_PER_SM)
        allocated_regs_per_warp =
            round_up(regs * THREADS_PER_WARP, A100_REGS_PER_ALLOCATION)
        allocated_warps * allocated_regs_per_warp > A100_MAX_REGS_PER_BLOCK &&
            return 0
        per_scheduler =
            (A100_REGS_PER_SM ÷ A100_SCHEDULERS_PER_SM) ÷ allocated_regs_per_warp
        max_warps = min(max_warps, per_scheduler * A100_SCHEDULERS_PER_SM)
    end
    blocks = max_warps ÷ warps_per_block
    warps_per_block == 2 && (blocks = 4 * (blocks ÷ 4))
    return min(blocks, A100_MAX_BLOCKS_PER_SM) * warps_per_block
end

# What the driver says, which is the number the model is being checked against.
function driver_warps_per_sm(k, block)
    blocks = CUDA.active_blocks(k.fun, block)
    return blocks * cld(block, THREADS_PER_WARP)
end

const BLOCKS_MEASURED = [128, 256]

function kernel_record(name, build)
    k = build()
    regs = CUDA.registers(k)
    rec = Dict{String, Any}(
        "registers" => regs,
        "local_bytes" => CUDA.memory(k).local,
        "max_threads" => CUDA.maxthreads(k),
    )
    for b in BLOCKS_MEASURED
        driver = b <= CUDA.maxthreads(k) ? driver_warps_per_sm(k, b) : 0
        model = model_warps_per_sm(regs, b)
        rec["driver_warps_per_sm_b$b"] = driver
        rec["model_warps_per_sm_b$b"] = model
        rec["model_agrees_b$b"] = driver == model
    end
    println(
        "  $name: $(regs) regs, $(rec["local_bytes"]) B local, " *
        "warps/SM driver/model " *
        join(
            [
                "b$b $(rec["driver_warps_per_sm_b$b"])/$(rec["model_warps_per_sm_b$b"])"
                for b in BLOCKS_MEASURED
            ],
            " ",
        ),
    )
    return rec
end

# ---------------------------------------------------------------------------
# The simulation.
# ---------------------------------------------------------------------------
config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]

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

function warm_up(cs, n)
    for i in 1:n
        @info "warmup step $i / $n"
        step!(cs)
    end
    return nothing
end
warm_up(cs, WARMUP_STEPS)

# The blocks below run the real model with whatever the solver was built with,
# which is the split path.
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

solver = cs.model_sims.atmos_sim.integrator.p.radiation.rrtmgp_solver
lws = solver.lws
sws = solver.sws
gp = solver.grid_params
as = solver.as
lk = solver.lookups
ms = solver.deep_atmosphere_inverse_scaling
clear_lw = solver.clear_flux_acc_lw
clear_sw = solver.clear_flux_acc_sw
nlay, ncol = RRTMGP.AtmosphericStates.get_dims(as)
nlev = nlay + 1
FT = eltype(gp)

lws isa RTE.TwoStreamLWRTE ||
    error("expected a TwoStreamLWRTE longwave solver, got $(typeof(lws))")
sws isa RTE.TwoStreamSWRTE ||
    error("expected a TwoStreamSWRTE shortwave solver, got $(typeof(sws))")
isnothing(lws.band_flux) ||
    error("band_flux is being kept, so the grouped path is unreachable")
isnothing(sws.band_flux) ||
    error("band_flux is being kept, so the grouped path is unreachable")
isnothing(lws.gpt_groups) &&
    error("the longwave solver has no group workspace, so there is nothing to split")
isnothing(sws.gpt_groups) &&
    error("the shortwave solver has no group workspace, so there is nothing to split")

const NGROUPS = RTE.n_gpt_groups(lws.gpt_groups)
RTE.n_gpt_groups(sws.gpt_groups) == NGROUPS ||
    error("the two bands were built with different group counts")

# One workspace, two arms: everything but the flag is the same object, so the
# arms cannot differ in what they allocated or in what state they read.
function with_split(g::G, split::Bool) where {G}
    return G(
        (
            f === :split_optics ? split : getfield(g, f) for f in fieldnames(G)
        )...,
    )
end

with_groups(lw::RTE.TwoStreamLWRTE, groups) = RTE.TwoStreamLWRTE(
    lw.context, lw.op, lw.src, lw.bcs, lw.fluxb, lw.flux, lw.band_flux,
    lw.state_cache, groups,
)
with_groups(sw::RTE.TwoStreamSWRTE, groups) = RTE.TwoStreamSWRTE(
    sw.context, sw.op, sw.src, sw.bcs, sw.fluxb, sw.flux, sw.band_flux,
    sw.state_cache, groups,
)

lw_solve!(lw) = RTESolver.solve_lw_both_skies!(
    lw, clear_lw, as, lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero, ms,
)
sw_solve!(sw) = RTESolver.solve_sw_both_skies!(
    sw, clear_sw, as, lk.lookup_sw, lk.lookup_sw_cld, lk.lookup_sw_aero, ms,
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

# Bit-for-bit, not within a tolerance: the split moves no arithmetic, so a
# difference of one bit is a bug and not roundoff. Compared as integers, since
# `==` on arrays of floats would call two NaNs unequal.
bits(a) = reinterpret(UInt64, Float64.(a))
# A diagnostic the solver was built without is nothing, and comparing nothing
# to nothing is a pass rather than an error.
bits_or_empty(a) = isnothing(a) ? UInt64[] : bits(Array(a))

lw_snapshot(lw) = Dict(
    "flux_up" => bits(Array(lw.flux.flux_up)),
    "flux_dn" => bits(Array(lw.flux.flux_dn)),
    "flux_net" => bits(Array(lw.flux.flux_net)),
    "clear_up" => bits(Array(clear_lw.flux_up)),
    "clear_dn" => bits(Array(clear_lw.flux_dn)),
    "cld_cover" => bits_or_empty(as.cloud_state.cld_cover_lw),
)

sw_snapshot(sw) = Dict(
    "flux_up" => bits(Array(sw.flux.flux_up)),
    "flux_dn" => bits(Array(sw.flux.flux_dn)),
    "flux_net" => bits(Array(sw.flux.flux_net)),
    "flux_dn_dir" => bits(Array(sw.flux.flux_dn_dir)),
    "clear_up" => bits(Array(clear_sw.flux_up)),
    "clear_dn_dir" => bits(Array(clear_sw.flux_dn_dir)),
    "cld_cover" => bits_or_empty(as.cloud_state.cld_cover_sw),
    "aod_ext" => isnothing(lk.lookup_sw_aero) ? UInt64[] :
                 bits_or_empty(as.aerosol_state.aod_sw_ext),
)

function differing_values(a, b)
    n = 0
    for key in keys(a)
        x, y = a[key], b[key]
        size(x) == size(y) || error("`$key` changed shape between arms")
        n += count(x .!= y)
    end
    return n
end

total_values(a) = sum(length(v) for v in values(a))

function worst_relative(a, b)
    worst = 0.0
    for key in keys(a)
        x = reinterpret(Float64, a[key])
        y = reinterpret(Float64, b[key])
        for i in eachindex(x)
            d = abs(x[i] - y[i])
            d == 0 && continue
            scale = max(abs(x[i]), abs(y[i]))
            worst = max(worst, scale == 0 ? d : d / scale)
        end
    end
    return worst
end

# ---------------------------------------------------------------------------
# The three arms, per band.
# ---------------------------------------------------------------------------
function band_arms(solve!, with, rte, groups, snapshot)
    arms = Dict{String, Any}()
    fused = with(rte, with_split(groups, false))
    split = with(rte, with_split(groups, true))
    ungrouped = with(rte, nothing)

    t_fused = timed_ms(() -> solve!(fused))
    shot_fused = snapshot(fused)
    t_split = timed_ms(() -> solve!(split))
    shot_split = snapshot(split)
    t_ungrouped = timed_ms(() -> solve!(ungrouped))

    for (name, t) in
        (("ungrouped", t_ungrouped), ("fused", t_fused), ("split", t_split))
        arms[name] = Dict{String, Any}(
            "solve_ms" => t.median,
            "solve_ms_min" => t.min,
            "solve_ms_max" => t.max,
        )
    end
    arms["repeat_spread_pct"] =
        100.0 * (t_fused.max - t_fused.min) / t_fused.median
    arms["split_vs_fused_pct"] =
        100.0 * (t_split.median - t_fused.median) / t_fused.median
    arms["split_vs_ungrouped_pct"] =
        100.0 * (t_split.median - t_ungrouped.median) / t_ungrouped.median
    arms["fused_vs_ungrouped_pct"] =
        100.0 * (t_fused.median - t_ungrouped.median) / t_ungrouped.median
    arms["split_saved_per_event_ms"] = t_fused.median - t_split.median
    arms["values_compared"] = total_values(shot_fused)
    arms["values_differing"] = differing_values(shot_fused, shot_split)
    arms["bit_for_bit"] = arms["values_differing"] == 0
    arms["worst_relative_difference"] = worst_relative(shot_fused, shot_split)
    return arms
end

# Correctness before timing, and before the blocks: the two grouped arms must
# agree exactly, and if they do not there is nothing worth timing. This costs
# four solves against eleven minutes of blocks, and it leaves the live buffers
# holding what the split arm would have written anyway -- the arms are
# bit-identical, so the next coupler step reads the state it would have read.
function assert_bit_for_bit(solve!, with, rte, groups, snapshot, band)
    split = with(rte, with_split(groups, true))
    fused = with(rte, with_split(groups, false))
    solve!(split)
    CUDA.synchronize()
    a = snapshot(split)
    solve!(fused)
    CUDA.synchronize()
    b = snapshot(fused)
    n = differing_values(a, b)
    n == 0 || error(
        "$band: the split and fused grouped solves differ in $n of " *
        "$(total_values(a)) values, so the split is not the same computation",
    )
    println("  $band: $(total_values(a)) values compared, none differ")
    return nothing
end

println("\n=== bit-for-bit gate ===")
assert_bit_for_bit(
    lw_solve!, with_groups, lws, lws.gpt_groups, lw_snapshot, "longwave",
)
assert_bit_for_bit(
    sw_solve!, with_groups, sws, sws.gpt_groups, sw_snapshot, "shortwave",
)

block_seconds = timed_blocks(cs, WALLCLOCK_BLOCKS, block_steps)

println("\n=== longwave arms ===")
lw_arms = band_arms(lw_solve!, with_groups, lws, lws.gpt_groups, lw_snapshot)
println("  split vs fused: $(round(lw_arms["split_vs_fused_pct"], digits = 2))%")

println("\n=== shortwave arms ===")
sw_arms = band_arms(sw_solve!, with_groups, sws, sws.gpt_groups, sw_snapshot)
println("  split vs fused: $(round(sw_arms["split_vs_fused_pct"], digits = 2))%")

# ---------------------------------------------------------------------------
# What ptxas and the driver say about each kernel the arms launch.
# ---------------------------------------------------------------------------
ext = Base.get_extension(RRTMGP, :RRTMGPCUDAExt)
isnothing(ext) && error("RRTMGPCUDAExt is not loaded, so no kernel can be built")

lwg = lws.gpt_groups
swg = sws.gpt_groups
n_gpt_lw = length(lk.lookup_lw.band_data.major_gpt2bnd)
n_gpt_sw = length(lk.lookup_sw.band_data.major_gpt2bnd)

function kernel_records()
    recs = Dict{String, Any}()
    println("\n=== kernels ===")
    recs["lw_fused"] = kernel_record(
        "lw_fused",
        () -> (CUDA.@cuda launch = false always_inline = true ext.rte_lw_2stream_solve_grouped_CUDA!(
            lwg.fluxb, lwg.acc, lwg.acc_clear, lwg.src, lws.bcs, lwg.op,
            lwg.mask, lwg.cloudy, nlay, ncol, as, lws.state_cache,
            lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero, NGROUPS,
        )),
    )
    recs["lw_optics"] = kernel_record(
        "lw_optics",
        () -> (CUDA.@cuda launch = false always_inline = true ext._lw_group_optics_CUDA!(
            lwg.op, lwg.src, as, lws.state_cache, lk.lookup_lw,
            lk.lookup_lw_aero, ncol, NGROUPS, n_gpt_lw, 1,
        )),
    )
    recs["lw_sweeps"] = kernel_record(
        "lw_sweeps",
        () -> (CUDA.@cuda launch = false always_inline = true ext._lw_group_sweeps_CUDA!(
            lwg.fluxb, lwg.acc, lwg.acc_clear, lwg.src, lws.bcs, lwg.op,
            lwg.mask, lwg.cloudy, nlay, ncol, as, lk.lookup_lw,
            lk.lookup_lw_cld, NGROUPS, n_gpt_lw, 1,
        )),
    )
    recs["sw_fused"] = kernel_record(
        "sw_fused",
        () -> (CUDA.@cuda launch = false always_inline = true ext.rte_sw_2stream_solve_grouped_CUDA!(
            swg.fluxb, swg.acc, swg.acc_clear, swg.src, sws.bcs, swg.op,
            swg.mask, swg.cloudy, nlay, ncol, as, sws.state_cache,
            lk.lookup_sw, lk.lookup_sw_cld, lk.lookup_sw_aero, NGROUPS,
        )),
    )
    recs["sw_optics"] = kernel_record(
        "sw_optics",
        () -> (CUDA.@cuda launch = false always_inline = true ext._sw_group_optics_CUDA!(
            swg.op, as, sws.state_cache, lk.lookup_sw, lk.lookup_sw_aero,
            ncol, NGROUPS, n_gpt_sw, 1,
        )),
    )
    recs["sw_sweeps"] = kernel_record(
        "sw_sweeps",
        () -> (CUDA.@cuda launch = false always_inline = true ext._sw_group_sweeps_CUDA!(
            swg.fluxb, swg.acc, swg.acc_clear, swg.src, sws.bcs, swg.op,
            swg.mask, swg.cloudy, nlay, ncol, as, lk.lookup_sw,
            lk.lookup_sw_cld, NGROUPS, n_gpt_sw, 1,
        )),
    )
    return recs
end

kernels = kernel_records()

# The block size each kernel is actually launched at, so the warps a reader
# compares are the ones the solve runs and not the best of a pair.
const LAUNCH_BLOCK = Dict(
    "lw_fused" => 256, "lw_optics" => ext.OPTICS_BLOCK_THREADS,
    "lw_sweeps" => 256, "sw_fused" => 256,
    "sw_optics" => ext.OPTICS_BLOCK_THREADS, "sw_sweeps" => 256,
)

function occupancy_summary(kernels)
    out = Dict{String, Any}()
    for (name, rec) in kernels
        b = LAUNCH_BLOCK[name]
        out[name] = Dict{String, Any}(
            "launch_block" => b,
            "registers" => rec["registers"],
            "local_bytes" => rec["local_bytes"],
            "driver_warps_per_sm" => rec["driver_warps_per_sm_b$b"],
            "model_warps_per_sm" => rec["model_warps_per_sm_b$b"],
            "model_agrees" => rec["model_agrees_b$b"],
        )
    end
    return out
end

at_launch = occupancy_summary(kernels)
model_checks = [
    kernels[n]["model_agrees_b$b"] for n in keys(kernels) for
    b in BLOCKS_MEASURED
]

# ---------------------------------------------------------------------------
# SYPD, by the coupler's own definition, as the g-point sweeps report it: the
# blocks above are measured with the split solver live, and the other two arms
# add their in-block difference back onto that measured block. The solve is
# synchronous on the step's critical path, so a millisecond of it is a
# millisecond of wall clock.
# ---------------------------------------------------------------------------
function wallclock_results(seconds)
    sorted = sort(seconds)
    n = length(sorted)
    mid = sorted[(n + 1) ÷ 2]
    q1 = sorted[max(1, (n + 3) ÷ 4)]
    q3 = sorted[min(n, (3 * n + 1) ÷ 4)]
    sim_seconds = block_steps * dt_cpl
    events = RAD_PERIODS_PER_BLOCK
    sypd(w) = sim_seconds / w / 365.25

    per_event(arm) =
        (lw_arms[arm]["solve_ms"] - lw_arms["split"]["solve_ms"]) +
        (sw_arms[arm]["solve_ms"] - sw_arms["split"]["solve_ms"])

    arms = Dict{String, Any}()
    for arm in ("fused", "ungrouped")
        extra = events * per_event(arm) / 1e3
        arms[arm] = Dict{String, Any}(
            "solve_cost_per_event_ms" => per_event(arm),
            "cost_per_block_s" => extra,
            "sypd" => sypd(mid + extra),
            "split_gain_pct" => 100.0 * extra / mid,
        )
    end

    iqr_pct = 100.0 * (q3 - q1) / mid
    return Dict{String, Any}(
        "block_steps" => block_steps,
        "blocks" => n,
        "rad_period_steps" => rad_period_steps,
        "radiation_events_per_block" => events,
        "dt_cpl_s" => dt_cpl,
        "block_seconds" => sorted,
        "block_median_s" => mid,
        "block_iqr_pct" => iqr_pct,
        "step_ms" => 1e3 * mid / block_steps,
        "sypd_split" => sypd(mid),
        "arms" => arms,
        "gain_over_fused_below_block_iqr" =>
            arms["fused"]["split_gain_pct"] < iqr_pct,
        "note" => "sypd_split is measured: a block of real coupler steps " *
                  "with the live split solver, timed with the wall clock the " *
                  "coupler's own SYPD is defined on. The other arms add " *
                  "their measured in-block solve difference back onto that " *
                  "same measured block rather than differencing a second " *
                  "run, because an AMIP pair cannot resolve a sub-percent " *
                  "effect here: one arm wandered 7.73% between two runs of " *
                  "identical code. These are steady-state figures and are " *
                  "NOT comparable to the pipeline's end-to-end SYPD, which " *
                  "includes first-call compilation.",
    )
end

wall = wallclock_results(block_seconds)

results = Dict{String, Any}(
    "ncol" => ncol,
    "nlay" => nlay,
    "n_gpt_lw" => n_gpt_lw,
    "n_gpt_sw" => n_gpt_sw,
    "ngroups" => NGROUPS,
    "rounds_lw" => cld(n_gpt_lw, NGROUPS),
    "rounds_sw" => cld(n_gpt_sw, NGROUPS),
    "optics_block_threads" => ext.OPTICS_BLOCK_THREADS,
    "float_type" => string(FT),
    "repeats" => REPEATS,
    "warmup_steps" => WARMUP_STEPS,
    "config_file" => config_file,
    "longwave" => lw_arms,
    "shortwave" => sw_arms,
    "kernels" => kernels,
    "at_launch" => at_launch,
    "wallclock" => wall,
    "split_wins_lw" => lw_arms["split_vs_fused_pct"] < 0,
    "split_wins_sw" => sw_arms["split_vs_fused_pct"] < 0,
    "split_is_bit_for_bit" =>
        lw_arms["bit_for_bit"] && sw_arms["bit_for_bit"],
    "occupancy_model_agrees_everywhere" => all(model_checks),
    "occupancy_model_checks" => length(model_checks),
    "note" => "Three arms on one simulation's state and buffers, so these " *
              "are in-block comparisons and not a difference of runs. " *
              "`fused` and `split` share one group workspace and differ only " *
              "in its split_optics flag. bit_for_bit counts values that " *
              "differ when the two arms' fluxes, cloud cover and 550 nm " *
              "optical depth are compared as integers: the split moves no " *
              "arithmetic, so anything but zero is a bug rather than " *
              "roundoff. driver_warps_per_sm is the CUDA driver's own " *
              "cuOccupancyMaxActiveBlocksPerMultiprocessor; " *
              "model_warps_per_sm is the allocator model ClimaCore's " *
              "max_active_blocks implements, which is what predicted this " *
              "split was worth building. Where they disagree the driver is " *
              "right. solve_ms times the whole both-skies solve, so the " *
              "grouped arms are charged for their reduction and their state " *
              "refresh, and the split arm for its extra launch per round.",
)

mkpath(dirname(out_path))
open(io -> TOML.print(io, results; sorted = true), out_path, "w")
@info(
    "wrote $out_path",
    lw_pct = lw_arms["split_vs_fused_pct"],
    sw_pct = sw_arms["split_vs_fused_pct"],
    bit_for_bit = results["split_is_bit_for_bit"],
    model_agrees = results["occupancy_model_agrees_everywhere"],
)
