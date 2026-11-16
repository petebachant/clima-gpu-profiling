# Is there a split boundary in the radiation solve that fits under the occupancy step?
#
# The radiation kernels are latency-bound, not compute- or bandwidth-bound: ncu
# puts them at 19.89% SM throughput and 9.00% DRAM throughput with 0.25 eligible
# warps per scheduler, so ~87% of the time both resident warps are stalled with
# nothing to switch to (section 4u). The cause is registers: 255 per thread pins
# one block per SM, which is 8 warps.
#
# At 256 threads per block the next occupancy step needs 65536 / (regs * 256) >= 2,
# so **registers must come down to 128**. There is no intermediate step. Forcing
# that budget has been tried twice on these kernels and lost twice -- 2 blocks/SM
# at 128 registers measured +3.93% and 128-thread blocks at 168 registers +2.91%
# -- because forcing it spills, and a spill adds latency to a dependency chain
# that has nothing to hide it behind.
#
# Splitting the kernel is the other route: fewer live values per piece rather
# than a forced budget. Whether it can work is a question about where the live
# state actually sits, and this script answers it before anything is built. It
# compiles each phase a split would produce AS ITS OWN KERNEL, with the real
# argument types from the live solver, and reports what ptxas gives it.
#
# The precedent that makes this worth measuring first: splitting the microphysics
# quadrature into per-point kernels was dead because each split kernel still
# contained one full CloudMicrophysics evaluation at ~246 registers, so every one
# of them ran at the same 8 warps/SM. If the radiation optics phase alone comes
# back near 255, the radiation split dies the same way and for the same reason.
#
# Note what this is NOT: a probe of an isolated synthetic body. Section 4ab
# records that measuring a layer rather than the kernel reversed a conclusion
# once -- an isolated body at 206 registers had room for a saving to show, while
# the real kernel was clipped at the cap and the same saving appeared only as 24
# bytes less spill. Every body here is called with the live solver's own
# arguments, so the types and the inlining are the model's.

import CUDA
import Printf
import RRTMGP
import TOML

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

const RTE = RRTMGP.RTE
const RTESolver = RRTMGP.RTESolver
const Optics = RRTMGP.Optics

# `let` is a hard scope, so assigning `out_path` inside one makes a local and
# leaves the global at its default: the flag parses, is deleted from ARGS by
# the mutation, and is then discarded, so nothing downstream complains.
function arg_value(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) && return default
    value = ARGS[i + 1]
    deleteat!(ARGS, i:(i + 1))
    return value
end

out_path = arg_value("--out", "results/radiation-phase-registers.toml")

const WARMUP_STEPS = 1
const A100_REGS_PER_SM = 65536
const A100_THREADS_PER_SM = 2048
const A100_MAX_BLOCKS_PER_SM = 32
const BLOCK = 256
const REGISTER_CAP = 255

# Warps per SM this register count buys at the solve's own block size, which is
# the only figure that decides whether a split changed anything.
#
# This follows ClimaCore's `max_active_blocks`, itself a port of NVIDIA's
# `cudaOccMaxActiveBlocksPerMultiprocessor`, because the allocation
# granularities move the thresholds: registers are allocated per warp in
# 256-register units, a block's warps round up to the scheduler count, and the
# register file is divided per scheduler rather than per multiprocessor.
# Dividing 65536 by regs*block instead puts the 128-thread step at 170
# registers; the granularity-correct answer is 168, which is the figure
# `launch-bounds-vs-maxregs` already recorded.
const A100_REGS_PER_ALLOCATION = 256
const A100_MAX_REGS_PER_BLOCK = 65536
const A100_SCHEDULERS_PER_SM = 4
const THREADS_PER_WARP = 32

round_up(n, divisor) = cld(n, divisor) * divisor

function warps_per_sm(regs, block = BLOCK)
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
    # Blocks of two warps are dealt unevenly to four schedulers unless each
    # gets the same number.
    warps_per_block == 2 && (blocks = 4 * (blocks ÷ 4))
    return min(blocks, A100_MAX_BLOCKS_PER_SM) * warps_per_block
end

# The largest budget that still reaches `want` warps, read off the model rather
# than divided out of the register file.
function budget_for_warps(want, block)
    r = findlast(r -> warps_per_sm(r, block) >= want, 1:REGISTER_CAP)
    return isnothing(r) ? 0 : r
end

# The register count a piece must reach for the next occupancy step.
const TARGET_REGS = budget_for_warps(16, BLOCK)

# The same question at 128 threads per block, where the ladder is finer: three
# blocks fit at 170 registers, so a phase that cannot reach 128 may still gain
# a step. At the cap both block sizes give 8 warps, so this costs nothing to
# ask and is the nearer bar.
const ALT_BLOCK = 128
const ALT_TARGET_REGS = budget_for_warps(12, ALT_BLOCK)
const BASE_WARPS = warps_per_sm(REGISTER_CAP, BLOCK)
const ALT_BASE_WARPS = warps_per_sm(REGISTER_CAP, ALT_BLOCK)

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)

function warmup!(cs, n)
    for i in 1:n
        @info "warmup step $i / $n"
        step!(cs)
    end
    return nothing
end
warmup!(cs, WARMUP_STEPS)

solver = cs.model_sims.atmos_sim.integrator.p.radiation.rrtmgp_solver
lws, sws = solver.lws, solver.sws
as, lk = solver.as, solver.lookups
nlay, ncol = RRTMGP.AtmosphericStates.get_dims(as)
nlev = nlay + 1
cloud_state = as.cloud_state

results = Dict{String, Any}()

function record!(label, build)
    local k
    try
        k = build()
    catch err
        msg = sprint(showerror, err)
        results[label] = Dict{String, Any}("error" => msg)
        # Print the whole thing: an InvalidIRError says what it could not
        # compile several lines in, and truncating it to a summary line threw
        # away the only diagnostic the run produced.
        println(rpad(label, 34), " FAILED")
        println(msg)
        return nothing
    end
    r = CUDA.registers(k)
    m = CUDA.memory(k)
    w = warps_per_sm(r)
    results[label] = Dict{String, Any}(
        "registers" => r,
        "local_bytes" => m.local,
        "warps_per_sm" => w,
        "at_cap" => r >= REGISTER_CAP,
        "reaches_target" => r <= TARGET_REGS && m.local == 0,
        "warps_per_sm_b128" => warps_per_sm(r, ALT_BLOCK),
        # A step is gained only without spill: forcing the budget with one is
        # what measured +3.93% and +2.91% on these kernels.
        "gains_step" => warps_per_sm(r, BLOCK) > BASE_WARPS && m.local == 0,
        "gains_step_b128" =>
            warps_per_sm(r, ALT_BLOCK) > ALT_BASE_WARPS && m.local == 0,
    )
    Printf.@printf(
        "%-34s regs=%3d  local=%6d B  warps/SM=%2d  %s\n",
        label, r, m.local, w,
        r <= TARGET_REGS ? (m.local == 0 ? "FITS" : "fits but spills") :
        (r <= ALT_TARGET_REGS && m.local == 0 ? "gains a step at 128 threads" :
         "too big"),
    )
    return r
end

# ---------------------------------------------------------------------------
# Longwave phases. Each is the body a split would put in its own kernel, called
# exactly as the fused kernel calls it. The intrinsics are qualified because
# this file imports CUDA rather than using it, and a bare `threadIdx()` here
# compiles to an undefined name whose dynamic dispatch buries every later
# Reason line in the error.
# ---------------------------------------------------------------------------

function k_lw_mask(cloud_state, mask_store, ncol)
    gcol = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if gcol <= ncol
        @inbounds RTESolver._build_cloud_mask!(
            cloud_state, Val(:mask_lw), gcol, 1, mask_store, gcol,
        )
    end
    return nothing
end

function k_lw_optics(op, as, state_cache, src, lkp, lkp_aero, ncol)
    gcol = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if gcol <= ncol
        @inbounds Optics.compute_optical_props!(
            op, as, state_cache, src, gcol, 1, lkp, nothing, lkp_aero, gcol,
        )
    end
    return nothing
end

function k_lw_optics_nogas(op, as, lkp_cld, mask_store, ncol)
    gcol = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if gcol <= ncol
        @inbounds Optics.add_cloud_optics_lw!(op, as, gcol, lkp_cld, 1, mask_store, gcol)
    end
    return nothing
end

function k_lw_sweep(op, flux, src, bcs, nlev, ncol)
    gcol = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if gcol <= ncol
        @inbounds RTESolver.rte_lw_2stream!(
            op, flux, src, bcs, gcol, 1, 1, nlev, ncol, gcol,
        )
    end
    return nothing
end

function k_lw_accumulate(acc, flux, nlev, ncol)
    gcol = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if gcol <= ncol
        @inbounds RTESolver._accumulate_fluxes!(acc, flux, gcol, nlev, 1, 1)
    end
    return nothing
end

println("\n=== longwave phases (target <= $TARGET_REGS registers, no spill) ===")
record!(
    "lw_cloud_mask",
    () -> (CUDA.@cuda launch = false always_inline = true k_lw_mask(
        cloud_state, cloud_state.mask_lw, ncol,
    )),
)
record!(
    "lw_gas_aerosol_optics",
    () -> (CUDA.@cuda launch = false always_inline = true k_lw_optics(
        lws.op, as, lws.state_cache, lws.src, lk.lookup_lw, lk.lookup_lw_aero,
        ncol,
    )),
)
record!(
    "lw_cloud_optics",
    () -> (CUDA.@cuda launch = false always_inline = true k_lw_optics_nogas(
        lws.op, as, lk.lookup_lw_cld, cloud_state.mask_lw, ncol,
    )),
)
record!(
    "lw_two_stream_sweep",
    () -> (CUDA.@cuda launch = false always_inline = true k_lw_sweep(
        lws.op, lws.fluxb, lws.src, lws.bcs, nlev, ncol,
    )),
)
record!(
    "lw_accumulate",
    () -> (CUDA.@cuda launch = false always_inline = true k_lw_accumulate(
        solver.clear_flux_acc_lw, lws.fluxb, nlev, ncol,
    )),
)

# ---------------------------------------------------------------------------
# Shortwave phases.
# ---------------------------------------------------------------------------

function k_sw_optics(op, as, state_cache, lkp, lkp_aero, ncol)
    gcol = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if gcol <= ncol
        @inbounds Optics.compute_optical_props!(
            op, as, state_cache, gcol, 1, lkp, nothing, lkp_aero, gcol, true,
        )
    end
    return nothing
end

function k_sw_sweep(op, src, bcs, flux, n_gpt, nlev, ncol)
    gcol = CUDA.threadIdx().x + (CUDA.blockIdx().x - 1) * CUDA.blockDim().x
    if gcol <= ncol
        @inbounds RTESolver.rte_sw_2stream!(
            op, src, bcs, flux, 1.0f0, 1, n_gpt, 1, nlev, gcol, gcol,
        )
    end
    return nothing
end

n_gpt_sw = length(lk.lookup_sw.band_data.major_gpt2bnd)

println("\n=== shortwave phases ===")
record!(
    "sw_gas_aerosol_optics",
    () -> (CUDA.@cuda launch = false always_inline = true k_sw_optics(
        sws.op, as, sws.state_cache, lk.lookup_sw, lk.lookup_sw_aero, ncol,
    )),
)
record!(
    "sw_two_stream_sweep",
    () -> (CUDA.@cuda launch = false always_inline = true k_sw_sweep(
        sws.op, sws.src, sws.bcs, sws.fluxb, n_gpt_sw, nlev, ncol,
    )),
)

# ---------------------------------------------------------------------------
# The whole fused kernels, as the control. These must come back at the cap, or
# the probe is not measuring what the model runs.
# ---------------------------------------------------------------------------

ext = Base.get_extension(RRTMGP, :RRTMGPCUDAExt)
isnothing(ext) && error("RRTMGPCUDAExt is not loaded, so the control cannot run")

println("\n=== whole fused kernels (control: expect the cap) ===")
record!(
    "lw_fused_whole",
    () -> (CUDA.@cuda launch = false always_inline = true ext.rte_lw_2stream_solve_both_skies_CUDA!(
        lws.fluxb, lws.flux, solver.clear_flux_acc_lw, nothing, lws.src,
        lws.bcs, lws.op, nlay, ncol, as, lws.state_cache, lk.lookup_lw,
        lk.lookup_lw_cld, lk.lookup_lw_aero,
    )),
)
record!(
    "sw_fused_whole",
    () -> (CUDA.@cuda launch = false always_inline = true ext.rte_sw_2stream_solve_both_skies_CUDA!(
        sws.fluxb, sws.flux, solver.clear_flux_acc_sw, nothing, sws.op,
        sws.bcs, sws.src, nlay, ncol, as, sws.state_cache, lk.lookup_sw,
        lk.lookup_sw_cld, lk.lookup_sw_aero,
    )),
)

# ---------------------------------------------------------------------------
# The grouped kernels, which are what the mod arm actually launches. The
# ungrouped controls above are the baseline's; both profile at the cap, so this
# records rather than discovers -- but a phase split would replace THESE.
# ---------------------------------------------------------------------------

grp_lw, grp_sw = lws.gpt_groups, sws.gpt_groups
ngroups_lw =
    isnothing(grp_lw) ? 0 : RRTMGP.RTE.n_gpt_groups(grp_lw)
ngroups_sw =
    isnothing(grp_sw) ? 0 : RRTMGP.RTE.n_gpt_groups(grp_sw)

println("\n=== grouped kernels (what the mod arm launches) ===")
if !isnothing(grp_lw)
    record!(
        "lw_grouped_whole",
        () -> (CUDA.@cuda launch = false always_inline = true ext.rte_lw_2stream_solve_grouped_CUDA!(
            grp_lw.fluxb, grp_lw.acc, grp_lw.acc_clear, grp_lw.src, lws.bcs,
            grp_lw.op, grp_lw.mask, grp_lw.cloudy, nlay, ncol, as,
            lws.state_cache, lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero,
            ngroups_lw,
        )),
    )
end
if !isnothing(grp_sw)
    record!(
        "sw_grouped_whole",
        () -> (CUDA.@cuda launch = false always_inline = true ext.rte_sw_2stream_solve_grouped_CUDA!(
            grp_sw.fluxb, grp_sw.acc, grp_sw.acc_clear, grp_sw.src, sws.bcs,
            grp_sw.op, grp_sw.mask, grp_sw.cloudy, nlay, ncol, as,
            sws.state_cache, lk.lookup_sw, lk.lookup_sw_cld, lk.lookup_sw_aero,
            ngroups_sw,
        )),
    )
end

# ---------------------------------------------------------------------------
# What forcing the budget costs, as a number rather than as the memory of two
# runs. Capping these kernels measured +3.93% and +2.91%; the reason given was
# spill, which was never quantified. Both levers are compiled here: the hard
# cap ptxas must honor by spilling, and the launch bounds it may instead reach
# by rematerializing. If neither lands near 128 without spill, forcing the
# occupancy step is closed and the split is the only route left.
# ---------------------------------------------------------------------------

println("\n=== the same kernels, forced to the 2-blocks/SM budget ===")
record!(
    "lw_fused_maxregs_128",
    () -> (CUDA.@cuda launch = false always_inline = true maxregs = 128 ext.rte_lw_2stream_solve_both_skies_CUDA!(
        lws.fluxb, lws.flux, solver.clear_flux_acc_lw, nothing, lws.src,
        lws.bcs, lws.op, nlay, ncol, as, lws.state_cache, lk.lookup_lw,
        lk.lookup_lw_cld, lk.lookup_lw_aero,
    )),
)
record!(
    "lw_fused_launch_bounds_2",
    () -> (CUDA.@cuda launch = false always_inline = true maxthreads = 256 blocks_per_sm = 2 ext.rte_lw_2stream_solve_both_skies_CUDA!(
        lws.fluxb, lws.flux, solver.clear_flux_acc_lw, nothing, lws.src,
        lws.bcs, lws.op, nlay, ncol, as, lws.state_cache, lk.lookup_lw,
        lk.lookup_lw_cld, lk.lookup_lw_aero,
    )),
)
record!(
    "sw_fused_maxregs_128",
    () -> (CUDA.@cuda launch = false always_inline = true maxregs = 128 ext.rte_sw_2stream_solve_both_skies_CUDA!(
        sws.fluxb, sws.flux, solver.clear_flux_acc_sw, nothing, sws.op,
        sws.bcs, sws.src, nlay, ncol, as, sws.state_cache, lk.lookup_sw,
        lk.lookup_sw_cld, lk.lookup_sw_aero,
    )),
)
record!(
    "sw_fused_launch_bounds_2",
    () -> (CUDA.@cuda launch = false always_inline = true maxthreads = 256 blocks_per_sm = 2 ext.rte_sw_2stream_solve_both_skies_CUDA!(
        sws.fluxb, sws.flux, solver.clear_flux_acc_sw, nothing, sws.op,
        sws.bcs, sws.src, nlay, ncol, as, sws.state_cache, lk.lookup_sw,
        lk.lookup_sw_cld, lk.lookup_sw_aero,
    )),
)

# ---------------------------------------------------------------------------
# The verdict, computed rather than read off by eye.
# ---------------------------------------------------------------------------

phases_lw = [
    "lw_cloud_mask", "lw_gas_aerosol_optics", "lw_cloud_optics",
    "lw_two_stream_sweep", "lw_accumulate",
]
phases_sw = ["sw_gas_aerosol_optics", "sw_two_stream_sweep"]

# The phase that decides it: gas optics is ~99% of RRTMGP time (section 4l item
# 5), so a split only helps if THAT phase fits. A split where the expensive
# phase stays at the cap moves 1% of the time to higher occupancy.
fits(p) = get(get(results, p, Dict{String, Any}()), "reaches_target", false)
at_cap(p) = get(get(results, p, Dict{String, Any}()), "at_cap", false)
gains(p) = get(get(results, p, Dict{String, Any}()), "gains_step_b128", false)

field(p, k, d) = get(get(results, p, Dict{String, Any}()), k, d)

# Requiring zero local memory is the wrong bar, and it reported a phase at 32 B
# as a failure. Section 8t records that `CUDA.memory(k).local` covers the call
# frame and local arrays as well as spilled registers, so it bounds spill
# rather than measuring it. The reference that means something is the kernel
# the phase would replace: the grouped controls carry 304-360 B and the forced
# variants that measured +3.93% and +2.91% carry 672-712 B, so a phase is only
# paying what made forcing lose if it spills MORE than what it replaces.
no_worse_spill(p, control) =
    field(p, "local_bytes", typemax(Int)) <= field(control, "local_bytes", 0)
fits_vs(p, control) =
    field(p, "registers", REGISTER_CAP) <= TARGET_REGS && no_worse_spill(p, control)
gains_vs(p, control) =
    field(p, "warps_per_sm_b128", 0) > ALT_BASE_WARPS && no_worse_spill(p, control)

lw_optics_fits = fits_vs("lw_gas_aerosol_optics", "lw_grouped_whole")
sw_optics_fits = fits_vs("sw_gas_aerosol_optics", "sw_grouped_whole")
lw_optics_gains = gains_vs("lw_gas_aerosol_optics", "lw_grouped_whole")
sw_optics_gains = gains_vs("sw_gas_aerosol_optics", "sw_grouped_whole")

results["summary"] = Dict{String, Any}(
    "ncol" => ncol,
    "nlay" => nlay,
    "block_size" => BLOCK,
    "register_cap" => REGISTER_CAP,
    "target_registers" => TARGET_REGS,
    "target_warps_per_sm" => warps_per_sm(TARGET_REGS),
    "current_warps_per_sm" => warps_per_sm(REGISTER_CAP),
    "alt_block_size" => ALT_BLOCK,
    "alt_target_registers" => ALT_TARGET_REGS,
    "alt_target_warps_per_sm" => warps_per_sm(ALT_TARGET_REGS, ALT_BLOCK),
    "alt_current_warps_per_sm" => ALT_BASE_WARPS,
    "control_lw_at_cap" => at_cap("lw_fused_whole"),
    "control_sw_at_cap" => at_cap("sw_fused_whole"),
    "control_lw_grouped_at_cap" => at_cap("lw_grouped_whole"),
    "control_sw_grouped_at_cap" => at_cap("sw_grouped_whole"),
    "failed" => sort([k for (k, v) in results if haskey(v, "error")]),
    "ngroups_lw" => ngroups_lw,
    "ngroups_sw" => ngroups_sw,
    "forced" => Dict{String, Any}(
        k => Dict{String, Any}(
            "registers" => get(get(results, k, Dict()), "registers", -1),
            "local_bytes" => get(get(results, k, Dict()), "local_bytes", -1),
            "warps_per_sm" => get(get(results, k, Dict()), "warps_per_sm", -1),
        ) for k in [
            "lw_fused_maxregs_128", "lw_fused_launch_bounds_2",
            "sw_fused_maxregs_128", "sw_fused_launch_bounds_2",
        ]
    ),
    # A forced budget is only worth running if something reaches it clean.
    "forced_budget_is_clean" => any(
        fits(k) for k in [
            "lw_fused_maxregs_128", "lw_fused_launch_bounds_2",
            "sw_fused_maxregs_128", "sw_fused_launch_bounds_2",
        ]
    ),
    "lw_optics_phase_fits" => lw_optics_fits,
    "sw_optics_phase_fits" => sw_optics_fits,
    "lw_optics_phase_fits_no_spill_at_all" => fits("lw_gas_aerosol_optics"),
    "sw_optics_phase_fits_no_spill_at_all" => fits("sw_gas_aerosol_optics"),
    "lw_optics_local_bytes" => field("lw_gas_aerosol_optics", "local_bytes", -1),
    "sw_optics_local_bytes" => field("sw_gas_aerosol_optics", "local_bytes", -1),
    "lw_grouped_local_bytes" => field("lw_grouped_whole", "local_bytes", -1),
    "sw_grouped_local_bytes" => field("sw_grouped_whole", "local_bytes", -1),
    "lw_optics_registers" => field("lw_gas_aerosol_optics", "registers", -1),
    "sw_optics_registers" => field("sw_gas_aerosol_optics", "registers", -1),
    "lw_optics_warps_per_sm_b128" =>
        field("lw_gas_aerosol_optics", "warps_per_sm_b128", -1),
    "sw_optics_warps_per_sm" => field("sw_gas_aerosol_optics", "warps_per_sm", -1),
    # The whole question, in one field.
    "split_is_viable" => lw_optics_fits || sw_optics_fits,
    # And the same question against the nearer bar, which a split can clear
    # without reaching the budget two forced attempts already lost on.
    "lw_optics_phase_gains_step_b128" => lw_optics_gains,
    "sw_optics_phase_gains_step_b128" => sw_optics_gains,
    "split_gains_a_step_b128" => lw_optics_gains || sw_optics_gains,
    "phases_that_fit" => [
        p for p in vcat(phases_lw, phases_sw) if fits(p)
    ],
    "phases_that_gain_a_step_b128" => [
        p for p in vcat(phases_lw, phases_sw) if gains(p)
    ],
    "note" => "Each phase is compiled as its own kernel with the live " *
              "solver's argument types, which is what a split would produce. " *
              "target_registers is what the next occupancy step needs at this " *
              "block size: 65536 / (regs * block) >= 2. reaches_target " *
              "requires BOTH the register count and zero spill, because " *
              "forcing the budget with a spill is the thing that measured " *
              "+3.93% and +2.91% on these kernels. The phase that decides " *
              "viability is the gas and aerosol optics: section 4l item 5 " *
              "measured 99% of RRTMGP time there and 0.1% in the vertical " *
              "sweeps, so a split whose expensive phase stays at the cap " *
              "moves almost none of the time to higher occupancy. The " *
              "controls must come back at the cap or the probe is not " *
              "measuring the kernel the model runs.",
)

mkpath(dirname(out_path))
open(io -> TOML.print(io, results; sorted = true), out_path, "w")

Printf.@printf(
    "\nverdict: optics phase fits under %d regs with no spill -- longwave %s, shortwave %s\n",
    TARGET_REGS, lw_optics_fits, sw_optics_fits,
)
println("wrote $out_path")
