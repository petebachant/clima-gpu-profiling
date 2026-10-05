# Are the radiation kernel's 255 registers doing arithmetic, or holding data?
#
# ncu says the kernel is latency-bound, not compute- or bandwidth-bound: 19.89%
# compute throughput, 9.00% DRAM, 0.25 eligible warps per scheduler, IPC 0.92 of
# 4. Occupancy is 12.40% against a 12.50% ceiling, and the ceiling is set by 255
# registers per thread. So the one lever that addresses the measured bottleneck
# is fewer registers -> more warps resident -> more latency hidden.
#
# Occupancy work has been tried from the other end and lost twice: capping
# registers at 128 (16 warps) cost 3.93%, at 168 (12 warps) 2.91%, because the
# cap forces spill (docs/learnings.md 4v). Capping asks the compiler to fit the
# same work in fewer registers. This asks a different question: is the work
# itself carrying data it does not need?
#
# The precedent is in the microphysics kernel. measure-cm-registers.jl found the
# same kernel at 246 registers with parameters as runtime arguments and 101 with
# them as const globals the compiler could fold -- and slimming the evaluator
# struct from 472 B to 40 B was worth -4.2%, more than skipping 88% of the
# physics (4r). Payload beat algorithm there.
#
# This compiles the fused kernels with `launch=false` and reads registers back,
# so it measures without running anything, and varies only which optional
# lookups are present. The signature flattens to on the order of a hundred array
# descriptors, each a pointer plus dimensions; the deltas below say how much of
# the budget they cost.

import CUDA
import ClimaAtmos as CA
import RRTMGP
import RRTMGP.RTESolver as RTE
import TOML
using Printf

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

out_path = "results/radiation-registers.toml"
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i)
        out_path = ARGS[i + 1]
        deleteat!(ARGS, i:(i + 1))
    end
end

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)
# A few steps so the cloud and aerosol state hold real values. Register counts
# depend on types rather than values, but building the state is what fixes the
# types, so this runs the model rather than inventing arrays.
for i in 1:3
    step!(cs)
end

s = cs.model_sims.atmos_sim.integrator.p.radiation.rrtmgp_solver
lk, as, lws, sws = s.lookups, s.as, s.lws, s.sws
nlay, ncol = RRTMGP.AtmosphericStates.get_dims(as)
ext = Base.get_extension(RRTMGP, :RRTMGPCUDAExt)
isnothing(ext) && error("the CUDA extension is not loaded; is this running on a GPU?")

# A100: 65536 registers per SM, 128-thread blocks, so warps resident per SM is
# bounded by registers per thread. This is the number occupancy actually turns
# on, which is why it is reported beside the raw count.
warps_per_sm(r) = r == 0 ? 0 : min(64, 4 * fld(65536, 128 * r))

results = Dict{String, Any}(
    "note" => "registers per thread for the radiation kernels, compiled with " *
              "launch=false so nothing is executed. Variants differ only in " *
              "which optional lookups are passed, which is what reveals how " *
              "much of the budget is descriptors for data rather than live " *
              "arithmetic.",
    "ncol" => ncol,
    "nlay" => nlay,
)

function record!(label, kernel, args)
    k = CUDA.@cuda launch = false always_inline = true kernel(args...)
    r = CUDA.registers(k)
    m = CUDA.memory(k)
    results[label] = Dict(
        "registers" => r,
        "local_bytes" => m.local,
        "shared_bytes" => m.shared,
        "const_bytes" => m.constant,
        "warps_per_sm" => warps_per_sm(r),
    )
    @printf(
        "%-34s regs=%3d  local=%5d B  const=%6d B  warps/SM=%2d\n",
        label, r, m.local, m.constant, warps_per_sm(r),
    )
    return r
end

println("\n=== fused longwave ===")
lw_args(cld, aero) = (
    lws.fluxb, lws.flux, s.clear_flux_acc_lw, lws.band_flux, lws.src, lws.bcs,
    lws.op, nlay, ncol, as, lws.state_cache, lk.lookup_lw, cld, aero,
)
record!("lw_fused_cloud_aerosol",
        ext.rte_lw_2stream_solve_both_skies_CUDA!,
        lw_args(lk.lookup_lw_cld, lk.lookup_lw_aero))
record!("lw_fused_cloud_only",
        ext.rte_lw_2stream_solve_both_skies_CUDA!,
        lw_args(lk.lookup_lw_cld, nothing))
record!("lw_fused_bare",
        ext.rte_lw_2stream_solve_both_skies_CUDA!,
        lw_args(nothing, nothing))

println("\n=== fused shortwave ===")
sw_args(cld, aero) = (
    sws.fluxb, sws.flux, s.clear_flux_acc_sw, sws.band_flux, sws.src, sws.bcs,
    sws.op, nlay, ncol, as, sws.state_cache, lk.lookup_sw, cld, aero,
)
record!("sw_fused_cloud_aerosol",
        ext.rte_sw_2stream_solve_both_skies_CUDA!,
        sw_args(lk.lookup_sw_cld, lk.lookup_sw_aero))
record!("sw_fused_cloud_only",
        ext.rte_sw_2stream_solve_both_skies_CUDA!,
        sw_args(lk.lookup_sw_cld, nothing))
record!("sw_fused_bare",
        ext.rte_sw_2stream_solve_both_skies_CUDA!,
        sw_args(nothing, nothing))

# What the aerosol and cloud lookups cost in registers. If these are large, the
# budget is going on descriptors and a slimmer payload is worth building; if
# they are ~0, the registers are arithmetic and this line of work is dead.
for band in ("lw", "sw")
    full = results["$(band)_fused_cloud_aerosol"]["registers"]
    nocld = results["$(band)_fused_bare"]["registers"]
    noaero = results["$(band)_fused_cloud_only"]["registers"]
    results["$(band)_aerosol_register_cost"] = full - noaero
    results["$(band)_cloud_register_cost"] = noaero - nocld
    @printf(
        "\n%s: aerosol lookups cost %d registers, cloud lookups %d (full %d, bare %d)\n",
        band, full - noaero, noaero - nocld, full, nocld,
    )
end

# 255 is the hardware maximum per thread. A kernel pinned there may simply want
# more, so the spill figure says whether the compiler was squeezed.
results["at_register_ceiling"] =
    results["lw_fused_cloud_aerosol"]["registers"] >= 255
results["spilling"] = results["lw_fused_cloud_aerosol"]["local_bytes"] > 0

open(out_path, "w") do io
    TOML.print(io, results; sorted = true)
end
@info "wrote $out_path"
