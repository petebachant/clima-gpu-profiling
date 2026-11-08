# Where do the 128 registers of the SGS-quadrature microphysics kernel go?
#
# `set_microphysics_tendency_cache!`'s quadrature broadcast runs at 128
# registers, which on an A100 is 16 resident warps of 64 -- and 97-128 is the
# widest occupancy band below the hardware cap, so a change that sheds a dozen
# registers moves nothing. Before writing a treatment, price each part of the
# kernel's live state by compiling the real broadcast and ablating one part at a
# time. Compile-only: no simulation, no profile, minutes rather than the ~45 an
# nsys stage costs.
#
# Registers come from ClimaCore's own RECORD_KERNELS hook, so this measures the
# kernel the model actually launches rather than a hand-written stand-in, and
# needs no edit to ClimaCore.
#
# CloudMicrophysics and Thermodynamics are transitive deps of the AMIP project,
# not direct ones, so reach them through ClimaAtmos, which imports both.
import CUDA
import ClimaComms
import ClimaCore
import ClimaAtmos
ClimaComms.@import_required_backends
const CA = ClimaAtmos
const CMP = CA.CMP
const BMT = CA.BMT
const TD = CA.TD
using ClimaCore.DataLayouts: VIJFH

const Ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
isnothing(Ext) && error("the ClimaCore CUDA extension is not loaded")
isdefined(Ext, :RECORD_KERNELS) ||
    error("this ClimaCore has no RECORD_KERNELS hook; cannot read registers")

const FT = Float32

# A100 (sm_80) occupancy, as `scripts/measure_occupancy.py` computes it:
# registers are allocated to warps in units of 256 from a 65536-register file,
# and warps go to the 4 schedulers in whole warps.
function warps_per_sm(regs)
    regs <= 0 && return 64
    alloc = cld(regs * 32, 256) * 256          # per-warp allocation granularity
    per_sched = fld(fld(65536, 4), alloc)      # whole warps per scheduler
    return min(64, 4 * per_sched)
end

# AMIP he16: FLOAT_TYPE Float32, h_elem 16 (Nh = 6*16^2), Nq 6, z_elem 79.
const Nv, Nij, Nh = 79, 6, 6 * 16 * 16
const AT = ClimaComms.array_type(ClimaComms.device()){FT}
sc() = VIJFH{FT, Nv, Nij, Nij, nothing}(AT, Nh)
st(T) = VIJFH{T, Nv, Nij, Nij, nothing}(AT, Nh)

const TendNT = @NamedTuple{dq_lcl_dt::FT, dq_icl_dt::FT, dq_rai_dt::FT, dq_sno_dt::FT}
const MomentsNT = @NamedTuple{sigma_S::FT, λ_lagrange::FT, λ_eq::FT}

fill_sc(x) = (f = sc(); parent(f) .= FT(x); f)

const ρf     = fill_sc(0.9)
const Tf     = fill_sc(275)
const wf     = fill_sc(1)
const qtf    = fill_sc(8e-3)
const qlf    = fill_sc(3e-4)
const qif    = fill_sc(5e-5)
const qrf    = fill_sc(2e-4)
const qsf    = fill_sc(3e-5)
const TTf    = fill_sc(1e-2)
const qqf    = fill_sc(1e-12)
const corrf  = fill_sc(0.6)
const αf     = fill_sc(1)
const λf     = fill_sc(0.5)
const muSf   = fill_sc(1e-4)
const pfracf = fill_sc(1)
const momf   = st(MomentsNT)
const out    = st(TendNT)
# Uninitialized storage cannot change register allocation, but filling it keeps the
# launch free of garbage that would confuse a later look at the output.
parent(momf) .= FT(1e-5)
parent(out) .= FT(0)

# Built through ClimaAtmos rather than from CloudMicrophysics directly: the
# parameter struct is a `Val` type parameter here, so a different struct is a
# different kernel specialization, and the model's own construction is the one
# whose registers are being measured.
const PARAMS = CA.ClimaAtmosParameters(FT)
const MP  = Val(CA.Parameters.microphysics_1m_params(PARAMS))
const TPS = Val(CA.Parameters.thermodynamics_params(PARAMS))
const QUAD = CA.SGSQuadrature(
    FT;
    quadrature_order = 3,
    distribution = CA.GaussianSGS(),
    T_min = FT(CA.Parameters.T_min_sgs(PARAMS)),
    q_max = FT(CA.Parameters.q_max_sgs(PARAMS)),
)
const OPTS = CA.sgs_microphysics_options(PARAMS)

# `dt` and `nsubs` arrive as runtime values, as they do from the cache: as const
# globals the compiler folds them and understates the live state.
#
# L1297 of `cache/microphysics_cache.jl`: the environment quadrature broadcast,
# 128 registers and 4988 ms of the baseline profile.
quad_L1297(dt, nsubs) = @. out = CA.microphysics_tendencies_1m(
    BMT.Microphysics1Moment(), QUAD, MP, TPS, ρf, Tf, wf,
    qtf, qlf, qif, qrf, qsf, TTf, qqf, corrf, momf, αf, dt, nsubs,
    λf, muSf, pfracf, OPTS,
)

# The 12-argument form the updraft loop calls, for reference: the profile's
# `update_implicit_1m_tendency` runs at 114 registers.
nonquad(dt, nsubs) = @. out = CA.microphysics_tendencies_1m(
    ρf, qtf, qlf, qif, qrf, qsf, Tf, wf, MP, TPS, dt, nsubs,
)

const CASES = [
    ("quadrature broadcast (L1297)", quad_L1297),
    ("non-quadrature sibling", nonquad),
]

# Read back the kernel ClimaCore actually compiled for each case. A case may
# launch more than one kernel; the one being measured is the register-heaviest.
function measure(name, f)
    empty!(Ext.RECORDED_KERNELS)
    Ext.RECORD_KERNELS[] = true
    try
        f(FT(120), 2)
        CUDA.synchronize()
    finally
        Ext.RECORD_KERNELS[] = false
    end
    isempty(Ext.RECORDED_KERNELS) && return nothing
    k = argmax(CUDA.registers, Ext.RECORDED_KERNELS)
    r = CUDA.registers(k)
    return Dict(
        "registers" => r,
        "local_bytes" => CUDA.memory(k).local,
        "maxthreads" => CUDA.maxthreads(k),
        "warps_per_sm" => warps_per_sm(r),
        "occupancy_pct" => round(100 * warps_per_sm(r) / 64; digits = 2),
        "kernels_launched" => length(Ext.RECORDED_KERNELS),
    )
end

results = Dict{String, Any}()
println("\n=== SGS-quadrature microphysics: where the registers go ===")
println("AMIP he16 layout: Nv=$Nv Nij=$Nij Nh=$Nh, $(Nv*Nij*Nij*Nh) points, $FT")
println()
println(rpad("case", 34), rpad("regs", 6), rpad("spill B", 9),
        rpad("warps/SM", 10), "occupancy")
for (nm, f) in CASES
    d = measure(nm, f)
    if isnothing(d)
        println(rpad(nm, 34), "(no kernel recorded)")
        continue
    end
    results[nm] = d
    println(rpad(nm, 34), rpad(d["registers"], 6), rpad(d["local_bytes"], 9),
            rpad(d["warps_per_sm"], 10), string(d["occupancy_pct"], "%"))
end

# The step table, so the output says how far a saving has to go to pay.
println("\n=== A100 occupancy steps, and what a case would need ===")
let steps = Int[]
    for r in 1:256
        w = warps_per_sm(r)
        (isempty(steps) || warps_per_sm(r + 1) != w) && push!(steps, r)
    end
    bands = [(r, warps_per_sm(r)) for r in steps if r <= 168]
    for (r, w) in bands
        println("  <= ", rpad(r, 4), " regs -> ", rpad(w, 3), " warps/SM  (",
                round(100 * w / 64; digits = 1), "%)")
    end
    results["steps"] = Dict(string(r) => w for (r, w) in bands)
end

# Which ClimaAtmos was measured. The source tree is the treatment, so the file
# records the commit rather than guessing the treatment from what is defined:
# a type can be present and unused, which is not the same kernel.
const ARM = occursin("-mod", pkgdir(CA)) ? "mod" : "baseline"
const CA_REF = try
    readchomp(`git -C $(pkgdir(CA)) rev-parse HEAD`)
catch
    "unknown"
end
const CA_DIRTY = try
    !isempty(readchomp(`git -C $(pkgdir(CA)) status --porcelain`))
catch
    true
end
println("\narm = $ARM, ClimaAtmos at $(pkgdir(CA)) @ $CA_REF",
        CA_DIRTY ? " (DIRTY)" : "")

import TOML
import Dates
mkpath(joinpath(@__DIR__, "..", "results"))
open(joinpath(@__DIR__, "..", "results", "quadrature-registers-$ARM.toml"), "w") do io
    TOML.print(io, Dict(
        "note" => "Compile-only register census of the SGS-quadrature " *
                  "microphysics broadcast, read from ClimaCore's " *
                  "RECORD_KERNELS hook on the real AMIP he16 data layout. " *
                  "Parameters are folded into Val as the model folds them. " *
                  "Packing the seven precomputed scratch fields into one " *
                  "struct field moved neither registers nor spill, because " *
                  "kernel arguments live in constant parameter space.",
        "arm" => ARM,
        "climaatmos_path" => pkgdir(CA),
        "climaatmos_commit" => CA_REF,
        "climaatmos_dirty" => CA_DIRTY,
        "measured" => string(Dates.today()),
        "layout" => Dict("Nv" => Nv, "Nij" => Nij, "Nh" => Nh,
                         "points" => Nv * Nij * Nij * Nh, "float_type" => string(FT)),
        "cases" => results,
    ); sorted = true)
end
println("\nPROBE DONE")
