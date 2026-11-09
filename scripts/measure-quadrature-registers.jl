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
import Random
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

# `VIJFH` stores components on the fourth axis, so a scalar field is component 1.
function fill_component!(field, k, seed, lo, hi)
    v = view(parent(field), :, :, :, k, :)
    h = rand(Random.MersenneTwister(seed), FT, size(v)) .* FT(hi - lo) .+ FT(lo)
    copyto!(v, h)
    return field
end
varied(seed, lo, hi) = fill_component!(sc(), 1, seed, lo, hi)

const ρf     = varied(1, 0.25, 1.2)      # kg/m^3, surface to model top
const Tf     = varied(2, 205, 300)       # K, spanning the mixed-phase band
const wf     = varied(3, -2, 3)          # m/s
const qtf    = varied(4, 1e-6, 2e-2)     # kg/kg
const qlf    = varied(5, 0, 1e-3)
const qif    = varied(6, 0, 3e-4)
const qrf    = varied(7, 0, 1e-3)
const qsf    = varied(8, 0, 3e-4)
const TTf    = varied(9, 1e-4, 1e-1)     # K^2
const qqf    = varied(10, 1e-14, 1e-10)  # (kg/kg)^2
const corrf  = varied(11, -0.9, 0.9)
const αf     = varied(12, 0.2, 1)
const λf     = varied(13, 0, 1)
const muSf   = varied(14, -5e-3, 5e-3)
const pfracf = varied(15, 1e-2, 1)
const momf   = st(MomentsNT)
const out    = st(TendNT)
fill_component!(momf, 1, 16, 1e-6, 3e-3)   # sigma_S, positive
fill_component!(momf, 2, 17, -3e-3, 3e-3)  # λ_lagrange
fill_component!(momf, 3, 18, -3e-3, 3e-3)  # λ_eq
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

# The same broadcast asking ClimaCore for the cap, through the API the treatment
# uses rather than through `cufunction` directly. The capped recompiles below
# prove only that `ptxas` CAN fit this body in 128 registers; this proves the
# request survives `Base.copyto!` -> `DataLayouts.copyto!` ->
# `foreach_slice_with_options` -> `with_register_cap` and reaches the compile.
# A silently dropped keyword would look exactly like no cap at all, and would
# only show up as a wasted profiling pair.
quad_capped(dt, nsubs) = ClimaCore.DataLayouts.copyto_capped!(
    out,
    Base.Broadcast.instantiate(
        Base.broadcasted(
            CA.microphysics_tendencies_1m,
            BMT.Microphysics1Moment(), QUAD, MP, TPS, ρf, Tf, wf,
            qtf, qlf, qif, qrf, qsf, TTf, qqf, corrf, momf, αf, dt, nsubs,
            λf, muSf, pfracf, OPTS,
        ),
    ),
    Val(128),
)

# The same quadrature with `BMT.Instantaneous()`, which is the dispatch
# `update_implicit_1m_tendency!` calls inside the Newton loop -- and the kernel
# the removals pay off on. The hash of THIS case is what makes the bit-for-bit
# claim cover the kernel the headline number comes from; the averaged case above
# exercises a different method of the same node path.
quad_instantaneous(dt, nsubs) = @. out = CA.microphysics_tendencies_1m(
    BMT.Microphysics1Moment(), QUAD, MP, TPS, ρf, Tf, wf,
    qtf, qlf, qif, qrf, qsf, TTf, qqf, corrf, momf, αf, dt, nsubs,
    λf, muSf, pfracf, OPTS, BMT.Instantaneous(),
)

# The 12-argument form the updraft loop calls, for reference: the profile's
# `update_implicit_1m_tendency` runs at 114 registers.
nonquad(dt, nsubs) = @. out = CA.microphysics_tendencies_1m(
    ρf, qtf, qlf, qif, qrf, qsf, Tf, wf, MP, TPS, dt, nsubs,
)

# The kernel's registers are the largest frame on its call graph, not the sum of
# its parts: the node function is `@noinline`, so `ptxas` allocates for the peak
# of {traversal, node function}. A barrier can only split a frame, never shrink
# one, so `max(traversal, BMT)` is the floor of every barrier arrangement
# reachable from ClimaAtmos -- which is what these three cases price.
#
# None of them edits ClimaAtmos: `sum_over_conditional_nodes` takes the node
# function as an argument, so the real traversal is measured by passing a stub in
# its place, and the two inner frames are measured by calling them directly.

# Stands in for `Microphysics1MEvaluator` to leave the traversal frame alone.
# `@noinline` because the real node function is, and the scaled arguments are
# consumed so that the rule's mass rescaling stays live rather than dead code.
struct StubNodeFunction{FT}
    λ_lagrange::FT
    α::FT
    shaft_frac::FT
end
@noinline function (s::StubNodeFunction{FT})(
    T_hat, q_tot_hat, flag_p, q_sat_hat...,
) where {FT}
    return (;
        dq_lcl_dt = s.λ_lagrange * T_hat,
        dq_icl_dt = s.α * q_tot_hat,
        dq_rai_dt = s.shaft_frac * FT(flag_p),
        dq_sno_dt = zero(FT),
    )
end

# The real conditional rule, mass rescaling, shaft fraction and node traversal,
# with the node function replaced by the stub: everything the kernel holds live
# across the call, including the 8-node table.
@inline function traversal_frame(
    ρ, T, w, q_tot, q_lcl, q_icl, q_rai, q_sno, TT, qq, corr, mom, α, dt, nsubs,
    λ, mu_S, precip_frac, opts,
)
    FT = typeof(ρ)
    a_p = max(FT(precip_frac), FT(opts.precip_frac_floor))
    σ_S = FT(mom.sigma_S)
    return CA._sum_over_conditional_sgs(
        (λ_scaled, α_scaled, shaft_frac) ->
            StubNodeFunction{FT}(λ_scaled, α_scaled, shaft_frac),
        TPS, ρ, T, mu_S, σ_S, mom.λ_lagrange, mom.λ_eq, α, q_lcl + q_icl,
        TT, qq, corr, FT(opts.precip_incloud_fraction),
        CA.sgs_precip_shaft_threshold(a_p, σ_S), QUAD,
    )
end

# One node function evaluation: the saturation, condensate reconstruction, shaft
# placement and early-out, plus the BMT call inlined into the same frame. This is
# the frame the barrier at the functor currently caps the kernel with.
@inline function node_function_frame(
    ρ, T, w, q_tot, q_lcl, q_icl, q_rai, q_sno, mom, α, dt, nsubs,
    λ, mu_S, shaft_frac, opts,
)
    FT = typeof(ρ)
    ev = CA.Microphysics1MEvaluator(
        BMT.LinearizedAverage(), BMT.Microphysics1Moment(), MP, TPS, ρ, w,
        q_rai, q_sno, λ, opts.ξ_liq, opts.ξ_ice, q_lcl, q_icl,
        mom.λ_lagrange, mom.λ_eq, mu_S, α,
        FT(opts.precip_incloud_fraction), shaft_frac, dt, nsubs, (),
    )
    return ev(T, q_tot, zero(FT))
end

# CloudMicrophysics' substep-averaged bulk tendency on its own, with nothing of
# ClimaAtmos's around it. Neither arm devs CloudMicrophysics, so this frame is
# the floor no ClimaAtmos-side change can lower.
@inline bmt_frame(ρ, T, w, q_tot, q_lcl, q_icl, q_rai, q_sno, dt, nsubs) =
    CA._call_bmt_1m(
        BMT.LinearizedAverage(), BMT.Microphysics1Moment(),
        CA._unwrapped(MP), CA._unwrapped(TPS), ρ, T, w,
        q_tot, q_lcl, q_icl, q_rai, q_sno, dt, nsubs, (),
    )

traversal_only(dt, nsubs) = @. out = traversal_frame(
    ρf, Tf, wf, qtf, qlf, qif, qrf, qsf, TTf, qqf, corrf, momf, αf, dt, nsubs,
    λf, muSf, pfracf, OPTS,
)
node_function_only(dt, nsubs) = @. out = node_function_frame(
    ρf, Tf, wf, qtf, qlf, qif, qrf, qsf, momf, αf, dt, nsubs,
    λf, muSf, pfracf, OPTS,
)
bmt_only(dt, nsubs) = @. out = bmt_frame(
    ρf, Tf, wf, qtf, qlf, qif, qrf, qsf, dt, nsubs,
)

# Two case sets, because the arm pair and the frame census answer different
# questions and write different files. The arm pair is an A/B across the two
# ClimaAtmos checkouts and its output is the record of a treatment, so it must
# stay regenerable exactly; the frame census is one run on one arm.
const CASE_SETS = Dict(
    "arms" => [
        ("quadrature broadcast (L1297)", quad_L1297),
        ("quadrature broadcast, ClimaCore cap", quad_capped),
        ("quadrature, instantaneous mode", quad_instantaneous),
        ("non-quadrature sibling", nonquad),
    ],
    "frames" => [
        ("quadrature broadcast (L1297)", quad_L1297),
        ("traversal frame (stub node function)", traversal_only),
        ("node function frame (recon + BMT)", node_function_only),
        ("BMT frame (CloudMicrophysics only)", bmt_only),
    ],
)
const CASE_SET = get(ENV, "QUAD_PROBE_CASES", "arms")
haskey(CASE_SETS, CASE_SET) ||
    error("QUAD_PROBE_CASES=$CASE_SET is not one of $(keys(CASE_SETS))")
const CASES = CASE_SETS[CASE_SET]

# Device time of the broadcast from CUDA events, best of `reps`: the layout is
# the model's, so this is directly comparable to the profile's per-launch time.
function time_ms(f; warmup = 2, reps = 7)
    for _ in 1:warmup
        f(FT(120), 2)
    end
    CUDA.synchronize()
    best = Inf
    for _ in 1:reps
        best = min(best, CUDA.@elapsed f(FT(120), 2))
    end
    return 1e3 * best
end

# FNV-1a over the raw output bits. A treatment that only reuses a value already
# computed is bit-for-bit by construction, and this is the statement of it that
# an approximate comparison of 4.4 million values is not.
function out_bits_hash(field)
    a = Array(parent(field))
    h = UInt64(0xcbf29ce484222325)
    for x in reinterpret(UInt32, vec(a))
        h = (h ⊻ UInt64(x)) * 0x100000001b3
    end
    return string(h, base = 16, pad = 16)
end

# Register caps to re-price each recorded kernel at. The quadrature broadcast
# sits at the TOP of the 97-128 band, so ptxas choosing 130 instead of 128 costs
# it a quarter of its occupancy (512-thread blocks and 16 warps become 384 and
# 12). A cap is the only deterministic way back into the band: source-level
# live-state reasoning has mispredicted this allocator twice here, once by 2
# registers and once by 33. What a cap costs is spill, so that is what these
# record -- 128 is the band edge and 96 is the next step down, which shows how
# fast the spill grows and therefore whether the edge is a cliff or a slope.
const CAPS = [128, 96]

# Re-compile a recorded kernel under a register cap. `HostKernel{F, TT}` carries
# the signature in its second type parameter, so the same specialization can be
# handed back to `cufunction` with a cap and nothing about the broadcast has to
# be rebuilt -- and no ClimaCore edit is needed to screen one.
function capped_compile(k, cap)
    tt = typeof(k).parameters[2]
    return CUDA.cufunction(k.f, tt; always_inline = true, maxregs = cap)
end

# Read back the kernel ClimaCore actually compiled for each case. A case may
# launch more than one kernel; the one being measured is the register-heaviest.
function measure(name, f)
    parent(out) .= FT(0)
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
    ms = time_ms(f)
    capped = Dict{String, Any}()
    for cap in CAPS
        try
            ck = capped_compile(k, cap)
            cr = CUDA.registers(ck)
            capped[string(cap)] = Dict{String, Any}(
                "registers" => cr,
                "local_bytes" => CUDA.memory(ck).local,
                "warps_per_sm" => warps_per_sm(cr),
                "honored" => cr <= cap,
            )
        catch err
            capped[string(cap)] = Dict{String, Any}(
                "error" => first(sprint(showerror, err), 200),
            )
        end
    end
    return Dict(
        "capped" => capped,
        "registers" => r,
        "device_ms" => round(ms; digits = 4),
        "out_bits_hash" => out_bits_hash(out),
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
println(rpad("case", 38), rpad("regs", 6), rpad("spill B", 9),
        rpad("warps/SM", 10), rpad("occupancy", 11), rpad("device ms", 11),
        "output bits")
for (nm, f) in CASES
    d = measure(nm, f)
    if isnothing(d)
        println(rpad(nm, 38), "(no kernel recorded)")
        continue
    end
    results[nm] = d
    println(rpad(nm, 38), rpad(d["registers"], 6), rpad(d["local_bytes"], 9),
            rpad(d["warps_per_sm"], 10), rpad(string(d["occupancy_pct"], "%"), 11),
            rpad(d["device_ms"], 11), d["out_bits_hash"])
    for cap in CAPS
        c = d["capped"][string(cap)]
        haskey(c, "error") && (println(rpad("  capped at $cap", 38), c["error"]); continue)
        println(rpad("  capped at $cap", 38), rpad(c["registers"], 6),
                rpad(c["local_bytes"], 9), rpad(c["warps_per_sm"], 10))
    end
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
const LABEL = get(ENV, "QUAD_PROBE_LABEL", CASE_SET == "arms" ? ARM : CASE_SET)
open(joinpath(@__DIR__, "..", "results", "quadrature-registers-$LABEL.toml"), "w") do io
    TOML.print(io, Dict(
        "note" => "Compile-only census of the SGS-quadrature microphysics " *
                  "broadcast on the real AMIP he16 data layout, with " *
                  "parameters folded into Val as the model folds them. " *
                  "Registers, spill and block size are read from " *
                  "ClimaCore's RECORD_KERNELS launch record, so they are " *
                  "the kernel the model launches. `device_ms` is the best " *
                  "of seven CUDA-event timings of that launch on a varied " *
                  "state in which almost every cell carries condensate, so " *
                  "it prices the full-work path and runs about twice the " *
                  "profile's per-launch time on the model's mixed state. " *
                  "`out_bits_hash` is FNV-1a over the raw output bits of " *
                  "all points, so two arms agreeing on it computed the " *
                  "same thing exactly.",
        "arm" => ARM,
        "case_set" => CASE_SET,
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
