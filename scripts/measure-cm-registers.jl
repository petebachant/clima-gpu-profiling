# Where do the ~246 registers of the hot microphysics kernel actually go?
#
# The hot AMIP kernel is pinned at 8 warps/SM because a single 1M evaluation
# needs ~246 registers, and 168 is the next occupancy step. This compiles the
# evaluation's constituent layers as separate kernels and reports registers for
# each, so we know which layer to attack -- without editing CloudMicrophysics.
#
# Needs no coupled simulation -- CMP.Microphysics1MParams(FT) constructs
# standalone -- so this runs in minutes rather than the ~45 an AMIP stage costs.
#
# Params are passed as kernel ARGUMENTS, not const globals. That matters: as
# const globals the compiler folds their fields and reports 101 registers for
# the full LinearizedAverage instead of 163, understating pressure by 62.
# CloudMicrophysics and Thermodynamics are transitive deps of the AMIP project,
# not direct ones, so reach them through ClimaAtmos, which imports both.
import CUDA
import ClimaAtmos
const CMP = ClimaAtmos.CMP
const BMT = ClimaAtmos.BMT
const TD = ClimaAtmos.TD

const FT = Float32
const MP = CMP.Microphysics1MParams(FT)
const TPS = TD.Parameters.ThermodynamicsParameters(FT)

# Representative cloudy state.
const ST = (ρ = FT(0.9), T = FT(275.0), q_tot = FT(8.0e-3),
            q_lcl = FT(3.0e-4), q_icl = FT(5.0e-5),
            q_rai = FT(2.0e-4), q_sno = FT(3.0e-5))
const DT = FT(30)

# One kernel per layer. `always_inline=true` matches how ClimaCore compiles.
layers = Any[]
# NOTE: params now arrive as kernel arguments (mp, tps), not const globals.
push!(layers, ("source terms only", (out, s, mp, tps) -> begin
    r = BMT._microphysics_source_terms(BMT.Microphysics1Moment(), mp, tps,
        s.ρ, s.T, s.q_tot, s.q_lcl, s.q_icl, s.q_rai, s.q_sno)
    out[1] = r.S_acnv_lcl_rai; nothing end))
push!(layers, ("+ aggregate (Instantaneous)", (out, s, mp, tps) -> begin
    r = BMT.bulk_microphysics_tendencies(BMT.Instantaneous(), BMT.Microphysics1Moment(),
        mp, tps, s.ρ, s.T, s.q_tot, s.q_lcl, s.q_icl, s.q_rai, s.q_sno)
    out[1] = r.dq_lcl_dt; nothing end))
push!(layers, ("one linearized implicit step", (out, s, mp, tps) -> begin
    r = BMT._linearized_implicit_step(BMT.Microphysics1Moment(), mp, tps,
        s.ρ, s.T, s.q_tot, s.q_lcl, s.q_icl, s.q_rai, s.q_sno, DT)
    out[1] = first(r); nothing end))
for nsub in (1, 2, 3)
    push!(layers, ("LinearizedAverage nsub=$nsub", (out, s, mp, tps) -> begin
        r = BMT.bulk_microphysics_tendencies(BMT.LinearizedAverage(), BMT.Microphysics1Moment(),
            mp, tps, s.ρ, s.T, s.q_tot, s.q_lcl, s.q_icl, s.q_rai, s.q_sno, DT, nsub)
        out[1] = r.dq_lcl_dt; nothing end))
end

# sm_80 occupancy steps: <=255 -> 8 warps/SM, <=168 -> 12, <=128 -> 16.
warps(r) = r <= 128 ? 16 : r <= 168 ? 12 : r <= 255 ? 8 : 0
out = CUDA.zeros(FT, 1)
println("\n=== registers, params passed as RUNTIME kernel arguments ===")
println(rpad("layer", 32), rpad("regs", 7), rpad("local B", 9), "warps/SM")
results = Dict{String, Any}()
for (nm, body) in layers
    f(o, s, mp, tps) = (body(o, s, mp, tps); nothing)
    k = CUDA.@cuda launch=false always_inline=true f(out, ST, MP, TPS)
    r = CUDA.registers(k); m = CUDA.memory(k)
    println(rpad(nm, 32), rpad(r, 7), rpad(m.local, 9), warps(r))
    results[nm] = Dict("registers" => r, "local_bytes" => m.local, "warps_per_sm" => warps(r))
end

# ---------------------------------------------------------------------------
# Can ptxas reach the 168-register occupancy step WITHOUT spilling, now that the
# fusion has removed the spill?
#
# This was measured before and the answer was no: ptxas reached 168 but paid
# +344 B of spill and ran 20.5% slower, so the launch-bounds guard rejected it
# and learnings.md §4 recorded occupancy as a dead end for this kernel. But that
# was measured on a kernel ALREADY spilling 21% at its natural register count --
# it had no slack. The fused kernel spills nothing (ncu 2026-09-03: 0 requests,
# 0% overhead), so the question is open again, and ncu now ranks occupancy as
# the single largest available win at 69% estimated speedup.
#
# Compile the full nine-point quadrature body both ways and compare. A
# `maxthreads` of 384 with one block per SM is the 12-warp target the
# launch-bounds mechanism asks for.
const QUAD = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3)
# corr and alpha are runtime scalars; their values do not affect register
# allocation, so use representative constants rather than building a full
# ClimaAtmosParameters (the docstring puts T-q correlation around 0.6).
const CORR = FT(0.6)
const ALPHA = FT(1)

quad_body(out, s, mp, tps, quad) = begin
    r = ClimaAtmos.microphysics_tendencies_1m(
        BMT.Microphysics1Moment(), quad, mp, tps,
        s.ρ, s.T, s.q_tot, s.q_lcl, s.q_icl, s.q_rai, s.q_sno,
        FT(1.0e-2), FT(1.0e-12), CORR, FT(1.0e-5), ALPHA, DT, 2,
    )
    out[1] = r.dq_lcl_dt
    nothing
end

println("\n=== can ptxas hit 12 warps/SM without spilling? ===")
println(rpad("compilation", 34), rpad("regs", 7), rpad("local B", 9), "warps/SM")
lb = Dict{String, Any}()
try
    unb = CUDA.@cuda launch=false always_inline=true quad_body(out, ST, MP, TPS, QUAD)
    ru, mu = CUDA.registers(unb), CUDA.memory(unb).local
    println(rpad("unbounded", 34), rpad(ru, 7), rpad(mu, 9), warps(ru))
    lb["unbounded"] = Dict("registers" => ru, "local_bytes" => mu, "warps_per_sm" => warps(ru))

    bnd = CUDA.@cuda launch=false always_inline=true maxthreads=384 blocks_per_sm=1 quad_body(
        out, ST, MP, TPS, QUAD)
    rb, mb = CUDA.registers(bnd), CUDA.memory(bnd).local
    println(rpad("__launch_bounds__ 12 warps/SM", 34), rpad(rb, 7), rpad(mb, 9), warps(rb))
    lb["bounded_12warps"] =
        Dict("registers" => rb, "local_bytes" => mb, "warps_per_sm" => warps(rb))
    lb["spill_growth_bytes"] = mb - mu
    println("\nspill growth asking for 12 warps/SM: $(mb - mu) bytes",
            mb - mu <= 256 ? "  (within the 256 B budget the guard allows)" :
                             "  (OVER the 256 B budget -- ptxas is buying occupancy with spill)")
catch err
    @warn "launch-bounds probe failed" exception = (err, catch_backtrace())
end
results["launch_bounds_probe"] = lb

# Record for the pipeline: these numbers are cited by the kernel-limits question.
# ---------------------------------------------------------------------------
# Second half: the ClimaCore side. A standalone CloudMicrophysics evaluation is
# only part of the hot kernel's register demand; the rest is the broadcast it
# runs inside. Build up through the layers on the real AMIP data layout so each
# delta names its own cost.
#
# Register counts come from ClimaCore's launch-bounds decision record, which
# stores the unbounded count for every kernel it compiles. That machinery only
# exists on the -mod branch, so this half is skipped elsewhere.
import ClimaComms
import ClimaCore
import ClimaCore.DataLayouts: VIJFH
ClimaComms.@import_required_backends

const Ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
mknt4(a, b, c, d) = (dq_lcl_dt = a, dq_icl_dt = b, dq_rai_dt = c, dq_sno_dt = d)

if isnothing(Ext) || !isdefined(Ext, :LAUNCH_BOUNDS_CACHE)
    @info "ClimaCore lacks the launch-bounds record; skipping the broadcast layers"
else
    AT = ClimaComms.array_type(ClimaComms.device()){FT}
    # AMIP he16: 6*16*16 = 1536 elements, 4x4 nodes, 63 levels.
    Nv, Nij, Nh = 63, 4, 1536
    sc() = VIJFH{FT, Nv, Nij, Nij, nothing}(AT, Nh)
    NTT = @NamedTuple{dq_lcl_dt::FT, dq_icl_dt::FT, dq_rai_dt::FT, dq_sno_dt::FT}
    ρf, qt, ql, qi, qr, qs, Tf = ntuple(_ -> sc(), 7)
    parent(ρf) .= FT(0.9); parent(Tf) .= FT(275); parent(qt) .= FT(8e-3)
    parent(ql) .= FT(3e-4); parent(qi) .= FT(5e-5)
    parent(qr) .= FT(2e-4); parent(qs) .= FT(3e-5)
    outs = sc()
    outnt = VIJFH{NTT, Nv, Nij, Nij, nothing}(AT, Nh)
    # Extra inputs the quadrature form takes: the two SGS variances and the
    # Lagrange multiplier.
    TTf, qqf, lamf = ntuple(_ -> sc(), 3)
    parent(TTf) .= FT(1.0e-2); parent(qqf) .= FT(1.0e-12); parent(lamf) .= FT(1.0e-5)

    bcast_layers = [
        ("A bare broadcast (3 in, 1 out)", () -> @. outs = ρf + qt * Tf),
        ("B 7 fields in, 1 out", () -> @. outs = ρf + qt + ql + qi + qr + qs + Tf),
        ("C 7 in, NamedTuple out",
         () -> @. outnt = mknt4(ρf + qt, ql + qi, qr + qs, Tf)),
        ("D microphysics_tendencies_1m",
         () -> @. outnt = ClimaAtmos.microphysics_tendencies_1m(
             ρf, qt, ql, qi, qr, qs, Tf, MP, TPS, DT, 2)),
        # E is the kernel that actually matters: the nine-point SGS-quadrature
        # form, broadcast over the same layout. D (the single-call form) matches
        # the real L970 exactly at 153 registers, so if E lands near D the
        # `@noinline` barrier on Microphysics1MEvaluator is holding and the
        # inflation is elsewhere; if it lands at 255 the barrier is being
        # defeated in the broadcast context, which is where L1013's remaining
        # register pressure would then be.
        ("E microphysics_tendencies_1m (9-pt quadrature)",
         () -> @. outnt = ClimaAtmos.microphysics_tendencies_1m(
             BMT.Microphysics1Moment(), QUAD, MP, TPS, ρf, Tf, qt, ql, qi, qr, qs,
             TTf, qqf, CORR, lamf, ALPHA, DT, 2)),
    ]
    println("\n=== ClimaCore broadcast layers, AMIP layout ===")
    println(rpad("layer", 34), rpad("regs", 7), "warps/SM")
    for (nm, f) in bcast_layers
        before = Set(keys(Ext.LAUNCH_BOUNDS_CACHE))
        f()
        CUDA.synchronize()
        fresh = [d for (k, d) in Ext.LAUNCH_BOUNDS_CACHE if !(k in before)]
        r = isempty(fresh) ? -1 : maximum(d.unbounded_regs for d in fresh)
        println(rpad(nm, 34), rpad(r, 7), r > 0 ? string(warps(r)) : "?")
        r > 0 && (results[nm] = Dict("registers" => r, "warps_per_sm" => warps(r)))
    end
end

# TOML, not JSON: TOML is stdlib and JSON is not a dependency of the AMIP
# project. Matches the other small evidence artifacts in results/.
import TOML
open(joinpath(@__DIR__, "..", "results", "cm-registers.toml"), "w") do io
    TOML.print(io, Dict("note" => "params passed as runtime kernel arguments; " *
                                  "as const globals the compiler folds their fields " *
                                  "and reports 101 instead of 163 for LinearizedAverage",
                        "layers" => results); sorted = true)
end
println("\nBISECT DONE")
