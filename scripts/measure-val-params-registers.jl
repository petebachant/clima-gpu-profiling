# Does Val-wrapping the microphysics parameters buy what const globals buy?
#
# measure-cm-registers.jl records the gap this asks about: the same kernel needs
# 246 registers with the parameters arriving as runtime arguments and 101 when
# they are const globals the compiler can fold. 101 clears two occupancy steps.
#
# Const globals are not available to the model, because the parameters come from
# configuration at run time. Val is, since Julia accepts an isbits struct as a
# type parameter, and it folds for the same reason: the values are in the type,
# so the kernel body sees literals.
#
# It would also need no CloudMicrophysics change, which is what blocks every
# other route into this kernel -- neither arm devs CM. ClimaAtmos owns the
# broadcast, so it can wrap on its own side and let inlining fold the constants
# into CM's code.
#
# Three compilations of one body, differing only in how the parameters arrive.
# Compile-only, no simulation, and no coupled model: CMP.Microphysics1MParams
# constructs standalone.

import CUDA
import ClimaAtmos
import TOML
using Printf

const CMP = ClimaAtmos.CMP
const BMT = ClimaAtmos.BMT
const TD = ClimaAtmos.TD

const FT = Float32
const MP = CMP.Microphysics1MParams(FT)
const TPS = TD.Parameters.ThermodynamicsParameters(FT)
const ST = (ρ = FT(0.9), T = FT(275.0), w = FT(1.0), q_tot = FT(8.0e-3),
            q_lcl = FT(1.0e-4), q_icl = FT(1.0e-5),
            q_rai = FT(1.0e-5), q_sno = FT(1.0e-6))
const DT = FT(30)

out_path = "results/val-params-registers.toml"
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i)
        out_path = ARGS[i + 1]
        deleteat!(ARGS, i:(i + 1))
    end
end

# Val needs an isbits value. If either parameter set nests an array this route
# is closed outright, so it is the first thing reported rather than assumed.
results = Dict{String, Any}(
    "mp_isbits" => isbits(MP),
    "tps_isbits" => isbits(TPS),
    "mp_sizeof_bytes" => sizeof(MP),
    "tps_sizeof_bytes" => sizeof(TPS),
    "note" => "registers for one microphysics body compiled three ways: " *
              "parameters as runtime arguments, as const globals, and via " *
              "Val. The const-global column is the target Val is trying to " *
              "reach; runtime arguments are what the model does today.",
)
@printf("MP isbits=%s (%d B)   TPS isbits=%s (%d B)\n",
        isbits(MP), sizeof(MP), isbits(TPS), sizeof(TPS))
if !isbits(MP) || !isbits(TPS)
    @warn "a parameter set is not isbits, so Val cannot wrap it; recording and stopping"
    open(out_path, "w") do io
        TOML.print(io, results; sorted = true)
    end
    exit(0)
end

# sm_80, 128-thread blocks: <=128 -> 16 warps/SM, <=168 -> 12, <=255 -> 8.
warps(r) = r <= 128 ? 16 : r <= 168 ? 12 : r <= 255 ? 8 : 0
out = CUDA.zeros(FT, 1)

# The body the three variants share, written once so the comparison is only
# about how the parameters arrive.
@inline function body(o, s, mp, tps, nsub)
    r = BMT.bulk_microphysics_tendencies(
        BMT.LinearizedAverage(), BMT.Microphysics1Moment(), mp, tps,
        s.ρ, s.T, s.w, s.q_tot, s.q_lcl, s.q_icl, s.q_rai, s.q_sno, DT, nsub,
    )
    o[1] = r.dq_lcl_dt
    return nothing
end

# (a) what the model does now
k_arg(o, s, mp, tps) = body(o, s, mp, tps, 3)
# (b) the target: folded from const globals
k_const(o, s) = body(o, s, MP, TPS, 3)
# (c) the proposal: folded from type parameters
k_val(o, s, ::Val{M}, ::Val{T}) where {M, T} = body(o, s, M, T, 3)

function record!(label, k)
    r = CUDA.registers(k)
    m = CUDA.memory(k)
    results[label] = Dict(
        "registers" => r,
        "local_bytes" => m.local,
        "warps_per_sm" => warps(r),
    )
    @printf("%-26s regs=%3d  local=%5d B  warps/SM=%2d\n", label, r, m.local, warps(r))
    return r
end

println("\n=== how the parameters arrive ===")
r_arg = record!("runtime_arguments",
                CUDA.@cuda launch = false always_inline = true k_arg(out, ST, MP, TPS))
r_const = record!("const_globals",
                  CUDA.@cuda launch = false always_inline = true k_const(out, ST))
r_val = record!("val_type_parameters",
                CUDA.@cuda launch = false always_inline = true k_val(
                    out, ST, Val(MP), Val(TPS)))

results["val_matches_const"] = r_val <= r_const
results["val_saving_vs_arguments"] = r_arg - r_val
results["const_saving_vs_arguments"] = r_arg - r_const
results["val_gains_occupancy_step"] = warps(r_val) > warps(r_arg)
@printf(
    "\nVal saves %d registers against runtime arguments (const globals save %d).\n",
    r_arg - r_val, r_arg - r_const,
)
@printf(
    "warps/SM: arguments %d -> Val %d%s\n",
    warps(r_arg), warps(r_val),
    warps(r_val) > warps(r_arg) ? "  (gains an occupancy step)" : "  (no step gained)",
)

# --- the path the model actually runs -----------------------------------
# microphysics_tendencies_1m with a quadrature, which builds a
# Microphysics1MEvaluator holding cmp and thp and loops the SGS points. All of
# it is ClimaAtmos code, so a change here needs no CloudMicrophysics release.
const QUAD = ClimaAtmos.SGSQuadrature(FT; quadrature_order = 3)
const AUX = (T2 = FT(0.5), q2 = FT(1.0e-8), corr = FT(0.3),
             lam = FT(0.5), alpha = FT(1.0), nsubs = 3,
             xi_liq = FT(0.0), xi_ice = FT(1.0))

@inline function quad_body(o, s, mp, tps)
    r = ClimaAtmos.microphysics_tendencies_1m(
        BMT.Microphysics1Moment(), QUAD, mp, tps,
        s.ρ, s.T, s.w, s.q_tot, s.q_lcl, s.q_icl, s.q_rai, s.q_sno,
        AUX.T2, AUX.q2, AUX.corr, AUX.lam, AUX.alpha,
        AUX.xi_liq, AUX.xi_ice, DT, AUX.nsubs,
    )
    o[1] = r.dq_lcl_dt
    return nothing
end

q_arg(o, s, mp, tps) = quad_body(o, s, mp, tps)
q_const(o, s) = quad_body(o, s, MP, TPS)
q_val(o, s, ::Val{M}, ::Val{T}) where {M, T} = quad_body(o, s, M, T)

println("\n=== the quadrature path (9 points), as the model calls it ===")
qr_arg = record!("quad_runtime_arguments",
                 CUDA.@cuda launch = false always_inline = true q_arg(out, ST, MP, TPS))
qr_const = record!("quad_const_globals",
                   CUDA.@cuda launch = false always_inline = true q_const(out, ST))
qr_val = record!("quad_val_type_parameters",
                 CUDA.@cuda launch = false always_inline = true q_val(
                     out, ST, Val(MP), Val(TPS)))

results["quad_val_matches_const"] = qr_val <= qr_const
results["quad_val_saving_vs_arguments"] = qr_arg - qr_val
results["quad_val_gains_occupancy_step"] = warps(qr_val) > warps(qr_arg)
@printf(
    "\nquadrature: Val saves %d registers (const globals %d); warps/SM %d -> %d%s\n",
    qr_arg - qr_val, qr_arg - qr_const, warps(qr_arg), warps(qr_val),
    warps(qr_val) > warps(qr_arg) ? "  (gains a step)" : "  (no step)",
)

# --- is the 255 an unroll, rather than a payload? -------------------------
# The quadrature's point count is a type parameter, so the nine evaluations can
# inline into one body and each one's live values compete for the same
# registers. If that is what pins the kernel at 255, neither folding the
# parameters nor splitting the loop per point addresses it -- an inlining
# barrier does, which is what cut this kernel 54% once before.
#
# Compiling the same body with always_inline off is the cheapest way to ask.
println("\n=== the same quadrature body, without forced inlining ===")
qn_arg = record!("quad_noinline_arguments",
                 CUDA.@cuda launch = false q_arg(out, ST, MP, TPS))
qn_val = record!("quad_noinline_val",
                 CUDA.@cuda launch = false q_val(out, ST, Val(MP), Val(TPS)))

results["quad_inlining_cost_registers"] = qr_arg - qn_arg
results["quad_noinline_gains_step"] = warps(qn_arg) > warps(qr_arg)
results["quad_noinline_val_gains_step"] = warps(qn_val) > warps(qr_arg)
@printf(
    "\nforced inlining costs %d registers (%d -> %d); warps/SM %d -> %d, and %d folded\n",
    qr_arg - qn_arg, qr_arg, qn_arg, warps(qr_arg), warps(qn_arg), warps(qn_val),
)

# --- what actually defeats the fold ---------------------------------------
# sum_over_quadrature_points already loops dynamically rather than unrolling,
# and says so: each iteration releases the previous one's registers. So the 255
# is one evaluation plus accumulators, not nine evaluations.
#
# Which leaves a candidate the earlier variants could not separate.
# microphysics_tendencies_1m folds its parameters into a Microphysics1MEvaluator
# as FIELDS, so a constant handed to it is stored in a struct and read back at
# run time -- the fold is erased at the boundary, whatever the caller knew.
#
# A and B differ only in where the parameters live. Both run the real
# integrate_over_sgs over the real quadrature.
struct FieldEval{P, T, FT}
    mp::P
    tps::T
    ρ::FT
    w::FT
    q_lcl::FT
    q_icl::FT
    q_rai::FT
    q_sno::FT
    dt::FT
    nsub::Int
end
@inline (e::FieldEval)(T_hat, q_hat) = BMT.bulk_microphysics_tendencies(
    BMT.LinearizedAverage(), BMT.Microphysics1Moment(), e.mp, e.tps,
    e.ρ, T_hat, e.w, q_hat, e.q_lcl, e.q_icl, e.q_rai, e.q_sno, e.dt, e.nsub,
)

struct TypeEval{M, TP, FT}
    ρ::FT
    w::FT
    q_lcl::FT
    q_icl::FT
    q_rai::FT
    q_sno::FT
    dt::FT
    nsub::Int
end
@inline (e::TypeEval{M, TP})(T_hat, q_hat) where {M, TP} =
    BMT.bulk_microphysics_tendencies(
        BMT.LinearizedAverage(), BMT.Microphysics1Moment(), M, TP,
        e.ρ, T_hat, e.w, q_hat, e.q_lcl, e.q_icl, e.q_rai, e.q_sno, e.dt, e.nsub,
    )

const EV_ARGS = (ST.ρ, ST.w, ST.q_lcl, ST.q_icl, ST.q_rai, ST.q_sno, DT, 3)
# mp and tps arrive as KERNEL ARGUMENTS here. Reading them from the const
# globals instead would let the compiler fold them in this variant too, which
# is how the first version of this comparison came out 93 against 93 and
# measured nothing.
f_field(o, s, mp, tps) = begin
    r = ClimaAtmos.integrate_over_sgs(
        FieldEval(mp, tps, EV_ARGS...), QUAD,
        s.q_tot, s.T, AUX.q2, AUX.T2, AUX.corr,
    )
    o[1] = r.dq_lcl_dt
    nothing
end
f_type(o, s) = begin
    r = ClimaAtmos.integrate_over_sgs(
        TypeEval{MP, TPS, FT}(EV_ARGS...), QUAD,
        s.q_tot, s.T, AUX.q2, AUX.T2, AUX.corr,
    )
    o[1] = r.dq_lcl_dt
    nothing
end

# The variant that matches what the model now does: the Val reaches
# microphysics_tendencies_1m still wrapped, so it is stored in the evaluator as a
# zero-size singleton and the parameters fold. q_val above unwraps it first,
# which puts a bare constant into the field and erases the fold -- the two look
# almost identical and measure opposite things.
q_wrapped(o, s, vm::Val, vt::Val) = quad_body(o, s, vm, vt)

println("\n=== the quadrature path with Val carried INTO the wrapper ===")
qw = record!("quad_val_carried_in",
             CUDA.@cuda launch = false always_inline = true q_wrapped(
                 out, ST, Val(MP), Val(TPS)))
results["quad_val_carried_saving"] = qr_arg - qw
results["quad_val_carried_gains_step"] = warps(qw) > warps(qr_arg)
@printf(
    "\ncarrying the Val in saves %d registers (%d -> %d); warps/SM %d -> %d%s\n",
    qr_arg - qw, qr_arg, qw, warps(qr_arg), warps(qw),
    warps(qw) > warps(qr_arg) ? "  (gains a step)" : "  (no step)",
)

println("\n=== where the parameters live, through the real quadrature loop ===")
fe = record!("evaluator_params_as_fields",
             CUDA.@cuda launch = false always_inline = true f_field(out, ST, MP, TPS))
te = record!("evaluator_params_in_type",
             CUDA.@cuda launch = false always_inline = true f_type(out, ST))
results["evaluator_fold_saving"] = fe - te
results["evaluator_fold_gains_step"] = warps(te) > warps(fe)
@printf(
    "\nmoving the parameters out of the evaluator's fields saves %d registers (%d -> %d); warps/SM %d -> %d%s\n",
    fe - te, fe, te, warps(fe), warps(te),
    warps(te) > warps(fe) ? "  (gains a step)" : "  (no step)",
)

open(out_path, "w") do io
    TOML.print(io, results; sorted = true)
end
@info "wrote $out_path"
