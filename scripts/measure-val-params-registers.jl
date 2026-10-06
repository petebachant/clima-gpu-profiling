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
const ST = (ρ = FT(0.9), T = FT(275.0), q_tot = FT(8.0e-3),
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
        s.ρ, s.T, s.q_tot, s.q_lcl, s.q_icl, s.q_rai, s.q_sno, DT, nsub,
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

open(out_path, "w") do io
    TOML.print(io, results; sorted = true)
end
@info "wrote $out_path"
