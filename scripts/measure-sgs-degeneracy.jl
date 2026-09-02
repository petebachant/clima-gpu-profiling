# Is the SGS quadrature doing work, and is it doing it in warp-sized clumps?
#
# The nine-point quadrature exists to resolve the `max(0, λ_lagrange + α·S′)`
# kink at saturation, so it earns its cost only where the SGS PDF straddles that
# kink. The criterion that matters is therefore the distance from saturation
# measured in units of the spread of the saturation excess -- |mu_S| vs σ_S --
# not σ_q or σ_T against an absolute constant, which is dimensionally arbitrary
# and was the mistake in the first version of this script.
#
# A per-point criterion also only pays when it fires for a whole warp: a warp
# covers 32 grid points and costs the maximum over its lanes, so a criterion
# true at 50% of scattered points saves nothing (0.5^32 ≈ 0). learnings.md §4
# records a 77.7% point-fraction returning 21.5% for exactly this reason. Center
# fields are VIJFH with the vertical varying fastest and the hot kernel
# grid-strides over the layout's linear index, so 32 consecutive lanes are 32
# consecutive levels in a column -- which is where any clustering must come from.

import CUDA
import ClimaCore as CC
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import Thermodynamics as TD
import TOML
using Printf

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

out_path = "results/sgs-degeneracy.toml"
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i)
        out_path = ARGS[i + 1]
        deleteat!(ARGS, i:(i + 1))   # keep ArgParse from choking on it
    end
end

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)

for i in 1:3
    @info "warmup step $i / 3"
    step!(cs)
end

integrator = cs.model_sims.atmos_sim.integrator
p = integrator.p
Y = integrator.u

flat(f) = vec(Array(parent(f)))
# Pick the first name that exists, so this cannot silently read a stale scratch
# buffer or die on a rename.
function pick(nt, names...)
    for nm in names
        hasproperty(nt, nm) && return (getproperty(nt, nm), nm)
    end
    error("none of $(names) in $(propertynames(nt))")
end

ᶜT′T′, _ = pick(p.precomputed, :ᶜT′T′)
ᶜq′q′, _ = pick(p.precomputed, :ᶜq′q′)
ᶜT, nT = pick(p.precomputed, :ᶜT⁰, :ᶜT)
ᶜqt, nq = pick(p.precomputed, :ᶜq_tot_nonneg⁰, :ᶜq_tot_nonneg)
@info "using fields" T = nT q_tot = nq

TT, qq, Tm, qt = flat(ᶜT′T′), flat(ᶜq′q′), flat(ᶜT), flat(ᶜqt)
# Grid-mean density: prognostic, so stable, unlike the scratch-held ᶜρ⁰ which is
# reused within the step. The environment density differs by the updraft area
# fraction, far below the decade-scale separations this asks about.
ρ = flat(Y.c.ρ)
n = length(TT)
@assert length(qq) == length(Tm) == length(qt) == length(ρ) == n

FT = eltype(TT)
σ_T = sqrt.(max.(FT(0), TT))
σ_q = sqrt.(max.(FT(0), qq))

function warp_fraction(mask::AbstractVector{Bool}, warp = 32)
    nw = length(mask) ÷ warp
    full = 0
    @inbounds for w in 0:(nw - 1)
        ok = true
        for k in 1:warp
            mask[w * warp + k] || (ok = false; break)
        end
        full += ok
    end
    return full / nw
end

thp = CAP.thermodynamics_params(p.params)
corr = FT(CA.correlation_Tq(p.params))
Rv = FT(TD.Parameters.R_v(thp))
Lv = FT(TD.Parameters.LH_v0(thp))

q_sat = similar(Tm)
dqdT = similar(Tm)
@inbounds for i in eachindex(Tm)
    qs = FT(TD.q_vap_saturation(thp, Tm[i], ρ[i]))
    q_sat[i] = qs
    dqdT[i] = qs * Lv / (Rv * Tm[i]^2)   # Clausius-Clapeyron
end

mu_S = qt .- q_sat
# σ_S² = σ_q² + (∂q_sat/∂T)²σ_T² − 2·corr·σ_q·σ_T·(∂q_sat/∂T)
σ_S = sqrt.(
    max.(
        FT(0),
        σ_q .^ 2 .+ (dqdT .* σ_T) .^ 2 .- 2 .* corr .* σ_q .* σ_T .* dqdT,
    ),
)
ratio = abs.(mu_S) ./ max.(σ_S, floatmin(FT))

println("\ncells = $n   (warps of 32: $(n ÷ 32))")
@printf("σ_T    median %.4g   max %.4g\n", sort(σ_T)[n ÷ 2], maximum(σ_T))
@printf("σ_q    median %.4g   max %.4g\n", sort(σ_q)[n ÷ 2], maximum(σ_q))
@printf("σ_S    median %.4g   max %.4g\n", sort(σ_S)[n ÷ 2], maximum(σ_S))
@printf("|mu_S| median %.4g\n", sort(abs.(mu_S))[n ÷ 2])

srt = sort(ratio)
println("\n|mu_S| / σ_S percentiles (how far the PDF sits from the kink):")
for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95)
    @printf("   p%-3d %12.4g\n", round(Int, 100q), srt[max(1, round(Int, q * n))])
end

println("\n" * "-"^70)
@printf("%-40s %12s %12s\n", "criterion", "point frac", "warp frac")
println("-"^70)
m = (σ_T .== 0) .& (σ_q .== 0)
@printf(
    "%-40s %11.2f%% %11.2f%%\n", "EXACT  σ_T == 0 && σ_q == 0",
    100count(m) / n, 100warp_fraction(m)
)
for k in (FT(2), FT(3), FT(5), FT(10))
    m = ratio .> k
    @printf(
        "%-40s %11.2f%% %11.2f%%\n", "|mu_S| > $(Int(k))·σ_S   (clear of the kink)",
        100count(m) / n, 100warp_fraction(m)
    )
end
println("-"^70)
println("\nA warp fraction near the point fraction means the criterion clusters")
println("vertically and a collapse would pay; a warp fraction near zero means it")
println("is scattered and would not, however large the point fraction.")

# Write the result as a declared stage output, so the finding is versioned and
# regenerable rather than living only in a scheduler log.
result = Dict{String, Any}(
    "note" => "SGS PDF distance from the saturation kink, in units of the " *
              "saturation-excess spread. warp_frac is the fraction of 32-lane " *
              "warps in which EVERY lane satisfies the criterion, which is what " *
              "decides whether a collapse pays; see docs/learnings.md 6a.",
    "cells" => n,
    "warps" => n ÷ 32,
    "sigma" => Dict{String, Any}(
        "T_median" => sort(σ_T)[n ÷ 2], "T_max" => maximum(σ_T),
        "q_median" => sort(σ_q)[n ÷ 2], "q_max" => maximum(σ_q),
        "S_median" => sort(σ_S)[n ÷ 2], "S_max" => maximum(σ_S),
        "abs_mu_S_median" => sort(abs.(mu_S))[n ÷ 2],
    ),
    "mu_over_sigma_percentiles" => Dict{String, Any}(
        "p$(round(Int, 100q))" => srt[max(1, round(Int, q * n))]
        for q in (0.01, 0.05, 0.25, 0.50, 0.75, 0.95)
    ),
)
crit = Dict{String, Any}()
m = (σ_T .== 0) .& (σ_q .== 0)
crit["exact_sigma_zero"] =
    Dict("point_frac" => count(m) / n, "warp_frac" => warp_fraction(m))
for k in (FT(2), FT(3), FT(5), FT(10))
    m = ratio .> k
    crit["clear_of_kink_$(Int(k))sigma"] =
        Dict("point_frac" => count(m) / n, "warp_frac" => warp_fraction(m))
end
result["criteria"] = crit
mkpath(dirname(out_path))
open(out_path, "w") do io
    TOML.print(io, result; sorted = true)
end
@info "wrote $out_path"
