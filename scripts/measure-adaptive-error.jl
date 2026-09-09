# What does collapsing the SGS quadrature where the PDF is clear of the
# saturation kink actually cost in accuracy?
#
# `measure-sgs-degeneracy.jl` established that the PDF sits a median 449 sigma
# from the kink, and that at 10 sigma 83.2% of 32-lane warps are entirely clear
# of it -- so an adaptive collapse could pay. That says nothing about whether
# the collapsed answer is right.
#
# Collapsing is an approximation, not an identity: clear of the kink the
# integrand is smooth but still varies across the PDF, so the error is
# O(sigma^2 f''). This measures it on the REAL state rather than on synthetic
# inputs. A first attempt using randomized states fired the branch in only 4% of
# cases -- its synthetic variances were far larger than the real field's, so it
# sampled only the marginal corner and reported a 194% relative error on
# tendencies of order 1e-8, which is a small-denominator artefact rather than a
# result.
#
# Reports the error distribution over every cell where the branch fires,
# normalized by the tendency's own scale rather than pointwise, so near-zero
# tendencies cannot dominate.

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import Thermodynamics as TD
import TOML
using Printf

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

out_path = "results/adaptive-error.toml"
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i)
        out_path = ARGS[i + 1]
        deleteat!(ARGS, i:(i + 1))
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
function pick(nt, names...)
    for nm in names
        hasproperty(nt, nm) && return getproperty(nt, nm)
    end
    error("none of $(names) in $(propertynames(nt))")
end

TT = flat(pick(p.precomputed, :ᶜT′T′))
qq = flat(pick(p.precomputed, :ᶜq′q′))
Tm = flat(pick(p.precomputed, :ᶜT⁰, :ᶜT))
qt = flat(pick(p.precomputed, :ᶜq_tot_nonneg⁰, :ᶜq_tot_nonneg))
ql = flat(pick(p.precomputed, :ᶜq_lcl⁰, :ᶜq_lcl))
qi = flat(pick(p.precomputed, :ᶜq_icl⁰, :ᶜq_icl))
qr = flat(pick(p.precomputed, :ᶜq_rai⁰, :ᶜq_rai))
qs = flat(pick(p.precomputed, :ᶜq_sno⁰, :ᶜq_sno))
lam = flat(p.precomputed.ᶜsgs_moments.λ_lagrange)
ρ = flat(Y.c.ρ)
n = length(TT)

FT = eltype(TT)
thp = CAP.thermodynamics_params(p.params)
cmp = CAP.microphysics_1m_params(p.params)
corr = FT(CA.correlation_Tq(p.params))
α = FT(CA.sgs_variance_fidelity(CAP.cloud_fraction_steepness_scale(p.params)))
dt = FT(float(integrator.dt))
nsubs = p.atmos.microphysics_n_substeps_quadrature
quad = p.atmos.water.sgs_quad

FIELDS = (:dq_lcl_dt, :dq_icl_dt, :dq_rai_dt, :dq_sno_dt)

function tendencies(k)
    CA.ADAPTIVE_QUADRATURE_SIGMA[] = k
    out = [Vector{FT}(undef, n) for _ in FIELDS]
    @inbounds for i in 1:n
        r = CA.microphysics_tendencies_1m(
            BMT.Microphysics1Moment(), quad, cmp, thp, ρ[i], Tm[i], qt[i],
            ql[i], qi[i], qr[i], qs[i], TT[i], qq[i], corr, lam[i], α, dt, nsubs,
        )
        for (j, f) in enumerate(FIELDS)
            out[j][i] = getproperty(r, f)
        end
    end
    return out
end

@info "evaluating $n cells with the full quadrature"
full = tendencies(0.0)

results = Dict{String, Any}("cells" => n)
for k in (3.0, 10.0)
    @info "evaluating with the adaptive collapse at $(k) sigma"
    adapt = tendencies(k)
    fired = falses(n)
    for j in eachindex(FIELDS), i in 1:n
        full[j][i] == adapt[j][i] || (fired[i] = true)
    end
    nf = count(fired)
    entry = Dict{String, Any}("fired_frac" => nf / n)
    @printf("\nk = %.1f sigma: branch changed %d of %d cells (%.2f%%)\n",
            k, nf, n, 100nf / n)
    for (j, f) in enumerate(FIELDS)
        # Normalize by the RMS of the field itself, so a tendency that is
        # near-zero everywhere cannot manufacture a huge relative error.
        scale = sqrt(sum(abs2, full[j]) / n)
        err = [abs(full[j][i] - adapt[j][i]) for i in 1:n if fired[i]]
        if isempty(err) || scale == 0
            continue
        end
        sort!(err)
        rel_max = err[end] / scale
        rel_p99 = err[max(1, floor(Int, 0.99 * length(err)))] / scale
        rel_rms = sqrt(sum(abs2, err) / length(err)) / scale
        @printf("  %-12s rms/scale %9.3e   p99/scale %9.3e   max/scale %9.3e\n",
                String(f), rel_rms, rel_p99, rel_max)
        entry[String(f)] = Dict(
            "rms_over_scale" => rel_rms,
            "p99_over_scale" => rel_p99,
            "max_over_scale" => rel_max,
            "field_rms" => scale,
        )
    end
    results["sigma_$(Int(k))"] = entry
end
CA.ADAPTIVE_QUADRATURE_SIGMA[] = 10.0

results["note"] = "Error from collapsing the SGS quadrature to its centre node " *
    "where |mu_S| > k*sigma_S, measured against the full nine-point rule on a " *
    "real AMIP state. Errors are normalized by each tendency's own RMS over " *
    "the whole field, not pointwise, so near-zero tendencies cannot dominate."
open(out_path, "w") do io
    TOML.print(io, results)
end
@info "wrote $out_path"
