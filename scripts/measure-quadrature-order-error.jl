# What accuracy does reducing the SGS quadrature from 3x3 to 2x2 cost?
#
# The performance side is measured and tagged:
#   exp/2026-09-08-quadrature-order-2-isolated  +6.14% on upstream code
#   exp/2026-09-09-order2-full-stack            +8.05% with the numerics work
#
# Neither says anything about accuracy, and the adaptive-collapse measurement in
# results/adaptive-error.toml is a DIFFERENT change (3x3 -> 1 point, selectively)
# and must not be cited for this one.
#
# Method: build the model once, evaluate the microphysics cache with the shipped
# 3x3 rule, swap the quadrature for a 2x2 rule, evaluate again, and diff the
# tendency field. Errors are normalized by each tendency's own RMS over the whole
# field, so tendencies that are near zero everywhere cannot manufacture large
# relative errors.
#
# CAVEAT, and it bounds what this can claim: the Lagrange multiplier
# `λ_lagrange` in p.precomputed.ᶜsgs_moments was fitted under the 3x3 rule and is
# held fixed here. A real order-2 run refits it, so this measures the DIRECT
# effect of changing the rule, not the self-consistent one. The self-consistent
# error is what a science reviewer ultimately wants and needs two full runs
# compared field by field.

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
import CloudMicrophysics.BulkMicrophysicsTendencies as BMT
import TOML
using Printf

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

out_path = "results/quadrature-order-error.toml"
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
FT = eltype(Y.c.ρ)

flatten(f, nm) = vec(Array(parent(getproperty(f, nm))))
flat(f) = vec(Array(parent(f)))

# Run the real cache function first: it leaves the environment inputs it built
# in p.scratch, so the final broadcast can be repeated with a different rule on
# BYTE-IDENTICAL inputs. Reconstructing those inputs by hand is what this avoids
# -- rho0 is TD.air_density(...) and the condensates are specific_env_value
# reconstructions, easy to get subtly wrong. The scratch mapping is read
# straight from set_microphysics_tendency_cache! in microphysics_cache.jl.
CA.set_microphysics_tendency_cache!(
    Y, p, p.atmos.microphysics_model, p.atmos.turbconv_model,
)
ref = deepcopy(p.precomputed.ᶜmp_tendency⁰)

ᶜρ⁰      = p.scratch.ᶜtemp_scalar
ᶜq_lcl⁰  = p.scratch.ᶜtemp_scalar_2
ᶜq_icl⁰  = p.scratch.ᶜtemp_scalar_3
ᶜq_rai⁰  = p.scratch.ᶜtemp_scalar_4
ᶜq_sno⁰  = p.scratch.ᶜtemp_scalar_5
ᶜλ⁰      = p.scratch.ᶜtemp_scalar_6
ᶜmu_S⁰   = p.scratch.ᶜtemp_scalar_7
ᶜT⁰      = p.precomputed.ᶜT⁰
ᶜqt⁰     = p.precomputed.ᶜq_tot_nonneg⁰
ᶜT′T′    = p.precomputed.ᶜT′T′
ᶜq′q′    = p.precomputed.ᶜq′q′
λ_lag    = p.precomputed.ᶜsgs_moments.λ_lagrange

thp   = CAP.thermodynamics_params(p.params)
cmp   = CAP.microphysics_1m_params(p.params)
corr  = CA.correlation_Tq(p.params)
α     = CA.sgs_variance_fidelity(CAP.cloud_fraction_steepness_scale(p.params))
dt    = p.dt
nsubs = p.atmos.water.microphysics_model.n_substeps_quad
out   = similar(ref)

shipped_order = length(p.atmos.sgs_quadrature.a)
@info "shipped quadrature order" shipped_order

function evaluate_with(quad)
    @. out = CA.microphysics_tendencies_1m(
        BMT.Microphysics1Moment(), quad, cmp, thp, ᶜρ⁰, ᶜT⁰, ᶜqt⁰,
        ᶜq_lcl⁰, ᶜq_icl⁰, ᶜq_rai⁰, ᶜq_sno⁰, ᶜT′T′, ᶜq′q′, corr,
        λ_lag, α, dt, nsubs, ᶜλ⁰, ᶜmu_S⁰,
    )
    return deepcopy(out)
end

# Self-check: re-running the shipped rule through this path must reproduce the
# cache function bit for bit. If it does not, the scratch mapping is wrong and
# every number below is measured against the wrong inputs.
check = evaluate_with(p.atmos.sgs_quadrature)
names_ = propertynames(ref)
n = length(flatten(ref, first(names_)))
mismatch = sum(
    count(!=(0), flatten(check, nm) .- flatten(ref, nm)) for nm in names_
)
if mismatch != 0
    error("scratch-path self-check FAILED: $mismatch values differ from the " *
          "cache function; the scratch mapping is wrong.")
end
@info "scratch-path self-check passed (bit-identical to the cache function)"

results = Dict{String, Any}(
    "cells" => n,
    "reference_order" => shipped_order,
    "note" => "Direct effect of the SGS quadrature order on the 1M microphysics " *
        "tendencies, against the shipped rule, on a real AMIP state with " *
        "byte-identical inputs. Errors normalized by each tendency's own RMS " *
        "over the whole field. lambda_lagrange is held at its 3x3 fit, so this " *
        "is the direct effect, not the self-consistent one.",
)

for order in (2, 1)
    quad = CA.SGSQuadrature(eltype(Y.c.ρ); quadrature_order = order)
    @info "evaluating quadrature order $order"
    got = evaluate_with(quad)
    entry = Dict{String, Any}()
    changed = falses(n)
    for nm in names_
        a, b = flatten(ref, nm), flatten(got, nm)
        @inbounds for i in 1:n
            a[i] == b[i] || (changed[i] = true)
        end
    end
    nc = count(changed)
    entry["changed_frac"] = nc / n
    @printf("\norder %d vs %d: answer changed in %d of %d cells (%.2f%%)\n",
            order, shipped_order, nc, n, 100nc / n)
    for nm in names_
        a, b = flatten(ref, nm), flatten(got, nm)
        scale = sqrt(sum(abs2, a) / n)
        scale == 0 && continue
        err = Float64[abs(a[i] - b[i]) for i in 1:n]
        sort!(err)
        rms = sqrt(sum(abs2, err) / n) / scale
        p99 = err[floor(Int, 0.99n)] / scale
        mx = err[end] / scale
        @printf("  %-12s rms/scale %9.3e   p99/scale %9.3e   max/scale %9.3e\n",
                String(nm), rms, p99, mx)
        entry[String(nm)] = Dict("rms_over_scale" => rms,
            "p99_over_scale" => p99, "max_over_scale" => mx, "field_rms" => scale)
    end
    results["order_$order"] = entry
end

open(out_path, "w") do io
    TOML.print(io, results)
end
@info "wrote $out_path"
