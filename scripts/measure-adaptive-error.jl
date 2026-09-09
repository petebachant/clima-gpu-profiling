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

# Rather than reconstructing the kernel's inputs -- which are scratch fields
# built inside the cache function (environment density, the specific_env_value
# reconstructions) and easy to get subtly wrong -- call the real cache function
# and diff the field it writes. This exercises the exact broadcast the model
# runs, on the GPU, through ClimaCore.
mp_model = p.atmos.microphysics_model
tm = p.atmos.turbconv_model
tend() = p.precomputed.ᶜmp_tendency⁰

function evaluate(k)
    CA.ADAPTIVE_QUADRATURE_SIGMA[] = k
    CA.set_microphysics_tendency_cache!(Y, p, mp_model, tm)
    return deepcopy(tend())
end

# A threshold of zero disables the branch (guarded in microphysics_wrappers.jl).
# Verified below rather than trusted: if the reference arm were itself collapsing
# cells, every error number here would be measured against the wrong baseline --
# which is exactly what happened on the first attempt.
@info "evaluating the full nine-point quadrature (branch disabled)"
full = evaluate(0.0)
flatten(f, name) = vec(Array(parent(getproperty(f, name))))
names_ = propertynames(full)
@info "tendency fields" names_
n = length(flatten(full, first(names_)))

results = Dict{String, Any}("cells" => n)
for k in (3.0, 10.0)
    @info "evaluating with the adaptive collapse at $(k) sigma"
    adapt = evaluate(k)
    entry = Dict{String, Any}()
    fired = falses(n)
    for nm in names_
        a, b = flatten(full, nm), flatten(adapt, nm)
        @inbounds for i in 1:n
            a[i] == b[i] || (fired[i] = true)
        end
    end
    nf = count(fired)
    entry["changed_frac"] = nf / n
    # Sanity: a STRICTER threshold must collapse fewer cells, so it must change
    # fewer. If this ordering inverts, the reference arm is wrong.
    entry["note"] = "changed_frac is where the answer differs, not where the " *
        "branch fired; most collapses are bit-identical because the quadrature " *
        "really is degenerate there."
    @printf("\nk = %.1f sigma: answer changed in %d of %d cells (%.3f%%)\n",
            k, nf, n, 100nf / n)
    for nm in names_
        a, b = flatten(full, nm), flatten(adapt, nm)
        scale = sqrt(sum(abs2, a) / n)
        scale == 0 && continue
        err = Float64[abs(a[i] - b[i]) for i in 1:n if fired[i]]
        isempty(err) && continue
        sort!(err)
        rel_rms = sqrt(sum(abs2, err) / length(err)) / scale
        rel_p99 = err[max(1, floor(Int, 0.99 * length(err)))] / scale
        rel_max = err[end] / scale
        @printf("  %-12s rms/scale %9.3e   p99/scale %9.3e   max/scale %9.3e\n",
                String(nm), rel_rms, rel_p99, rel_max)
        entry[String(nm)] = Dict(
            "rms_over_scale" => rel_rms, "p99_over_scale" => rel_p99,
            "max_over_scale" => rel_max, "field_rms" => scale,
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
