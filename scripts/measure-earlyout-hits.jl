# Why did the clear-air early-out decay from -21.5% to -3.27%?
#
# The guard skips a quadrature point only when it is provably zero: subsaturated,
# no cloud condensate, no precipitation. It fired at 77.7% of points on
# 2026-08-23 and is worth only -3.27% now, with the kernel essentially the same
# size (3425.9 ms at the September tag, 3434.8 ms now). So the saving did not
# shrink because the kernel got cheaper -- far fewer points qualify.
#
# The suspect is upstream ClimaAtmos #4779, which changed how `λ_lagrange` is
# fitted. The guard requires q_lcl_hat and q_icl_hat to be EXACTLY zero. If the
# refitted closure leaves them tiny but nonzero, the guard stops firing at points
# whose tendencies are still negligible.
#
# That distinction decides what to do next, so this separates the two cases:
#
#   exact_zero_condensate   guard is correct and simply has less to skip
#   tiny_condensate         a threshold early-out would recover the gain, but it
#                           is an approximation and needs science review
#
# Also reports how often the full CloudMicrophysics call returns all-zero
# tendencies anyway, which bounds what ANY skip could ever be worth.

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import ClimaAtmos.Parameters as CAP
# Reached through ClimaAtmos rather than imported directly: CloudMicrophysics
# is no longer dev'd by either arm, so it is in the manifest but not a direct
# dependency of the AMIP project and cannot be imported by name.
const BMT = CA.BMT
const TD = CA.TD
import TOML
using Printf

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

out_path = "results/earlyout-hits.toml"
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

# Populate p.scratch with the environment inputs, exactly as the kernel sees
# them. Reconstructing these by hand is the easy way to measure the wrong thing.
CA.set_microphysics_tendency_cache!(
    Y, p, p.atmos.microphysics_model, p.atmos.turbconv_model,
)

flat(f) = vec(Array(parent(f)))
ρ⁰     = flat(p.scratch.ᶜtemp_scalar)
q_lcl⁰ = flat(p.scratch.ᶜtemp_scalar_2)
q_icl⁰ = flat(p.scratch.ᶜtemp_scalar_3)
q_rai⁰ = flat(p.scratch.ᶜtemp_scalar_4)
q_sno⁰ = flat(p.scratch.ᶜtemp_scalar_5)
λ⁰     = flat(p.scratch.ᶜtemp_scalar_6)
mu_S⁰  = flat(p.scratch.ᶜtemp_scalar_7)
T⁰     = flat(p.precomputed.ᶜT⁰)
qt⁰    = flat(p.precomputed.ᶜq_tot_nonneg⁰)
T′T′   = flat(p.precomputed.ᶜT′T′)
q′q′   = flat(p.precomputed.ᶜq′q′)
λ_lag  = flat(p.precomputed.ᶜsgs_moments.λ_lagrange)

thp   = CAP.thermodynamics_params(p.params)
cmp   = CAP.microphysics_1m_params(p.params)
corr  = CA.correlation_Tq(p.params)
α     = CA.sgs_variance_fidelity(CAP.cloud_fraction_steepness_scale(p.params))
dt    = p.dt
nsubs = p.atmos.water.microphysics_model.n_substeps_quad
quad  = p.atmos.sgs_quadrature

ncells = length(ρ⁰)

# Wrapped in a function, not run at top level: `npts += 1` inside a top-level
# `for` creates a new local and the loop dies on the first iteration.
function tally(
    ncells, ρ⁰, T⁰, qt⁰, q_rai⁰, q_sno⁰, λ⁰, mu_S⁰, λ_lag, T′T′, q′q′,
    quad, thp, cmp, corr, α, dt, nsubs, ::Type{FT},
) where {FT}
    npts = 0
    guard_fires = 0          # all five conditions: what the early-out skips
    subsat_noprecip = 0      # subsaturated and precipitation-free
    exact_zero = 0           #   ... of those, condensate exactly zero
    tiny_nonzero = 0         #   ... condensate in (0, 1e-10]
    larger = 0               #   ... condensate > 1e-10
    zero_tendency = 0        # CM returns all-zero tendencies regardless
    max_tiny = zero(FT)

    for c in 1:ncells
        ρ, T, qt = ρ⁰[c], T⁰[c], qt⁰[c]
        q_rai, q_sno = q_rai⁰[c], q_sno⁰[c]
        λ, mu_S, λl = λ⁰[c], mu_S⁰[c], λ_lag[c]
        transform =
            CA.build_physical_transform(quad, qt, T, q′q′[c], T′T′[c], corr)
        vals = CA.quadrature_point_values(
            (T_hat, q_hat) -> (T_hat, q_hat), transform, quad,
        )
        for (T_hat, q_tot_hat_raw) in vals
            npts += 1
            q_tot_hat = max(FT(0), q_tot_hat_raw)
            q_sat_hat = TD.q_vap_saturation(thp, T_hat, ρ)
            S′_hat = q_tot_hat - q_sat_hat - mu_S
            shifted = max(FT(0), λl + α * S′_hat)
            q_lcl_hat = λ * shifted
            q_icl_hat = (FT(1) - λ) * shifted

            noprecip = iszero(q_rai) && iszero(q_sno)
            subsat = q_tot_hat <= q_sat_hat
            if subsat && noprecip
                subsat_noprecip += 1
                cond = max(q_lcl_hat, q_icl_hat)
                if iszero(q_lcl_hat) && iszero(q_icl_hat)
                    exact_zero += 1
                    guard_fires += 1
                elseif cond <= FT(1e-10)
                    tiny_nonzero += 1
                    max_tiny = max(max_tiny, cond)
                else
                    larger += 1
                end
            end
            t = BMT.bulk_microphysics_tendencies(
                BMT.LinearizedAverage(), BMT.Microphysics1Moment(), cmp, thp,
                ρ, T_hat, q_tot_hat, q_lcl_hat, q_icl_hat,
                q_rai, q_sno, dt, nsubs,
            )
            if all(iszero, (t.dq_lcl_dt, t.dq_icl_dt, t.dq_rai_dt, t.dq_sno_dt))
                zero_tendency += 1
            end
        end
    end
    return (; npts, guard_fires, subsat_noprecip, exact_zero, tiny_nonzero,
            larger, zero_tendency, max_tiny)
end

r = tally(
    ncells, ρ⁰, T⁰, qt⁰, q_rai⁰, q_sno⁰, λ⁰, mu_S⁰, λ_lag, T′T′, q′q′,
    quad, thp, cmp, corr, α, dt, nsubs, FT,
)
(; npts, guard_fires, subsat_noprecip, exact_zero, tiny_nonzero,
   larger, zero_tendency, max_tiny) = r

pct(x) = 100 * x / npts
@printf("\npoints evaluated                 %d over %d cells\n", npts, ncells)
@printf("guard fires (shipped early-out)  %.2f%%\n", pct(guard_fires))
@printf("subsaturated & precip-free       %.2f%%\n", pct(subsat_noprecip))
@printf("  condensate exactly zero        %.2f%%\n", pct(exact_zero))
@printf("  condensate in (0, 1e-10]       %.2f%%   max %.3e\n",
        pct(tiny_nonzero), max_tiny)
@printf("  condensate > 1e-10             %.2f%%\n", pct(larger))
@printf("CM call returns all-zero anyway  %.2f%%   <- ceiling for any skip\n",
        pct(zero_tendency))

open(out_path, "w") do io
    TOML.print(io, Dict(
        "points" => npts,
        "cells" => ncells,
        "guard_fires_pct" => pct(guard_fires),
        "subsat_noprecip_pct" => pct(subsat_noprecip),
        "exact_zero_pct" => pct(exact_zero),
        "tiny_nonzero_pct" => pct(tiny_nonzero),
        "tiny_nonzero_max" => Float64(max_tiny),
        "larger_pct" => pct(larger),
        "zero_tendency_pct" => pct(zero_tendency),
    ))
end
@info "wrote $out_path"
