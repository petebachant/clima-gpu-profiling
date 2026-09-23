# Does the fused longwave solve agree with the two solves it replaces?
#
# solve_lw_both! computes the gas and aerosol optics once and sweeps twice. It
# is not bit-identical to two solve_lw! calls -- the cloud and aerosol
# increments are weighted sums applied per component, so adding aerosol before
# cloud rather than after changes summation order -- so the check is a bound on
# the difference, not equality.
#
# This matters more than a state comparison would: the clear-sky fluxes feed
# the cloud radiative effect diagnostics and nothing else, so an error in them
# never reaches the prognostic state and a run would look perfectly healthy.

import CUDA
import ClimaAtmos as CA
import RRTMGP
import RRTMGP.RTESolver as RTE
import TOML
using Printf

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

out_path = "results/fused-lw-check.toml"
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i)
        out_path = ARGS[i + 1]
        deleteat!(ARGS, i:(i + 1))
    end
end

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)
# Long enough for condensate to form. At three steps -- 90 simulated seconds --
# the sky is still clear, the cloud increment is a no-op, and both code paths
# compute the same cloud-free radiation: the check passes without testing
# anything. That is the failure mode docs/learnings.md 4h/4i is about.
const WARMUP = 60
for i in 1:WARMUP
    i % 10 == 0 && @info "warmup step $i / $WARMUP"
    step!(cs)
end

s = cs.model_sims.atmos_sim.integrator.p.radiation.rrtmgp_solver
lk, as, lws = s.lookups, s.as, s.lws

snap(f) = (up = Array(f.flux_up), dn = Array(f.flux_dn), net = Array(f.flux_net))

# Reference: the two solves the fused version replaces
RTE.solve_lw!(lws, as, lk.lookup_lw, nothing, lk.lookup_lw_aero, nothing)
ref_clear = snap(lws.flux)
RTE.solve_lw!(lws, as, lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero, nothing)
ref_allsky = snap(lws.flux)

# The fused solve, filling both skies in one pass
RTE.solve_lw_both!(
    lws, s.clear_acc_lw, as,
    lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero, nothing,
)
got_allsky = snap(lws.flux)
got_clear = snap(s.clear_acc_lw)

# Relative to the field's own scale: fluxes span orders of magnitude, so an
# absolute difference says nothing
function compare(ref, got)
    out = Dict{String, Any}()
    for k in (:up, :dn, :net)
        r, g = getproperty(ref, k), getproperty(got, k)
        scale = maximum(abs, r)
        out[String(k)] = Dict(
            "max_abs_diff" => Float64(maximum(abs, r .- g)),
            "max_rel_diff" => Float64(maximum(abs, r .- g) / max(scale, eps())),
            "field_max_abs" => Float64(scale),
        )
    end
    return out
end

# Does the reference pair actually differ? If clouds are doing nothing, both
# code paths compute cloud-free radiation and agreeing proves nothing
sky_contrast = maximum(abs, ref_allsky.net .- ref_clear.net) /
               max(maximum(abs, ref_allsky.net), eps())

results = Dict{String, Any}(
    "note" => "fused solve_lw_both! against two solve_lw! calls; differences " *
              "are expected at roundoff, not zero",
    "warmup_steps" => WARMUP,
    "reference_sky_contrast" => Float64(sky_contrast),
    "allsky" => compare(ref_allsky, got_allsky),
    "clearsky" => compare(ref_clear, got_clear),
)
worst = maximum(
    results[sky][k]["max_rel_diff"] for sky in ("allsky", "clearsky") for
    k in ("up", "dn", "net")
)
results["worst_rel_diff"] = worst
# Float32 roundoff over a 256-g-point accumulation; anything much larger is a
# bug, not summation order
# Both conditions: the fused result matches, AND the comparison was capable of
# detecting a mismatch in the first place
results["cloud_effect_present"] = sky_contrast > 1e-3
results["passes"] = worst < 1e-4 && results["cloud_effect_present"]

for sky in ("allsky", "clearsky"), k in ("up", "dn", "net")
    @printf("%-9s %-4s max rel diff %.3e (field max %.1f)\n", sky, k,
            results[sky][k]["max_rel_diff"], results[sky][k]["field_max_abs"])
end
@printf("reference all-sky vs clear-sky contrast %.3e (needs > 1e-3 to be a real test)\n",
        sky_contrast)
@printf("worst %.3e -> %s\n", worst, results["passes"] ? "PASS" : "FAIL")

open(out_path, "w") do io
    TOML.print(io, results)
end
