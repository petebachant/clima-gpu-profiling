# Does the fused longwave solve agree with the two solves it replaces?
#
# solve_lw_both! computes the gas and aerosol optics once and sweeps twice.
#
# It cannot be compared to two solve_lw! calls cell by cell. The cloud mask is
# McICA-sampled with Random.rand(), keyed per kernel launch, so the fused solve
# (one launch) draws a different cloud sample than the reference all-sky pass
# (the second of two launches). Any two all-sky solves differ for that reason
# alone.
#
# So the reference is compared against ITSELF -- two all-sky solves, differing
# only by resampling -- and the fused solve has to sit inside that spread. The
# clear sky has no cloud sampling and must match exactly.
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

# The fused solve is named solve_*_both_skies! on pb/fused-clear-sky (#631) and
# solve_*_both! on pb/optics-split-rebased, with identical argument lists. Pick
# whichever the dev'd RRTMGP has, and do it BEFORE the 24 minutes of setup: this
# script failed once on a renamed entry point with the simulation already built.
function pick(names, what, has)
    i = findfirst(has, names)
    isnothing(i) && error(
        "none of $(join(names, ", ")) found for the $what. The dev'd RRTMGP is " *
        "$(pkgdir(RRTMGP)); pb/fused-clear-sky and pb/optics-split-rebased name " *
        "these differently, so a third naming needs adding here.",
    )
    return names[i]
end

lw_both_name = pick((:solve_lw_both_skies!, :solve_lw_both!),
                    "fused longwave solve", n -> isdefined(RTE, n))
sw_both_name = pick((:solve_sw_both_skies!, :solve_sw_both!),
                    "fused shortwave solve", n -> isdefined(RTE, n))
solve_lw_both! = getproperty(RTE, lw_both_name)
solve_sw_both! = getproperty(RTE, sw_both_name)
@info "fused entry points" lw = lw_both_name sw = sw_both_name

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
# What update_lw_fluxes!/update_sw_fluxes! pass as metric_scaling.
ms = s.deep_atmosphere_inverse_scaling
# Renamed alongside the entry points.
clear_lw_name = pick((:clear_flux_acc_lw, :clear_acc_lw),
                     "clear-sky longwave accumulator", n -> hasproperty(s, n))
clear_sw_name = pick((:clear_flux_acc_sw, :clear_acc_sw),
                     "clear-sky shortwave accumulator", n -> hasproperty(s, n))
clear_lw_acc = getproperty(s, clear_lw_name)
clear_sw_acc = getproperty(s, clear_sw_name)

snap(f) = (up = Array(f.flux_up), dn = Array(f.flux_dn), net = Array(f.flux_net))

# --- longwave --------------------------------------------------------------
# Reference: the two solves the fused version replaces
RTE.solve_lw!(lws, as, lk.lookup_lw, nothing, lk.lookup_lw_aero, ms)
ref_clear = snap(lws.flux)
RTE.solve_lw!(lws, as, lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero, ms)
ref_allsky = snap(lws.flux)
# The control: the same solve again. Whatever it differs from itself by is what
# McICA resampling costs, and is the yardstick for the fused solve.
RTE.solve_lw!(lws, as, lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero, ms)
ref_allsky_again = snap(lws.flux)

# The fused solve, filling both skies in one pass
solve_lw_both!(
    lws, clear_lw_acc, as,
    lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero, ms,
)
got_allsky = snap(lws.flux)
got_clear = snap(clear_lw_acc)

# Relative to the field's own scale: fluxes span orders of magnitude, so an
# absolute difference says nothing
function compare(ref, got)
    out = Dict{String, Any}()
    for k in (:up, :dn, :net)
        r, g = getproperty(ref, k), getproperty(got, k)
        scale = maximum(abs, r)
        n = length(r)
        out[String(k)] = Dict(
            "max_abs_diff" => Float64(maximum(abs, r .- g)),
            "max_rel_diff" => Float64(maximum(abs, r .- g) / max(scale, eps())),
            # The field mean is what survives McICA sampling: individual cells
            # are resampled, the field is not supposed to move
            "mean_abs_diff" => Float64(sum(abs, r .- g) / n),
            "mean_rel_diff" => Float64(sum(abs, r .- g) / n / max(scale, eps())),
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
    "warmup_steps" => WARMUP,
    "lw_entry_point" => String(lw_both_name),
    "sw_entry_point" => String(sw_both_name),
    "rrtmgp_dir" => string(pkgdir(RRTMGP)),
    "reference_sky_contrast" => Float64(sky_contrast),
    "allsky" => compare(ref_allsky, got_allsky),
    "allsky_control" => compare(ref_allsky, ref_allsky_again),
    "clearsky" => compare(ref_clear, got_clear),
)
# The fused all-sky against the reference, and the reference against itself
fused_mean = maximum(results["allsky"][k]["mean_rel_diff"] for k in ("up", "dn", "net"))
control_mean = maximum(results["allsky_control"][k]["mean_rel_diff"] for k in ("up", "dn", "net"))
clear_worst = maximum(results["clearsky"][k]["max_rel_diff"] for k in ("up", "dn", "net"))
worst = clear_worst
results["worst_rel_diff"] = worst
# Which test applies is decided by the measurement, not by assumption. Keyed
# MCICA (pb/fused-clear-sky) makes the reference bit-reproducible, and then the
# fused solve must agree to Float32 roundoff over a 256-g-point accumulation.
# pb/optics-split-rebased has no keyed sampling, the two reference solves draw
# different cloud masks, and roundoff equality is unreachable by construction;
# the test is then the one this script's header describes -- the fused draw must
# sit in the reference's own resampling spread.
#
# The factor of 2 bounds gross error only. A wrong increment or wrong shared
# optics is orders of magnitude out AND breaks the clear-sky equality below,
# which is the sharp test: the clear sky has no sampling, so it must be exact.
const ROUNDOFF = 64 * eps(Float32)
consistent(fused, control) =
    control <= ROUNDOFF ? fused <= ROUNDOFF : fused <= 2 * control
# Both conditions: the fused result matches, AND the comparison was capable of
# detecting a mismatch in the first place
results["cloud_effect_present"] = sky_contrast > 1e-3
results["fused_mean_rel_diff"] = fused_mean
results["control_mean_rel_diff"] = control_mean
# Three conditions: the clear sky is exact, the all-sky sits within the
# reference's own resampling spread, and clouds were actually doing something
results["clearsky_exact"] = clear_worst < 1e-6
# Keyed sampling makes the reference reproducible, so control_mean is 0 and
# the old "inside the resampling band" test degenerates into demanding
# bit-equality. The fused path applies the cloud increment to optics it has
# already swept, so it agrees to roundoff, not exactly. 64 ulp of Float32 is
# still four orders below what a different cloud draw produced (0.33%).
results["sampling_deterministic"] = control_mean <= ROUNDOFF
results["allsky_consistent"] = consistent(fused_mean, control_mean)
# --- shortwave -------------------------------------------------------------
sws = s.sws
RTE.solve_sw!(sws, as, lk.lookup_sw, nothing, lk.lookup_sw_aero, ms)
sw_ref_clear = snap(sws.flux)
RTE.solve_sw!(sws, as, lk.lookup_sw, lk.lookup_sw_cld, lk.lookup_sw_aero, ms)
sw_ref_allsky = snap(sws.flux)
RTE.solve_sw!(sws, as, lk.lookup_sw, lk.lookup_sw_cld, lk.lookup_sw_aero, ms)
sw_ref_allsky_again = snap(sws.flux)
solve_sw_both!(
    sws, clear_sw_acc, as,
    lk.lookup_sw, lk.lookup_sw_cld, lk.lookup_sw_aero, ms,
)
sw_got_allsky = snap(sws.flux)
sw_got_clear = snap(clear_sw_acc)

results["sw_allsky"] = compare(sw_ref_allsky, sw_got_allsky)
results["sw_allsky_control"] = compare(sw_ref_allsky, sw_ref_allsky_again)
results["sw_clearsky"] = compare(sw_ref_clear, sw_got_clear)
sw_fused_mean = maximum(results["sw_allsky"][k]["mean_rel_diff"] for k in ("up", "dn", "net"))
sw_control_mean = maximum(results["sw_allsky_control"][k]["mean_rel_diff"] for k in ("up", "dn", "net"))
sw_clear_worst = maximum(results["sw_clearsky"][k]["max_rel_diff"] for k in ("up", "dn", "net"))
results["sw_fused_mean_rel_diff"] = sw_fused_mean
results["sw_control_mean_rel_diff"] = sw_control_mean
results["sw_clearsky_exact"] = sw_clear_worst < 1e-6
results["sw_sampling_deterministic"] = sw_control_mean <= ROUNDOFF
results["sw_allsky_consistent"] = consistent(sw_fused_mean, sw_control_mean)

# Which of the two tests applied depends on whether the branch under test keys
# its MCICA draw, so the file says which one it was rather than asserting one.
results["note"] =
    results["sampling_deterministic"] && results["sw_sampling_deterministic"] ?
    "the reference solve is bit-reproducible on this branch, so the all-sky " *
    "comparison is roundoff equality against a threshold of $(ROUNDOFF); the " *
    "clear sky carries no sampling and must match exactly" :
    "the cloud mask is McICA-sampled per launch, so all-sky fluxes are " *
    "compared against the reference resampled against itself, not against " *
    "equality; the clear sky has no sampling and must match exactly"

@printf("SW: fused mean rel diff %.3e vs resampling control %.3e\n",
        sw_fused_mean, sw_control_mean)
@printf("SW: clear sky max rel diff %.3e (must be exact)\n", sw_clear_worst)

results["passes"] = results["clearsky_exact"] &&
                    results["allsky_consistent"] &&
                    results["cloud_effect_present"] &&
                    results["sw_clearsky_exact"] &&
                    results["sw_allsky_consistent"]

for sky in ("allsky", "allsky_control", "clearsky"), k in ("up", "dn", "net")
    @printf("%-9s %-4s max rel diff %.3e (field max %.1f)\n", sky, k,
            results[sky][k]["max_rel_diff"], results[sky][k]["field_max_abs"])
end
@printf("reference all-sky vs clear-sky contrast %.3e (needs > 1e-3 to be a real test)\n",
        sky_contrast)
@printf("fused mean rel diff %.3e vs resampling control %.3e\n", fused_mean, control_mean)
@printf("clear sky max rel diff %.3e (must be exact)\n", clear_worst)
@printf("-> %s\n", results["passes"] ? "PASS" : "FAIL")

open(out_path, "w") do io
    TOML.print(io, results)
end

# The verdict has to reach the pipeline. Writing passes = false and exiting 0
# let `calkit status` report green over a failing check.
results["passes"] || exit(1)
