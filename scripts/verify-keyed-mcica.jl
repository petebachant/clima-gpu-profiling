# Does coordinate-keyed McICA make the fused solve bit-identical to two solves?
#
# The fused solve differs from the two it replaces only because the cloud mask is
# drawn from the device RNG, whose state is keyed PER KERNEL LAUNCH: one fused
# launch draws a different sample than two separate launches did. RRTMGP.jl#316
# (open since 2022-09-20) asks for a seed to be passed into the sampler, and
# build_cloud_mask!'s own docstring names the fix -- "column-indexed seeding" --
# and declines it "to keep the kernel allocation-free".
#
# A stateless, counter-based draw needs no allocation: hash (column, g-point,
# layer) into a uniform. Then the same physical coordinate always draws the same
# number, no matter how many kernels launched or in what order, and the fused
# path should reproduce the two-solve path EXACTLY.
#
# That is the whole test: equality, not a resampling band. If it holds, the
# accuracy question blocking the fusion disappears, and GPU runs become
# reproducible and restartable, which is what #316 and #544 were about.

import CUDA
import ClimaAtmos as CA
import RRTMGP
import RRTMGP.RTESolver as RTE
import TOML
using Printf

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

out_path = "results/keyed-mcica.toml"
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i)
        out_path = ARGS[i + 1]
        deleteat!(ARGS, i:(i + 1))
    end
end

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)
# 60 steps so there is real condensate. At three steps the sky is clear, the
# cloud increment is a no-op, and any two paths agree trivially -- the failure
# mode docs/learnings.md 4h/4i is about, and the reason an earlier version of
# the flux check passed while testing nothing.
const WARMUP = 60
for i in 1:WARMUP
    i % 10 == 0 && @info "warmup step $i / $WARMUP"
    step!(cs)
end

s = cs.model_sims.atmos_sim.integrator.p.radiation.rrtmgp_solver
lk, as, lws, sws = s.lookups, s.as, s.lws, s.sws

snap(f) = (
    up = Array(f.flux_up),
    dn = Array(f.flux_dn),
    net = Array(f.flux_net),
)

function compare(ref, got, label)
    out = Dict{String, Any}()
    worst = 0.0
    for k in (:up, :dn, :net)
        r, g = getproperty(ref, k), getproperty(got, k)
        d = maximum(abs, r .- g)
        out[String(k)] = Dict(
            "max_abs_diff" => Float64(d),
            "field_max_abs" => Float64(maximum(abs, r)),
            "identical" => d == 0,
        )
        worst = max(worst, Float64(d))
    end
    out["worst_max_abs_diff"] = worst
    out["all_identical"] = worst == 0
    @printf("%-28s worst |diff| = %.6e  %s\n", label, worst,
            worst == 0 ? "IDENTICAL" : "differs")
    return out
end

results = Dict{String, Any}("warmup_steps" => WARMUP)

# --- longwave ---------------------------------------------------------------
# The two-solve path: clear, then all-sky. This is what upstream does today.
RTE.solve_lw!(lws, as, lk.lookup_lw, nothing, lk.lookup_lw_aero, nothing)
two_clear_lw = snap(lws.flux)
RTE.solve_lw!(lws, as, lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero, nothing)
two_allsky_lw = snap(lws.flux)

# Repeating the all-sky solve is the reproducibility test on its own: with the
# per-launch device RNG this differs from the line above; keyed, it cannot.
RTE.solve_lw!(lws, as, lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero, nothing)
repeat_allsky_lw = snap(lws.flux)

RTE.solve_lw_both!(
    lws, s.clear_acc_lw, as,
    lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero, nothing,
)
fused_allsky_lw = snap(lws.flux)
fused_clear_lw = snap(s.clear_acc_lw)

results["lw_repeat_vs_two"] =
    compare(two_allsky_lw, repeat_allsky_lw, "LW all-sky, solve twice")
results["lw_fused_vs_two"] =
    compare(two_allsky_lw, fused_allsky_lw, "LW all-sky, fused vs two")
results["lw_clear"] = compare(two_clear_lw, fused_clear_lw, "LW clear sky")

# --- shortwave --------------------------------------------------------------
RTE.solve_sw!(sws, as, lk.lookup_sw, nothing, lk.lookup_sw_aero, nothing)
two_clear_sw = snap(sws.flux)
RTE.solve_sw!(sws, as, lk.lookup_sw, lk.lookup_sw_cld, lk.lookup_sw_aero, nothing)
two_allsky_sw = snap(sws.flux)
RTE.solve_sw!(sws, as, lk.lookup_sw, lk.lookup_sw_cld, lk.lookup_sw_aero, nothing)
repeat_allsky_sw = snap(sws.flux)

RTE.solve_sw_both!(
    sws, s.clear_acc_sw, as,
    lk.lookup_sw, lk.lookup_sw_cld, lk.lookup_sw_aero, nothing,
)
fused_allsky_sw = snap(sws.flux)
fused_clear_sw = snap(s.clear_acc_sw)

results["sw_repeat_vs_two"] =
    compare(two_allsky_sw, repeat_allsky_sw, "SW all-sky, solve twice")
results["sw_fused_vs_two"] =
    compare(two_allsky_sw, fused_allsky_sw, "SW all-sky, fused vs two")
results["sw_clear"] = compare(two_clear_sw, fused_clear_sw, "SW clear sky")

# Guard against the vacuous pass: if there are no clouds, every comparison is
# trivially identical and this script proves nothing. The clear/all-sky contrast
# is what shows the cloud increment actually did something.
contrast = let r = two_clear_lw.net, g = two_allsky_lw.net
    Float64(maximum(abs, r .- g) / max(maximum(abs, r), eps()))
end
results["clear_allsky_contrast"] = contrast
results["clouds_present"] = contrast > 0.01

results["repeat_reproducible"] =
    results["lw_repeat_vs_two"]["all_identical"] &&
    results["sw_repeat_vs_two"]["all_identical"]
results["fused_identical"] =
    results["lw_fused_vs_two"]["all_identical"] &&
    results["sw_fused_vs_two"]["all_identical"]
results["clear_identical"] =
    results["lw_clear"]["all_identical"] && results["sw_clear"]["all_identical"]
results["passes"] =
    results["clouds_present"] &&
    results["repeat_reproducible"] &&
    results["fused_identical"] &&
    results["clear_identical"]

open(out_path, "w") do io
    TOML.print(io, results; sorted = true)
end
@printf("\nclear/all-sky contrast %.3f (clouds present: %s)\n",
        contrast, results["clouds_present"])
println("repeated solve reproducible: ", results["repeat_reproducible"])
println("fused == two solves:         ", results["fused_identical"])
println("clear sky identical:         ", results["clear_identical"])
println(results["passes"] ? "PASS" : "FAIL")
@info "wrote $out_path"
