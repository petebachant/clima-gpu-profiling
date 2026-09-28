# Is the fused solve's all-sky flux the SAME ESTIMATOR as the two solves it
# replaces, or a biased one?
#
# The per-call check (verify-fused-lw.jl) established that one fused solve
# differs from one two-solve reference by about what resampling costs. That
# bounds the noise but says nothing about bias: a systematically shifted
# estimator would also sit inside a single draw's spread.
#
# Bias and noise separate under averaging. Draw K cloud samples from each path
# on the SAME atmospheric state and average: sampling noise in the difference of
# the two K-means falls as 1/sqrt(K), and bias does not fall at all. No model
# stepping is needed, because McICA redraws on every solve call, so this is
# nearly free once the simulation is built.
#
# The control is the same measurement with both sides drawn from the REFERENCE
# path. It is the shape 1/sqrt(K) actually takes here, rather than the ideal, so
# the treatment is read against it instead of against theory.

import CUDA
import ClimaAtmos as CA
import RRTMGP
import RRTMGP.RTESolver as RTE
import TOML
using Printf

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

out_path = "results/fused-flux-bias.toml"
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i)
        out_path = ARGS[i + 1]
        deleteat!(ARGS, i:(i + 1))
    end
end
draws = 200
let i = findfirst(==("--draws"), ARGS)
    if !isnothing(i)
        draws = parse(Int, ARGS[i + 1])
        deleteat!(ARGS, i:(i + 1))
    end
end

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)
# Same 60 steps as verify-fused-lw.jl, and for the same reason: at three steps
# the sky is clear, the cloud increment is a no-op, and both paths agree
# trivially (docs/learnings.md 4h/4i).
const WARMUP = 60
for i in 1:WARMUP
    i % 10 == 0 && @info "warmup step $i / $WARMUP"
    step!(cs)
end

s = cs.model_sims.atmos_sim.integrator.p.radiation.rrtmgp_solver
lk, as, lws, sws = s.lookups, s.as, s.lws, s.sws

const CHECKPOINTS = [1, 2, 5, 10, 25, 50, 100, 200]

# Accumulation lives in a function and the counts come back as a return value:
# `acc .+= x` at top level inside a `for` creates a new local and dies on the
# first iteration (docs/learnings.md, four separate scripts).
function accumulate_draws(draw_ref!, draw_fused!, read_flux, n_draws, checkpoints)
    # Two reference accumulators, so the control is two independent sets of
    # draws from the SAME path -- the null this treatment is read against.
    acc_ref_a, acc_ref_b, acc_fused = nothing, nothing, nothing
    records = Dict{String, Any}[]
    for k in 1:n_draws
        draw_ref!()
        fa = Float64.(read_flux())
        draw_ref!()
        fb = Float64.(read_flux())
        draw_fused!()
        ff = Float64.(read_flux())
        if isnothing(acc_ref_a)
            acc_ref_a, acc_ref_b, acc_fused = copy(fa), copy(fb), copy(ff)
        else
            acc_ref_a .+= fa
            acc_ref_b .+= fb
            acc_fused .+= ff
        end
        if k in checkpoints
            mean_a = acc_ref_a ./ k
            mean_b = acc_ref_b ./ k
            mean_f = acc_fused ./ k
            n = length(mean_a)
            scale = max(maximum(abs, mean_a), eps())
            push!(
                records,
                Dict(
                    "draws" => k,
                    "treatment_rel_diff" =>
                        sum(abs, mean_a .- mean_f) / n / scale,
                    "control_rel_diff" =>
                        sum(abs, mean_a .- mean_b) / n / scale,
                    "treatment_max_abs_diff" => maximum(abs, mean_a .- mean_f),
                    "control_max_abs_diff" => maximum(abs, mean_a .- mean_b),
                ),
            )
            @printf(
                "  K=%3d  treatment %.3e  control %.3e\n",
                k,
                records[end]["treatment_rel_diff"],
                records[end]["control_rel_diff"],
            )
        end
    end
    return records
end

checkpoints = filter(<=(draws), CHECKPOINTS)
draws in checkpoints || push!(checkpoints, draws)

@info "longwave: averaging $draws draws per path"
lw = accumulate_draws(
    () -> RTE.solve_lw!(
        lws, as, lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero, nothing,
    ),
    () -> RTE.solve_lw_both!(
        lws, s.clear_acc_lw, as,
        lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero, nothing,
    ),
    () -> Array(lws.flux.flux_net),
    draws,
    checkpoints,
)

@info "shortwave: averaging $draws draws per path"
sw = accumulate_draws(
    () -> RTE.solve_sw!(
        sws, as, lk.lookup_sw, lk.lookup_sw_cld, lk.lookup_sw_aero, nothing,
    ),
    () -> RTE.solve_sw_both!(
        sws, s.clear_acc_sw, as,
        lk.lookup_sw, lk.lookup_sw_cld, lk.lookup_sw_aero, nothing,
    ),
    () -> Array(sws.flux.flux_net),
    draws,
    checkpoints,
)

# A biased estimator's disagreement stops falling while the control keeps
# falling, so the ratio of the two is the statistic: near one at every K means
# the fused path is the same estimator drawn differently.
function verdict(records)
    last = records[end]
    first = records[1]
    ratio = last["treatment_rel_diff"] / max(last["control_rel_diff"], eps())
    return Dict(
        "by_draws" => records,
        "final_draws" => last["draws"],
        "final_treatment_rel_diff" => last["treatment_rel_diff"],
        "final_control_rel_diff" => last["control_rel_diff"],
        "final_ratio_to_control" => ratio,
        # How far each fell from a single draw. Sampling noise falls as
        # 1/sqrt(K); a bias floor shows up as a treatment that stops falling
        # while the control continues.
        "treatment_shrink_factor" =>
            first["treatment_rel_diff"] / max(last["treatment_rel_diff"], eps()),
        "control_shrink_factor" =>
            first["control_rel_diff"] / max(last["control_rel_diff"], eps()),
        "ideal_shrink_factor" => sqrt(Float64(last["draws"])),
        # Generous, and deliberately so: the claim is only that the treatment
        # behaves like resampling, and a ratio near one is what that looks like.
        "no_detectable_bias" => ratio <= 2.0,
    )
end

results = Dict(
    "warmup_steps" => WARMUP,
    "draws_per_path" => draws,
    "note" => "Averaging K McICA draws from each path on one atmospheric " *
              "state. Sampling noise in the difference of the two K-means " *
              "falls as 1/sqrt(K); bias does not. The control draws both " *
              "sides from the reference path, so the treatment is read " *
              "against the shrinkage actually observed rather than the ideal.",
    "lw" => verdict(lw),
    "sw" => verdict(sw),
)
results["passes"] =
    results["lw"]["no_detectable_bias"] && results["sw"]["no_detectable_bias"]

open(out_path, "w") do io
    TOML.print(io, results; sorted = true)
end
@printf(
    "\nLW: treatment/control at K=%d is %.2f (shrank %.1fx vs control %.1fx, ideal %.1fx)\n",
    results["lw"]["final_draws"],
    results["lw"]["final_ratio_to_control"],
    results["lw"]["treatment_shrink_factor"],
    results["lw"]["control_shrink_factor"],
    results["lw"]["ideal_shrink_factor"],
)
@printf(
    "SW: treatment/control at K=%d is %.2f (shrank %.1fx vs control %.1fx, ideal %.1fx)\n",
    results["sw"]["final_draws"],
    results["sw"]["final_ratio_to_control"],
    results["sw"]["treatment_shrink_factor"],
    results["sw"]["control_shrink_factor"],
    results["sw"]["ideal_shrink_factor"],
)
println(results["passes"] ? "no detectable bias" : "BIAS DETECTED")
@info "wrote $out_path"
