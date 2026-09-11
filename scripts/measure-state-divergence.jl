# Do the baseline and mod arms compute the same model state?
#
# Motivation (docs/learnings.md 4g): on a 120-step window the mod arm's
# microphysics kernel degrades 2.4x while EVERY other kernel in both arms is
# flat to within 1%, and baseline's microphysics is flat too. Constant time in
# baseline means constant work. If the two arms computed the same state, the mod
# kernel would see the same constant work -- so either it has a state
# sensitivity baseline lacks, or the arms have diverged and the mod arm is doing
# genuinely more work.
#
# The fusion and the evaluator refactor were both verified bit-identical, but at
# UNIT level (2000 randomized trials, 400 states) and never in the full model
# across coupled steps. This checks the latter.
#
# Dumps per-field checksums of the prognostic state at intervals, so divergence
# can be dated rather than just detected. Run once per arm and diff the JSON.

import ClimaComms
ClimaComms.@import_required_backends
import TOML
using Printf

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

out_path = "results/state.toml"
n_steps = 120
every = 20
# NB `global` is required: `let` at top level is a hard scope, so a plain
# assignment here creates a local and the option silently does nothing. That bug
# shipped in four scripts in this repo and was masked because every caller passed
# the default value.
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i); global out_path = ARGS[i+1]; deleteat!(ARGS, i:(i+1)); end
end
let i = findfirst(==("--steps"), ARGS)
    if !isnothing(i); global n_steps = parse(Int, ARGS[i+1]); deleteat!(ARGS, i:(i+1)); end
end

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)

flat(f) = vec(Array(parent(f)))

"""Checksums that are sensitive to any bit difference but cheap to compare."""
function fingerprint(Y)
    out = Dict{String, Any}()
    for space in (:c, :f)
        hasproperty(Y, space) || continue
        blk = getproperty(Y, space)
        for nm in propertynames(blk)
            v = getproperty(blk, nm)
            local a
            try
                a = flat(v)
            catch
                continue   # nested (e.g. sgsʲs); the scalars below suffice
            end
            eltype(a) <: AbstractFloat || continue
            out["$space.$nm"] = Dict(
                "sum" => Float64(sum(a)),
                "sumabs" => Float64(sum(abs, a)),
                "max" => Float64(maximum(a)),
                "min" => Float64(minimum(a)),
            )
        end
    end
    return out
end

integrator = cs.model_sims.atmos_sim.integrator
snapshots = Dict{String, Any}()
snapshots["step_0"] = fingerprint(integrator.u)
for i in 1:n_steps
    step!(cs)
    if i % every == 0 || i == n_steps
        snapshots["step_$i"] = fingerprint(integrator.u)
        @info "snapshot at step $i"
    end
end

open(out_path, "w") do io
    TOML.print(io, Dict("steps" => n_steps, "every" => every,
                        "snapshots" => snapshots))
end
@info "wrote $out_path"
