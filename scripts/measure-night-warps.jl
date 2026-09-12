# How much of the shortwave solve can a night-column skip actually save?
#
# Skipping the ~224-g-point solve where the sun is below the horizon gave only
# -2.63% on the shortwave kernels (longwave control: -0.20%). If roughly half the
# globe is dark, perfect skipping would give ~-50%, so the skip must be firing
# without whole warps agreeing.
#
# A GPU warp costs the maximum over its 32 lanes, so a criterion true at
# scattered points saves nothing -- the same reason the microphysics clear-air
# early-out returned +1.79% off a 77.7% POINT fraction with a 21.5% WARP
# fraction (docs/learnings.md 6a). This measures both for the night criterion,
# in the column order RRTMGP actually sees.

import ClimaComms
ClimaComms.@import_required_backends
import ClimaAtmos as CA
import TOML
# RRTMGP is a transitive dep (via ClimaAtmos) and cannot be imported directly
# from the AMIP project; reach it through the loaded-module table instead.
const RRTMGP = let m = nothing
    for (pkgid, mod) in Base.loaded_modules
        pkgid.name == "RRTMGP" && (m = mod)
    end
    m === nothing && error("RRTMGP not loaded")
    m
end
using Printf

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

out_path = "results/night-warps.toml"
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i); global out_path = ARGS[i+1]; deleteat!(ARGS, i:(i+1)); end
end

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)
for i in 1:3
    @info "warmup step $i / 3"
    step!(cs)
end

p = cs.model_sims.atmos_sim.integrator.p
μ = Array(RRTMGP.cos_zenith(p.radiation.rrtmgp_solver))
FT = eltype(μ)
n = length(μ)
night = μ .<= eps(FT)

function warp_fraction(mask, warp = 32)
    nw = length(mask) ÷ warp
    full = 0
    @inbounds for w in 0:(nw - 1)
        ok = true
        for k in 1:warp
            mask[w * warp + k] || (ok = false; break)
        end
        full += ok
    end
    return full / nw, nw
end

nf = count(night) / n
wf, nw = warp_fraction(night)
# Also: how many warps are MIXED (neither all-night nor all-day)?
# In a function, not a top-level loop: assignment inside a top-level `for` is a
# soft-scope local and silently does not update the global. That bug is recorded
# in docs/learnings.md 4e and was reproduced here anyway.
function count_mixed(night, warp = 32)
    mixed = 0
    for w in 0:(length(night) ÷ warp - 1)
        s = count(@view night[(w * warp + 1):(w * warp + warp)])
        0 < s < warp && (mixed += 1)
    end
    return mixed
end
mixed = count_mixed(night)

@printf("\n  columns              %d\n", n)
@printf("  night point fraction %.4f\n", nf)
@printf("  night WARP fraction  %.4f   <- what the skip can actually save\n", wf)
@printf("  mixed warps          %.4f of %d\n", mixed / nw, nw)
@printf("  min/max cos_zenith   %.3e / %.3f\n", minimum(μ), maximum(μ))

TOML.print(open(out_path, "w"), Dict(
    "columns" => n, "night_point_frac" => nf, "night_warp_frac" => wf,
    "mixed_warp_frac" => mixed / nw, "warps" => nw,
    "note" => "Fraction of columns with cos_zenith <= eps (ClimaAtmos's night " *
        "sentinel) and the fraction of 32-lane warps entirely so. The warp " *
        "fraction is what a GPU early-out can save; the point fraction is not."))
@info "wrote $out_path"
