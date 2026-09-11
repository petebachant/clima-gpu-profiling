# Where does the CloudMicrophysics fusion disagree with the path it replaced?
#
# docs/learnings.md 4h: pb/1m-spill-fuse changes the model's snow/rain
# partitioning -- 22% less snow by coupled step 20 -- while
# test/bulk_tendencies_tests.jl passes 2000 randomized trials per float type.
# The randomized test samples a distribution the author chose; the model samples
# the one the physics produces. This compares the two paths on states taken FROM
# THE MODEL, after enough steps to reach the regime where the arms diverge.
#
#   fused:     _fused_linearize(mp, tps, ρ, T, q_tot, q_lcl, q_icl, q_rai, q_sno, q_min)
#   reference: _linearize(_microphysics_source_terms(...), q_lcl, q_icl, q_rai, q_sno, q_min)
#
# Reports how many cells disagree, in which accumulator, by how much, and what
# distinguishes the disagreeing cells -- which is what identifies the regime the
# randomized sampler misses.

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

out_path = "results/fusion-mismatch.toml"
warm = 20
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i); global out_path = ARGS[i+1]; deleteat!(ARGS, i:(i+1)); end
end
let i = findfirst(==("--warmup"), ARGS)
    if !isnothing(i); global warm = parse(Int, ARGS[i+1]); deleteat!(ARGS, i:(i+1)); end
end

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)
for i in 1:warm
    @info "warmup step $i / $warm"
    step!(cs)
end

integrator = cs.model_sims.atmos_sim.integrator
p = integrator.p; Y = integrator.u
flat(f) = vec(Array(parent(f)))
pick(nt, ns...) = for n in ns; hasproperty(nt,n) && return getproperty(nt,n); end

# Drive the real cache function so the scratch fields hold the environment
# inputs the kernel actually sees, then read them back (same approach as
# scripts/measure-quadrature-order-error.jl, which self-checks this mapping).
CA.set_microphysics_tendency_cache!(Y, p, p.atmos.microphysics_model, p.atmos.turbconv_model)
ρ  = flat(p.scratch.ᶜtemp_scalar)
ql = flat(p.scratch.ᶜtemp_scalar_2); qi = flat(p.scratch.ᶜtemp_scalar_3)
qr = flat(p.scratch.ᶜtemp_scalar_4); qs = flat(p.scratch.ᶜtemp_scalar_5)
T  = flat(p.precomputed.ᶜT⁰); qt = flat(p.precomputed.ᶜq_tot_nonneg⁰)

thp = CAP.thermodynamics_params(p.params)
mp  = CAP.microphysics_1m_params(p.params)
q_min = TD.Parameters.q_min(thp)
n = length(ρ)
@info "comparing $n cells after $warm coupled steps"

FIELDS = (:M11,:M12,:M22,:M31,:M33,:M34,:M41,:M42,:M43,:M44,:e1,:e2,:e4)
nbad = 0; worst = 0.0; worst_i = 0; worst_f = :none
bad_idx = Int[]
per_field = Dict(String(f) => 0 for f in FIELDS)

for i in 1:n
    a = BMT._fused_linearize(mp, thp, ρ[i], T[i], qt[i], ql[i], qi[i], qr[i], qs[i], q_min)
    src = BMT._microphysics_source_terms(BMT.Microphysics1Moment(), mp, thp,
                                         ρ[i], T[i], qt[i], ql[i], qi[i], qr[i], qs[i])
    b = BMT._linearize(src, ql[i], qi[i], qr[i], qs[i], q_min)
    hit = false
    for f in FIELDS
        x, y = getproperty(a,f), getproperty(b,f)
        if x != y
            hit = true; per_field[String(f)] += 1
            d = abs(x-y)/max(abs(x),abs(y),eps(Float32))
            if d > worst; worst = Float64(d); worst_i = i; worst_f = f; end
        end
    end
    if hit
        nbad += 1
        length(bad_idx) < 2000 && push!(bad_idx, i)
    end
end

@printf("\n  cells compared      %d\n", n)
@printf("  cells disagreeing   %d (%.4f%%)\n", nbad, 100nbad/n)
@printf("  worst rel diff      %.3e  in %s at cell %d\n", worst, worst_f, worst_i)
println("\n  disagreements by accumulator:")
for (k,v) in sort(collect(per_field), by=x->-x[2])
    v > 0 && @printf("    %-5s %d\n", k, v)
end

# What distinguishes the disagreeing cells?
function stats(idx, v, label)
    isempty(idx) && return
    s = sort([v[i] for i in idx])
    @printf("    %-8s min %11.4e  p50 %11.4e  max %11.4e\n", label, s[1], s[length(s)÷2+1], s[end])
end
if nbad > 0
    println("\n  state of DISAGREEING cells:")
    for (v,l) in ((T,"T"),(qt,"q_tot"),(ql,"q_lcl"),(qi,"q_icl"),(qr,"q_rai"),(qs,"q_sno"))
        stats(bad_idx, v, l)
    end
    println("  state of ALL cells:")
    for (v,l) in ((T,"T"),(qt,"q_tot"),(ql,"q_lcl"),(qi,"q_icl"),(qr,"q_rai"),(qs,"q_sno"))
        stats(1:n, v, l)
    end
end

TOML.print(open(out_path,"w"), Dict(
    "cells"=>n, "warmup_steps"=>warm, "disagreeing"=>nbad,
    "disagreeing_frac"=>nbad/n, "worst_rel_diff"=>worst,
    "worst_field"=>String(worst_f), "by_field"=>per_field,
    "note"=>"_fused_linearize vs _linearize(_microphysics_source_terms(...)) on "*
            "states taken from the model after $warm coupled steps."))
@info "wrote $out_path"
