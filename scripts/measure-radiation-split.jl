# How much of an RRTMGP solve is optics, and how much is the two-stream sweep?
#
# With `rad: allskywithclear` the solver runs twice per radiation step: once
# with the cloud lookup passed as `nothing` (clear sky, for the cloud radiative
# effect diagnostics) and once with it. Both passes recompute the gas optics and
# the aerosol optics from scratch; only the cloud increment differs. Fusing them
# -- compute the optics once, run the sweep twice -- would save one copy of the
# optics and nothing else, so the prize is exactly the optics share.
#
# The optics-only entry points live in RRTMGP itself (solve_lw_optics_only!).
# A kernel defined out here cannot be compiled: calling RRTMGP's @inline
# internals from a script leaves them as dynamic invocations, which a GPU
# kernel cannot make.
#
# Not a proposal and not a benchmark of a change: it sizes one before it is
# written. See docs/learnings.md "The RRTMGP ledger" for what has been tried.

import CUDA
import ClimaAtmos as CA
import RRTMGP
import RRTMGP.RTESolver as RTE
import TOML
using Printf

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

out_path = "results/radiation-split.toml"
let i = findfirst(==("--out"), ARGS)
    if !isnothing(i)
        out_path = ARGS[i + 1]
        deleteat!(ARGS, i:(i + 1))
    end
end

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)

# Radiation fires every dt_rad, so step until it has run and the state it reads
# is settled
for i in 1:3
    @info "warmup step $i / 3"
    step!(cs)
end

p = cs.model_sims.atmos_sim.integrator.p
solver = p.radiation.rrtmgp_solver
lk = solver.lookups
as = solver.as
lws, sws = solver.lws, solver.sws
nlay, ncol = RRTMGP.AtmosphericStates.get_dims(as)

# Median of repeats: one timing on this machine is worth little, and the mean is
# dragged by whatever else lands on the GPU
function timed_ms(f; repeats = 7)
    f()
    CUDA.synchronize()
    ts = Float64[]
    for _ in 1:repeats
        CUDA.synchronize()
        push!(ts, CUDA.@elapsed(f()) * 1e3)
    end
    sort!(ts)
    return ts[(length(ts) + 1) ÷ 2]
end

results = Dict{String, Any}(
    "ncol" => ncol,
    "nlay" => nlay,
    "n_gpt_lw" => length(lk.lookup_lw.band_data.major_gpt2bnd),
    "n_gpt_sw" => length(lk.lookup_sw.band_data.major_gpt2bnd),
    "note" => "optics_ms is the solve with the two-stream sweep and flux " *
              "accumulation removed; sweep_ms is the remainder. A fused " *
              "all-sky/clear-sky solve would save one copy of optics_ms.",
)

for (band, rte, lkp, lkp_cld, lkp_aero, solve!, optics_only!) in (
    ("lw", lws, lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero,
     RTE.solve_lw!, RTE.solve_lw_optics_only!),
    ("sw", sws, lk.lookup_sw, lk.lookup_sw_cld, lk.lookup_sw_aero,
     RTE.solve_sw!, RTE.solve_sw_optics_only!),
)
    allsky = timed_ms(() -> solve!(rte, as, lkp, lkp_cld, lkp_aero, nothing))
    clear = timed_ms(() -> solve!(rte, as, lkp, nothing, lkp_aero, nothing))
    optics = timed_ms(() -> optics_only!(rte, as, lkp, lkp_cld, lkp_aero))
    results[band] = Dict{String, Any}(
        "allsky_ms" => allsky,
        "clearsky_ms" => clear,
        "optics_ms" => optics,
        "sweep_ms" => allsky - optics,
        "optics_frac_of_allsky" => optics / allsky,
        # Both passes pay for the optics today; a fused solve would pay once
        "fused_saving_ms" => optics,
        "fused_saving_frac_of_pair" => optics / (allsky + clear),
    )
    @printf("%s: allsky %.2f ms, clear %.2f ms, optics %.2f ms (%.1f%% of allsky)\n",
            band, allsky, clear, optics, 100 * optics / allsky)
end

open(out_path, "w") do io
    TOML.print(io, results)
end
@info "wrote $out_path"
