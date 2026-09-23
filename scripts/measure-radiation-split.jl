# How much of an RRTMGP solve is optics, and how much is the two-stream sweep?
#
# With `rad: allskywithclear` the solver runs twice per radiation step: once
# with the cloud lookup passed as `nothing` (clear sky, for the cloud radiative
# effect diagnostics) and once with it. Both passes recompute the gas optics and
# the aerosol optics from scratch; only the cloud increment differs. Fusing them
# -- compute the optics once, run the sweep twice -- would save one copy of the
# optics, and nothing else, so the prize is exactly the optics share.
#
# This measures that share by timing the real solve against a kernel that does
# the same per-column, per-g-point work with the sweep removed. The difference
# is the sweep plus flux accumulation.
#
# Not a proposal and not a benchmark of a change: it sizes one before it is
# written. See docs/learnings.md "The RRTMGP ledger" for what has already been
# tried on these kernels.

import CUDA
import ClimaComms
import ClimaAtmos as CA
import RRTMGP
import RRTMGP.RTESolver as RTE
import RRTMGP.Optics: compute_optical_props!
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

# Radiation fires every dt_rad, so step until it has run at least once and the
# state it reads is settled
for i in 1:3
    @info "warmup step $i / 3"
    step!(cs)
end

p = cs.model_sims.atmos_sim.integrator.p
solver = p.radiation.rrtmgp_solver
lk = solver.lookups
as = solver.as
nlay, ncol = RRTMGP.AtmosphericStates.get_dims(as)

# Optics only: the body of the real kernel with rte_*_2stream! and the flux
# accumulation removed. Everything else -- the masks, the g-point loop, the
# per-layer gas/cloud/aerosol optics -- is what the real kernel does.
function lw_optics_only!(op, as, state_cache, src, lkp, lkp_cld, lkp_aero, ncol, n_gpt)
    gcol = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if gcol ≤ ncol
        RTE._compute_aero_mask!(as.aerosol_state, gcol)
        @inbounds for igpt in 1:n_gpt
            RTE._build_cloud_mask!(as.cloud_state, Val(:mask_lw), gcol)
            compute_optical_props!(
                op, as, state_cache, src, gcol, igpt, lkp, lkp_cld, lkp_aero,
            )
        end
    end
    return nothing
end

function sw_optics_only!(op, as, state_cache, lkp, lkp_cld, lkp_aero, ncol, n_gpt)
    gcol = threadIdx().x + (blockIdx().x - 1) * blockDim().x
    if gcol ≤ ncol
        RTE._compute_aero_mask!(as.aerosol_state, gcol)
        @inbounds for igpt in 1:n_gpt
            RTE._build_cloud_mask!(as.cloud_state, Val(:mask_sw), gcol)
            compute_optical_props!(
                op, as, state_cache, gcol, igpt, lkp, lkp_cld, lkp_aero,
            )
        end
    end
    return nothing
end

const THREADS = 256
nblocks(n) = cld(n, THREADS)

# Median of repeats: one timing on this machine is worth little, and the mean
# is dragged by whatever else lands on the GPU
function timed_ms(f; repeats = 7)
    f()
    CUDA.synchronize()
    ts = Float64[]
    for _ in 1:repeats
        CUDA.synchronize()
        t = CUDA.@elapsed f()
        push!(ts, t * 1e3)
    end
    sort!(ts)
    return ts[(length(ts) + 1) ÷ 2]
end

n_gpt_lw = length(lk.lookup_lw.band_data.major_gpt2bnd)
n_gpt_sw = length(lk.lookup_sw.band_data.major_gpt2bnd)
lws, sws = solver.lws, solver.sws

results = Dict{String, Any}(
    "ncol" => ncol,
    "nlay" => nlay,
    "n_gpt_lw" => n_gpt_lw,
    "n_gpt_sw" => n_gpt_sw,
    "note" => "optics_ms is the real kernel with the two-stream sweep and flux " *
              "accumulation removed; sweep_ms is the remainder. A fused " *
              "all-sky/clear-sky solve would save one copy of optics_ms.",
)

# The longwave optics kernel takes the source terms, the shortwave one does
# not, so each band gets its own closure rather than dispatching on a symbol
bands = (
    (
        "lw", lws, lk.lookup_lw, lk.lookup_lw_cld, lk.lookup_lw_aero, n_gpt_lw,
        RTE.solve_lw!,
        (o, sc, lkp, lkp_cld, lkp_aero, n) -> CUDA.@cuda(
            threads = THREADS,
            blocks = nblocks(ncol),
            lw_optics_only!(o, as, sc, lws.src, lkp, lkp_cld, lkp_aero, ncol, n),
        ),
    ),
    (
        "sw", sws, lk.lookup_sw, lk.lookup_sw_cld, lk.lookup_sw_aero, n_gpt_sw,
        RTE.solve_sw!,
        (o, sc, lkp, lkp_cld, lkp_aero, n) -> CUDA.@cuda(
            threads = THREADS,
            blocks = nblocks(ncol),
            sw_optics_only!(o, as, sc, lkp, lkp_cld, lkp_aero, ncol, n),
        ),
    ),
)

for (band, solver_obj, lkp, lkp_cld, lkp_aero, n_gpt, solve!, optics_only) in bands
    allsky = timed_ms(() -> solve!(solver_obj, as, lkp, lkp_cld, lkp_aero, nothing))
    clear = timed_ms(() -> solve!(solver_obj, as, lkp, nothing, lkp_aero, nothing))
    op, state_cache = solver_obj.op, solver_obj.state_cache
    optics = timed_ms(
        () -> optics_only(op, state_cache, lkp, lkp_cld, lkp_aero, n_gpt),
    )
    results[band] = Dict{String, Any}(
        "allsky_ms" => allsky,
        "clearsky_ms" => clear,
        "optics_ms" => optics,
        "sweep_ms" => allsky - optics,
        "optics_frac_of_allsky" => optics / allsky,
        # Both passes pay optics today; a fused solve pays it once
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
