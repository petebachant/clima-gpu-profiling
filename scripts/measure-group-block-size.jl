# Is 128 threads the right block for the grouped radiation rounds, or does
# another size reach the same occupancy?
#
# `GROUP_BLOCK_THREADS` is one constant shared by both bands and both halves of
# a round, and its comment justifies it from the longwave's register counts
# alone. The longwave optics need 156 registers and its sweeps 140, which at
# 128 threads reach 12 warps per multiprocessor against the 8 they get at the
# solve's usual 256. The shortwave's two kernels are cheaper in registers and
# are not mentioned, so whether 128 is also right for them is unasked.
#
# Occupancy is a step function of the block size, not a curve, so several sizes
# can land on the same step and a size that looks finer can land below it. Two
# things decide where the steps are: registers are allocated per warp in
# 256-register units, and a block's warps round up to the multiprocessor's four
# schedulers -- so a 2-warp block pays a 4-warp allocation and can reach fewer
# warps per multiprocessor than a wider one. This reports the driver's own
# answer, `cuOccupancyMaxActiveBlocksPerMultiprocessor`, for every kernel at
# every size, and times the rounds beside it.
#
# The timing launches the shipped kernels through a copy of `_grouped_rounds!`
# that takes the block size as an argument. Nothing in the package changes, so
# the kernels timed here are the ones the solve launches, and the only
# difference between two points is the launch configuration.
#
# Block size cannot change arithmetic: a thread's wide column is its global id
# whatever the block, and a full pass over the rounds assigns on the first
# g-point of each group rather than accumulating into what was there. So every
# point must be bit-for-bit with the shipped one, and this asserts it rather
# than assuming it.

import CUDA
import RRTMGP
import TOML

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

const RTE = RRTMGP.RTE
const RTESolver = RRTMGP.RTESolver

# `let` is a hard scope, so assigning `out_path` inside one makes a local and
# leaves the global at its default: the flag parses and is then discarded.
function arg_value(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) && return default
    value = ARGS[i + 1]
    deleteat!(ARGS, i:(i + 1))
    return value
end

out_path = arg_value("--out", "results/group-block-size.toml")

const REPEATS = 9
const WARMUP_STEPS = 3

# Every multiple of 32 the hardware allows below the solve's own 256, the two
# sizes the comment compares, and the two Teja asks about. 512 is included
# because the shortwave's registers leave room for it and the longwave's do
# not, which is the asymmetry the shared constant hides.
const BLOCK_SIZES = [32, 64, 96, 128, 160, 192, 256, 288, 384, 512]

const THREADS_PER_WARP = 32

# ---------------------------------------------------------------------------
# The simulation. Three steps so radiation has fired and the state the solve
# reads is settled, as the g-point sweeps do.
# ---------------------------------------------------------------------------
config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]

cs = CoupledSimulation(config_file)

function warm_up(cs, n)
    for i in 1:n
        @info "warmup step $i / $n"
        step!(cs)
    end
    return nothing
end
warm_up(cs, WARMUP_STEPS)

solver = cs.model_sims.atmos_sim.integrator.p.radiation.rrtmgp_solver
lws = solver.lws
sws = solver.sws
as = solver.as
lk = solver.lookups
gp = solver.grid_params
nlay, ncol = RRTMGP.AtmosphericStates.get_dims(as)
FT = eltype(gp)

lws isa RTE.TwoStreamLWRTE ||
    error("expected a TwoStreamLWRTE longwave solver, got $(typeof(lws))")
sws isa RTE.TwoStreamSWRTE ||
    error("expected a TwoStreamSWRTE shortwave solver, got $(typeof(sws))")
isnothing(lws.gpt_groups) &&
    error("the longwave solver has no group workspace, so there are no rounds")
isnothing(sws.gpt_groups) &&
    error("the shortwave solver has no group workspace, so there are no rounds")

const NGROUPS = RTE.n_gpt_groups(lws.gpt_groups)
RTE.n_gpt_groups(sws.gpt_groups) == NGROUPS ||
    error("the two bands were built with different group counts")

ext = Base.get_extension(RRTMGP, :RRTMGPCUDAExt)
isnothing(ext) && error("RRTMGPCUDAExt is not loaded, so no kernel can be built")

# The shipped value, read rather than restated, so the document can name it
# without a second copy drifting from the package.
const SHIPPED_BLOCK = ext.GROUP_BLOCK_THREADS
SHIPPED_BLOCK in BLOCK_SIZES ||
    error("the shipped block size $SHIPPED_BLOCK is not among the sizes swept")

# ---------------------------------------------------------------------------
# The two argument tuples a round launches, built exactly as
# `_grouped_rounds!` builds them. One definition feeds both the timing and the
# compile-only build below, so a signature change cannot make the occupancy
# table describe a kernel other than the one being timed.
# ---------------------------------------------------------------------------
band_optics_args(b, round) = (
    ext._optics_workspaces(b.groups)...,
    as,
    b.state_cache,
    b.lookup,
    b.lookup_aero,
    ncol,
    NGROUPS,
    b.n_gpt,
    round,
)

band_sweeps_args(b, round) = (
    b.groups.fluxb,
    b.groups.acc,
    b.groups.acc_clear,
    b.groups.src,
    b.bcs,
    b.groups.op,
    b.groups.mask,
    b.groups.cloudy,
    nlay,
    ncol,
    as,
    b.lookup,
    b.lookup_cld,
    b.lookup_precip,
    NGROUPS,
    b.n_gpt,
    round,
)

# A copy of `_grouped_rounds!` with the block size as an argument. The launch
# arithmetic is `_configure_threadblock`'s, restated here because it is
# internal to the extension.
function timed_rounds!(b, block)
    nwide = ncol * NGROUPS
    tx = min(block, nwide)
    bx = cld(nwide, tx)
    optics! = ext._optics_kernel(b.groups)
    sweeps! = ext._sweeps_kernel(b.groups)
    for round in 1:cld(b.n_gpt, NGROUPS)
        CUDA.@cuda always_inline = true threads = (tx) blocks = (bx) optics!(
            band_optics_args(b, round)...,
        )
        CUDA.@cuda always_inline = true threads = (tx) blocks = (bx) sweeps!(
            band_sweeps_args(b, round)...,
        )
    end
    return nothing
end

# The bands, each carrying everything a round of it needs. `lookup_precip` is
# `nothing` because that is what the AMIP solve passes: `solve_lw_both_skies!`
# defaults it and the coupler calls it without one.
longwave = (
    name = "lw",
    groups = lws.gpt_groups,
    bcs = lws.bcs,
    state_cache = lws.state_cache,
    lookup = lk.lookup_lw,
    lookup_cld = lk.lookup_lw_cld,
    lookup_aero = lk.lookup_lw_aero,
    lookup_precip = nothing,
    n_gpt = length(lk.lookup_lw.band_data.major_gpt2bnd),
)
shortwave = (
    name = "sw",
    groups = sws.gpt_groups,
    bcs = sws.bcs,
    state_cache = sws.state_cache,
    lookup = lk.lookup_sw,
    lookup_cld = lk.lookup_sw_cld,
    lookup_aero = lk.lookup_sw_aero,
    lookup_precip = nothing,
    n_gpt = length(lk.lookup_sw.band_data.major_gpt2bnd),
)
const BANDS = (longwave, shortwave)

# The optics kernels read the transposed state cache, which the solve refreshes
# on entry. Calling the rounds directly skips that, so refresh it once here
# against the settled state every point then shares.
for b in BANDS
    RRTMGP.AtmosphericStates.refresh_transposed_state!(b.state_cache, as)
end

# The real solves, run once so the aerosol mask the rounds read is the one the
# solve would have hoisted, and so the live buffers hold a complete answer
# rather than one band's rounds without its reduction.
RTESolver.solve_lw_both_skies!(
    lws,
    solver.clear_flux_acc_lw,
    as,
    lk.lookup_lw,
    lk.lookup_lw_cld,
    lk.lookup_lw_aero,
    solver.deep_atmosphere_inverse_scaling,
)
RTESolver.solve_sw_both_skies!(
    sws,
    solver.clear_flux_acc_sw,
    as,
    lk.lookup_sw,
    lk.lookup_sw_cld,
    lk.lookup_sw_aero,
    solver.deep_atmosphere_inverse_scaling,
)
CUDA.synchronize()

# ---------------------------------------------------------------------------
# What ptxas and the driver say about each kernel, at every block size.
# ---------------------------------------------------------------------------
driver_warps_per_sm(k, block) =
    CUDA.active_blocks(k.fun, block) * cld(block, THREADS_PER_WARP)

function kernel_record(name, k)
    regs = CUDA.registers(k)
    maxthreads = CUDA.maxthreads(k)
    warps = Dict{String, Any}()
    for block in BLOCK_SIZES
        warps[string(block)] =
            block <= maxthreads ? driver_warps_per_sm(k, block) : 0
    end
    reached = maximum(values(warps))
    rec = Dict{String, Any}(
        "registers" => regs,
        "local_bytes" => CUDA.memory(k).local,
        "max_threads" => maxthreads,
        "warps_per_sm" => warps,
        "best_warps_per_sm" => reached,
        "blocks_at_best" =>
            sort([parse(Int, b) for (b, w) in warps if w == reached]),
    )
    println(
        "  $name: $regs regs, $(rec["local_bytes"]) B local, " *
        "max $maxthreads threads, best $reached warps/SM at " *
        "$(rec["blocks_at_best"])",
    )
    return rec
end

function kernel_records()
    recs = Dict{String, Any}()
    println("\n=== kernels ===")
    for b in BANDS
        optics! = ext._optics_kernel(b.groups)
        sweeps! = ext._sweeps_kernel(b.groups)
        ko = CUDA.@cuda launch = false always_inline = true optics!(
            band_optics_args(b, 1)...,
        )
        ks = CUDA.@cuda launch = false always_inline = true sweeps!(
            band_sweeps_args(b, 1)...,
        )
        recs["$(b.name)_optics"] = kernel_record("$(b.name)_optics", ko)
        recs["$(b.name)_sweeps"] = kernel_record("$(b.name)_sweeps", ks)
    end
    return recs
end

kernels = kernel_records()

# A block size every kernel can take, and at which every kernel reaches the
# most warps per multiprocessor it reaches anywhere in this sweep. This is the
# question asked directly: which other sizes give the same occupancy.
function blocks_at_everyones_best(kernels)
    return [
        block for block in BLOCK_SIZES if all(
            rec["warps_per_sm"][string(block)] == rec["best_warps_per_sm"] for
            rec in values(kernels)
        )
    ]
end

equal_blocks = blocks_at_everyones_best(kernels)

# ---------------------------------------------------------------------------
# Timing. Median of repeats: one timing on this machine is worth little, and
# the mean is dragged by whatever else lands on the GPU.
# ---------------------------------------------------------------------------
function timed_ms(f; repeats = REPEATS)
    f()
    CUDA.synchronize()
    ts = Float64[]
    for _ in 1:repeats
        CUDA.synchronize()
        push!(ts, CUDA.@elapsed(f()) * 1e3)
    end
    sort!(ts)
    return (median = ts[(length(ts) + 1) ÷ 2], min = first(ts), max = last(ts))
end

# What a pass over the rounds wrote, on the host, so a later point cannot
# overwrite the comparison. Every field the rounds touch is here: both sets of
# partial fluxes and the per-thread cloudy tally.
function rounds_snapshot(groups)
    shot = Dict{String, Any}()
    for (name, nt) in (("acc", groups.acc), ("acc_clear", groups.acc_clear))
        for f in propertynames(nt)
            v = getproperty(nt, f)
            v isa CUDA.CuArray || continue
            shot["$name.$f"] = Array(v)
        end
    end
    shot["cloudy"] = Array(groups.cloudy)
    return shot
end

function differing_values(a, b)
    n = 0
    for key in keys(a)
        x, y = a[key], b[key]
        size(x) == size(y) || error("`$key` changed shape between block sizes")
        n += count(.!isequal.(x, y))
    end
    return n
end

total_values(a) = sum(length(v) for v in values(a))

# The largest block any of this band's kernels can take. A size past it is not
# a slow point, it is a launch failure, so it is skipped and recorded as such.
function band_max_threads(kernels, band)
    return min(
        kernels["$(band)_optics"]["max_threads"],
        kernels["$(band)_sweeps"]["max_threads"],
    )
end

# Everything that accumulates across the sweep lives in here. A `for` at top
# level makes its assignments local, so a reference captured in one iteration
# is not visible in the next -- the UndefVarError that has cost this project
# four reservations.
function sweep_band(b, kernels)
    points = Dict{String, Any}()
    limit = band_max_threads(kernels, b.name)
    println("\n=== $(b.name) rounds, max $limit threads ===")

    # The shipped size first, so it is the reference every other point is
    # compared against in both time and bits.
    reference = nothing
    reference_ms = 0.0
    for block in vcat(SHIPPED_BLOCK, filter(!=(SHIPPED_BLOCK), BLOCK_SIZES))
        if block > limit
            points[string(block)] = Dict{String, Any}(
                "block_threads" => block,
                "skipped_too_many_registers" => true,
                "band_max_threads" => limit,
            )
            println("  $block: skipped, past this band's $limit-thread limit")
            continue
        end
        t = timed_ms(() -> timed_rounds!(b, block))
        CUDA.synchronize()
        shot = rounds_snapshot(b.groups)
        if isnothing(reference)
            reference = shot
            reference_ms = t.median
        end
        differing = differing_values(reference, shot)
        points[string(block)] = Dict{String, Any}(
            "block_threads" => block,
            "rounds_ms" => t.median,
            "rounds_ms_min" => t.min,
            "rounds_ms_max" => t.max,
            "change_pct" => 100.0 * (t.median - reference_ms) / reference_ms,
            "warps_per_sm_optics" =>
                kernels["$(b.name)_optics"]["warps_per_sm"][string(block)],
            "warps_per_sm_sweeps" =>
                kernels["$(b.name)_sweeps"]["warps_per_sm"][string(block)],
            "values_differing_from_shipped" => differing,
            "bit_for_bit_with_shipped" => differing == 0,
        )
        println(
            "  $block: $(round(t.median, digits = 2)) ms, " *
            "$(round(points[string(block)]["change_pct"], digits = 2))% vs " *
            "$SHIPPED_BLOCK, $differing of $(total_values(shot)) values differ",
        )
    end
    return points, total_values(reference)
end

function sweep(kernels)
    bands = Dict{String, Any}()
    for b in BANDS
        points, compared = sweep_band(b, kernels)
        measured = [
            block for block in BLOCK_SIZES if
            !get(points[string(block)], "skipped_too_many_registers", false)
        ]
        # The spread of the reference point's own repeats, which is what a
        # difference between two points has to clear to be a difference.
        spread_pct =
            100.0 *
            (
                points[string(SHIPPED_BLOCK)]["rounds_ms_max"] -
                points[string(SHIPPED_BLOCK)]["rounds_ms_min"]
            ) / points[string(SHIPPED_BLOCK)]["rounds_ms"]
        best = minimum(points[string(n)]["change_pct"] for n in measured)
        best_block = minimum(
            n for n in measured if points[string(n)]["change_pct"] == best
        )
        bands[b.name] = Dict{String, Any}(
            "band" => b.name,
            "n_gpt" => b.n_gpt,
            "rounds" => cld(b.n_gpt, NGROUPS),
            "measured_blocks" => measured,
            "band_max_threads" => band_max_threads(kernels, b.name),
            "repeat_spread_pct" => spread_pct,
            "shipped_rounds_ms" =>
                points[string(SHIPPED_BLOCK)]["rounds_ms"],
            "best_change_pct" => best,
            "best_block_threads" => best_block,
            "best_beats_shipped" => best < -spread_pct,
            "values_compared" => compared,
            "all_bit_for_bit" => all(
                get(points[string(n)], "bit_for_bit_with_shipped", true) for
                n in measured
            ),
            "points" => points,
        )
    end
    return bands
end

bands = sweep(kernels)

# Does any block size beat the shipped one in a way the repeats can resolve, in
# either band? This is the decision the sweep exists to make, written as a
# field rather than left to a reader comparing two percentages.
function verdict(bands)
    beaten = [name for (name, b) in bands if b["best_beats_shipped"]]
    return Dict{String, Any}(
        "bands_with_a_faster_block" => sort(beaten),
        "shipped_block_is_best" => isempty(beaten),
        "bands_disagree_on_best" =>
            length(unique(b["best_block_threads"] for b in values(bands))) > 1,
    )
end

results = Dict{String, Any}(
    "ncol" => ncol,
    "nlay" => nlay,
    "ngroups" => NGROUPS,
    "float_type" => string(FT),
    "repeats" => REPEATS,
    "warmup_steps" => WARMUP_STEPS,
    "config_file" => config_file,
    "block_sizes" => BLOCK_SIZES,
    "shipped_block_threads" => SHIPPED_BLOCK,
    "blocks_at_equal_occupancy" => equal_blocks,
    "n_blocks_at_equal_occupancy" => length(equal_blocks),
    "shipped_is_at_equal_occupancy" => SHIPPED_BLOCK in equal_blocks,
    "kernels" => kernels,
    "bands" => bands,
    "verdict" => verdict(bands),
    "note" => "rounds_ms times the grouped g-point rounds of one band -- " *
              "every round's optics and sweeps launch, which is what the " *
              "block size configures -- and not the whole solve, so it " *
              "excludes the reduction and the state refresh, whose launches " *
              "the block size does not reach. Every point shares one " *
              "simulation's state, lookups and workspaces and differs only " *
              "in the launch configuration, so these are in-block " *
              "comparisons and not a difference of runs. change_pct is " *
              "against the shipped block size, and a point is only faster if " *
              "it clears repeat_spread_pct, the shipped point's own spread. " *
              "warps_per_sm is the CUDA driver's " *
              "cuOccupancyMaxActiveBlocksPerMultiprocessor times the block's " *
              "warps, not a model of the register file. A block size past a " *
              "kernel's register budget cannot be launched at all and is " *
              "recorded as skipped_too_many_registers rather than timed. " *
              "bit_for_bit_with_shipped must hold everywhere: a thread's " *
              "wide column is its global id whatever the block, so a " *
              "difference would be a bug and not roundoff.",
)

mkpath(dirname(out_path))
open(io -> TOML.print(io, results; sorted = true), out_path, "w")
verdict_summary = results["verdict"]
@info "wrote $out_path" equal_blocks verdict_summary
