# Does storing the grouped cloud mask column-fast buy time, and does the saving
# survive the register cap that raised occupancy?
#
# Every wide workspace a grouped round writes is column-fast. The optics arrays
# are built by `_coalesced_3d_zeros(DA, FT, ncol, nlay, 3)` and sliced
# `view(op.τ, wcol, :)`; the partial accumulators are `(ncolw, nlay + 1,
# planes)`. The mask was the one exception, `(nlay, ncolw)`, so a thread's
# column was contiguous and a warp's threads stood `nlay` bytes apart. A warp
# then touched 32 sectors per one-byte mask access where it could touch one.
#
# What that costs is wavefronts, not bandwidth. The whole tile a warp walks is
# the same either way and L1 absorbs it -- the radiation sweeps run DRAM at 29%
# of peak -- so the layer-fast layout does not move more bytes off the device,
# it issues 32 times the sector requests to move them. On a kernel whose stalls
# are 80% `long_scoreboard` that is paid in latency.
#
# Which is why the cap state is an axis rather than a constant. Raising
# occupancy from 12 warps to 20 made the radiation sweeps MORE latency-exposed,
# not less (`long_scoreboard` 59% -> 80% across the phase split), so a treatment
# that removes memory requests should be worth more at the shipped caps than
# without them, and a 2x2 says so instead of assuming it. The two arms are timed
# in both cap states against the layer-fast mask of the same state.
#
# The two layouts coexist because the layout is a type. `WideMask` wraps the
# column-fast store and `_mask_column` slices each correctly, so both are
# constructible at once and the comparison is paired inside one job rather than
# taken across two.
#
# A layout cannot change the answer. The McICA draw is keyed on
# `(key, gcol, igpt, ilay)` and not on memory order, so every accumulator and
# the mask itself must come out identical; the mask is compared in a canonical
# orientation so a transposition error cannot read as agreement.
#
# Timing is interleaved and paired for the reason the register-cap experiment
# states: a blocked design on this machine attributes the device's own drift to
# the treatment, and that drift has exceeded the effects being measured.

import CUDA
import RRTMGP
import TOML

project_dir = dirname(Base.active_project())
include(joinpath(project_dir, "code_loading.jl"))

const RTE = RRTMGP.RTE
const RTESolver = RRTMGP.RTESolver
const AS = RRTMGP.AtmosphericStates

# `let` is a hard scope, so assigning `out_path` inside one makes a local and
# leaves the global at its default: the flag parses and is then discarded.
function arg_value(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) && return default
    value = ARGS[i + 1]
    deleteat!(ARGS, i:(i + 1))
    return value
end

out_path = arg_value("--out", "results/radiation-mask-layout.toml")

# Four configurations only, and a round of them is about a second, so the
# rounds are cheap and there are more of them than the register sweep needed.
const ROUNDS = 20
const INNER = 3
const WARMUP_STEPS = 3

const THREADS_PER_WARP = 32

# ---------------------------------------------------------------------------
# The simulation. Three steps so radiation has fired and the state the solve
# reads is settled.
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
nlay, ncol = AS.get_dims(as)
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

# The block size the solve ships, read rather than restated.
const SHIPPED_BLOCK = ext.GROUP_BLOCK_THREADS

# ---------------------------------------------------------------------------
# The two argument tuples a round launches, built exactly as
# `_grouped_rounds!` builds them, except that the mask store is an argument:
# it is the treatment, and the sweeps kernel takes it positionally, so neither
# workspace struct has to be rebuilt to swap it.
# ---------------------------------------------------------------------------
band_optics_args(b, round_index) = (
    ext._optics_workspaces(b.groups)...,
    as,
    b.state_cache,
    b.lookup,
    b.lookup_aero,
    ncol,
    NGROUPS,
    b.n_gpt,
    round_index,
)

band_sweeps_args(b, round_index, mask_store) = (
    b.groups.fluxb,
    b.groups.acc,
    b.groups.acc_clear,
    b.groups.src,
    b.bcs,
    b.groups.op,
    mask_store,
    b.groups.cloudy,
    nlay,
    ncol,
    as,
    b.lookup,
    b.lookup_cld,
    b.lookup_precip,
    NGROUPS,
    b.n_gpt,
    round_index,
)

# The bands, each carrying everything a round of it needs. `lookup_precip` is
# `nothing` because that is what the AMIP solve passes.
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
    AS.refresh_transposed_state!(b.state_cache, as)
end

# The real solves, run once so the aerosol mask the rounds read is the one the
# solve would have hoisted, and so the live buffers hold a complete answer.
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
# The two mask stores. The shipped one is whatever the workspace was built
# with, read rather than reconstructed, so this cannot measure a store the
# solve does not use; its counterpart is allocated here in the other layout.
# Neither needs seeding: a round writes the mask before it reads it.
# ---------------------------------------------------------------------------
const NCOLW = ncol * NGROUPS

layer_fast_store() = CUDA.CuArray{Bool}(undef, nlay, NCOLW)
column_fast_store() = AS.WideMask(CUDA.CuArray{Bool}(undef, NCOLW, nlay))

# The canonical orientation is `(nlay, ncolw)`, so the two stores' contents are
# comparable element by element and a transposition error cannot read as
# agreement.
canonical_mask(m::AS.WideMask) = permutedims(Array(m.data))
canonical_mask(m) = Array(m)

function band_mask_stores(b)
    shipped = b.groups.mask
    if shipped isa AS.WideMask
        size(shipped.data) == (NCOLW, nlay) || error(
            "the shipped column-fast mask is $(size(shipped.data)), " *
            "expected $((NCOLW, nlay))",
        )
        return (column_fast = shipped, layer_fast = layer_fast_store())
    end
    size(shipped) == (nlay, NCOLW) || error(
        "the shipped layer-fast mask is $(size(shipped)), " *
        "expected $((nlay, NCOLW))",
    )
    return (layer_fast = shipped, column_fast = column_fast_store())
end

# ---------------------------------------------------------------------------
# The kernels. The optics half does not touch the mask, so it is built once per
# cap state; the sweeps half is built once per cap state and layout, because
# the mask's type is part of its signature.
# ---------------------------------------------------------------------------
function build_kernel(f, args, cap)
    k = if isnothing(cap)
        CUDA.@cuda launch = false always_inline = true f(args...)
    else
        CUDA.@cuda launch = false always_inline = true maxregs = cap f(args...)
    end
    blocks = CUDA.active_blocks(k.fun, SHIPPED_BLOCK)
    rec = Dict{String, Any}(
        "registers" => CUDA.registers(k),
        "local_bytes" => CUDA.memory(k).local,
        "max_threads" => CUDA.maxthreads(k),
        "blocks_per_sm" => blocks,
        "warps_per_sm" => blocks * cld(SHIPPED_BLOCK, THREADS_PER_WARP),
    )
    isnothing(cap) || (rec["cap"] = cap)
    return k, rec
end

# The caps the solve ships, asked of the extension rather than restated, so
# this experiment cannot run at a budget the solve does not.
band_caps(b) = (
    optics = ext._optics_maxregs(b.groups),
    sweeps = ext._sweeps_maxregs(b.groups),
)

function band_kernels(b, stores)
    caps = band_caps(b)
    optics! = ext._optics_kernel(b.groups)
    sweeps! = ext._sweeps_kernel(b.groups)
    built = Dict{String, Any}()
    records = Dict{String, Any}()
    oargs = band_optics_args(b, 1)
    for (state, cap) in (("uncapped", nothing), ("capped", caps.optics))
        k, rec = build_kernel(optics!, oargs, cap)
        built["optics_$state"] = k
        records["$(b.name)_optics_$state"] = rec
    end
    for (state, cap) in (("uncapped", nothing), ("capped", caps.sweeps))
        for layout in ("layer_fast", "column_fast")
            sargs = band_sweeps_args(b, 1, getproperty(stores, Symbol(layout)))
            k, rec = build_kernel(sweeps!, sargs, cap)
            rec["layout"] = layout
            built["sweeps_$(state)_$(layout)"] = k
            records["$(b.name)_sweeps_$(state)_$(layout)"] = rec
        end
    end
    println("\n=== $(b.name) kernels, at $SHIPPED_BLOCK threads ===")
    println("  shipped caps: optics $(caps.optics), sweeps $(caps.sweeps)")
    for key in sort(collect(keys(records)))
        r = records[key]
        println(
            "  $key: $(r["registers"]) regs, $(r["local_bytes"]) B local, " *
            "$(r["warps_per_sm"]) warps/SM",
        )
    end
    return built, records, caps
end

# ---------------------------------------------------------------------------
# The grid. Two cap states by two layouts, and each layout is differenced
# against the layer-fast arm of its own cap state -- the comparison is the
# layout, with the cap state held.
# ---------------------------------------------------------------------------
const CAP_STATES = ("uncapped", "capped")
const LAYOUTS = ("layer_fast", "column_fast")

configuration_name(state, layout) = "$(state)_$(layout)"

function grid_configs(stores)
    configs = Any[]
    for state in CAP_STATES
        for layout in LAYOUTS
            push!(
                configs,
                (
                    name = configuration_name(state, layout),
                    optics = "optics_$state",
                    sweeps = "sweeps_$(state)_$(layout)",
                    mask = getproperty(stores, Symbol(layout)),
                    state = state,
                    layout = layout,
                ),
            )
        end
    end
    return configs
end

# ---------------------------------------------------------------------------
# Timing.
# ---------------------------------------------------------------------------
median_of(ts) = (s = sort(ts); s[(length(s) + 1) ÷ 2])

function timed_rounds!(b, ko, ks, mask_store)
    tx = min(SHIPPED_BLOCK, NCOLW)
    bx = cld(NCOLW, tx)
    for round_index in 1:cld(b.n_gpt, NGROUPS)
        ko(band_optics_args(b, round_index)...; threads = tx, blocks = bx)
        ks(
            band_sweeps_args(b, round_index, mask_store)...;
            threads = tx,
            blocks = bx,
        )
    end
    return nothing
end

function time_config(b, built, cfg)
    ko = built[cfg.optics]
    ks = built[cfg.sweeps]
    run() = timed_rounds!(b, ko, ks, cfg.mask)
    run()
    CUDA.synchronize()
    ts = Float64[]
    for _ in 1:INNER
        CUDA.synchronize()
        push!(ts, CUDA.@elapsed(run()) * 1e3)
    end
    return median_of(ts)
end

# One round visits every configuration once, rotated by the round index, so a
# drift slow compared with a round falls on both arms of every difference.
function visit_order(configs, index)
    n = length(configs)
    return [configs[mod1(i + index - 1, n)] for i in 1:n]
end

function interleave(b, built, configs)
    samples = Dict{String, Vector{Float64}}()
    println(
        "\n=== $(b.name) rounds, $(length(configs)) configurations, " *
        "$ROUNDS interleaved rounds ===",
    )
    for index in 1:ROUNDS
        for cfg in visit_order(configs, index)
            push!(
                get!(samples, cfg.name, Float64[]),
                time_config(b, built, cfg),
            )
        end
        line = join(
            [
                "$(cfg.name) $(round(samples[cfg.name][end], digits = 2))" for
                cfg in configs
            ],
            ", ",
        )
        println("  round $index: $line")
    end
    return samples
end

# ---------------------------------------------------------------------------
# Equivalence. The mask is part of the snapshot, in the canonical orientation,
# because it is the array the treatment moves and comparing only the fluxes
# would pass a transposition that happened to be unread.
# ---------------------------------------------------------------------------
function rounds_snapshot(groups, mask_store)
    shot = Dict{String, Any}()
    for (name, nt) in (("acc", groups.acc), ("acc_clear", groups.acc_clear))
        for f in propertynames(nt)
            v = getproperty(nt, f)
            v isa CUDA.CuArray || continue
            shot["$name.$f"] = Array(v)
        end
    end
    shot["cloudy"] = Array(groups.cloudy)
    shot["mask"] = canonical_mask(mask_store)
    return shot
end

function differing_values(a, b)
    n = 0
    for key in keys(a)
        x, y = a[key], b[key]
        size(x) == size(y) ||
            error("`$key` changed shape between configurations")
        n += count(.!isequal.(x, y))
    end
    return n
end

total_values(a) = sum(length(v) for v in values(a))

# Bits once per configuration, outside the timed rounds, because copying every
# workspace back is not something to put inside a paired difference.
function compare_bits(b, built, configs)
    differing = Dict{String, Int}()
    reference = nothing
    compared = 0
    for cfg in configs
        timed_rounds!(b, built[cfg.optics], built[cfg.sweeps], cfg.mask)
        CUDA.synchronize()
        shot = rounds_snapshot(b.groups, cfg.mask)
        if isnothing(reference)
            reference = shot
            compared = total_values(shot)
        end
        differing[cfg.name] = differing_values(reference, shot)
    end
    return differing, compared
end

# ---------------------------------------------------------------------------
# Analysis.
# ---------------------------------------------------------------------------

# Two-sided sign test over the differences, exactly. Ties are dropped, which is
# the conservative reading: a round that could not tell the two apart is not
# evidence either way.
function sign_test_p(diffs)
    nonzero = filter(!iszero, diffs)
    n = length(nonzero)
    n == 0 && return 1.0
    k = count(<(0), nonzero)
    tail = min(k, n - k)
    total = sum(binomial(big(n), big(i)) for i in 0:tail)
    return min(1.0, 2.0 * Float64(total / big(2)^n))
end

function effect(samples, name, reference)
    diffs = samples[name] .- samples[reference]
    base = median_of(samples[reference])
    half = length(diffs) ÷ 2
    return Dict{String, Any}(
        "rounds_ms" => median_of(samples[name]),
        "reference_rounds_ms" => base,
        "median_diff_ms" => median_of(diffs),
        "change_pct" => 100.0 * median_of(diffs) / base,
        "rounds_negative" => count(<(0), diffs),
        "rounds_positive" => count(>(0), diffs),
        "sign_test_p" => sign_test_p(diffs),
        # An effect whose two halves disagree is reporting a drift as well as a
        # layout, so a magnitude is quoted against the agreement of its halves.
        "change_pct_first_half" => 100.0 * median_of(diffs[1:half]) / base,
        "change_pct_second_half" =>
            100.0 * median_of(diffs[(half + 1):end]) / base,
        "per_round_diff_ms" => diffs,
    )
end

function analyze_band(b)
    stores = band_mask_stores(b)
    built, records, caps = band_kernels(b, stores)
    configs = grid_configs(stores)
    samples = interleave(b, built, configs)
    differing, compared = compare_bits(b, built, configs)

    points = Dict{String, Any}()
    for state in CAP_STATES
        reference = configuration_name(state, "layer_fast")
        for layout in LAYOUTS
            name = configuration_name(state, layout)
            rec = effect(samples, name, reference)
            rec["layout"] = layout
            rec["cap_state"] = state
            rec["differing_from_first_configuration"] = differing[name]
            rec["bit_for_bit"] = differing[name] == 0
            points[name] = rec
        end
    end
    # What the cap state is worth on its own, at the layer-fast mask the cap
    # experiment measured, so the two treatments can be read against each other
    # in one table.
    cap_only = effect(
        samples,
        configuration_name("capped", "layer_fast"),
        configuration_name("uncapped", "layer_fast"),
    )
    # And both together, against neither.
    both = effect(
        samples,
        configuration_name("capped", "column_fast"),
        configuration_name("uncapped", "layer_fast"),
    )

    uncapped_gain = points[configuration_name("uncapped", "column_fast")]
    capped_gain = points[configuration_name("capped", "column_fast")]
    println(
        "\n  $(b.name): column-fast mask is $(round(uncapped_gain["change_pct"], digits = 2))% " *
        "uncapped and $(round(capped_gain["change_pct"], digits = 2))% at the shipped caps; " *
        "both treatments together $(round(both["change_pct"], digits = 2))%",
    )
    return Dict{String, Any}(
        "band" => b.name,
        "n_gpt" => b.n_gpt,
        "rounds" => cld(b.n_gpt, NGROUPS),
        "shipped_cap_optics" => caps.optics,
        "shipped_cap_sweeps" => caps.sweeps,
        "configurations" => [cfg.name for cfg in configs],
        "points" => points,
        "cap_only" => cap_only,
        "both_treatments" => both,
        "values_compared" => compared,
        "all_bit_for_bit" => all(==(0), values(differing)),
        # The question the 2x2 exists for: a treatment that removes memory
        # requests should be worth MORE where occupancy exposed more memory
        # latency, not less.
        "layout_gain_pct_uncapped" => uncapped_gain["change_pct"],
        "layout_gain_pct_capped" => capped_gain["change_pct"],
        "layout_worth_more_at_the_caps" =>
            capped_gain["change_pct"] < uncapped_gain["change_pct"],
        "layout_is_resolved_at_the_caps" =>
            capped_gain["change_pct"] < 0 && capped_gain["sign_test_p"] < 0.05,
    ), records
end

function run_all()
    bands = Dict{String, Any}()
    records = Dict{String, Any}()
    for b in BANDS
        br, kr = analyze_band(b)
        bands[b.name] = br
        merge!(records, kr)
    end
    return bands, records
end

(bands, kernel_records) = run_all()

results = Dict{String, Any}(
    "config_file" => config_file,
    "float_type" => string(FT),
    "ncol" => ncol,
    "nlay" => nlay,
    "ngroups" => NGROUPS,
    "block_threads" => SHIPPED_BLOCK,
    "rounds_interleaved" => ROUNDS,
    "timings_per_round" => INNER,
    "warmup_steps" => WARMUP_STEPS,
    "kernels" => kernel_records,
    "bands" => bands,
    "verdict" => Dict{String, Any}(
        "bands_where_column_fast_wins" => [
            name for (name, b) in bands if b["layout_is_resolved_at_the_caps"]
        ],
        "column_fast_is_worth_shipping" =>
            all(b["layout_is_resolved_at_the_caps"] for b in values(bands)),
        "worth_more_at_the_caps_everywhere" =>
            all(b["layout_worth_more_at_the_caps"] for b in values(bands)),
        "all_bit_for_bit" => all(b["all_bit_for_bit"] for b in values(bands)),
    ),
    "note" => "Asks what the grouped cloud mask's memory layout is worth, and " *
              "whether it is worth more once the register caps have raised " *
              "occupancy. Every other wide workspace a grouped round writes " *
              "is column-fast -- the optics arrays come from " *
              "`_coalesced_3d_zeros(DA, FT, ncol, nlay, 3)` and the partial " *
              "accumulators are `(ncolw, nlay + 1, planes)` -- so a " *
              "`(nlay, ncolw)` mask stands a warp's threads `nlay` bytes " *
              "apart and makes one one-byte access touch 32 sectors where it " *
              "could touch one. What that costs is wavefronts rather than " *
              "bandwidth: the tile a warp walks is the same either way and " *
              "L1 absorbs it, so the layer-fast layout does not move more " *
              "bytes off the device, it issues 32 times the requests to move " *
              "them. The cap state is an axis and not a constant because " *
              "raising occupancy over the phase split made these kernels MORE " *
              "latency-exposed (`long_scoreboard` 59% -> 80%), so a treatment " *
              "that removes requests should be worth more at the caps than " *
              "without them; `layout_worth_more_at_the_caps` is that " *
              "comparison and `cap_only` and `both_treatments` put the two " *
              "treatments in one table. Each layout is differenced against " *
              "the layer-fast arm of its OWN cap state, so a layout figure " *
              "is never carrying a cap. Timing is interleaved and paired as " *
              "in results/radiation-register-cap.toml, and for the same " *
              "reason: a blocked design on this machine attributes the " *
              "device's drift to the treatment. A layout cannot change the " *
              "answer -- the McICA draw is keyed on `(key, gcol, igpt, " *
              "ilay)` and not on memory order -- so `all_bit_for_bit` must " *
              "hold, and the mask itself is compared in the canonical " *
              "`(nlay, ncolw)` orientation so that a transposition error " *
              "cannot read as agreement.",
)

mkpath(dirname(out_path))
open(io -> TOML.print(io, results; sorted = true), out_path, "w")
verdict_summary = results["verdict"]
@info "wrote $out_path" verdict_summary
