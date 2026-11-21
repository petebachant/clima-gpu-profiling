# Does shedding registers off a grouped radiation kernel buy time, now that the
# phase split has taken them off the 255-register cap?
#
# The split left four kernels well below the ceiling -- the longwave optics at
# 156 registers and its sweeps at 140, the shortwave at 124 and 117 -- and each
# of them sits at the most warps per multiprocessor its own count permits, so
# nothing is lost to the launch configuration. What remains is the count.
# `results/group-block-size.toml` closed the launch side of this; this asks the
# register side.
#
# The question is not whether occupancy would rise. That is arithmetic, and the
# driver will confirm it before anything is timed. The question is what `ptxas`
# charges to meet the budget. Two earlier attempts on the FUSED solvers were
# measured and lost, both at 128 registers, because cramming a 255-register
# live state into 128 roughly doubled a local frame that was already large. A
# cut of a dozen registers off a kernel carrying 32 bytes of local memory is a
# different proposition, and `register-cap-filter` found that `ptxas` meets a
# cut that small almost entirely without spilling.
#
# That is also the trap. A near-zero spill does not make a cap free: the
# allocator can meet a tighter budget by rematerializing values instead, which
# costs instructions and no local memory at all, so the local frame reads clean
# in exactly the case that needs detecting. A ClimaLand kernel in this project's
# history regressed a fifth while spilling a few dozen bytes. So the local
# frame is recorded as a diagnostic and the decision rests on the clock.
#
# Caps are not hand-picked. For each kernel the highest register count strictly
# below its own that the allocation granularity says reaches more warps is
# computed, and then the driver is asked whether it does -- `predicted_warps`
# and `warps_per_sm` are both recorded, so a step this script got wrong shows up
# as a disagreement rather than as a silent premise. A second, deeper step is
# compiled beside it to show the shape of the trade.
#
# Timing is interleaved and paired, not blocked. Timing every repeat of one
# configuration and then every repeat of the next attributes to the cap
# whatever else moved on the device between the two blocks, and on this machine
# that drift has been larger than the effect being measured. Each round visits
# every configuration once, the visiting order rotates by one per round, and a
# configuration's effect is the median of its per-round differences against the
# uncapped point in the same round. The sign of that effect is tested over the
# rounds, and the rounds are split in half and reported separately, because an
# effect that holds its magnitude across a run is worth more than one that does
# not.
#
# A cap cannot change arithmetic -- the allocator spills and rematerializes, it
# does not reassociate -- so every point should be bit-for-bit with the uncapped
# one. That is recorded rather than assumed: if it fails, the cap is doing
# something other than what this measures.

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

out_path = arg_value("--out", "results/radiation-register-cap.toml")

# Rounds of the interleaved visit, and timings per configuration per round. The
# round is the unit of pairing, so there are more rounds than there would be
# repeats in a blocked design, and few timings inside each.
const ROUNDS = 12
const INNER = 3
const WARMUP_STEPS = 3

const THREADS_PER_WARP = 32

# An A100 multiprocessor's register file, and the unit a warp's registers are
# allocated in. Both are properties of compute capability 8.0.
const REGS_PER_SM = 65536
const ALLOC_UNIT = 256
const MAX_WARPS_PER_SM = 64

# Candidate caps are multiples of this, because a count between two multiples
# buys no warps: it rounds up to the same allocation unit as the one above it.
const CAP_GRANULE = ALLOC_UNIT ÷ THREADS_PER_WARP

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

# The block size the solve ships, read rather than restated. Every point here
# launches at it: this experiment varies the register budget and nothing else,
# and the block size was settled separately.
const SHIPPED_BLOCK = ext.GROUP_BLOCK_THREADS

# ---------------------------------------------------------------------------
# The two argument tuples a round launches, built exactly as
# `_grouped_rounds!` builds them. One definition feeds both the compiles and
# the timing, so a signature change cannot make the register table describe a
# kernel other than the one being timed.
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
    RRTMGP.AtmosphericStates.refresh_transposed_state!(b.state_cache, as)
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
# What a register count buys at this block size, and which cap to ask for.
# ---------------------------------------------------------------------------

# The granularity-correct warps per multiprocessor: a warp's registers round up
# to `ALLOC_UNIT`, and a multiprocessor admits whole blocks. Dividing the
# register file by the thread count instead gets several A100 steps wrong, which
# is why this is only ever a candidate generator -- the driver decides.
function model_warps_per_sm(registers, block_threads)
    per_warp = cld(registers * THREADS_PER_WARP, ALLOC_UNIT) * ALLOC_UNIT
    warps_per_block = cld(block_threads, THREADS_PER_WARP)
    blocks = (REGS_PER_SM ÷ (per_warp * warps_per_block))
    return min(blocks * warps_per_block, MAX_WARPS_PER_SM)
end

# The `how_many` highest multiples of the granule strictly below `registers`
# that the model says each reach more warps than the step above. Returned
# largest-first: a kernel with no step left returns an empty vector. Four steps
# rather than two because the first pass found longwave still falling at the
# bottom of a two-step range, so a two-step search cannot see its own floor.
function candidate_caps(registers, block_threads, how_many)
    caps = Int[]
    reached = model_warps_per_sm(registers, block_threads)
    cap = (Int(registers) ÷ CAP_GRANULE) * CAP_GRANULE
    cap == Int(registers) && (cap -= CAP_GRANULE)
    while cap >= CAP_GRANULE && length(caps) < how_many
        warps = model_warps_per_sm(cap, block_threads)
        if warps > reached
            push!(caps, cap)
            reached = warps
        end
        cap -= CAP_GRANULE
    end
    return caps
end

driver_warps_per_sm(k, block) =
    CUDA.active_blocks(k.fun, block) * cld(block, THREADS_PER_WARP)

# Build one kernel, capped or not, and read back what it cost. A compile that
# fails is recorded with its whole error rather than killing the run: a cap the
# allocator cannot meet is a result, and truncating the error once cost this
# project a diagnosis.
function build_kernel(f, args, cap)
    try
        k = if isnothing(cap)
            CUDA.@cuda launch = false always_inline = true f(args...)
        else
            CUDA.@cuda launch = false always_inline = true maxregs = cap f(args...)
        end
        rec = Dict{String, Any}(
            "registers" => CUDA.registers(k),
            "local_bytes" => CUDA.memory(k).local,
            "max_threads" => CUDA.maxthreads(k),
            "warps_per_sm" => driver_warps_per_sm(k, SHIPPED_BLOCK),
            "blocks_per_sm" => CUDA.active_blocks(k.fun, SHIPPED_BLOCK),
            "compiled" => true,
        )
        isnothing(cap) || (rec["cap"] = cap)
        return k, rec
    catch err
        buf = IOBuffer()
        showerror(buf, err)
        return nothing,
        Dict{String, Any}(
            "compiled" => false,
            "cap" => isnothing(cap) ? -1 : cap,
            "error" => String(take!(buf)),
        )
    end
end

# Every kernel of a band's round, uncapped and at each candidate cap. The
# uncapped build has to succeed -- it is the kernel the solve launches -- and a
# capped one need not.
function band_kernels(b, how_many)
    halves = (
        ("optics", ext._optics_kernel(b.groups), band_optics_args(b, 1)),
        ("sweeps", ext._sweeps_kernel(b.groups), band_sweeps_args(b, 1)),
    )
    built = Dict{String, Any}()
    records = Dict{String, Any}()
    for (half, f, args) in halves
        base_k, base_rec = build_kernel(f, args, nothing)
        isnothing(base_k) && error(
            "the uncapped $(b.name) $half kernel did not compile, so there " *
            "is no reference to measure against:\n$(base_rec["error"])",
        )
        natural = base_rec["registers"]
        base_rec["predicted_warps"] =
            model_warps_per_sm(natural, SHIPPED_BLOCK)
        base_rec["model_agrees_with_driver"] =
            base_rec["predicted_warps"] == base_rec["warps_per_sm"]
        built["$(half)_base"] = base_k
        records["$(b.name)_$(half)"] = base_rec
        caps = candidate_caps(natural, SHIPPED_BLOCK, how_many)
        println(
            "  $(b.name) $half: $natural regs, " *
            "$(base_rec["local_bytes"]) B local, " *
            "$(base_rec["warps_per_sm"]) warps/SM " *
            "(model says $(base_rec["predicted_warps"])), caps to try $caps",
        )
        for cap in caps
            k, rec = build_kernel(f, args, cap)
            rec["requested_cap"] = cap
            rec["natural_registers"] = natural
            rec["predicted_warps"] = model_warps_per_sm(cap, SHIPPED_BLOCK)
            if rec["compiled"]
                rec["met_cap"] = rec["registers"] <= cap
                rec["registers_shed"] = natural - rec["registers"]
                rec["local_bytes_added"] =
                    rec["local_bytes"] - base_rec["local_bytes"]
                rec["warps_gained"] =
                    rec["warps_per_sm"] - base_rec["warps_per_sm"]
                rec["model_agrees_with_driver"] =
                    rec["predicted_warps"] == rec["warps_per_sm"]
                built["$(half)_$(cap)"] = k
                println(
                    "    cap $cap: $(rec["registers"]) regs, " *
                    "$(rec["local_bytes"]) B local " *
                    "(+$(rec["local_bytes_added"])), " *
                    "$(rec["warps_per_sm"]) warps/SM " *
                    "(+$(rec["warps_gained"]), model says " *
                    "$(rec["predicted_warps"]))",
                )
            else
                println("    cap $cap: did not compile\n$(rec["error"])")
            end
            records["$(b.name)_$(half)_cap$(cap)"] = rec
        end
    end
    return built, records
end

function all_kernels(how_many)
    built = Dict{String, Any}()
    records = Dict{String, Any}()
    println("\n=== kernels, at $SHIPPED_BLOCK threads ===")
    for b in BANDS
        bb, br = band_kernels(b, how_many)
        built[b.name] = bb
        merge!(records, br)
    end
    return built, records
end

built, kernel_records = all_kernels(4)

# ---------------------------------------------------------------------------
# The configurations. One per band: the uncapped pair, then each cap applied to
# the sweeps alone, the optics alone, and both. A cap the driver says buys no
# warps is not timed -- it would spend registers for nothing, which is a
# conclusion the compile already reached.
# ---------------------------------------------------------------------------
const UNCAPPED = "uncapped"

function band_configs(b, records)
    configs = [(name = UNCAPPED, optics = "optics_base", sweeps = "sweeps_base")]
    skipped = Dict{String, Any}()
    caps = Dict{String, Vector{Int}}()
    for half in ("optics", "sweeps")
        caps[half] = sort(
            [
                rec["requested_cap"] for (key, rec) in records if
                startswith(key, "$(b.name)_$(half)_cap") &&
                rec["compiled"] &&
                get(rec, "warps_gained", 0) > 0
            ];
            rev = true,
        )
        for (key, rec) in records
            startswith(key, "$(b.name)_$(half)_cap") || continue
            if !rec["compiled"]
                skipped[key] = "did not compile"
            elseif get(rec, "warps_gained", 0) <= 0
                skipped[key] = "the driver gives it no more warps"
            end
        end
    end
    for cap in caps["sweeps"]
        push!(
            configs,
            (
                name = "sweeps_$(cap)",
                optics = "optics_base",
                sweeps = "sweeps_$(cap)",
            ),
        )
    end
    for cap in caps["optics"]
        push!(
            configs,
            (
                name = "optics_$(cap)",
                optics = "optics_$(cap)",
                sweeps = "sweeps_base",
            ),
        )
    end
    for cap in intersect(caps["optics"], caps["sweeps"])
        push!(
            configs,
            (
                name = "both_$(cap)",
                optics = "optics_$(cap)",
                sweeps = "sweeps_$(cap)",
            ),
        )
    end
    return configs, skipped
end

# ---------------------------------------------------------------------------
# Timing.
# ---------------------------------------------------------------------------
median_of(ts) = (s = sort(ts); s[(length(s) + 1) ÷ 2])

function timed_rounds!(b, ko, ks)
    nwide = ncol * NGROUPS
    tx = min(SHIPPED_BLOCK, nwide)
    bx = cld(nwide, tx)
    for round in 1:cld(b.n_gpt, NGROUPS)
        ko(band_optics_args(b, round)...; threads = tx, blocks = bx)
        ks(band_sweeps_args(b, round)...; threads = tx, blocks = bx)
    end
    return nothing
end

function time_config(b, built_band, cfg)
    ko = built_band[cfg.optics]
    ks = built_band[cfg.sweeps]
    run() = timed_rounds!(b, ko, ks)
    run()
    CUDA.synchronize()
    ts = Float64[]
    for _ in 1:INNER
        CUDA.synchronize()
        push!(ts, CUDA.@elapsed(run()) * 1e3)
    end
    return median_of(ts)
end

# What a pass over the rounds wrote, on the host, so a later configuration
# cannot overwrite the comparison.
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
        size(x) == size(y) ||
            error("`$key` changed shape between configurations")
        n += count(.!isequal.(x, y))
    end
    return n
end

total_values(a) = sum(length(v) for v in values(a))

# One round visits every configuration once, rotated by the round index, so a
# drift slow compared with a round falls on both arms of every difference.
function visit_order(configs, index)
    n = length(configs)
    return [configs[mod1(i + index - 1, n)] for i in 1:n]
end

function interleave(b, built_band, configs)
    samples = Dict{String, Vector{Float64}}()
    println(
        "\n=== $(b.name) rounds, $(length(configs)) configurations, " *
        "$ROUNDS interleaved rounds ===",
    )
    for index in 1:ROUNDS
        for cfg in visit_order(configs, index)
            ms = time_config(b, built_band, cfg)
            push!(get!(samples, cfg.name, Float64[]), ms)
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

# Bits once per configuration, outside the timed rounds, because copying every
# workspace back is not something to put inside a paired difference.
function compare_bits(b, built_band, configs)
    differing = Dict{String, Int}()
    reference = nothing
    compared = 0
    for cfg in configs
        timed_rounds!(b, built_band[cfg.optics], built_band[cfg.sweeps])
        CUDA.synchronize()
        shot = rounds_snapshot(b.groups)
        if isnothing(reference)
            reference = shot
            compared = total_values(shot)
        end
        differing[cfg.name] = differing_values(reference, shot)
    end
    return differing, compared
end

# ---------------------------------------------------------------------------
# Analysis. The claim is about the per-round differences, so those are what is
# tested.
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
    first_half = diffs[1:half]
    second_half = diffs[(half + 1):end]
    return Dict{String, Any}(
        "rounds_ms" => median_of(samples[name]),
        "median_diff_ms" => median_of(diffs),
        "change_pct" => 100.0 * median_of(diffs) / base,
        "rounds_negative" => count(<(0), diffs),
        "rounds_positive" => count(>(0), diffs),
        "sign_test_p" => sign_test_p(diffs),
        # An effect whose two halves disagree is reporting a drift as well as a
        # cap, so a magnitude is quoted against the agreement of its halves.
        "change_pct_first_half" => 100.0 * median_of(first_half) / base,
        "change_pct_second_half" => 100.0 * median_of(second_half) / base,
        "per_round_diff_ms" => diffs,
    )
end

function analyze_band(b, built_band, records)
    configs, skipped = band_configs(b, records)
    if length(configs) == 1
        println(
            "\n=== $(b.name): no cap buys a warp, so there is nothing to " *
            "time ===",
        )
        return Dict{String, Any}(
            "band" => b.name,
            "n_gpt" => b.n_gpt,
            "rounds" => cld(b.n_gpt, NGROUPS),
            "configurations" => [UNCAPPED],
            "skipped" => skipped,
            "any_cap_timed" => false,
        ), nothing
    end
    samples = interleave(b, built_band, configs)
    differing, compared = compare_bits(b, built_band, configs)
    points = Dict{String, Any}()
    for cfg in configs
        rec = effect(samples, cfg.name, UNCAPPED)
        rec["optics_kernel"] = cfg.optics
        rec["sweeps_kernel"] = cfg.sweeps
        rec["values_differing_from_uncapped"] = differing[cfg.name]
        rec["bit_for_bit_with_uncapped"] = differing[cfg.name] == 0
        points[cfg.name] = rec
        println(
            "  $(cfg.name): $(round(rec["change_pct"], digits = 2))% " *
            "(halves $(round(rec["change_pct_first_half"], digits = 2))% / " *
            "$(round(rec["change_pct_second_half"], digits = 2))%), " *
            "$(rec["rounds_negative"]) of $ROUNDS rounds faster, " *
            "sign test p=$(round(rec["sign_test_p"], digits = 5)), " *
            "$(differing[cfg.name]) of $compared values differ",
        )
    end
    timed = [cfg.name for cfg in configs if cfg.name != UNCAPPED]
    best = minimum(points[n]["change_pct"] for n in timed)
    best_name = first(n for n in timed if points[n]["change_pct"] == best)
    return Dict{String, Any}(
        "band" => b.name,
        "n_gpt" => b.n_gpt,
        "rounds" => cld(b.n_gpt, NGROUPS),
        "configurations" => [cfg.name for cfg in configs],
        "skipped" => skipped,
        "any_cap_timed" => true,
        "uncapped_rounds_ms" => points[UNCAPPED]["rounds_ms"],
        "values_compared" => compared,
        "all_bit_for_bit" =>
            all(points[n]["bit_for_bit_with_uncapped"] for n in timed),
        "best_change_pct" => best,
        "best_configuration" => best_name,
        # A configuration is only faster if its rounds agree on the sign. The
        # spread of one point's own repeats is set by the device's drift, so it
        # is not the bar here; the bar is the sign test.
        "best_is_resolved" =>
            best < 0.0 && points[best_name]["sign_test_p"] < 0.05,
        "points" => points,
    ), samples
end

function run_bands(built, records)
    bands = Dict{String, Any}()
    for b in BANDS
        rec, _ = analyze_band(b, built[b.name], records)
        bands[b.name] = rec
    end
    return bands
end

bands = run_bands(built, kernel_records)

# Did any cap earn its place, in either band? Written as a field rather than
# left to a reader comparing percentages.
function verdict(bands)
    won = [
        name for (name, b) in bands if
        get(b, "any_cap_timed", false) && b["best_is_resolved"]
    ]
    return Dict{String, Any}(
        "bands_with_a_faster_cap" => sort(won),
        "any_cap_is_worth_shipping" => !isempty(won),
        "model_agrees_with_driver_everywhere" => all(
            get(rec, "model_agrees_with_driver", true) for
            rec in values(kernel_records) if get(rec, "compiled", false)
        ),
        "all_bit_for_bit" => all(
            get(b, "all_bit_for_bit", true) for b in values(bands)
        ),
    )
end

results = Dict{String, Any}(
    "ncol" => ncol,
    "nlay" => nlay,
    "ngroups" => NGROUPS,
    "float_type" => string(FT),
    "block_threads" => SHIPPED_BLOCK,
    "rounds_interleaved" => ROUNDS,
    "timings_per_round" => INNER,
    "warmup_steps" => WARMUP_STEPS,
    "config_file" => config_file,
    "regs_per_sm" => REGS_PER_SM,
    "register_alloc_unit" => ALLOC_UNIT,
    "kernels" => kernel_records,
    "bands" => bands,
    "verdict" => verdict(bands),
    "note" => "Asks what `ptxas` charges to shed registers off the four " *
              "kernels the g-point-grouped radiation solves launch, now that " *
              "the phase split has taken them off the 255-register cap. " *
              "Every point launches at the shipped block size and differs " *
              "only in the `maxregs` passed to one or both halves of a " *
              "round, so the comparison is a register budget and not a " *
              "launch configuration -- that question is closed separately in " *
              "results/group-block-size.toml. Caps are computed, not chosen: " *
              "for each kernel the four highest multiples of 8 below its " *
              "own count that the allocation granularity says each reach " *
              "more warps than the step above. `predicted_warps` is that " *
              "model " *
              "and `warps_per_sm` is the driver's own " *
              "cuOccupancyMaxActiveBlocksPerMultiprocessor, so a step the " *
              "model gets wrong reads as a disagreement rather than as a " *
              "silent premise; a cap the driver gives no extra warps is " *
              "recorded in `skipped` and never timed. `local_bytes_added` " *
              "bounds what the cap cost in spill but does not measure it: " *
              "the allocator can also meet a budget by rematerializing, " *
              "which costs instructions and no local memory, so a clean " *
              "local frame is not a free cap and the decision rests on the " *
              "clock. Timing is interleaved and paired rather than blocked: " *
              "each round visits every configuration once in an order that " *
              "rotates by one per round, and `change_pct` is the median of " *
              "the per-round differences against the uncapped point of the " *
              "same round, so a drift slow compared with a round falls on " *
              "both arms of every difference. `sign_test_p` is an exact " *
              "two-sided sign test over those differences and is what " *
              "`best_is_resolved` tests, because the spread of one point's " *
              "own repeats is set by the device's drift and is not an error " *
              "bar for the difference. The rounds are split in half and " *
              "reported as `change_pct_first_half` and " *
              "`change_pct_second_half`: a magnitude whose halves disagree " *
              "is reporting drift as well as a cap. A register budget " *
              "cannot change arithmetic -- the allocator spills and " *
              "rematerializes, it does not reassociate -- so " *
              "`bit_for_bit_with_uncapped` should hold everywhere, and a " *
              "failure means the cap is doing something other than what " *
              "this measures. The grid holds a point per kernel as well as " *
              "the matched pair, because the first pass found the two halves " *
              "compose: at every cap it tried, the matched `both_` point " *
              "landed within 0.03 points of the sum of the two " *
              "single-kernel points, against effects of 2 to 17 points. So " *
              "a band's best budget is read off its single-kernel points " *
              "independently, which is why they are swept deeper than the " *
              "matched pairs need, and the matched pairs are kept as the " *
              "check on that composition rather than as the candidates.",
)

mkpath(dirname(out_path))
open(io -> TOML.print(io, results; sorted = true), out_path, "w")
verdict_summary = results["verdict"]
@info "wrote $out_path" verdict_summary
