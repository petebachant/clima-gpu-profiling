# What the widened g-point workspaces cost in device memory, as a law rather
# than a figure.
#
# `default_ngpt_groups` sizes the split from the column count alone, so the
# memory it commits rises with a level count it never reads. This measures
# bytes per wide column and per level for both bands, fits them on grids small
# enough to hold several at once, and then prices the shipped default on the
# grids that actually run it -- including one measured point held out of the fit
# so the prediction has something to be wrong against.
#
# Nothing here steps a model: the workspaces are constructed from grid
# parameters alone, which is the whole of what their size depends on.

import ClimaComms
ClimaComms.@import_required_backends
import ClimaParams
import CUDA
import RRTMGP
import RRTMGP.RTE
import RRTMGP.Parameters.RRTMGPParameters
import TOML

const FT = Float32

# Fit grid: column counts either side of the configurations priced below, and
# level counts spanning the high-resolution benchmark's 63 and AMIP's 80.
const FIT_NCOLS = [6144, 24576]
const FIT_NLAYS = [31, 63, 80, 127]
const FIT_GROUPS = [2, 4, 8, 16]

# Held out of the fit: the high-resolution benchmark CI grid at the group count
# the shipped default gives it.
const HELD_OUT = (ncol = 86400, nlay = 63)

# Configurations worth pricing, by the grid they run on.
const PRICED = [
    (name = "amip_he16", ncol = 24576, nlay = 80),
    (name = "ci_highres", ncol = 86400, nlay = 63),
    (name = "coarse_he8", ncol = 6144, nlay = 80),
]

# Byte budgets to report a group count against, and whole-device sizes to
# report a fit against. The P100s are the small device in CliMA's CI.
const BUDGET_GIB = [0.5, 1.0, 2.0, 4.0]
const DEVICE_GIB = [(name = "p100_16gib", gib = 16.0), (name = "a100_40gib", gib = 40.0)]

function arg_value(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) && return default
    value = ARGS[i + 1]
    deleteat!(ARGS, i:(i + 1))
    return value
end

grid(ncol, nlay) = RRTMGP.RRTMGPGridParams(
    FT;
    context = ClimaComms.context(),
    domain_nlay = nlay,
    ncol = ncol,
)

# The accumulators alone, from their shapes: two `(ncol * ngroups, nlev, k)`
# arrays, k = 2 in the longwave and 3 in the shortwave. A floor under the
# measurement, and a check that the delta is reading the right thing.
accumulator_bytes(ncol, nlay, ngroups, planes) =
    2 * (ncol * ngroups) * (nlay + 1) * planes * sizeof(FT)

# Construct one band's workspace and charge it the free-memory it took. The
# predicted floor gates the point: a workspace that does not fit would kill the
# job rather than report a miss.
function measure_one(band, ncol, nlay, ngroups, params)
    planes = band === :lw ? 2 : 3
    floor_bytes = accumulator_bytes(ncol, nlay, ngroups, planes)
    GC.gc()
    CUDA.reclaim()
    free_before = CUDA.free_memory()
    # The measured points run about 2.6x the accumulator floor; budget 3x and a
    # 2 GiB margin.
    needed = 3 * floor_bytes + 2 * 2^30
    if free_before < needed
        return Dict{String, Any}(
            "band" => String(band),
            "ncol" => ncol,
            "nlay" => nlay,
            "ngroups" => ngroups,
            "skipped_out_of_memory" => true,
            "predicted_floor_bytes" => floor_bytes,
            "free_bytes" => free_before,
        )
    end
    gp = grid(ncol, nlay)
    groups =
        band === :lw ? RTE.LWGptGroups(gp, ngroups; params) :
        RTE.SWGptGroups(gp, ngroups)
    CUDA.synchronize()
    bytes = free_before - CUDA.free_memory()
    # Keep the workspace alive across the delta; the compiler is free to drop a
    # value nothing reads, and then the delta reads zero.
    ngroups_seen = RTE.n_gpt_groups(groups)
    groups = nothing
    GC.gc()
    CUDA.reclaim()
    return Dict{String, Any}(
        "band" => String(band),
        "ncol" => ncol,
        "nlay" => nlay,
        "nlev" => nlay + 1,
        "ngroups" => ngroups_seen,
        "wide_columns" => ncol * ngroups,
        "bytes" => bytes,
        "gib" => bytes / 2^30,
        "accumulator_bytes" => floor_bytes,
        "over_accumulator_floor" => bytes / floor_bytes,
    )
end

function fit_points(params)
    points = Dict{String, Any}[]
    for ncol in FIT_NCOLS, nlay in FIT_NLAYS, ngroups in FIT_GROUPS
        for band in (:lw, :sw)
            @info "fit point" band ncol nlay ngroups
            push!(points, measure_one(band, ncol, nlay, ngroups, params))
        end
    end
    return points
end

# Least squares by normal equations, eliminated in place. `LinearAlgebra` is
# not a dependency of the AMIP project, and three unknowns do not need it.
function solve_normal(A, y)
    n = size(A, 2)
    m = zeros(Float64, n, n + 1)
    for i in 1:n
        for j in 1:n
            m[i, j] = sum(A[:, i] .* A[:, j])
        end
        m[i, n + 1] = sum(A[:, i] .* y)
    end
    for i in 1:n
        p = argmax(abs.(m[i:n, i])) + i - 1
        abs(m[p, i]) > 0 || error("normal equations are singular at column $i")
        if p != i
            for j in 1:(n + 1)
                m[i, j], m[p, j] = m[p, j], m[i, j]
            end
        end
        for k in (i + 1):n
            f = m[k, i] / m[i, i]
            for j in i:(n + 1)
                m[k, j] -= f * m[i, j]
            end
        end
    end
    x = zeros(Float64, n)
    for i in n:-1:1
        acc = m[i, n + 1]
        for j in (i + 1):n
            acc -= m[i, j] * x[j]
        end
        x[i] = acc / m[i, i]
    end
    return x
end

# bytes = wide_columns * (a * nlev + b) + c, least squares over the measured
# points of one band. `a` is the quantity the default ignores.
function fit_law(points, band)
    rows = filter(
        p -> p["band"] == String(band) && !haskey(p, "skipped_out_of_memory"),
        points,
    )
    length(rows) >= 3 || error("need at least 3 points to fit $band, got $(length(rows))")
    A = zeros(Float64, length(rows), 3)
    y = zeros(Float64, length(rows))
    for (i, p) in enumerate(rows)
        w = Float64(p["wide_columns"])
        A[i, 1] = w * p["nlev"]
        A[i, 2] = w
        A[i, 3] = 1.0
        y[i] = Float64(p["bytes"])
    end
    coef = solve_normal(A, y)
    residual = A * coef - y
    worst = maximum(abs.(residual ./ y))
    return Dict{String, Any}(
        "bytes_per_wide_column_per_level" => coef[1],
        "bytes_per_wide_column" => coef[2],
        "bytes_fixed" => coef[3],
        "n_points" => length(rows),
        "worst_relative_residual" => worst,
    )
end

predict(law, ncol, nlay, ngroups) =
    (ncol * ngroups) *
    (law["bytes_per_wide_column_per_level"] * (nlay + 1) + law["bytes_per_wide_column"]) +
    law["bytes_fixed"]

# Both bands are built, so the committed memory is their sum.
predict_both(lw, sw, ncol, nlay, ngroups) =
    predict(lw, ncol, nlay, ngroups) + predict(sw, ncol, nlay, ngroups)

# The shipped default itself, not a copy of its arithmetic: it reads the level
# count as well as the column count, so a mirror here would drift from it.
shipped_default(ncol, nlay) = RTE.default_ngpt_groups(grid(ncol, nlay))

# The column target alone, rounded both ways. `cld` is what the default uses --
# the target is a floor on parallelism -- and the difference is what the review
# thread asks about, so both are recorded.
ceil_columns_only(ncol) =
    clamp(cld(RTE.TARGET_GPT_GROUP_COLUMNS, ncol), 1, RTE.MAX_NGPT_GROUPS)
floor_columns_only(ncol) =
    clamp(fld(RTE.TARGET_GPT_GROUP_COLUMNS, ncol), 1, RTE.MAX_NGPT_GROUPS)

function largest_within(lw, sw, ncol, nlay, budget_bytes)
    best = 0
    for n in 1:RTE.MAX_NGPT_GROUPS
        predict_both(lw, sw, ncol, nlay, n) <= budget_bytes && (best = n)
    end
    return best
end

function price_configs(lw, sw)
    out = Dict{String, Any}()
    for cfg in PRICED
        n = shipped_default(cfg.ncol, cfg.nlay)
        bytes = predict_both(lw, sw, cfg.ncol, cfg.nlay, n)
        wide = cfg.ncol * n
        budgets = Dict{String, Any}()
        for gib in BUDGET_GIB
            budgets[string(gib)] = largest_within(lw, sw, cfg.ncol, cfg.nlay, gib * 2^30)
        end
        devices = Dict{String, Any}()
        for dev in DEVICE_GIB
            devices[dev.name] = Dict{String, Any}(
                "device_gib" => dev.gib,
                "workspace_share_pct" => 100.0 * (bytes / 2^30) / dev.gib,
            )
        end
        # The level term, isolated: this group count at the other grids' level
        # counts. The default now responds to levels, so these are what it
        # would have committed had it not, which is the span the byte bound
        # exists to cover.
        by_nlay = Dict{String, Any}()
        for nlay in FIT_NLAYS
            by_nlay[string(nlay)] = Dict{String, Any}(
                "gib_at_this_ngroups" =>
                    predict_both(lw, sw, cfg.ncol, nlay, n) / 2^30,
                "default_ngroups_there" => shipped_default(cfg.ncol, nlay),
                "gib_at_that_default" =>
                    predict_both(
                        lw,
                        sw,
                        cfg.ncol,
                        nlay,
                        shipped_default(cfg.ncol, nlay),
                    ) / 2^30,
            )
        end
        out[cfg.name] = Dict{String, Any}(
            "ncol" => cfg.ncol,
            "nlay" => cfg.nlay,
            "nlev" => cfg.nlay + 1,
            "default_ngroups" => n,
            "ceil_columns_only_ngroups" => ceil_columns_only(cfg.ncol),
            "floor_columns_only_ngroups" => floor_columns_only(cfg.ncol),
            "byte_bound_binds" => n < ceil_columns_only(cfg.ncol),
            "wide_columns" => wide,
            "wide_columns_over_target" =>
                100.0 * (wide - RTE.TARGET_GPT_GROUP_COLUMNS) /
                RTE.TARGET_GPT_GROUP_COLUMNS,
            "predicted_bytes" => bytes,
            "predicted_gib" => bytes / 2^30,
            "largest_ngroups_within_gib" => budgets,
            "devices" => devices,
            "predicted_gib_by_nlay" => by_nlay,
        )
    end
    return out
end

function run()
    device = ClimaComms.device(ClimaComms.context())
    device isa ClimaComms.CUDADevice ||
        error("this measures device memory and needs a CUDA device, got $device")
    params = RRTMGPParameters(FT)
    @info "device" name = CUDA.name(CUDA.device()) total_gib =
        CUDA.total_memory() / 2^30

    points = fit_points(params)
    lw = fit_law(points, :lw)
    sw = fit_law(points, :sw)

    # The held-out grid at its shipped group count: measured against predicted,
    # which is what says the law holds off the grid it was fitted on.
    n_held = shipped_default(HELD_OUT.ncol, HELD_OUT.nlay)
    held = Dict{String, Any}()
    for (band, law) in (("lw", lw), ("sw", sw))
        sym = band == "lw" ? :lw : :sw
        p = measure_one(sym, HELD_OUT.ncol, HELD_OUT.nlay, n_held, params)
        if !haskey(p, "skipped_out_of_memory")
            pred = predict(law, HELD_OUT.ncol, HELD_OUT.nlay, n_held)
            p["predicted_bytes"] = pred
            p["prediction_error_pct"] = 100.0 * (pred - p["bytes"]) / p["bytes"]
        end
        held[band] = p
    end

    configs = price_configs(lw, sw)
    amip = configs["amip_he16"]
    verdict = Dict{String, Any}(
        # The level count is in the law, and now in the default too, which is
        # the objection answered: the same column count no longer commits
        # whatever the grid's height asks for.
        "default_reads_nlev" => true,
        "amip_gib_at_31_levels" =>
            amip["predicted_gib_by_nlay"]["31"]["gib_at_this_ngroups"],
        "amip_gib_at_127_levels" =>
            amip["predicted_gib_by_nlay"]["127"]["gib_at_this_ngroups"],
        "amip_nlev_span_ratio" =>
            amip["predicted_gib_by_nlay"]["127"]["gib_at_this_ngroups"] /
            amip["predicted_gib_by_nlay"]["31"]["gib_at_this_ngroups"],
        # What the bound buys where it binds: the tallest grid priced, at the
        # count the default now gives it rather than the one columns alone would.
        "amip_gib_at_127_under_bound" =>
            amip["predicted_gib_by_nlay"]["127"]["gib_at_that_default"],
        "bound_binds_at_127_levels" =>
            amip["predicted_gib_by_nlay"]["127"]["default_ngroups_there"] <
            amip["default_ngroups"],
        "worst_relative_residual" =>
            max(lw["worst_relative_residual"], sw["worst_relative_residual"]),
        "target_gpt_group_columns" => RTE.TARGET_GPT_GROUP_COLUMNS,
        "max_ngpt_groups" => RTE.MAX_NGPT_GROUPS,
    )

    results = Dict{String, Any}(
        "float_type" => string(FT),
        "device" => CUDA.name(CUDA.device()),
        "device_total_gib" => CUDA.total_memory() / 2^30,
        "fit" => Dict{String, Any}("lw" => lw, "sw" => sw),
        "points" => points,
        "held_out" => held,
        "configs" => configs,
        "verdict" => verdict,
        "note" => "bytes is the device free-memory delta across one band's " *
                  "workspace construction, so it charges the whole widening -- " *
                  "optics, sources, band fluxes, cloud mask and the two " *
                  "accumulators -- and not the accumulators alone, whose " *
                  "shapes give accumulator_bytes as a floor. The fit is " *
                  "bytes = wide_columns * (a * nlev + b) + c per band, least " *
                  "squares over the fit grid; worst_relative_residual is how " *
                  "far the worst fitted point misses. held_out is the " *
                  "high-resolution CI grid at its shipped group count, " *
                  "measured and compared against the law it was excluded " *
                  "from. configs prices the shipped default by prediction, " *
                  "and predicted_gib_by_nlay varies the level count at a " *
                  "fixed group count, which is the span the default's byte " *
                  "bound covers, beside the count the default itself gives " *
                  "each of those grids.",
    )

    out_path = arg_value("--out", "results/group-workspace-memory.toml")
    mkpath(dirname(out_path))
    open(out_path, "w") do io
        TOML.print(io, results; sorted = true)
    end
    verdict_summary = verdict
    @info "wrote $out_path" verdict_summary
    return nothing
end

run()
