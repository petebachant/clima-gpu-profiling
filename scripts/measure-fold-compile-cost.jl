# What does folding the terminal-velocity parameters into the type cost in
# compile time?
#
# CliMA/ClimaAtmos.jl#4947 wraps the Chen 2022 coefficients and the 1M
# precipitation parameters in `Val` at the sedimentation call sites, so the
# kernel body sees literals instead of reading a struct per point. The register
# saving is measured elsewhere; this measures the price. Putting a runtime value
# in a type parameter makes a fresh specialization that no pkgimage can hold, so
# whatever it costs to infer and codegen is paid at run time, on every start.
#
# The two arms differ ONLY in how the parameters arrive, which is why the body
# is written once: `_unwrapped` accepts both a struct and a `Val`, so the folded
# and unfolded calls are the same shipped methods.
#
# One variant per process, because a compile time is only a compile time the
# first time. Running both arms in one session would let the second arm reuse
# the first's inference of every shared callee, which is most of the work.
#
# Only the modes that actually read the parameters are measured. The `Fixed`
# modes take `args...` and ignore `cmc`/`cmp` by signature, so the fold cannot
# change what they compile -- and `fold_relevant_in_benchmark` records which of
# these the benchmark configuration actually selects.
#
# Usage:
#   julia --project=<env> measure-fold-compile-cost.jl \
#       --variant folded|unfolded --rep 1 --out <file.toml>

import CUDA
import ClimaAtmos
import TOML
import Dates
using Printf

const CA = ClimaAtmos
const MF = ClimaAtmos.MatrixFields

function argval(flag, default = nothing)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag requires a value")
    return ARGS[i + 1]
end

const VARIANT = argval("--variant", "unfolded")
const REP = parse(Int, argval("--rep", "1"))
const OUT_PATH = argval("--out", "fold-compile-cost.toml")
VARIANT in ("folded", "unfolded") ||
    error("--variant must be folded or unfolded, got $VARIANT")

const FT = Float32

# `cloud_parameters` and `Microphysics1MParams` both construct standalone from a
# default TOML dict, so this needs no coupled model and no configuration file.
const CMC = CA.cloud_parameters(FT)
const CMP1M = CA.microphys_1m_parameters(FT)
# The velocity members only: `aml` and `activation` are not read by any
# terminal-velocity method and a type parameter has to be isbits. This is the
# same subset #4947 wraps at the call site.
const TV = (; CMC.liquid, CMC.ice, CMC.stokes, CMC.Ch2022)

# sm_80, 128-thread blocks.
warps(r) = r <= 128 ? 16 : r <= 168 ? 12 : r <= 255 ? 8 : 0

# The fold-relevant paths: every mode that reads `cmc` or `cmp`. The `Fixed`
# modes are excluded by construction, not by choice -- their method ignores both
# payloads, so there is nothing for a fold to change.
const PATHS = (
    (name = "liquid_stokes", species = MF.@name(q_lcl),
     mode = CA.StokesTerminalVelocity()),
    (name = "ice_chen2022", species = MF.@name(q_icl),
     mode = CA.Chen2022TerminalVelocity()),
    (name = "rain_chen2022", species = MF.@name(q_rai),
     mode = CA.Chen2022TerminalVelocity()),
    (name = "snow_chen2022", species = MF.@name(q_sno),
     mode = CA.Chen2022TerminalVelocity()),
)

# `params` is an unused positional argument in every method measured here (only
# the `Fixed` methods read it, through `CAP.fixed_*_terminal_velocity`), so
# passing nothing keeps the compiled body identical to the model's without
# dragging in a full parameter set.
@inline function tv_body(o, mode, species, a, b, ρ, q)
    o[1] = CA.terminal_velocity(
        CA.NonEquilibriumMicrophysics1M(), mode, species, nothing, a, b, ρ, q,
    )
    return nothing
end

out = CUDA.zeros(FT, 1)

# Pay GPUCompiler's one-time costs -- loading the PTX toolchain, the first
# device code generation -- before anything is timed, so they do not land on
# whichever path happens to be measured first.
@inline function warmup_body(o)
    o[1] = o[1] + one(FT)
    return nothing
end
CUDA.@cuda launch = false always_inline = true warmup_body(out)

payload(::Val{:unfolded}) = (CMC, CMP1M)
payload(::Val{:folded}) = (Val(TV), Val(CMP1M))
const A, B = payload(Val(Symbol(VARIANT)))

results = Dict{String, Any}(
    "variant" => VARIANT,
    "rep" => REP,
    "float_type" => string(FT),
    "tv_isbits" => isbits(TV),
    "cmp_isbits" => isbits(CMP1M),
    "tv_sizeof_bytes" => sizeof(TV),
    "cmp_sizeof_bytes" => sizeof(CMP1M),
    "julia" => string(VERSION),
    "gpu" => try CUDA.name(CUDA.device()) catch; "unavailable" end,
    "slurm_job_id" => get(ENV, "SLURM_JOB_ID", "none"),
    "timestamp_utc" => string(Dates.now(Dates.UTC)),
    # Recorded so a reader can tell which of these paths the benchmark window
    # actually exercises: the rest are compiled only if a config enables them.
    "benchmark_modes" => Dict(
        "cloud_liquid" => "Fixed", "cloud_ice" => "Fixed",
        "rain" => "Fixed", "snow" => "Chen2022",
    ),
    "fold_relevant_in_benchmark" => ["snow_chen2022"],
    "note" => "Compile wall time and registers for the terminal-velocity " *
              "bodies #4947 folds, one arm per process so every compile is " *
              "cold. Only modes that read cmc/cmp are measured; the Fixed " *
              "modes ignore both by signature. `params` is passed as nothing " *
              "because no measured method reads it. The 1M payload is the\n" *
              "default process selection rather than the benchmark's pruned one; " *
              "both arms carry the identical payload, so the difference is\n" *
              "unaffected and only the per-path absolute is a proxy.",
)

if !isbits(TV) || !isbits(CMP1M)
    @warn "a payload is not isbits, so Val cannot wrap it; recording and stopping"
    open(OUT_PATH, "w") do io
        TOML.print(io, results; sorted = true)
    end
    exit(0)
end

@printf("variant=%s rep=%d  TV=%d B isbits=%s  CMP=%d B isbits=%s\n",
        VARIANT, REP, sizeof(TV), isbits(TV), sizeof(CMP1M), isbits(CMP1M))

# One cold compile per path. `@elapsed` around `@cuda launch=false` is inference
# plus codegen plus PTX assembly, which is what a fresh specialization costs at
# startup; nothing is launched.
function measure(path)
    local k
    t = @elapsed begin
        k = CUDA.@cuda launch = false always_inline = true tv_body(
            out, path.mode, path.species, A, B, FT(0.9), FT(1.0e-5),
        )
    end
    r = CUDA.registers(k)
    m = CUDA.memory(k)
    return Dict(
        "compile_s" => t,
        "registers" => r,
        "local_bytes" => m.local,
        "warps_per_sm" => warps(r),
    )
end

total = 0.0
for path in PATHS
    row = measure(path)
    results[path.name] = row
    total += row["compile_s"]
    @printf("%-16s compile=%7.3f s  regs=%3d  local=%4d B  warps/SM=%2d\n",
            path.name, row["compile_s"], row["registers"],
            row["local_bytes"], row["warps_per_sm"])
end
results["total_compile_s"] = total
results["paths_measured"] = [p.name for p in PATHS]
@printf("\n%s rep %d: %.3f s over %d paths\n", VARIANT, REP, total, length(PATHS))

open(OUT_PATH, "w") do io
    TOML.print(io, results; sorted = true)
end
println("wrote ", OUT_PATH)
