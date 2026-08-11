# Measure how much of a profiling job's wall time is compilation, and how much
# of that is recoverable by GPUCompiler's on-disk kernel cache.
#
# Motivation: an nsys stage takes ~40 minutes to produce ~1.8 s of measured
# stepping. Nearly all the remainder is setup plus first-call compilation. Julia
# host code is cached across processes by pkgimages, but GPU kernels are compiled
# by GPUCompiler at run time and discarded at process exit *unless* its
# `disk_cache` preference is enabled (it defaults to false). This script
# quantifies that split so the decision rests on measurement, not plausibility.
#
# Run one phase per process -- the whole point is to measure cross-process cache
# reuse, which cannot be observed within a single session.
#
# Usage:
#   julia --project=<AMIP env> measure-compile-time.jl \
#       --config_file <cfg> --label <phase> --out <results.toml>

import CUDA
import TOML
import Dates
import Printf

# --- Argument parsing --------------------------------------------------------
# Hand-rolled rather than ArgParse: this script must run inside the AMIP project
# and may only load that project's direct dependencies plus stdlibs.
function argval(flag, default = nothing)
    i = findfirst(==(flag), ARGS)
    i === nothing && return default
    i == length(ARGS) && error("$flag requires a value")
    return ARGS[i + 1]
end

const CONFIG_FILE = argval("--config_file")
const LABEL = argval("--label", "unlabeled")
const OUT_PATH = argval("--out", "compile-time.toml")
CONFIG_FILE === nothing && error("--config_file is required")

# ClimaCoupler's `Input.parse_commandline` validates the real ARGS strictly and
# rejects anything outside its own option set, so `--label`/`--out` would abort
# the run with "unrecognized option". Having consumed them, narrow ARGS to just
# what the coupler expects.
empty!(ARGS)
append!(ARGS, ["--config_file", CONFIG_FILE])

# --- Environment fingerprint -------------------------------------------------
# Recorded alongside the timings so a frozen result stays interpretable: these
# numbers are only meaningful for the toolchain that produced them.
function environment_info()
    info = Dict{String, Any}(
        "julia" => string(VERSION),
        "gpu_index" => get(ENV, "CUDA_VISIBLE_DEVICES", "unknown"),
        "hostname" => gethostname(),
        "slurm_job_id" => get(ENV, "SLURM_JOB_ID", "none"),
        "timestamp_utc" => string(Dates.now(Dates.UTC)),
    )
    # Individually guarded: a renamed CUDA.jl accessor must not cost us a phase
    # that takes ~40 minutes to reproduce.
    for (key, f) in (
        "cuda_runtime" => () -> string(CUDA.runtime_version()),
        "cuda_driver" => () -> string(CUDA.driver_version()),
        "gpu_name" => () -> CUDA.name(CUDA.device()),
        "gpu_uuid" => () -> string(CUDA.uuid(CUDA.device())),
        "gpu_capability" => () -> begin
            cap = CUDA.capability(CUDA.device())
            "sm_$(cap.major)$(cap.minor)"
        end,
    )
        info[key] = try
            f()
        catch ex
            "unavailable ($(sprint(showerror, ex)))"
        end
    end
    return info
end

# Report what GPUCompiler will actually do, read back from the live preference
# rather than from what we believe we set. GPUCompiler is an indirect dependency
# so it cannot be imported here; the preference is read straight from the
# active project's LocalPreferences.toml.
function disk_cache_setting()
    prefs_path = joinpath(dirname(Base.active_project()), "LocalPreferences.toml")
    isfile(prefs_path) || return "false (no LocalPreferences.toml)"
    prefs = TOML.parsefile(prefs_path)
    return string(get(get(prefs, "GPUCompiler", Dict()), "disk_cache", "false"))
end

# Collected up front so any problem here surfaces immediately rather than after
# the workload has already burned the phase.
const ENV_INFO = environment_info()
@info "Compile-time study phase" label = LABEL disk_cache = disk_cache_setting() ENV_INFO

# --- Workload ----------------------------------------------------------------
# Mirrors scripts/run.jl: load the coupler, build the simulation, take a step.
# Setup and the first step together trigger essentially all of the compilation;
# the profiled window in run.jl is under two seconds and is not the subject here.

project_dir = dirname(Base.active_project())

t_load = @timed include(joinpath(project_dir, "code_loading.jl"))
@info Printf.@sprintf("code_loading: %.1f s (%.1f s compiling)",
    t_load.time, t_load.compile_time)

t_setup = @timed CoupledSimulation(CONFIG_FILE)
cs = t_setup.value
@info Printf.@sprintf("CoupledSimulation: %.1f s (%.1f s compiling)",
    t_setup.time, t_setup.compile_time)

# First step: the bulk of GPU kernel compilation happens here.
t_first = @timed step!(cs)
@info Printf.@sprintf("first step!: %.1f s (%.1f s compiling)",
    t_first.time, t_first.compile_time)

# Second step: with everything compiled, this approximates steady-state cost and
# so bounds how much of the first step was genuinely compute rather than compile.
t_second = @timed step!(cs)
@info Printf.@sprintf("second step!: %.1f s (%.1f s compiling)",
    t_second.time, t_second.compile_time)

# --- Record ------------------------------------------------------------------
phase(t) = Dict{String, Any}(
    "wall_s" => t.time,
    "compile_s" => t.compile_time,
    "recompile_s" => t.recompile_time,
    "gc_s" => t.gctime,
    "bytes" => t.bytes,
)

total_wall = t_load.time + t_setup.time + t_first.time + t_second.time
total_host_compile =
    t_load.compile_time + t_setup.compile_time +
    t_first.compile_time + t_second.compile_time

record = Dict{String, Any}(
    "label" => LABEL,
    "disk_cache" => disk_cache_setting(),
    "config_file" => CONFIG_FILE,
    "environment" => ENV_INFO,
    "code_loading" => phase(t_load),
    "coupled_simulation" => phase(t_setup),
    "first_step" => phase(t_first),
    "second_step" => phase(t_second),
    "totals" => Dict{String, Any}(
        "wall_s" => total_wall,
        # Host-side JIT only. GPU kernel compilation is *not* counted here --
        # GPUCompiler runs outside Julia's compile-time accounting, which is
        # precisely why it must be inferred from the cross-phase wall delta.
        "host_compile_s" => total_host_compile,
        "steady_step_s" => t_second.time,
    ),
)

# Append this phase to the results file so one job accumulates all phases.
existing = isfile(OUT_PATH) ? TOML.parsefile(OUT_PATH) : Dict{String, Any}()
existing[LABEL] = record
mkpath(dirname(abspath(OUT_PATH)))
open(OUT_PATH, "w") do io
    TOML.print(io, existing; sorted = true)
end

@info Printf.@sprintf(
    "phase '%s': %.1f s wall, %.1f s host-compile, %.2f s steady step",
    LABEL, total_wall, total_host_compile, t_second.time)
@info "Wrote results" path = OUT_PATH
