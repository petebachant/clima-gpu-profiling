# What would ClimaCore's automatic register cap do to the whole model?
#
# `scripts/measure-quadrature-registers.jl` checks the policy against one kernel
# whose answer is already measured. This runs the real coupled model with the
# policy on and dumps every decision it made, so the thresholds meet the whole
# kernel population rather than the five kernels a 20-row profile can see --
# which kernels it fires on, which it declines and on which test.
#
# No profiler and no second arm: the policy records its own decisions, so the
# census is a by-product of one run. Compilation dominates the job either way
# (~35 min of JIT against seconds of stepping), so the step count is chosen to
# cover the long cycles rather than to be small.
import CUDA
import ClimaComms
import ClimaCore

project_dir = dirname(Base.active_project())
@info "Active project: $project_dir"
include(joinpath(project_dir, "code_loading.jl"))

const Ext = Base.get_extension(ClimaCore, :ClimaCoreCUDAExt)
isnothing(Ext) && error("the ClimaCore CUDA extension is not loaded")
isdefined(Ext, :auto_cap_decisions) ||
    error("this ClimaCore has no automatic register cap; wrong branch?")
# The policy is read from the environment in the extension's __init__, so it has
# to be set before this process starts: a Ref flipped here would miss every
# kernel compiled during setup.
Ext.auto_register_cap() ||
    error("set CLIMA_AUTO_REGISTER_CAP=true: the census is of the policy's own decisions")

function arg_value(flag, default)
    i = findfirst(==(flag), ARGS)
    isnothing(i) && return default
    value = ARGS[i + 1]
    deleteat!(ARGS, i:(i + 1))
    return value
end
const OUT_PATH = arg_value("--out", "results/auto-register-cap-census.toml")

config_file = Input.parse_commandline(Input.argparse_settings())["config_file"]
cs = CoupledSimulation(config_file)

# 35 steps covers one radiation and cloud-fraction cycle (30 steps) and two
# gravity-wave cycles (15), so every scheme fires at least once and every kernel
# it launches is decided. Stepping is seconds against the JIT, so a window that
# clears the longest cycle costs effectively nothing over a window that does not.
function step_model!(cs, n)
    for i in 1:n
        step!(cs)
        i % 5 == 0 && @info "step $i / $n"
    end
    return nothing
end
step_model!(cs, 35)
CUDA.synchronize()

# Group by the test that decided each kernel, and carry the device-side numbers
# so a declined kernel can be re-examined without another run.
function census(decisions)
    by_reason = Dict{String, Int}()
    rows = Dict{String, Any}()
    n_capped = 0
    for (i, (_, d)) in enumerate(decisions)
        reason = string(d.reason)
        by_reason[reason] = get(by_reason, reason, 0) + 1
        d.reason === :capped && (n_capped += 1)
        rows[string(i)] = Dict{String, Any}(
            "reason" => reason,
            "kernel" => something(d.name, "unnamed"),
            "registers" => d.regs,
            "warps_per_sm" => d.warps,
            "local_bytes" => d.local_bytes,
            "edge" => something(d.edge, -1),
            "shed" => isnothing(d.edge) ? -1 : d.regs - d.edge,
            "cap" => something(d.cap, -1),
            "local_budget" => d.local_budget,
            "capped_registers" => d.capped_regs,
            "capped_local_bytes" => d.capped_local,
            "capped_warps_per_sm" => d.capped_warps,
        )
    end
    return (; by_reason, rows, n_capped)
end
(; by_reason, rows, n_capped) = census(Ext.auto_cap_decisions())

println("\n=== automatic register-cap census: $(length(rows)) kernel specializations ===")
println("capped: $n_capped")
for (reason, n) in sort(collect(by_reason); by = last, rev = true)
    println(rpad("  $reason", 28), n)
end
println("\n", rpad("kernel", 58), rpad("regs", 6), rpad("shed", 6),
        rpad("warps", 10), rpad("local B", 18), "reason")
for (_, r) in sort(collect(rows); by = p -> (p[2]["reason"] != "capped", -p[2]["registers"]))
    w = r["capped_warps_per_sm"] < 0 ? string(r["warps_per_sm"]) :
        "$(r["warps_per_sm"])->$(r["capped_warps_per_sm"])"
    l = r["capped_local_bytes"] < 0 ? "$(r["local_bytes"])" :
        "$(r["local_bytes"])->$(r["capped_local_bytes"]) (bud $(r["local_budget"]))"
    println(rpad(first(r["kernel"], 58), 58), rpad(r["registers"], 6),
            rpad(r["shed"], 6), rpad(w, 10), rpad(l, 18), r["reason"])
end

import TOML
import Dates
mkpath(dirname(OUT_PATH))
open(OUT_PATH, "w") do io
    TOML.print(io, Dict(
        "note" => "Every decision ClimaCore's automatic register cap made over " *
                  "35 coupler steps of the benchmark configuration, with the " *
                  "policy on from process start. `reason` is the test that " *
                  "decided each kernel: `capped` launched capped, every other " *
                  "value launched unchanged. `registers` and `local_bytes` are " *
                  "what the kernel compiled to with no cap, `capped_*` what it " *
                  "compiled to at the band edge, and -1 marks a field the " *
                  "decision never reached. No profiler ran, so there are no " *
                  "kernel times here: this census says what the policy does, " *
                  "not what it is worth.",
        "measured" => string(Dates.today()),
        "steps" => 35,
        "config_file" => config_file,
        "climacore_path" => pkgdir(ClimaCore),
        "climacore_commit" =>
            try readchomp(`git -C $(pkgdir(ClimaCore)) rev-parse HEAD`) catch; "unknown" end,
        "max_shed" => Ext.AUTO_REGISTER_CAP_MAX_SHED[],
        "max_warps" => Ext.AUTO_REGISTER_CAP_MAX_WARPS[],
        "local_growth_pct" => Ext.AUTO_REGISTER_CAP_LOCAL_GROWTH_PCT[],
        "local_slack_bytes" => Ext.AUTO_REGISTER_CAP_LOCAL_SLACK[],
        "n_specializations" => length(rows),
        "n_capped" => n_capped,
        "by_reason" => by_reason,
        "decisions" => rows,
    ); sorted = true)
end
println("\nwrote $OUT_PATH")
println("CENSUS DONE")
