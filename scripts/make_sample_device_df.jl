
include("../ClimaAtmos.jl/perf/benchmark_step.jl");
import CSV

mkpath("results")

CSV.write("results/sample-device-df.csv", res.device)
