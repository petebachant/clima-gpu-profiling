# ClimaCore #2598 (`dy/spectral_fusion`) — kernel exception on AMIP

Tested at `0f2131304` ("Strip broadcast args against each node's own space, not
the root space") against ClimaAtmos `main` (8e81bb618), on an A100-SXM4-80GB.

## What works

The whole stack **precompiles cleanly** — ClimaAtmos, ClimaLand, ClimaCoupler —
and the run completes setup: state initialisation, cache build, callbacks, and
the tendency function. None of ClimaAtmos, ClimaCoupler or ClimaLand reference
the APIs the PR removes (`SpectralBroadcasted`, `todata`, `get_node`,
`reinstantiate_bc`), and the version stays 0.15.3 so downstream compat bounds
hold. The API surface of the refactor looks sound; this is a runtime fault.

## The failure

Crashes on the **first tendency evaluation**, before any stepping. Device
stacktrace (`julia -g2`; without it CUDA.jl reports only a thread/block index):

```
[1] error at ./error.jl:35
[2] error_mismatched_spaces at /home/pbachant/calkit/clima-gpu-profiling/ClimaCore.jl-mod/src/Fields/bro
[3] broadcast_shape at /home/pbachant/calkit/clima-gpu-profiling/ClimaCore.jl-mod/src/Fields/broadcast.j
[4] combine_axes at ./broadcast.jl:496
[5] combine_axes at ./broadcast.jl:496
[6] combine_axes at ./broadcast.jl:496
[7] combine_axes at ./broadcast.jl:496
[8] combine_axes at ./broadcast.jl:496
[9] combine_axes at ./broadcast.jl:496
[10] combine_axes at ./broadcast.jl:496
[11] _axes at ./broadcast.jl:236
[12] axes at ./broadcast.jl:234
[13] getidx at /home/pbachant/calkit/clima-gpu-profiling/ClimaCore.jl-mod/src/Operators/finitedifference
[14] get_level_value at /home/pbachant/calkit/clima-gpu-profiling/ClimaCore.jl-mod/src/Operators/integra
[15] single_column_accumulate! at /home/pbachant/calkit/clima-gpu-profiling/ClimaCore.jl-mod/src/Operato
[16] bycolumn_kernel! at /home/pbachant/calkit/clima-gpu-profiling/ClimaCore.jl-mod/ext/cuda/operators_i
```

Notable: this is a **deliberate device-side `error()`** — the space check fires
and rejects the broadcast — not a memory fault.

## How it is reached

```
[1] check_exceptions()
[2] device_synchronize(; blocking::Bool, spin::Bool)
[3] device_synchronize
[4] checked_cuModuleLoadDataEx(_module::Base.RefValue{…}, image::Ptr{…}, numOptions::Int64, options::Vec
[5] CUDA.CuModule(data::Vector{UInt8}, options::Dict{CUDA.CUjit_option_enum, Any})
[6] CuModule
[7] link(job::GPUCompiler.CompilerJob, compiled::@NamedTuple{image::Vector{UInt8}, entry::String})
[8] actual_compilation(cache::Dict{…}, src::Core.MethodInstance, world::UInt64, cfg::GPUCompiler.Compile
[9] cached_compilation(cache::Dict{…}, src::Core.MethodInstance, cfg::GPUCompiler.CompilerConfig{…}, com
[10] macro expansion
[11] macro expansion
[12] cufunction(f::ClimaCoreCUDAExt.var"#kernel_function#23"{…}, tt::Type{…}; kwargs::@Kwargs{…})
```

## Reading

The fault is not in the spectral operators the PR rewrites, but in a **column
integral**. `single_column_accumulate!` → `get_level_value` builds a broadcast
mixing a level-indexed field (`ᶜlevel`) with column fields; after per-node space
stripping those arguments no longer share a space, so `broadcast_shape` rejects
them. The head commit is the obvious suspect.

## Reproducing

Config: `ClimaCoupler.jl/config/benchmark_configs/amip_progedmf_1m_land_he16.yml`
— Float32, `h_elem=16`, `z_elem=63`, prognostic EDMFX, 1M microphysics, and
**`non_orographic_gravity_wave: true`**, which the host frames identify as the
trigger. A config without it may well pass, and would be a far faster reproducer
than a full AMIP setup.

Reaching the crash needs setup plus one tendency evaluation — about 7 minutes,
no stepping and no profiling. Run under `julia -g2` or the device stacktrace is
suppressed.

## Performance artifacts

Deferred until it runs. For reference, the baseline it would be measured against
(same machine and config, from `results/kernel-population.json`): 31,140 kernel
launches over 10 coupler steps (3,114/step), 343 distinct kernels, GPU idle
30.9%, and spectral element operators at 1,310 launches / 5.8% of kernel time.

The metric this PR should move is **launch count and idle fraction**, not
necessarily kernel time: one fused kernel doing several kernels' work need not be
faster on the GPU, but it removes launches, and idle time converts to wall clock
far more directly than kernel time does.
