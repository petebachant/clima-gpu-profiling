#!/usr/bin/env bash
# Dev our local submodules for the Coupler submodules

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
JULIA_PROJECT="experiments/AMIP"

# First update registry since dev installs don't do this
julia -e 'using Pkg; Pkg.Registry.update()'

# In ClimaCoupler.jl, dev ClimaCore.jl and ClimaAtmos.jl in the AMIP env
julia --project="$REPO_ROOT/ClimaCoupler.jl/$JULIA_PROJECT" -e "
    import Pkg
    Pkg.develop(path=\"./ClimaCore.jl\")
    Pkg.develop(path=\"./ClimaAtmos.jl\")
"

# Do the same for ClimaCoupler.jl-mod for the -mod suffix submodules.
# The dev'd set is deliberately the same on both sides: a package dev'd in only
# one arm makes the arms differ by its whole version, not by our change to it.
# CloudMicrophysics was dev'd here alone until 2026-09-14, which is how upstream
# drift rode along inside every CM experiment (docs/learnings.md 4i). RRTMGP was
# too, until the binary search merged upstream (CliMA/RRTMGP.jl#628). Neither is
# a treatment now, so both arms take the Coupler-pinned versions.
julia --project="$REPO_ROOT/ClimaCoupler.jl-mod/$JULIA_PROJECT" -e "
    import Pkg
    Pkg.develop(path=\"./ClimaCore.jl-mod\")
    Pkg.develop(path=\"./ClimaAtmos.jl-mod\")
"
