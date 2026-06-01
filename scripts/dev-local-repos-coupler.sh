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

# Do the same for ClimaCoupler.jl-mod for the -mod suffix submodules
julia --project="$REPO_ROOT/ClimaCoupler.jl-mod/$JULIA_PROJECT" -e "
    import Pkg
    Pkg.develop(path=\"./ClimaCore.jl-mod\")
    Pkg.develop(path=\"./ClimaAtmos.jl-mod\")
"
