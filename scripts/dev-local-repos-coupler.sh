#!/usr/bin/env bash
# Dev our local submodules for the Coupler submodules

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
JULIA_PROJECT="experiments/AMIP"

# The stack runs Julia 1.13, which keeps its own `Manifest-v1.13.toml`, and the
# login shell's julia is 1.11, which writes the unversioned `Manifest.toml`. The
# GPU stages load the module below, so dev'ing under the login shell wrote the
# dev paths into a file nothing reads and left the versioned manifest holding
# upstream's registry versions -- both arms then benchmark registry packages and
# every treatment silently disappears. Load the same toolchain the stages do.
module purge
module load climacommon/2026_11_06-1.13.3

# First update registry since dev installs don't do this
julia -e 'using Pkg; Pkg.Registry.update()'

# In ClimaCoupler.jl, dev ClimaCore.jl and ClimaAtmos.jl in the AMIP env
julia --project="$REPO_ROOT/ClimaCoupler.jl/$JULIA_PROJECT" -e "
    import Pkg
    Pkg.develop(path=\"./ClimaCore.jl\")
    Pkg.develop(path=\"./ClimaAtmos.jl\")
    Pkg.develop(path=\"./RRTMGP.jl\")
"

# Do the same for ClimaCoupler.jl-mod for the -mod suffix submodules.
# The dev'd set is deliberately the same on both sides: a package dev'd in only
# one arm makes the arms differ by its whole version, not by our change to it.
# CloudMicrophysics was dev'd here alone until 2026-09-14, which is how upstream
# drift rode along inside every CM experiment (docs/learnings.md 4i). RRTMGP was
# too, until the binary search merged upstream (CliMA/RRTMGP.jl#628). Neither is
# a treatment now, so both arms take the Coupler-pinned versions.
#
# RRTMGP is dev'd in BOTH arms from 2026-09-16. The registry pins v1.0.0, which
# predates the binary search that merged as CliMA/RRTMGP.jl#628, so a stack
# taking the registry version benchmarks radiation ~27% slower than what the
# nightly (which installs from main) actually runs. Dev'ing both arms at main
# fixes the baseline and makes RRTMGP the treatment again.
julia --project="$REPO_ROOT/ClimaCoupler.jl-mod/$JULIA_PROJECT" -e "
    import Pkg
    Pkg.develop(path=\"./ClimaCore.jl-mod\")
    Pkg.develop(path=\"./ClimaAtmos.jl-mod\")
    Pkg.develop(path=\"./RRTMGP.jl-mod\")
"

# Assert the dev paths landed in the manifest the stages read, rather than in
# whichever file the toolchain on PATH happens to prefer.
MANIFEST="$JULIA_PROJECT/Manifest-v1.13.toml"
for arm in ClimaCoupler.jl ClimaCoupler.jl-mod; do
  for pkg in ClimaCore ClimaAtmos RRTMGP; do
    if ! grep -A5 "^\[\[deps\.$pkg\]\]" "$REPO_ROOT/$arm/$MANIFEST" | grep -q "^path = "; then
      echo "ERROR: $arm/$MANIFEST does not dev $pkg; the wrong manifest was written" >&2
      exit 1
    fi
  done
done
echo "both arms dev ClimaCore, ClimaAtmos and RRTMGP in $MANIFEST"
