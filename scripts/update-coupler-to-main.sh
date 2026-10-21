#!/usr/bin/env bash
set -euo pipefail

# The benchmark window is this project's one deliberate deviation from coupler
# main, and `checkout origin/main -- .` below reverts it. That silently restored
# upstream's 40-step, dt=30s window more than once -- short enough that neither
# gravity-wave scheme fires and about half the wall clock is compilation -- so it
# is restored and the two arms are compared byte for byte before anything runs.
BENCH_CFG="config/benchmark_configs/amip_progedmf_1m_land_he16.yml"

git -C ClimaCoupler.jl fetch
git -C ClimaCoupler.jl-mod fetch
# `checkout <ref> -- .` never deletes, so drop whatever main removed as well.
for d in ClimaCoupler.jl ClimaCoupler.jl-mod; do
  git -C "$d" checkout origin/main -- .
  git -C "$d" diff --name-only --diff-filter=A origin/main -- . |
    xargs -r git -C "$d" rm -q --
  git -C "$d" checkout HEAD -- "$BENCH_CFG"
done

if ! cmp -s "ClimaCoupler.jl/$BENCH_CFG" "ClimaCoupler.jl-mod/$BENCH_CFG"; then
  echo "ERROR: the arms' benchmark configs differ; they must be byte-identical" >&2
  exit 1
fi
JULIA_PKG_PRECOMPILE_AUTO=0 bash ./scripts/dev-local-repos-coupler.sh
git -C ClimaCoupler.jl commit -am "Update to latest main"
git -C ClimaCoupler.jl push
git -C ClimaCoupler.jl-mod commit -am "Update to latest main"
git -C ClimaCoupler.jl-mod push
