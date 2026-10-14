#!/usr/bin/env bash
set -euo pipefail

git -C ClimaCoupler.jl fetch
git -C ClimaCoupler.jl-mod fetch
# `checkout <ref> -- .` never deletes, so drop whatever main removed as well.
for d in ClimaCoupler.jl ClimaCoupler.jl-mod; do
  git -C "$d" checkout origin/main -- .
  git -C "$d" diff --name-only --diff-filter=A origin/main -- . |
    xargs -r git -C "$d" rm -q --
done
bash ./scripts/dev-local-repos-coupler.sh
git -C ClimaCoupler.jl commit -am "Update to latest main"
git -C ClimaCoupler.jl push
git -C ClimaCoupler.jl-mod commit -am "Update to latest main"
git -C ClimaCoupler.jl-mod push
