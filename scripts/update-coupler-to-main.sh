#!/usr/bin/env bash
set -euo pipefail

git -C ClimaCoupler.jl fetch
git -C ClimaCoupler.jl-mod fetch
git -C ClimaCoupler.jl checkout origin/main -- .
git -C ClimaCoupler.jl-mod checkout origin/main -- .
bash ./scripts/dev-local-repos-coupler.sh
git -C ClimaCoupler.jl commit -am "Update to latest main"
git -C ClimaCoupler.jl push
git -C ClimaCoupler.jl-mod commit -am "Update to latest main"
git -C ClimaCoupler.jl-mod push
