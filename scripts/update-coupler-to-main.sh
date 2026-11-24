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

# The benchmark window is not the only deliberate deviation. ClimaAtmos's minor
# version runs ahead of coupler main's compat bound for days at a time -- 0.43.0
# landed 2026-11-20 and main still read "0.42.17" on 2026-11-23 -- so the AMIP
# environment cannot resolve the dev'd ClimaAtmos until the bound is widened,
# and `checkout origin/main -- .` above reverts the widening. The pull then dies
# inside `Pkg.develop` with "Unsatisfiable requirements", which names the
# package and not the cause. Carry our own bound across, and drop the deviation
# by itself once main's bound admits the dev'd version.
for d in ClimaCoupler.jl ClimaCoupler.jl-mod; do
  atmos="ClimaAtmos.jl${d#ClimaCoupler.jl}"
  python3 - "$d" "$atmos" <<'PY'
import re, subprocess, sys

coupler, atmos = sys.argv[1], sys.argv[2]
path = f"{coupler}/Project.toml"
# `"[0-9]` is what separates the [compat] bound from the [deps] UUID, which
# starts with a letter.
pat = re.compile(r'^ClimaAtmos = "[0-9][^"]*"$', re.M)

head = subprocess.run(["git", "-C", coupler, "show", "HEAD:Project.toml"],
                      capture_output=True, text=True, check=True).stdout
ours = pat.search(head)
assert ours, f"{coupler}: HEAD carries no ClimaAtmos compat bound"
ours = ours.group(0)

with open(path) as f:
    s = f.read()
theirs = pat.findall(s)
assert len(theirs) == 1, f"{coupler}: main carries {len(theirs)} ClimaAtmos bounds"
theirs = theirs[0]

with open(f"{atmos}/Project.toml") as f:
    minor = re.search(r'^version = "(\d+\.\d+)', f.read(), re.M)
assert minor, f"{atmos}: no version line"
minor = minor.group(1)

if minor in theirs:
    print(f"{coupler}: main's bound {theirs} admits {atmos} {minor}; taking main's")
    sys.exit(0)
if ours == theirs:
    print(f"{coupler}: bound already {theirs}")
    sys.exit(0)
out = pat.sub(lambda m: ours, s, count=1)
assert out != s, f"{coupler}: the bound substitution did not apply"
with open(path, "w") as f:
    f.write(out)
print(f"{coupler}: carried {ours} over main's {theirs}, for {atmos} {minor}")
PY
done

a=$(grep -m1 '^ClimaAtmos = "[0-9]' ClimaCoupler.jl/Project.toml)
b=$(grep -m1 '^ClimaAtmos = "[0-9]' ClimaCoupler.jl-mod/Project.toml)
if [ "$a" != "$b" ]; then
  echo "ERROR: the arms' ClimaAtmos bounds differ ($a vs $b)" >&2
  exit 1
fi
JULIA_PKG_PRECOMPILE_AUTO=0 bash ./scripts/dev-local-repos-coupler.sh
git -C ClimaCoupler.jl commit -am "Update to latest main"
git -C ClimaCoupler.jl push
git -C ClimaCoupler.jl-mod commit -am "Update to latest main"
git -C ClimaCoupler.jl-mod push
