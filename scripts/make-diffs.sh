#!/usr/bin/env bash
# Diff each repo's `mod` versus base and save the diff files in `diffs`

set -euo pipefail

mkdir -p diffs

for repo in ClimaAtmos.jl ClimaCore.jl ClimaCoupler.jl; do
    base="${repo}"
    mod="${repo}-mod"

    # Collect non-ignored files from both sides (tracked + untracked non-ignored)
    base_files=$(git -C "${base}" ls-files; git -C "${base}" ls-files --others --exclude-standard)
    mod_files=$(git -C "${mod}" ls-files; git -C "${mod}" ls-files --others --exclude-standard)
    all_files=$(printf '%s\n%s\n' "${base_files}" "${mod_files}" | sort -u)

    while IFS= read -r file; do
        [ -n "${file}" ] || continue
        base_f="${base}/${file}"
        mod_f="${mod}/${file}"
        [ -f "${base_f}" ] || base_f="/dev/null"
        [ -f "${mod_f}" ] || mod_f="/dev/null"
        git diff --no-index "${base_f}" "${mod_f}" || true
    done <<< "${all_files}" > "diffs/${repo}.diff"
done

# Diff CloudMicrophysics.jl-mod against the exact version that runs alongside
# the baseline ClimaAtmos inside the coupler AMIP environment. That version is
# pinned by `git-tree-sha1` in the baseline coupler's AMIP Manifest, so when the
# CM submodule is checked out to the matching commit this diff is empty
# (i.e. CloudMicrophysics has no influence on the experiment).
CM_MANIFEST="ClimaCoupler.jl/experiments/AMIP/Manifest-v1.11.toml"
CM_TREE=$(awk '
    /^\[\[deps\.CloudMicrophysics\]\]/ { f = 1; next }
    f && /^git-tree-sha1/ { print; exit }
' "${CM_MANIFEST}" | sed -E 's/.*"([0-9a-f]+)".*/\1/')

if [ -z "${CM_TREE}" ]; then
    echo "ERROR: could not read CloudMicrophysics git-tree-sha1 from ${CM_MANIFEST}" >&2
    exit 1
fi

# Two diffs, because the pinned tree answers a different question than "what did
# we change". `git diff <tree-ish>` against the manifest pin shows the TOTAL
# delta between the arms, which is what determines behaviour -- but it also
# sweeps in everything upstream landed on CloudMicrophysics main since that
# release: vendored docs/dev-guides, unrelated src modules, doc plots. That came
# to 39 files and 1534 insertions against 5 files of actual work, which buried
# the change that produces the benefit.
#
#   CloudMicrophysics.diff             <- our authored changes only
#   CloudMicrophysics-arm-delta.diff   <- full difference from what baseline runs
#
# Read the first to review the work; read the second to know what actually
# differs between the two arms of an experiment.
git -C CloudMicrophysics.jl-mod diff "${CM_TREE}" -- . \
    > diffs/CloudMicrophysics-arm-delta.diff || true

# Authored changes: everything on the working tree that is not on upstream main.
CM_BASE=$(git -C CloudMicrophysics.jl-mod merge-base HEAD origin/main)
if [ -z "${CM_BASE}" ]; then
    echo "ERROR: could not find CloudMicrophysics merge-base with origin/main" >&2
    exit 1
fi
git -C CloudMicrophysics.jl-mod diff "${CM_BASE}" -- . > diffs/CloudMicrophysics.diff || true
