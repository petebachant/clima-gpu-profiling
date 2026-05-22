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

# `git diff <tree-ish>` compares that tree to the working tree. The tree object
# is present locally via the registered release commit (e.g. v0.35.0).
git -C CloudMicrophysics.jl-mod diff "${CM_TREE}" -- . > diffs/CloudMicrophysics.diff || true
