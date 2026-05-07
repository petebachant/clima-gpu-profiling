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

# Now simply diff CloudMicrophysics.jl-mod against origin/main
git -C CloudMicrophysics.jl-mod fetch origin main
git -C CloudMicrophysics.jl-mod diff origin/main -- . > diffs/CloudMicrophysics.diff || true
