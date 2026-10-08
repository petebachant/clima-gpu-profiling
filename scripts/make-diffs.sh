#!/usr/bin/env bash
# Diff each repo's `mod` against its base and save the diffs in `diffs`.
#
# Only packages carrying an actual treatment land at the top level, so `ls
# diffs` answers "what is under test" without a reader having to open files to
# find out. A package whose arms differ only in the AMIP Manifest's dev paths is
# wired, not treated, and goes under diffs/not-treatments; an identical package
# gets no file at all. diffs/README.md lists every package either way, so an
# absent diff is stated rather than inferred from a missing file.

set -euo pipefail

rm -rf diffs
mkdir -p diffs/not-treatments

MANIFEST_PATH="experiments/AMIP/Manifest-v1.11.toml"
index=""

# RRTMGP is here because both arms have their own submodule at the same base,
# so the plain base-vs-mod directory comparison works. That symmetry is
# deliberate: CloudMicrophysics has a mod-only submodule against a registry
# baseline, which is why its diff needs the special handling below and why
# upstream drift rode along inside every CM experiment (docs/learnings.md 4i).
for repo in ClimaAtmos.jl ClimaCore.jl ClimaCoupler.jl RRTMGP.jl; do
    base="${repo}"
    mod="${repo}-mod"
    tmp="$(mktemp)"

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
    done <<< "${all_files}" > "${tmp}"

    # Which files the diff actually touches, as paths inside the package.
    changed=$(grep '^diff --git ' "${tmp}" | sed -E "s|^diff --git a/${base}/||; s| b/.*$||" | sort -u || true)
    other=$(printf '%s\n' "${changed}" | grep -v "^${MANIFEST_PATH}$" | grep -v '^$' || true)
    n_files=$(printf '%s\n' "${changed}" | grep -c '[^[:space:]]' || true)
    n_lines=$(grep -c '^[+-][^+-]' "${tmp}" || true)

    if [ -z "${changed}" ]; then
        rm -f "${tmp}"
        index="${index}- **${repo}** --- identical in both arms, so no diff file.\n"
    elif [ -z "${other}" ]; then
        mv "${tmp}" "diffs/not-treatments/${repo}.diff"
        index="${index}- **${repo}** --- not-treatments/${repo}.diff: ${n_lines} changed lines, all in ${MANIFEST_PATH}. This is the dev-path rewiring that points the mod arm at the -mod submodules. It is how the arm is built, not what is under test.\n"
    else
        mv "${tmp}" "diffs/${repo}.diff"
        index="${index}- **${repo}** --- ${repo}.diff: ${n_lines} changed lines across ${n_files} file(s). **Under test.**\n"
    fi
done

# CloudMicrophysics only reaches a run if the mod Coupler devs it. When it does
# not, diffing its checkout describes nothing that ran -- and a 615-line diff
# sitting beside the results reads as a treatment. Say so in the index instead
# of writing a file that has to be opened to learn it is not a diff.
if ! grep -q 'path = "../../../CloudMicrophysics.jl-mod"' \
    ClimaCoupler.jl-mod/experiments/AMIP/Manifest-v1.11.toml; then
    index="${index}- **CloudMicrophysics.jl** --- not dev'd by either arm, so its submodule does not reach the run and there is no arm difference to report. Both arms take the version pinned in the Coupler AMIP manifest.\n"
else
    CM_MANIFEST="ClimaCoupler.jl/experiments/AMIP/Manifest-v1.11.toml"
    CM_TREE=$(awk '
        /^\[\[deps\.CloudMicrophysics\]\]/ { f = 1; next }
        f && /^git-tree-sha1/ { print; exit }
    ' "${CM_MANIFEST}" | sed -E 's/.*"([0-9a-f]+)".*/\1/')

    if [ -z "${CM_TREE}" ]; then
        echo "ERROR: could not read CloudMicrophysics git-tree-sha1 from ${CM_MANIFEST}" >&2
        exit 1
    fi

    # Two diffs, because the pinned tree answers a different question than "what
    # did we change". `git diff <tree-ish>` against the manifest pin shows the
    # TOTAL delta between the arms, which is what determines behavior -- but it
    # also sweeps in everything upstream landed on CloudMicrophysics main since
    # that release: vendored docs/dev-guides, unrelated src modules, doc plots.
    # That came to 39 files and 1534 insertions against 5 files of actual work,
    # which buried the change that produces the benefit.
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
    index="${index}- **CloudMicrophysics.jl** --- CloudMicrophysics.diff (authored changes) and CloudMicrophysics-arm-delta.diff (full difference from what the baseline runs). **Under test.**\n"
fi

rmdir diffs/not-treatments 2>/dev/null || true

{
    echo "# What differs between the two arms"
    echo
    echo "Generated by scripts/make-diffs.sh. A package with no diff file below"
    echo "is identical in both arms; the list is complete either way, so nothing"
    echo "has to be inferred from a file's absence."
    echo
    printf '%b' "${index}"
} > diffs/README.md

echo "--- diffs/ ---"
ls diffs
echo "--- index ---"
printf '%b' "${index}"
