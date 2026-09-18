#!/usr/bin/env bash
# Install the repo's hooks. Hooks live in scripts/git-hooks so they are
# versioned; .git/hooks is not.
set -euo pipefail
cd "$(dirname "$0")/.."
for h in scripts/git-hooks/*; do
    n=$(basename "$h")
    ln -sf "../../scripts/git-hooks/$n" ".git/hooks/$n"
    echo "  installed $n"
done
