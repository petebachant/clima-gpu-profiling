"""Emit GitHub permalinks to the results files a PR body cites, pinned to a commit.

A figure quoted in a PR body is unverifiable on its own: the reader cannot tell
which run produced it, and `results/summary.json` on main holds whichever run wrote
it last. A permalink to the file at the commit that produced it is checkable by
anyone, with no Calkit account.

Line anchors are resolved by walking the file rather than counted by hand, so they
cannot drift from the key they are supposed to point at.
"""

import json
import subprocess
import sys

REPO = "https://github.com/petebachant/clima-gpu-profiling"

# The figures the #4879 body quotes, as (file, dotted key or None for whole file).
CITED = [
    ("results/summary.json", "speedup_pct"),
    ("results/summary.json", "baseline.sypd"),
    ("results/summary.json", "mod.sypd"),
    ("results/step-comparison.json", "saved_pct"),
    ("results/step-comparison.json", "typical_step.conversion_pct"),
    ("results/param-fold.json", "fold_attributable.ms_saved"),
    ("results/kernel-resources.csv", None),
]


def line_of(path, dotted):
    """Line number of a dotted key in an indent-2 pretty-printed JSON file.

    Depth comes from indentation, which is what `json.dump(indent=2)` guarantees, so
    the path is tracked without a position-aware parser.
    """
    want = dotted.split(".")
    stack = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            stripped = line.lstrip()
            if not stripped.startswith('"'):
                continue
            key = stripped.split('"')[1]
            depth = (len(line) - len(stripped)) // 2 - 1
            stack = stack[:depth] + [key]
            if stack == want:
                return n
    raise SystemExit(f"{path}: key {dotted} not found")


def value_of(path, dotted):
    d = json.load(open(path))
    for part in dotted.split("."):
        d = d[part]
    return d


def main():
    sha = sys.argv[1] if len(sys.argv) > 1 else subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--"] + sorted({f for f, _ in CITED}),
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if dirty:
        raise SystemExit(
            "cited results files differ from the commit being linked:\n" + dirty +
            "\ncommit them first, or the permalink will not show these values"
        )

    print(f"Evidence at `{sha[:7]}`:\n")
    for path, dotted in CITED:
        if dotted is None:
            print(f"- [`{path}`]({REPO}/blob/{sha}/{path})")
        else:
            n = line_of(path, dotted)
            v = value_of(path, dotted)
            shown = f"{v:.4g}" if isinstance(v, float) else v
            print(f"- `{dotted}` = {shown} — [`{path}` L{n}]({REPO}/blob/{sha}/{path}#L{n})")


main()
