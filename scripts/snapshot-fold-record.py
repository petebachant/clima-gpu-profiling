"""Pin the parameter fold's measured figures where no later run can overwrite them.

The fold's gain lives in three derived files -- the AMIP SYPD pair, the step
breakdown, and the per-kernel register and time table. All three are written by
stages that re-run whenever either arm's code changes, and the arms are now
identical: ClimaAtmos main carries the fold, so the next pass is a null test and
will rewrite all three with ~0%. The answers that cite them would then render the
null test's numbers under the fold's prose, which is the drift this project keeps
rediscovering.

`git_ref` is not enough on its own, because calkit renders a templated value from
the WORKING TREE and ignores the pin -- a pinned citation can display one number
while the tag it cites holds another. So the figures are copied out of the tag
instead, verbatim and under the same keys, and this stage re-copies them on every
run. The prose stays templated and the number it renders is the tag's by
construction.

Verbatim matters: a citation moves from results/summary.json to
results/fold-record/summary.json by changing the path alone, so the value keys
cannot be mistyped in the move.
"""

import json
import subprocess
import sys
from pathlib import Path

TAG = "exp/2026-10-21-fold-same-revision"
OUT = Path("results/fold-record")
FILES = ("summary.json", "step-comparison.json", "param-fold.json")

# One key per file that must be present, so a tag whose pipeline was not green
# fails here rather than producing a record with holes in it.
REQUIRED = {
    "summary.json": ("speedup_pct",),
    "step-comparison.json": ("typical_step",),
    "param-fold.json": ("kernels", "fold_attributable"),
}


def at(ref, path):
    out = subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, text=True
    )
    if out.returncode != 0:
        sys.exit(f"{ref}:{path} does not resolve: {out.stderr.strip()}")
    return out.stdout


def main():
    commit = subprocess.check_output(
        ["git", "rev-list", "-n1", TAG], text=True
    ).strip()

    OUT.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        blob = at(TAG, f"results/{name}")
        obj = json.loads(blob)
        for key in REQUIRED[name]:
            if key not in obj:
                sys.exit(f"{TAG}:results/{name} has no {key!r}")
        (OUT / name).write_text(blob)

    # The submodule pointers are what makes the record reproducible rather than
    # merely archived, so they travel with it.
    pointers = {}
    for line in subprocess.check_output(
        ["git", "ls-tree", TAG], text=True
    ).splitlines():
        meta, _, path = line.partition("\t")
        mode, kind, sha = meta.split()
        if kind == "commit":
            pointers[path] = sha

    summary = json.loads((OUT / "summary.json").read_text())
    provenance = {
        "tag": TAG,
        "commit": commit,
        "subject": subprocess.check_output(
            ["git", "tag", "-l", "--format=%(contents:subject)", TAG], text=True
        ).strip(),
        "files": list(FILES),
        "submodules": pointers,
        "speedup_pct": summary["speedup_pct"],
        "note": (
            "Copied verbatim out of the tag by scripts/snapshot-fold-record.py, "
            "which re-runs on every pass. The stage has no file inputs: its "
            "source is a git tag, which DVC cannot hash, so it always runs."
        ),
    }
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"wrote {len(FILES) + 1} files to {OUT} from {TAG} ({commit[:9]})")
    print(f"  speedup_pct {summary['speedup_pct']:.4f}")


if __name__ == "__main__":
    main()
