"""Pin a tag's measured figures where no later run can overwrite them.

A measurement's gain lives in derived files -- the AMIP SYPD pair, the step
breakdown, and the per-kernel register and time table -- all written by stages
that re-run whenever either arm's code changes. So each new pass rewrites the
numbers the previous pass's answers cite, which is the drift this project keeps
rediscovering.

`git_ref` is not enough on its own, because calkit renders a templated value from
the WORKING TREE and ignores the pin -- a pinned citation can display one number
while the tag it cites holds another. So the figures are copied out of the tag
instead, verbatim and under the same keys, and this stage re-copies them on every
run. The prose stays templated and the number it renders is the tag's by
construction.

Verbatim matters: a citation moves from results/summary.json to
results/<record>/summary.json by changing the path alone, so the value keys
cannot be mistyped in the move.

Two records are pinned. The fold's, because ClimaAtmos main now carries the fold
and the arms are identical, so every later pass measures ~0% and would render the
null test's numbers under the fold's prose. And the null test's own, because it
is what the fold's figures must be read against -- a treatment of nothing,
measured three ways that disagree by several points, is the noise floor, and it
is as perishable as the result it calibrates.
"""

import json
import subprocess
import sys
from pathlib import Path

FILES = ("summary.json", "step-comparison.json", "param-fold.json")

RECORDS = {
    "results/fold-record": "exp/2026-10-21-fold-same-revision",
    "results/null-record": "exp/2026-10-22-null-test-on-main",
}

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


def snapshot(out_dir, tag):
    out = Path(out_dir)
    commit = subprocess.check_output(
        ["git", "rev-list", "-n1", tag], text=True
    ).strip()

    out.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        blob = at(tag, f"results/{name}")
        obj = json.loads(blob)
        for key in REQUIRED[name]:
            if key not in obj:
                sys.exit(f"{tag}:results/{name} has no {key!r}")
        (out / name).write_text(blob)

    # The submodule pointers are what makes the record reproducible rather than
    # merely archived, so they travel with it.
    pointers = {}
    for line in subprocess.check_output(
        ["git", "ls-tree", tag], text=True
    ).splitlines():
        meta, _, path = line.partition("\t")
        mode, kind, sha = meta.split()
        if kind == "commit":
            pointers[path] = sha

    summary = json.loads((out / "summary.json").read_text())
    provenance = {
        "tag": tag,
        "commit": commit,
        "subject": subprocess.check_output(
            ["git", "tag", "-l", "--format=%(contents:subject)", tag], text=True
        ).strip(),
        "files": list(FILES),
        "submodules": pointers,
        "speedup_pct": summary["speedup_pct"],
        "note": (
            "Copied verbatim out of the tag by scripts/snapshot-tag-records.py, "
            "which re-runs on every pass. The stage has no file inputs: its "
            "source is a git tag, which DVC cannot hash, so it always runs."
        ),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"wrote {len(FILES) + 1} files to {out} from {tag} ({commit[:9]})")
    print(f"  speedup_pct {summary['speedup_pct']:+.4f}")


def main():
    for out_dir, tag in RECORDS.items():
        snapshot(out_dir, tag)


if __name__ == "__main__":
    main()
