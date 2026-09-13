"""Assert DVC hashes exactly the git-tracked files of each submodule dependency.

Two failure modes, opposite and both silent:

  dvc > tracked   A machine-local file (a resolved Manifest.toml, tooling state,
                  editor scratch) is inside a dependency directory. DVC honours
                  .dvcignore, not .gitignore, so it is hashed -- and the stage
                  goes stale only on clones that lack the file. A tag can look
                  clean where it was made and stale everywhere else.

  dvc < tracked   A .dvcignore pattern is swallowing tracked files, so real
                  changes to them no longer invalidate the stage. This is the
                  worse one: it does not announce itself at all.

Run after editing .dvcignore or adding a submodule dependency.
"""

import subprocess
import sys

import yaml


def tracked(path: str) -> int | None:
    """Count git-tracked files in a submodule, or None if it is not one."""
    r = subprocess.run(
        ["git", "-C", path, "ls-files"], capture_output=True, text=True
    )
    return None if r.returncode else len(r.stdout.split())


def main() -> int:
    lock = yaml.safe_load(open("dvc.lock"))
    seen, bad = {}, 0
    for stage in lock["stages"].values():
        for dep in stage.get("deps", []):
            if "nfiles" not in dep or "/" in dep["path"].rstrip("/"):
                continue  # only whole-submodule deps have a git file list
            seen[dep["path"]] = dep["nfiles"]
    for path, n in sorted(seen.items()):
        t = tracked(path)
        if t is None:
            continue
        if t != n:
            bad += 1
            extra = "hashing untracked files" if n > t else "ignoring tracked files"
            print(f"FAIL {path}: dvc={n} tracked={t} ({extra})")
        else:
            print(f"ok   {path}: {n}")
    if bad:
        print(f"\n{bad} dependency directories disagree with git.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
