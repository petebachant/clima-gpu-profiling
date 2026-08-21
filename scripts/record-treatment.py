"""Record what actually ran, so an experiment can be identified without archaeology.

A superproject commit already pins every submodule SHA, so the code state *is*
captured by git. What git does not capture is the part that decides whether those
SHAs matter: which packages the two Coupler environments actually `dev` (a
submodule can be checked out and still be irrelevant, as CloudMicrophysics is
whenever it is not dev'd), whether a working tree was dirty when the job ran, and
which CLIMA_* environment variables the stages export. Each of those has silently
changed the meaning of a result at least once in this project.

Writes results/treatment.json. Cheap, CPU-only, no GPU or network.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "treatment.json"
COUPLERS = {"baseline": "ClimaCoupler.jl", "mod": "ClimaCoupler.jl-mod"}


def git(*args, cwd=ROOT):
    """Run git, returning stripped stdout, or None if the command fails."""
    try:
        r = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True
        )
        return r.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def submodule_paths():
    """Submodule paths from .gitmodules, in declaration order."""
    cfg = git("config", "--file", ".gitmodules", "--get-regexp", "path") or ""
    return [line.split(" ", 1)[1] for line in cfg.splitlines() if " " in line]


def describe_submodule(path):
    d = ROOT / path
    if not (d / ".git").exists():
        return {"sha": None, "ref": None, "dirty": None, "note": "not initialized"}
    sha = git("rev-parse", "HEAD", cwd=d)
    branch = git("symbolic-ref", "-q", "--short", "HEAD", cwd=d)
    # A detached HEAD is normal here (we pin CloudMicrophysics to a tag), so
    # record the tag when there is one rather than reporting "detached".
    tag = git("describe", "--tags", "--exact-match", cwd=d) if branch is None else None
    status = git("status", "--porcelain", cwd=d)
    return {
        "sha": sha,
        "ref": branch or tag or "detached",
        "dirty": bool(status),
        # Uncommitted changes make the SHA a lie, so name the files rather than
        # just flagging it -- that is the difference between "rerun this commit"
        # and "this result is not reproducible".
        "dirty_files": [l[3:] for l in status.splitlines()][:20] if status else [],
    }


def dev_packages(coupler):
    """Packages the AMIP environment resolves to a local path, and pinned versions.

    A `path = ...` entry in the manifest is what makes a submodule's contents
    reach the run. Everything else comes from the registry, no matter what is
    checked out.
    """
    manifest = ROOT / coupler / "experiments" / "AMIP" / "Manifest-v1.11.toml"
    if not manifest.exists():
        return {"dev": [], "pinned": {}, "note": "manifest missing"}
    text = manifest.read_text()
    dev, pinned = [], {}
    # Manifest entries look like [[deps.Name]] followed by keys until the next
    # blank-line-separated block.
    for block in re.split(r"\n(?=\[\[deps\.)", text):
        m = re.match(r"\[\[deps\.([A-Za-z0-9_]+)\]\]", block)
        if not m:
            continue
        name = m.group(1)
        if re.search(r"^path = ", block, re.M):
            dev.append(name)
        elif ver := re.search(r'^version = "([^"]+)"', block, re.M):
            pinned[name] = ver.group(1)
    return {"dev": sorted(dev), "pinned": pinned}


def declared_env():
    """CLIMA_* variables the pipeline exports, per stage.

    Read from calkit.yaml rather than os.environ: this script runs on the login
    node, while the stages that matter run inside SLURM jobs, so the ambient
    environment here says nothing about what they saw.
    """
    text = (ROOT / "calkit.yaml").read_text()
    env, stage = {}, None
    for line in text.splitlines():
        if m := re.match(r"^    ([a-z][a-z0-9-]*):$", line):
            stage = m.group(1)
        for var, val in re.findall(r"(CLIMA_[A-Z_]+)=([^,\s]+)", line):
            env.setdefault(stage, {})[var] = val
    return env


def diff_digest():
    """Digest over diffs/, which is the make-diffs stage's record of local edits.

    Two experiments with identical submodule SHAs but different digests means
    someone ran with uncommitted changes.
    """
    d = ROOT / "diffs"
    if not d.is_dir():
        return None
    h = hashlib.sha256()
    for f in sorted(d.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(d).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def tree_digest(path):
    """Content digest of a submodule's tracked files, ignoring the `-mod` suffix.

    Lets a `-mod` submodule be compared against its sibling for actual sameness.
    Paths and manifest entries differ by that suffix by construction, so leaving
    it in would report every pair as different and make a null test impossible
    to recognize.
    """
    d = ROOT / path
    files = git("ls-files", cwd=d)
    if files is None:
        return None
    h = hashlib.sha256()
    for rel in sorted(files.splitlines()):
        f = d / rel
        if not f.is_file():
            continue
        h.update(rel.replace("-mod", "").encode())
        h.update(f.read_bytes().replace(b"-mod", b""))
    return h.hexdigest()[:16]


def main():
    subs = {p: describe_submodule(p) for p in submodule_paths()}
    envs = {arm: dev_packages(c) for arm, c in COUPLERS.items()}

    # The comparison is only meaningful if the two arms differ in a way we can
    # name. Compute it rather than trusting the commit message.
    delta = []
    for path in subs:
        if not path.endswith("-mod"):
            continue
        sibling = path[: -len("-mod")]
        if sibling in subs:
            # SHAs are useless here: the two Couplers live on separate branches
            # and always have different commits even when their content matches.
            # Compare content instead, normalizing the `-mod` suffix that appears
            # in paths and manifests by construction.
            if tree_digest(path) != tree_digest(sibling):
                delta.append(path)
        else:
            # No sibling (CloudMicrophysics): it is a treatment only when the mod
            # environment actually dev's it. Checked out but not dev'd means the
            # registry version is what runs, so it changes nothing.
            pkg = sibling.removesuffix(".jl")
            if pkg in envs["mod"]["dev"] and pkg not in envs["baseline"]["dev"]:
                delta.append(path)
    if envs["baseline"]["dev"] != envs["mod"]["dev"]:
        delta.append("dev-package-set")

    record = {
        "superproject": {
            "sha": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        },
        "submodules": subs,
        "environments": envs,
        "declared_env": declared_env(),
        "diff_digest": diff_digest(),
        # Empty means the arms are identical: a null test, not a treatment.
        "arms_differ_in": sorted(set(delta)),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n")

    dirty = [p for p, i in subs.items() if i["dirty"]]
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  arms differ in: {record['arms_differ_in'] or 'NOTHING (null test)'}")
    if dirty:
        print(f"  WARNING: dirty submodules, SHAs do not describe this run: {dirty}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
