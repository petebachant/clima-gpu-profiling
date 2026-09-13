"""Check that every Q&A evidence item resolves at its git_ref with the value claimed.

A `git show <ref>:<path>` that merely succeeds proves the file existed, not that
it holds the number the answer quotes. On 2026-09-13 a tag was published whose
`top-kernels.csv` carried the previous configuration's radiation figures: the
stage that writes it had not re-run, and a presence-only check passed.

Keyed evidence is compared against the answer text. Tables are reported with the
figures a reader can check by eye, since their claim lives in prose.
"""

import json
import re
import subprocess
import sys
import tomllib

import yaml


def at_ref(ref: str, path: str) -> str | None:
    """Return a file's contents at a git ref, or None if it is not there."""
    out = subprocess.run(
        ["git", "show", f"{ref}:{path}"], capture_output=True, text=True
    )
    return out.stdout if out.returncode == 0 else None


def dig(blob: str, path: str, key: str):
    """Pull a dotted key out of a JSON or TOML blob."""
    obj = json.loads(blob) if path.endswith(".json") else tomllib.loads(blob)
    for part in key.split("."):
        obj = obj[part]
    return obj


def radiation_pct(blob: str) -> float | None:
    """Radiation's change in total_ms, for the kernel table."""
    import csv
    import io

    rows = list(csv.DictReader(io.StringIO(blob)))
    if not rows or "total_ms_baseline" not in rows[0]:
        return None
    base = sum(
        float(r["total_ms_baseline"]) for r in rows if "rte_" in r["kernel"]
    )
    mod = sum(float(r["total_ms_mod"]) for r in rows if "rte_" in r["kernel"])
    return None if base == 0 else 100 * (mod / base - 1)



def in_answer(value, answer: str) -> bool:
    """Whether a numeric value appears in the answer in some plausible form.

    A value reaches prose rounded, and often as a percentage of itself: 0.01196
    is quoted as "1.20%". Check the renderings a writer would actually use
    rather than one canonical string.
    """
    if not isinstance(value, (int, float)):
        return str(value) in answer
    text = answer.replace("−", "-")
    candidates = {
        f"{value:.2f}", f"{value:.3f}", f"{value:g}",
        f"{value * 100:.2f}", f"{value * 100:.1f}",
        f"{abs(value):.2f}", f"{abs(value) * 100:.2f}",
        str(int(value)) if float(value).is_integer() else "",
    }
    return any(c and c in text for c in candidates)


def main() -> int:
    info = yaml.safe_load(open("calkit.yaml"))
    bad = 0
    for q in info.get("questions", []):
        answer = str(q.get("answer", ""))
        for e in q.get("evidence", []):
            ref, path, key = e.get("git_ref"), e.get("path"), e.get("key")
            if not ref:
                continue
            blob = at_ref(ref, path)
            if blob is None:
                print(f"MISSING  {ref}:{path}")
                bad += 1
                continue
            if key:
                try:
                    value = dig(blob, path, key)
                except Exception as exc:
                    print(f"KEY      {ref}:{path} {key}: {exc}")
                    bad += 1
                    continue
                mark = "ok" if in_answer(value, answer) else "NOT IN ANSWER"
                if mark != "ok":
                    bad += 1
                print(f"{mark:14s} {ref}:{path} {key}={value}")
            else:
                pct = radiation_pct(blob)
                extra = "" if pct is None else f" radiation={pct:+.2f}%"
                print(f"{'table':14s} {ref}:{path}{extra}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
