"""Check that every Q&A evidence item resolves at its git_ref with the value claimed.

A `git show <ref>:<path>` that merely succeeds proves the file existed, not that
it holds the number the answer quotes. On 2026-09-13 a tag was published whose
`top-kernels.csv` carried the previous configuration's radiation figures: the
stage that writes it had not re-run, and a presence-only check passed.

Keyed evidence is compared against the answer text. Tables are reported with the
figures a reader can check by eye, since their claim lives in prose.

Since the answers moved to templated values, the check that matters is different:
calkit injects a named value by rendering it from the WORKING TREE and ignoring
`git_ref`, so a pinned citation can display one number while the tag it cites
holds another. That is reported as DRIFT, and it found a real case --
`guard_fires_pct` rendered 81.80% while citing a tag carrying 88.31%.
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
    # A cell is blank when a kernel exists in one arm only, which is what a
    # successful fusion looks like: the clear-sky kernels are in the baseline
    # column and absent from the mod one. Blank means zero time, not no data.
    def ms(row, col):
        return float(row[col] or 0.0)

    base = sum(ms(r, "total_ms_baseline") for r in rows if "rte_" in r["kernel"])
    mod = sum(ms(r, "total_ms_mod") for r in rows if "rte_" in r["kernel"])
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
        # Prose writes large counts with thousands separators.
        f"{int(value):,}" if float(value).is_integer() else "",
        f"{value:,.1f}", f"{value:,.2f}",
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
                name = e.get("name")
                if name and "{" + name in answer:
                    # Templated. The number in prose cannot drift from the
                    # results file -- but calkit renders from the WORKING TREE
                    # and ignores git_ref, so it can drift from the pinned ref,
                    # which is the one the answer claims to rest on. That is
                    # the failure this branch exists to catch.
                    try:
                        live = dig(open(path).read(), path, key)
                    except Exception:
                        live = None
                    if live is None:
                        mark, note = "NO LIVE VALUE", " (renders blank)"
                    elif isinstance(value, float) and isinstance(live, float):
                        drifted = abs(live - value) > 1e-9 * max(1.0, abs(value))
                        mark = "DRIFT" if drifted else "ok"
                        note = f" renders {live}" if drifted else ""
                    else:
                        mark = "ok" if live == value else "DRIFT"
                        note = f" renders {live}" if mark == "DRIFT" else ""
                elif name:
                    # Declares a name the answer never uses: either a figure
                    # meant to be injected and left typed, or a promise to
                    # drop. Only clean if the literal is in the prose.
                    mark = "ok" if in_answer(value, answer) else "NOT IN ANSWER"
                    note = "" if mark == "ok" else (
                        f" (declares name '{name}' that the answer never uses)"
                    )
                elif in_answer(value, answer):
                    mark, note = "ok", ""
                else:
                    # No name and not quoted: supporting evidence, which is a
                    # legitimate thing for an answer to rest on without
                    # putting a number in prose. Reported, not failed -- a
                    # check that cries wolf on these is a check nobody reads.
                    mark, note = "supporting", ""
                if mark not in ("ok", "supporting"):
                    bad += 1
                print(f"{mark:14s} {ref}:{path} {key}={value}{note}")
            else:
                pct = radiation_pct(blob)
                extra = "" if pct is None else f" radiation={pct:+.2f}%"
                print(f"{'table':14s} {ref}:{path}{extra}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
