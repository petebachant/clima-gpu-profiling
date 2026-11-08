"""Append the treated kernels' profiled figures for one mod-arm variant.

Run by hand after a `mod-nsys` pass, like `scripts/measure_occupancy.py` writes
`results/launch-inventory.json`, and for the same reason: `results/nsys/*.sqlite`
holds whichever experiment profiled last, so a document that wants to compare
two variants of one treatment cannot read it. This accumulates one entry per
ClimaAtmos revision instead, keyed by commit, and the document reads the
accumulated file.

Both kernels that the SGS node path compiles into are recorded, because the
treatment reaches both and they move in opposite directions. The cache's
environment quadrature broadcast runs the substep-averaged bulk tendency and sits
at the top of its occupancy band with no headroom; `update_implicit_1m_tendency`
runs the instantaneous one, so its node function needs far fewer registers and it
has room to move up a band.

Usage:
    python scripts/snapshot-quadrature-profile.py [label]
"""

import json
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

MOD = "results/nsys/mod.sqlite"
OUT = Path("results/quadrature-profiles.json")

# The two kernels the treated node path compiles into, by the function whose
# source line their name embeds. Lines move -- the mod arm's sedimentation fold
# inserts them into the same file -- so match the function and not the line.
TREATED = {
    "quadrature_cache": ("set_microphysics_tendency_cache", 240),
    "implicit_tendency": ("update_implicit_1m_tendency", 720),
}

A100_REGS_PER_SM = 65536
A100_THREADS_PER_SM = 2048
A100_BLOCKS_PER_SM = 32


def warps_per_sm(regs, block):
    blocks = min(A100_REGS_PER_SM // max(regs * block, 1),
                 A100_THREADS_PER_SM // max(block, 1), A100_BLOCKS_PER_SM)
    return blocks * ((block + 31) // 32)


def git(*args):
    r = subprocess.run(["git", *args], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def read(path):
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    with con:
        total = con.execute(
            "SELECT SUM(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_KERNEL"
        ).fetchone()[0]
        rows = con.execute("""
            SELECT s.value, COUNT(*), SUM(k.end-k.start)/1e6,
                   k.registersPerThread, k.blockX*k.blockY*k.blockZ
            FROM CUPTI_ACTIVITY_KIND_KERNEL k
            JOIN StringIds s ON k.shortName = s.id
            GROUP BY s.value, k.registersPerThread,
                     k.blockX*k.blockY*k.blockZ
        """).fetchall()
    con.close()

    out = {"device_total_ms": round(total, 1)}
    for tag, (func, launches) in TREATED.items():
        # The function's whole device time, over every line it launches from,
        # and separately the one population that carries it.
        family = [r for r in rows if func in r[0]]
        if not family:
            sys.exit(f"{path}: no kernel from {func}")
        heavy = [r for r in family if r[1] == launches]
        if not heavy:
            sys.exit(f"{path}: no {func} population with {launches} launches")
        name, n, ms, regs, block = max(heavy, key=lambda r: r[2])
        out[tag] = {
            "line": int(m.group(1)) if (m := re.search(r"_L(\d+)$", name)) else None,
            "launches": n,
            "total_ms": round(ms, 1),
            "family_total_ms": round(sum(r[2] for r in family), 1),
            "registers": regs,
            "block_size": block,
            "warps_per_sm": warps_per_sm(regs, block),
        }
    return out


def main(label=None):
    ref = git("-C", "ClimaAtmos.jl-mod", "rev-parse", "HEAD")
    dirty = git("-C", "ClimaAtmos.jl-mod", "status", "--porcelain")
    if dirty:
        sys.exit("ClimaAtmos.jl-mod is dirty; the profile would not be reproducible")

    doc = json.loads(OUT.read_text()) if OUT.exists() else {
        "note": "Profiled figures for the treated SGS node kernels, one entry "
                "per ClimaAtmos.jl-mod revision. Accumulated by hand because "
                "results/nsys/mod.sqlite holds whichever experiment profiled "
                "last. The `pre-treatment` entry is transcribed from "
                "results/launch-inventory.json, whose profile is gone.",
        "variants": {},
    }
    entry = read(MOD)
    entry["climaatmos_commit"] = ref
    entry["climaatmos_branch"] = git("-C", "ClimaAtmos.jl-mod", "rev-parse",
                                     "--abbrev-ref", "HEAD")
    entry["superproject_commit"] = git("rev-parse", "HEAD")
    entry["label"] = label or ref[:9]
    doc["variants"][ref[:9]] = entry
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")

    print(f"wrote {OUT}: variant {ref[:9]} ({entry['label']})")
    for tag in TREATED:
        e = entry[tag]
        print(f"  {tag:18} {e['total_ms']:8} ms  {e['registers']:3} regs  "
              f"block {e['block_size']:4}  {e['warps_per_sm']:2} warps/SM")
    print(f"  device total       {entry['device_total_ms']} ms")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
