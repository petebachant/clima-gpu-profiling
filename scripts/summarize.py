"""Extract a summary from an nsys report file."""

import json
import sqlite3

fpath_in_baseline = "results/nsys/baseline.sqlite"
fpath_in_mod = "results/nsys/mod.sqlite"
fpath_out = "results/summary.json"


def get_summary(fpath_in) -> dict:
    """Extract summary from nsys report file."""
    conn = sqlite3.connect(fpath_in)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM ANALYSIS_DETAILS")
    row = cursor.fetchone()
    columns = [description[0] for description in cursor.description]
    summary = dict(zip(columns, row))
    summary["duration_s"] = summary["duration"] / 1e9  # Convert from ns to s
    conn.close()
    return summary


baseline = get_summary(fpath_in_baseline)
mod = get_summary(fpath_in_mod)

results = {
    "speedup_pct": (1 - mod["duration"] / baseline["duration"]) * 100,
    "baseline": baseline,
    "mod": mod,
}


with open(fpath_out, "w") as f:
    json.dump(results, f, indent=2)

print(f"🚀 Speedup: {results['speedup_pct']:.1f}%")
