"""A100 occupancy bands, and the register-cap policy's filters replayed offline.

ClimaCore's automatic register cap decides per kernel (see `auto_register_cap_for`
in its CUDA extension) and records every decision, so a census holds the register
count, the local frame and the warps of each specialization. Everything but the
spill test is arithmetic on the register count, which means a census taken at one
setting can be re-read at another without a GPU.

The arithmetic here mirrors `max_warps_per_sm_by_regs` and `band_edge_below`
exactly. `assert_matches_census` is what keeps it honest: a census records the
warps the extension computed, so a replay that disagrees on any row is wrong
about the device and not merely about the policy.
"""

A100 = dict(
    threads_per_warp=32,
    max_threads_per_sm=2048,
    max_regs_per_sm=65536,
    regs_per_allocation=256,
    max_regs_per_thread=256,
    schedulers_per_sm=4,
)

# The band edges of an A100, as the Julia test asserts them.
A100_BAND_EDGES = [32, 40, 48, 56, 64, 72, 80, 96, 128, 168, 256]


def warps(regs, attrs=A100):
    """Resident warps per SM that `regs` registers per thread allow."""
    per_sm = attrs["max_threads_per_sm"] // attrs["threads_per_warp"]
    if regs == 0:
        return per_sm
    if regs > attrs["max_regs_per_thread"]:
        return 0
    granule = attrs["regs_per_allocation"]
    allocated = -(-(regs * attrs["threads_per_warp"]) // granule) * granule
    per_scheduler = attrs["max_regs_per_sm"] // attrs["schedulers_per_sm"]
    return min(per_sm, (per_scheduler // allocated) * attrs["schedulers_per_sm"])


def band_edge_below(regs, attrs=A100):
    """The largest count below `regs` that allows more warps, or None."""
    here = warps(regs, attrs)
    for edge in range(regs - 1, 0, -1):
        if warps(edge, attrs) > here:
            return edge
    return None


def band_edges(attrs=A100):
    """Every count whose successor allows fewer warps."""
    top = attrs["max_regs_per_thread"]
    return [r for r in range(1, top + 1) if warps(r + 1, attrs) < warps(r, attrs)]


def filters_before_compile(regs, max_warps, max_shed, attrs=A100):
    """The policy's verdict from arithmetic alone.

    Returns (reason, edge, shed). `reason` is "candidate" when the kernel reaches
    the point of being compiled at its edge, which is where the spill test takes
    over and arithmetic runs out.
    """
    if warps(regs, attrs) > max_warps:
        return "already_resident", None, None
    edge = band_edge_below(regs, attrs)
    if edge is None:
        return "no_band_below", None, None
    if regs - edge > max_shed:
        return "edge_too_far", edge, regs - edge
    return "candidate", edge, regs - edge


# Verdicts the policy can only reach by compiling the kernel at its edge.
POST_COMPILE = ("capped", "cap_not_honored", "buys_no_warps",
                "spill_over_budget", "capped_compile_failed")


def replay(row, max_warps, max_shed, attrs=A100):
    """What the policy would decide for a census row at another setting.

    Exact only where the row's own census already paid for the capped compile:
    a row recorded as `already_resident` never reached it, so a setting that
    admits the row can be told it is a candidate and no more. A row recorded
    with a post-compile verdict carries that verdict at every setting that
    admits it, because the compile does not depend on the filters.
    """
    reason, edge, shed = filters_before_compile(
        row["registers"], max_warps, max_shed, attrs)
    if reason != "candidate":
        return reason
    if row["reason"] in POST_COMPILE:
        return row["reason"]
    return "candidate"


def assert_matches_census(rows, attrs=A100):
    """Check the arithmetic against the warps the extension itself recorded."""
    bad = [(r["kernel"], r["registers"], r["warps_per_sm"], warps(r["registers"], attrs))
           for r in rows if warps(r["registers"], attrs) != r["warps_per_sm"]]
    assert not bad, f"warps disagree on {len(bad)} of {len(rows)} rows: {bad[:3]}"
    recorded = [(r["registers"], r["edge"]) for r in rows if r["edge"] > 0]
    bad = [(regs, edge, band_edge_below(regs, attrs))
           for regs, edge in recorded if band_edge_below(regs, attrs) != edge]
    assert not bad, f"edges disagree on {len(bad)} of {len(recorded)} rows: {bad[:3]}"
    return len(rows), len(recorded)
