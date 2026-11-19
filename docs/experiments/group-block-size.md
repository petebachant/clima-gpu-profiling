# The grouped rounds' block size

`GROUP_BLOCK_THREADS` is one constant, shared by both radiation bands and by both halves of a round, and the comment above it justifies the value from the longwave's register counts alone: <!-- calkit value key=kernels.lw_optics.registers path=results/docs/group-block-size.json -->156<!-- /calkit value --> in the optics and <!-- calkit value key=kernels.lw_sweeps.registers path=results/docs/group-block-size.json -->140<!-- /calkit value --> in the sweeps reach <!-- calkit value key=kernels.lw_optics.ceiling_warps path=results/docs/group-block-size.json -->12<!-- /calkit value --> warps per multiprocessor at <!-- calkit value key=shipped_block path=results/docs/group-block-size.json -->128<!-- /calkit value --> threads against the <!-- calkit value key=bands.lw.points.256.warps path=results/docs/group-block-size.json -->8<!-- /calkit value --> they get at the solve's usual 256.
A review of CliMA/RRTMGP.jl#637 asked the obvious next question --- whether other block sizes reach the same occupancy, expecting 64 and 384 to --- which the PR had not measured.
This answers it two ways: the CUDA driver's own occupancy for every kernel at every size, and the clock.

The two are not the same question, and that turns out to be the result.

## Method

Every size is timed in one process against one settled AMIP state, with the same workspaces, the same lookups and the same state.
Only the launch configuration differs between points, so these are in-block comparisons and not differences of runs --- the only way to resolve a few percent on this machine.
The timing drives the shipped kernels through a copy of `_grouped_rounds!` that takes the block size as an argument, so nothing in the package changes and the kernels timed here are the ones the solve launches.
It times the rounds rather than the whole solve because the rounds are what this constant configures: the reduction and the state refresh compute their own launches from `ncol * nlev` and `ncol`.

Each size is timed twice, the second pass running the sweep backwards, because a sweep taken in one order cannot tell a property of a block size from a drift across the sweep.

```python calkit stage name=values environment=py outputs=[{path: results/docs/group-block-size.json, storage: git}] inputs=[results/group-block-size.toml]
import json
import tomllib

d = tomllib.load(open("results/group-block-size.toml", "rb"))
ks = d["kernels"]
shipped = str(d["shipped_block_threads"])


def say(blocks):
    """A list of block sizes as prose, since a marker renders a value and not a list."""
    blocks = [str(b) for b in blocks]
    if not blocks:
        return "none"
    if len(blocks) == 1:
        return blocks[0]
    return ", ".join(blocks[:-1]) + " and " + blocks[-1]


def kernel(name):
    k = ks[name]
    w = k["warps_per_sm"]
    return {
        "registers": k["registers"],
        "local_bytes": k["local_bytes"],
        "max_threads": k["max_threads"],
        "ceiling_warps": k["best_warps_per_sm"],
        "blocks_at_ceiling": say(k["blocks_at_best"]),
        "n_blocks_at_ceiling": len(k["blocks_at_best"]),
        "warps_at_shipped": w[shipped],
        "shipped_at_ceiling": w[shipped] == k["best_warps_per_sm"],
        # The two sizes the review names, each against this kernel's own
        # ceiling rather than against the shipped size.
        "warps_at_64": w["64"],
        "warps_at_384": w["384"],
        "sixtyfour_at_ceiling": w["64"] == k["best_warps_per_sm"],
        "threeeightyfour_at_ceiling": w["384"] == k["best_warps_per_sm"],
        "launchable_at_384": 384 <= k["max_threads"],
        "launchable_at_512": 512 <= k["max_threads"],
    }


def band(name):
    b = d["bands"][name]
    pts = b["points"]
    # What a difference between two points has to clear to be a difference:
    # the shipped point's own spread over every repeat of both passes.
    floor_pct = b["repeat_spread_pct"]

    def point(n):
        p = pts[str(n)]
        if p.get("skipped_too_many_registers", False):
            return {"block": n, "skipped": True}
        return {
            "block": n,
            "rounds_ms": round(p["rounds_ms"], 1),
            "change_pct": round(p["change_pct"], 2),
            "pass_gap_pct": round(p["pass_gap_pct"], 2),
            "warps": p["warps_per_sm_optics"],
            "warps_sweeps": p["warps_per_sm_sweeps"],
            "blocks_per_sm": p["blocks_per_sm_optics"],
            "grid_blocks": p["grid_blocks"],
            # Resolved against the shipped point's spread AND against this
            # point's own disagreement between the two passes, so a figure the
            # sweep cannot reproduce is not reported as a difference.
            "resolved": abs(p["change_pct"]) > max(floor_pct, p["pass_gap_pct"]),
            "bit_for_bit": p["bit_for_bit_with_shipped"],
        }

    points = {str(n): point(n) for n in b["measured_blocks"]}
    return {
        "band": name,
        "n_gpt": b["n_gpt"],
        "rounds": b["rounds"],
        "max_threads": b["band_max_threads"],
        "shipped_rounds_ms": round(b["shipped_rounds_ms"], 1),
        "repeat_spread_pct": round(floor_pct, 3),
        "worst_pass_gap_pct": round(b["worst_pass_gap_pct"], 2),
        "best_block": b["best_block_threads"],
        "best_change_pct": round(b["best_change_pct"], 2),
        "best_beats_shipped": b["best_beats_shipped"],
        "all_bit_for_bit": b["all_bit_for_bit"],
        "values_compared": b["values_compared"],
        "measured_blocks": say(b["measured_blocks"]),
        "points": points,
    }


kernels = {name: kernel(name) for name in ks}
bands = {name: band(name) for name in d["bands"]}

# The question as asked: which sizes give every kernel the occupancy its own
# registers allow. A size that is a ceiling for one band and a band below for
# the other is not an equal-occupancy alternative, however it looks per kernel.
equal = d["blocks_at_equal_occupancy"]

# Every claim the prose makes, as a field, so a pass that moves a figure moves
# the thing the sentence rests on with it.
equal_and_slower = [
    n
    for n in equal
    for b in bands.values()
    if b["points"][str(n)]["change_pct"] > 0 and b["points"][str(n)]["resolved"]
]
claims = {
    # The headline: same driver-reported occupancy, different time.
    "equal_occupancy_is_not_equal_time": len(equal_and_slower) > 0,
    "equal_occupancy_blocks_that_are_slower": say(sorted(set(equal_and_slower))),
    # The review's two sizes, each on its own terms.
    "sixtyfour_matches_occupancy_everywhere": all(
        k["sixtyfour_at_ceiling"] for k in kernels.values()
    ),
    "sixtyfour_is_slower_somewhere": any(
        b["points"]["64"]["change_pct"] > 0 and b["points"]["64"]["resolved"]
        for b in bands.values()
    ),
    "threeeightyfour_matches_occupancy_everywhere": all(
        k["threeeightyfour_at_ceiling"] for k in kernels.values()
    ),
    "threeeightyfour_costs_the_shortwave": not kernels["sw_optics"][
        "threeeightyfour_at_ceiling"
    ],
    "threeeightyfour_is_unlaunchable_in_the_longwave": not kernels["lw_optics"][
        "launchable_at_384"
    ],
    # Whether the comment's reasoning picks out 128 or merely admits it.
    "longwave_reason_admits_other_sizes": kernels["lw_optics"]["n_blocks_at_ceiling"] > 1,
    # Whether one shared constant can be read off either band alone.
    "bands_share_a_ceiling_set": sorted(ks["lw_optics"]["blocks_at_best"])
    == sorted(ks["sw_optics"]["blocks_at_best"]),
    "shipped_is_fastest_in_both": d["verdict"]["shipped_block_is_best"],
    "every_point_bit_for_bit": all(b["all_bit_for_bit"] for b in bands.values()),
}

out = {
    "ncol": d["ncol"],
    "nlay": d["nlay"],
    "ngroups": d["ngroups"],
    "repeats": d["repeats"],
    "passes": d["bands"]["lw"]["passes"],
    "shipped_block": d["shipped_block_threads"],
    "blocks_swept": say(d["block_sizes"]),
    "n_blocks_swept": len(d["block_sizes"]),
    "equal_occupancy_blocks": say(equal),
    "n_equal_occupancy_blocks": len(equal),
    "shipped_is_equal_occupancy": d["shipped_is_at_equal_occupancy"],
    "kernels": kernels,
    "bands": bands,
    "claims": claims,
    "bands_disagree_on_fastest": d["verdict"]["bands_disagree_on_best"],
    "faster_block_bands": say(d["verdict"]["bands_with_a_faster_block"]),
}

json.dump(out, open("results/docs/group-block-size.json", "w"), indent=2)
```

## What the driver says about occupancy

The four kernels a round launches do not sit in one band of the register file.

| kernel | registers | max threads | warps/SM ceiling | sizes that reach it |
|---|---|---|---|---|
| longwave optics | <!-- calkit value key=kernels.lw_optics.registers path=results/docs/group-block-size.json -->156<!-- /calkit value --> | <!-- calkit value key=kernels.lw_optics.max_threads path=results/docs/group-block-size.json -->384<!-- /calkit value --> | <!-- calkit value key=kernels.lw_optics.ceiling_warps path=results/docs/group-block-size.json -->12<!-- /calkit value --> | <!-- calkit value key=kernels.lw_optics.blocks_at_ceiling path=results/docs/group-block-size.json -->32, 64, 96, 128, 192 and 384<!-- /calkit value --> |
| longwave sweeps | <!-- calkit value key=kernels.lw_sweeps.registers path=results/docs/group-block-size.json -->140<!-- /calkit value --> | <!-- calkit value key=kernels.lw_sweeps.max_threads path=results/docs/group-block-size.json -->384<!-- /calkit value --> | <!-- calkit value key=kernels.lw_sweeps.ceiling_warps path=results/docs/group-block-size.json -->12<!-- /calkit value --> | <!-- calkit value key=kernels.lw_sweeps.blocks_at_ceiling path=results/docs/group-block-size.json -->32, 64, 96, 128, 192 and 384<!-- /calkit value --> |
| shortwave optics | <!-- calkit value key=kernels.sw_optics.registers path=results/docs/group-block-size.json -->124<!-- /calkit value --> | <!-- calkit value key=kernels.sw_optics.max_threads path=results/docs/group-block-size.json -->512<!-- /calkit value --> | <!-- calkit value key=kernels.sw_optics.ceiling_warps path=results/docs/group-block-size.json -->16<!-- /calkit value --> | <!-- calkit value key=kernels.sw_optics.blocks_at_ceiling path=results/docs/group-block-size.json -->32, 64, 128, 256 and 512<!-- /calkit value --> |
| shortwave sweeps | <!-- calkit value key=kernels.sw_sweeps.registers path=results/docs/group-block-size.json -->120<!-- /calkit value --> | <!-- calkit value key=kernels.sw_sweeps.max_threads path=results/docs/group-block-size.json -->512<!-- /calkit value --> | <!-- calkit value key=kernels.sw_sweeps.ceiling_warps path=results/docs/group-block-size.json -->16<!-- /calkit value --> | <!-- calkit value key=kernels.sw_sweeps.blocks_at_ceiling path=results/docs/group-block-size.json -->32, 64, 128, 256 and 512<!-- /calkit value --> |

The review's expectation about occupancy is correct, and more so than it claimed.

Several sizes tie: the longwave's ceiling of <!-- calkit value key=kernels.lw_optics.ceiling_warps path=results/docs/group-block-size.json -->12<!-- /calkit value --> warps per multiprocessor is reached at <!-- calkit value key=kernels.lw_optics.n_blocks_at_ceiling path=results/docs/group-block-size.json -->6<!-- /calkit value --> of the <!-- calkit value key=n_blocks_swept path=results/docs/group-block-size.json -->10<!-- /calkit value --> sizes swept, so the comment's reasoning does not pick 128 out --- it admits every one of them.
What the two bands do not do is agree about which sizes those are, and `bands_share_a_ceiling_set` records that as <!-- calkit value key=claims.bands_share_a_ceiling_set path=results/docs/group-block-size.json -->false<!-- /calkit value -->.
The shortwave's kernels are cheaper in registers, so their ceiling is <!-- calkit value key=kernels.sw_optics.ceiling_warps path=results/docs/group-block-size.json -->16<!-- /calkit value --> warps rather than 12, and it is reached at a different set of sizes.

That makes 384 the interesting case, and it is the one the review names.

It reaches the longwave's ceiling, but it leaves the shortwave at <!-- calkit value key=kernels.sw_optics.warps_at_384 path=results/docs/group-block-size.json -->12<!-- /calkit value --> of its own <!-- calkit value key=kernels.sw_optics.ceiling_warps path=results/docs/group-block-size.json -->16<!-- /calkit value --> --- a whole band below --- which `threeeightyfour_costs_the_shortwave` records as <!-- calkit value key=claims.threeeightyfour_costs_the_shortwave path=results/docs/group-block-size.json -->true<!-- /calkit value -->.
256 is the mirror image: the shortwave's ceiling, and <!-- calkit value key=bands.lw.points.256.warps path=results/docs/group-block-size.json -->8<!-- /calkit value --> of the longwave's <!-- calkit value key=kernels.lw_optics.ceiling_warps path=results/docs/group-block-size.json -->12<!-- /calkit value -->, which is the figure the existing comment quotes.
So a shared constant cannot be justified from either band's registers by itself, which is what the comment currently does.

Only <!-- calkit value key=n_equal_occupancy_blocks path=results/docs/group-block-size.json -->3<!-- /calkit value --> sizes give all four kernels the occupancy their own registers allow: <!-- calkit value key=equal_occupancy_blocks path=results/docs/group-block-size.json -->32, 64 and 128<!-- /calkit value -->.

## What the clock says

| block | LW warps/SM | LW vs <!-- calkit value key=shipped_block path=results/docs/group-block-size.json -->128<!-- /calkit value --> | SW warps/SM | SW vs <!-- calkit value key=shipped_block path=results/docs/group-block-size.json -->128<!-- /calkit value --> |
|---|---|---|---|---|
| 32 | <!-- calkit value key=bands.lw.points.32.warps path=results/docs/group-block-size.json -->12<!-- /calkit value --> | <!-- calkit value key=bands.lw.points.32.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+0.28<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.32.warps path=results/docs/group-block-size.json -->16<!-- /calkit value --> | <!-- calkit value key=bands.sw.points.32.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+1.89<!-- /calkit value -->% |
| 64 | <!-- calkit value key=bands.lw.points.64.warps path=results/docs/group-block-size.json -->12<!-- /calkit value --> | <!-- calkit value key=bands.lw.points.64.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+6.69<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.64.warps path=results/docs/group-block-size.json -->16<!-- /calkit value --> | <!-- calkit value key=bands.sw.points.64.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+1.57<!-- /calkit value -->% |
| 96 | <!-- calkit value key=bands.lw.points.96.warps path=results/docs/group-block-size.json -->12<!-- /calkit value --> | <!-- calkit value key=bands.lw.points.96.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+0.08<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.96.warps path=results/docs/group-block-size.json -->15<!-- /calkit value --> | <!-- calkit value key=bands.sw.points.96.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+0.66<!-- /calkit value -->% |
| 128 | <!-- calkit value key=bands.lw.points.128.warps path=results/docs/group-block-size.json -->12<!-- /calkit value --> | reference | <!-- calkit value key=bands.sw.points.128.warps path=results/docs/group-block-size.json -->16<!-- /calkit value --> | reference |
| 160 | <!-- calkit value key=bands.lw.points.160.warps path=results/docs/group-block-size.json -->10<!-- /calkit value --> | <!-- calkit value key=bands.lw.points.160.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+15.71<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.160.warps path=results/docs/group-block-size.json -->15<!-- /calkit value --> | <!-- calkit value key=bands.sw.points.160.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+0.62<!-- /calkit value -->% |
| 192 | <!-- calkit value key=bands.lw.points.192.warps path=results/docs/group-block-size.json -->12<!-- /calkit value --> | <!-- calkit value key=bands.lw.points.192.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+0.62<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.192.warps path=results/docs/group-block-size.json -->12<!-- /calkit value --> | <!-- calkit value key=bands.sw.points.192.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+15.08<!-- /calkit value -->% |
| 256 | <!-- calkit value key=bands.lw.points.256.warps path=results/docs/group-block-size.json -->8<!-- /calkit value --> | <!-- calkit value key=bands.lw.points.256.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+34.68<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.256.warps path=results/docs/group-block-size.json -->16<!-- /calkit value --> | <!-- calkit value key=bands.sw.points.256.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+1.18<!-- /calkit value -->% |
| 288 | <!-- calkit value key=bands.lw.points.288.warps path=results/docs/group-block-size.json -->9<!-- /calkit value --> | <!-- calkit value key=bands.lw.points.288.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+23.98<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.288.warps path=results/docs/group-block-size.json -->9<!-- /calkit value --> | <!-- calkit value key=bands.sw.points.288.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+40.23<!-- /calkit value -->% |
| 384 | <!-- calkit value key=bands.lw.points.384.warps path=results/docs/group-block-size.json -->12<!-- /calkit value --> | <!-- calkit value key=bands.lw.points.384.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+1.63<!-- /calkit value -->% | <!-- calkit value key=bands.sw.points.384.warps path=results/docs/group-block-size.json -->12<!-- /calkit value --> | <!-- calkit value key=bands.sw.points.384.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+16.44<!-- /calkit value -->% |
| 512 | past the <!-- calkit value key=bands.lw.max_threads path=results/docs/group-block-size.json -->384<!-- /calkit value -->-thread limit | --- | <!-- calkit value key=bands.sw.points.512.warps path=results/docs/group-block-size.json -->16<!-- /calkit value --> | <!-- calkit value key=bands.sw.points.512.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+2.80<!-- /calkit value -->% |

The shipped size is the fastest of the ten in both bands, and `shipped_is_fastest_in_both` records it as <!-- calkit value key=claims.shipped_is_fastest_in_both path=results/docs/group-block-size.json -->true<!-- /calkit value -->.
The resolution it is read against is each band's own: the longwave's shipped point reproduces itself to <!-- calkit value key=bands.lw.repeat_spread_pct format="{:.3f}" path=results/docs/group-block-size.json -->0.127<!-- /calkit value -->% across every repeat of both passes and the shortwave's to <!-- calkit value key=bands.sw.repeat_spread_pct format="{:.3f}" path=results/docs/group-block-size.json -->0.479<!-- /calkit value -->%, and the two passes disagree about no point by more than <!-- calkit value key=bands.lw.worst_pass_gap_pct format="{:.2f}" path=results/docs/group-block-size.json -->0.79<!-- /calkit value -->% in the longwave.

The result the question did not anticipate is that equal occupancy is not equal time, which `equal_occupancy_is_not_equal_time` records as <!-- calkit value key=claims.equal_occupancy_is_not_equal_time path=results/docs/group-block-size.json -->true<!-- /calkit value -->.

64 reaches every one of the four kernels' ceilings --- `sixtyfour_matches_occupancy_everywhere` is <!-- calkit value key=claims.sixtyfour_matches_occupancy_everywhere path=results/docs/group-block-size.json -->true<!-- /calkit value --> --- and costs the longwave <!-- calkit value key=bands.lw.points.64.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+6.69<!-- /calkit value -->% and the shortwave <!-- calkit value key=bands.sw.points.64.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+1.57<!-- /calkit value -->% anyway.
It is not a resolution artifact: the two passes' own medians for that point differ by <!-- calkit value key=bands.lw.points.64.pass_gap_pct format="{:.2f}" path=results/docs/group-block-size.json -->0.79<!-- /calkit value -->% in the longwave, against the <!-- calkit value key=bands.lw.points.64.change_pct format="{:+.2f}" path=results/docs/group-block-size.json -->+6.69<!-- /calkit value -->% being claimed.
At 64 threads the driver keeps <!-- calkit value key=bands.lw.points.64.blocks_per_sm path=results/docs/group-block-size.json -->6<!-- /calkit value --> blocks resident per multiprocessor against <!-- calkit value key=bands.lw.points.128.blocks_per_sm path=results/docs/group-block-size.json -->3<!-- /calkit value --> at 128, over a grid of <!-- calkit value key=bands.lw.points.64.grid_blocks format="{:,}" path=results/docs/group-block-size.json -->6,144<!-- /calkit value --> blocks rather than <!-- calkit value key=bands.lw.points.128.grid_blocks format="{:,}" path=results/docs/group-block-size.json -->3,072<!-- /calkit value --> --- the same warps, resident in twice as many blocks.
What that costs is measured here and not explained here; the point for the PR is that the driver's occupancy number does not settle the choice, so the block size has to be timed.

Every point is bit-for-bit with the shipped one, as it must be --- a thread's wide column is its global id whatever the block --- and `every_point_bit_for_bit` records that over <!-- calkit value key=bands.sw.values_compared format="{:,}" path=results/docs/group-block-size.json -->191,496,192<!-- /calkit value --> shortwave values as <!-- calkit value key=claims.every_point_bit_for_bit path=results/docs/group-block-size.json -->true<!-- /calkit value -->.

## Verdict

128 stays, and it is now a measurement rather than the first size that worked.

The answer to the review is that the expectation was right about occupancy and that occupancy was the wrong instrument: <!-- calkit value key=kernels.lw_optics.n_blocks_at_ceiling path=results/docs/group-block-size.json -->6<!-- /calkit value --> sizes reach the longwave's ceiling and <!-- calkit value key=n_equal_occupancy_blocks path=results/docs/group-block-size.json -->3<!-- /calkit value --> reach every kernel's, and of those the shipped one is the fastest in both bands by margins the sweep resolves.
What should change is the comment, which justifies the value from one band and from a property <!-- calkit value key=kernels.lw_optics.n_blocks_at_ceiling path=results/docs/group-block-size.json -->6<!-- /calkit value --> sizes share.
384 specifically is worse than it looks: it matches the longwave and drops the shortwave a band, which is the asymmetry a single shared constant hides and the reason the constant is worth keeping rather than tuning per band --- the two bands' ceiling sets intersect in <!-- calkit value key=n_equal_occupancy_blocks path=results/docs/group-block-size.json -->3<!-- /calkit value --> sizes and the fastest of them is the same for both.
