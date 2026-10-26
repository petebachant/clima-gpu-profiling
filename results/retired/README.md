# Retired results

Outputs of stages that no longer exist, preserved verbatim from the commit
before they were deleted. Nothing regenerates them, so they are historical
records rather than pipeline outputs: a figure here is checkable against the
commit named below, and cannot drift because nothing rewrites it.

They are kept because `docs/learnings.md` reasons from them. A conclusion whose
evidence survives only in git history has to be rediscovered by archaeology
every time it is questioned, which is how this project has twice re-litigated a
decision that was already made.

| file | produced by | retired in | reasoning that cites it |
|---|---|---|---|
| `sgs-quadrature-values.json` | the `values` stage of `docs/sgs-quadrature.md`, from `results/quadrature-order-error.toml`, `results/sgs-degeneracy.toml` and `results/kernel-shares.json` | `4934ef1` "Cut the pipeline back to what bears on AMIP SYPD" | `docs/learnings.md` §6a |

To regenerate `sgs-quadrature-values.json` rather than read it, check out
`meas/2026-09-09-quadrature-order-accuracy` (the pre-update comparison it reads)
or the parent of `4934ef1` (the post-update figures it reports), where the
stages and the document still exist.
