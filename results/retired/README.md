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

The parameter-fold and `Val` direction was retired on 2026-11-12, when the arms
stopped carrying it: CliMA/ClimaAtmos.jl#4947 merged the sedimentation fold
upstream, so both arms run it, and `ClimaAtmos.jl-mod` was returned to upstream
`main` in the same pass. Fifteen stages measured a treatment that no longer
exists, and three questions templated their figures from the working tree, which
would have rendered a null every run rather than the result they describe.

| file | produced by | reasoning that cites it |
|---|---|---|
| `param-fold.json`, `param-fold-prior.json` | `export-param-fold` | `docs/learnings.md` §8g; the live conversion now reads `results/fold-record/param-fold.json` |
| `sedimentation-fold.json` | `export-sedimentation-fold` | `docs/experiments/sedimentation-velocity-fold.md` |
| `fold-decision.json` | `fold-decision-doc` | `docs/fold-decision-rule.md` |
| `val-mileage.json`, `val-params-registers.toml` | `val-mileage`, `val-params-registers` | `docs/val-fold-scope.md` |
| `pointwise-registers.json`, `pointwise-registers.csv` | `pointwise-registers`, `export-pointwise-registers` | `docs/val-fold-scope.md` |
| `cm-registers.toml` | `cm-registers` | `docs/val-fold-scope.md` |
| `fold-compile-cost.json`, `fold-compile-cost/` | `fold-compile-cost`, `export-fold-compile-cost` | `docs/fold-compile-cost.md` |
| `fold-arch-portability.json` | `export-fold-arch-portability` | `docs/fold-compile-cost.md` |
| `fold-on-current-stack.json` | `export-fold-on-current-stack` | nothing; it was a null by construction once the fold went upstream |
| `val-fold-scope.json`, `sedimentation-velocity-fold.json`, `docs-fold-compile-cost.json` | the `values` stages of the three documents above | the documents' own value markers |

The four documents stay where they are, because `docs/learnings.md` links to
them. Nothing rewrites them now, so their figures are frozen rather than live --
the same guarantee the files in this directory carry. To rebuild any of it, check
out `67f8a3d`, where every stage above still exists and the fold was still the
mod arm's treatment.

To regenerate `sgs-quadrature-values.json` rather than read it, check out
`meas/2026-09-09-quadrature-order-accuracy` (the pre-update comparison it reads)
or the parent of `4934ef1` (the post-update figures it reports), where the
stages and the document still exist.
