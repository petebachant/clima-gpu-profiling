# Is the SGS quadrature earning its cost?

A note for whoever owns the subgrid covariance closure. This is a measurement
report, not a proposal — the performance side can say what the quadrature costs
and where the PDF sits, but not whether the variances are right, and the answer
to that determines which remedy is correct.

## The short version

In the flagship AMIP configuration (`amip_progedmf_1m_land_he16`: prognostic
EDMF, 1-moment microphysics, `quadrature_order: 3`), the environment
microphysics tendency is evaluated at 3×3 Gauss–Hermite points over the joint
subgrid PDF of (T, q_tot). That quadrature exists to resolve the `max(0, ·)`
kink at saturation, where the integrand is non-smooth.

**Measured across all 1,548,288 grid cells on a settled state, the PDF sits a
median of 449 standard deviations away from that kink.** The worst 1% of cells
are still ~8σ clear of it.

**The quadrature costs 7.93% of simulated-years-per-day.**

So either the covariance closure is producing variances far smaller than
intended, or the quadrature is not needed at this resolution. Those have
opposite implications, which is why this is a question rather than a patch.

## What was measured

The quadrature reconstructs local condensate from the centred saturation excess

    S′ = (q_tot − q_sat(T, ρ)) − mu_S

and partitions `max(0, λ_lagrange + α·S′)` by liquid fraction. The kink is at
the zero of that argument. The quadrature therefore earns its cost only where
the PDF has meaningful weight on both sides of it.

The relevant comparison is the distance from saturation measured in units of the
spread of the saturation excess:

    mu_S  = q_tot − q_sat(T, ρ)                                    (already computed in the kernel)
    σ_S²  = σ_q² + (∂q_sat/∂T)²σ_T² − 2·corr·σ_q·σ_T·(∂q_sat/∂T)

with `σ_q = √q′q′`, `σ_T = √T′T′` taken from `p.precomputed`, `corr` from
`correlation_Tq(params)`, and `∂q_sat/∂T` from Clausius–Clapeyron. Grid-mean
density is used for `q_sat`; the environment density differs by the updraft area
fraction, far below the decade-scale separations at issue.

An earlier version of this measurement compared `σ_q` and `σ_T` against absolute
constants and is not reported here — "σ is small" is dimensionally meaningless.
A tiny σ still needs the quadrature if the mean sits on the kink; a large one
does not if the cell is far from it. Only the ratio to `σ_S` says anything.

## The distribution

| quantity | median | max |
|---|---|---|
| `σ_T` | 0.0112 K | 0.465 K |
| `σ_q` | 2.25e-7 | 1.07e-3 |
| `σ_S` | 3.04e-6 | 0.0336 |
| `\|mu_S\|` | 1.64e-3 | — |

`|mu_S| / σ_S` percentiles:

| p1 | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|
| 7.8 | 42 | 266 | **449** | 899 | 2761 |

| criterion | cells | 32-cell groups where *all* qualify |
|---|---|---|
| `σ_T = 0` **and** `σ_q = 0` exactly | 0.00% | 0.00% |
| `\|mu_S\| > 2·σ_S` | 99.76% | 94.30% |
| `\|mu_S\| > 3·σ_S` | 99.63% | 92.24% |
| `\|mu_S\| > 5·σ_S` | 99.36% | 89.12% |
| `\|mu_S\| > 10·σ_S` | 98.74% | 83.22% |

(The second column matters for the GPU: work is done in groups of 32 cells that
share a cost, so a criterion true at scattered points saves nothing. Here it
clusters vertically, so it would actually pay. That is a performance detail, not
a physics one.)

## What it costs

Priced by a configuration-only run — `quadrature_order: 1`, collapsing 3×3
points to 1 — against an otherwise identical model:

| configuration | SYPD |
|---|---|
| current | 0.28874 |
| quadrature collapsed to one point | **≈ +7.93%** |

That is an upper bound and was reverted immediately; it changes results and is
not proposed as a change. It exists so the question carries a number.

For scale, the entire performance effort on this benchmark has produced +6.22%
to date. The quadrature alone is larger than everything else combined.

## The two readings, and why they differ

**If the variances are too small** — the closure is not delivering the subgrid
variability it is meant to represent, the quadrature is integrating a
near-degenerate PDF, and it is doing nothing because it has been given nothing
to do. Then the fix is in `_compute_sgs_moments` / the covariance closure, the
quadrature stays, and the 7.93% is the price of a correctly functioning scheme.

**If the variances are right** — subgrid variability genuinely is this small at
h_elem 16, the PDF genuinely never straddles saturation, and a 3×3 rule is
resolving a feature that is not there. Then the quadrature order is the thing to
reconsider, and 7.93% is recoverable.

We cannot distinguish these from the profile. A useful discriminator would be
whether `σ_q` and `σ_T` here match what the closure is expected to produce for
this resolution and these conditions, and whether the near-zero medians are
concentrated in the dry upper atmosphere (plausible and harmless) or also
present in the boundary layer and cloudy regions (not harmless).

## Caveats worth stating plainly

- **Cell counts are dominated by cells where nothing happens.** The median cell
  is dry upper atmosphere. `σ_q` reaches 1.07e-3, comparable to `q_tot`, so the
  cells that matter physically are the minority where σ is large — exactly the
  ones the quadrature is for. A 98.74% point fraction is not "the quadrature is
  useless 98.74% of the time" in any physically weighted sense.
- **One state, one configuration.** Three steps into an AMIP run at h_elem 16,
  Float32. Not a survey across resolutions, seasons, or configurations.
- **Collapsing is an approximation, not an identity.** Even clear of the kink,
  the evaporation and sublimation rates still vary across the PDF; the error is
  O(σ²·f″) and should be small at these widths, but it is not zero and has not
  been measured against the 9-point answer.
- **An adaptive scheme would recover less than the full 7.93%** — roughly the
  83% warp fraction of it, before the cost of the branch itself.

## Reproducing

```sh
calkit run sgs-degeneracy      # writes results/sgs-degeneracy.toml
```

Source: `scripts/measure-sgs-degeneracy.jl`. The cost figure comes from setting
`quadrature_order: 1` in the benchmark config and re-running `amip-mod`.
Performance-side detail, including the warp-clustering analysis and why the
exact-degeneracy collapse is unavailable, is in `docs/learnings.md` §6a.
