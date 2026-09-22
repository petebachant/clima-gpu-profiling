# Decision record: the SGS quadrature order in the 1-moment microphysics

A note for whoever owns the subgrid covariance closure. It records what was
measured, what was decided, and what would change the decision. It is not a
proposal: the performance side can say what the quadrature costs and what
reducing it changes, but not whether the resulting tendencies are acceptable.

Every number here is injected from the results files by the block below, and
rewritten whenever the pipeline runs, so this document cannot drift from the
measurements. The pre-update figures are read out of the Git tag that recorded
them rather than typed.

```python calkit stage name=values environment=py outputs=[{path: results/sgs-quadrature-values.json, storage: git}] inputs=[results/quadrature-order-error.toml, results/sgs-degeneracy.toml, results/kernel-shares.json]
import json, subprocess, tomllib

BEFORE = "meas/2026-09-09-quadrature-order-accuracy"
now = tomllib.load(open("results/quadrature-order-error.toml", "rb"))
# The same measurement as it stood before the 2026-09-21 update, from the tag
# that recorded it, so the comparison is not typed in
before = tomllib.loads(
    subprocess.check_output(
        ["git", "show", f"{BEFORE}:results/quadrature-order-error.toml"]
    ).decode()
)
degeneracy = tomllib.load(open("results/sgs-degeneracy.toml", "rb"))
shares = json.load(open("results/kernel-shares.json"))["baseline"]

def tendencies(src, order):
    return {
        name: {
            "rms_pct": round(src[order][name]["rms_over_scale"] * 100, 2),
            "max_x_rms": round(src[order][name]["max_over_scale"], 1),
        }
        for name in ("dq_lcl_dt", "dq_icl_dt", "dq_sno_dt", "dq_rai_dt")
    }

out = {
    "cells": now["cells"],
    "before_ref": BEFORE,
    "now": {
        "changed_pct": round(now["order_2"]["changed_frac"] * 100, 1),
        "order_2": tendencies(now, "order_2"),
        "order_1": tendencies(now, "order_1"),
    },
    "before": {
        "changed_pct": round(before["order_2"]["changed_frac"] * 100, 2),
        "order_2": tendencies(before, "order_2"),
        "order_1": tendencies(before, "order_1"),
    },
    "degeneracy": {
        "mu_over_sigma_p50": round(degeneracy["mu_over_sigma_percentiles"]["p50"]),
        "mu_over_sigma_p1": round(degeneracy["mu_over_sigma_percentiles"]["p1"], 1),
        "clear_warp_pct": round(
            degeneracy["criteria"]["clear_of_kink_10sigma"]["warp_frac"] * 100, 2
        ),
    },
    "cost": {"hot_kernel_pct_of_gpu_time": shares["microphysics_hot_kernel_pct"]},
}
json.dump(out, open("results/sgs-quadrature-values.json", "w"), indent=2)
```

<!-- calkit values path=results/sgs-quadrature-values.json -->

## Decision

**Do not reduce the quadrature order on the current configuration.** An earlier
version of this document recommended 2x2 subject to sign-off. That
recommendation is withdrawn.

## Context

In the flagship AMIP configuration (`amip_progedmf_1m_land_he16`: prognostic
EDMF, 1-moment microphysics, `quadrature_order: 3`), the environment
microphysics tendency is evaluated at 3x3 Gauss-Hermite points over the joint
subgrid PDF of (T, q_tot). The kernel that does it is <!-- calkit value key=cost.hot_kernel_pct_of_gpu_time -->12.13<!-- /calkit value -->% of GPU kernel time, the largest single kernel in the run, so the rule's order is worth asking about.

## What reducing it costs, on the current configuration

Measured over all <!-- calkit value key=cells format="{:,}" -->1,548,288<!-- /calkit value --> cells on a settled state, with byte-identical inputs and each error normalized by that tendency's own RMS over the whole field.

| tendency | 2x2 RMS error | worst cell | 1-point RMS error |
|---|---|---|---|
| cloud liquid | <!-- calkit value key=now.order_2.dq_lcl_dt.rms_pct -->1.3<!-- /calkit value -->% | <!-- calkit value key=now.order_2.dq_lcl_dt.max_x_rms -->3.9<!-- /calkit value -->x | <!-- calkit value key=now.order_1.dq_lcl_dt.rms_pct -->2.75<!-- /calkit value -->% |
| cloud ice | <!-- calkit value key=now.order_2.dq_icl_dt.rms_pct -->3.62<!-- /calkit value -->% | <!-- calkit value key=now.order_2.dq_icl_dt.max_x_rms -->13.1<!-- /calkit value -->x | <!-- calkit value key=now.order_1.dq_icl_dt.rms_pct -->7.14<!-- /calkit value -->% |
| snow | <!-- calkit value key=now.order_2.dq_sno_dt.rms_pct -->0.48<!-- /calkit value -->% | <!-- calkit value key=now.order_2.dq_sno_dt.max_x_rms -->1.5<!-- /calkit value -->x | <!-- calkit value key=now.order_1.dq_sno_dt.rms_pct -->1.41<!-- /calkit value -->% |
| rain | <!-- calkit value key=now.order_2.dq_rai_dt.rms_pct -->0.38<!-- /calkit value -->% | <!-- calkit value key=now.order_2.dq_rai_dt.max_x_rms -->2.4<!-- /calkit value -->x | <!-- calkit value key=now.order_1.dq_rai_dt.rms_pct -->1.76<!-- /calkit value -->% |

Reducing to 2x2 changes <!-- calkit value key=now.changed_pct -->58.1<!-- /calkit value -->% of cells at all.

## Why this reverses an earlier recommendation

The same measurement at `meas/2026-09-09-quadrature-order-accuracy`, before
ClimaAtmos main re-enabled cloud ice formation and liquid freezing in the
1-moment scheme on 2026-09-21, read very differently.

| | before | now |
|---|---|---|
| cells changed | <!-- calkit value key=before.changed_pct -->1.12<!-- /calkit value -->% | <!-- calkit value key=now.changed_pct -->58.1<!-- /calkit value -->% |
| cloud ice RMS | <!-- calkit value key=before.order_2.dq_icl_dt.rms_pct -->1.2<!-- /calkit value -->% | <!-- calkit value key=now.order_2.dq_icl_dt.rms_pct -->3.62<!-- /calkit value -->% |
| cloud liquid RMS | <!-- calkit value key=before.order_2.dq_lcl_dt.rms_pct -->0.65<!-- /calkit value -->% | <!-- calkit value key=now.order_2.dq_lcl_dt.rms_pct -->1.3<!-- /calkit value -->% |
| worst ice cell | <!-- calkit value key=before.order_2.dq_icl_dt.max_x_rms -->1.7<!-- /calkit value -->x | <!-- calkit value key=now.order_2.dq_icl_dt.max_x_rms -->13.1<!-- /calkit value -->x |

Those processes are nonlinear in temperature away from saturation, and the quadrature integrates over temperature as well as over total water, so it is now resolving structure the earlier configuration did not have.

## The degeneracy argument, and why it was not sufficient

The case for reducing the order rested on where the subgrid PDF sits relative to the saturation kink that the quadrature exists to resolve. That measurement still holds: the PDF sits a median of <!-- calkit value key=degeneracy.mu_over_sigma_p50 -->448<!-- /calkit value --> standard deviations from the kink, the worst percentile is <!-- calkit value key=degeneracy.mu_over_sigma_p1 -->7.4<!-- /calkit value -->σ clear of it, and <!-- calkit value key=degeneracy.clear_warp_pct -->81.9<!-- /calkit value -->% of 32-cell warps are entirely clear.

It was the inference that was wrong. Distance from the **saturation** kink bounds the non-smoothness of `max(0, ·)` and nothing else. The re-enabled processes add their own nonlinearities elsewhere in the integration domain, and the order-reduction measurement above sees them.

**Carry this forward: distance from one kink is necessary but not sufficient evidence that a quadrature is degenerate.** The test that settles it is reducing the rule and diffing the tendencies, which is cheap and is now a pipeline stage.

## What would change the decision

1. **A cost measurement on the current configuration.** The throughput figures this decision would trade against (+6.14% SYPD on upstream code at `exp/2026-09-08-quadrature-order-2-isolated`, +8.05% with this project's numerics work at `exp/2026-09-09-order2-full-stack`) predate the update, so the trade cannot be struck yet.
2. **A self-consistent run.** `lambda_lagrange` is held at its 3x3 fit here, so these are the direct effects of changing the rule. A real order-2 run refits it and would compensate in part; settling that needs two full runs diffed field by field.
3. **A judgement about the worst cells rather than the field.** The error is concentrated where the PDF reaches a kink, which is exactly where the quadrature is doing its job.

## Reproducing this

```sh
calkit run quadrature-order-error   # the accuracy table
calkit run sgs-degeneracy           # the distance-from-kink distribution
```
