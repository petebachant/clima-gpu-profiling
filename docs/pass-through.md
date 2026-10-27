# How much of a kernel-time saving reaches SYPD

Every experiment in this project that reduced GPU kernel time also measured SYPD end to end, so the ratio between the two is measured rather than assumed, over the <!-- calkit value key=n_runs path=results/docs/pass-through.json -->9<!-- /calkit value --> on the record.
The ratios do not cluster by subsystem, which is how this project first grouped them, and the parameter fold is what showed it.

Numbers here are injected from `results/docs/pass-through.json` by the block below, which reads them out of `results/pass-through.json` and `results/fold-record/param-fold.json` and rounds them to display precision.
They are not typed, and they are rewritten on every run.

The block is also what makes this document a node in the pipeline: its `inputs` are the results files the prose quotes, so the document goes stale when they do and `calkit run` restates it.

```python calkit stage name=values environment=py outputs=[{path: results/docs/pass-through.json, storage: git}] inputs=[results/pass-through.json, results/fold-record/param-fold.json]
import json

pt = json.load(open("results/pass-through.json"))
fold = json.load(open("results/fold-record/param-fold.json"))
quad = fold["kernels"]["microphysics_quadrature"]["baseline"]


def band(name):
    s = pt["by_subsystem"][name]
    return {
        "n": s["n"],
        "min_pct": round(s["min_pct"]),
        "max_pct": round(s["max_pct"]),
        "mean_pct": round(s["mean_pct"]),
    }


out = {
    "radiation": band("radiation"),
    "microphysics": band("microphysics"),
    "n_runs": len(pt["runs"]),
    "fold_quadrature": {
        "launches": quad["launches"],
        "mean_ms": round(quad["mean_ms"]),
    },
}
json.dump(out, open("results/docs/pass-through.json", "w"), indent=2)
```

<!-- calkit values path=results/docs/pass-through.json -->

## The bands

Radiation work converts at <!-- calkit value key=radiation.mean_pct -->69<!-- /calkit value -->% of its kernel-time saving, over a range of <!-- calkit value key=radiation.min_pct -->62<!-- /calkit value -->-<!-- calkit value key=radiation.max_pct -->74<!-- /calkit value -->% across <!-- calkit value key=radiation.n -->5<!-- /calkit value --> experiments.
Microphysics work converts at <!-- calkit value key=microphysics.mean_pct -->50<!-- /calkit value -->%, over <!-- calkit value key=microphysics.min_pct -->31<!-- /calkit value -->-<!-- calkit value key=microphysics.max_pct -->73<!-- /calkit value -->% across <!-- calkit value key=microphysics.n -->4<!-- /calkit value -->.

Both halves of every ratio are read out of the tag that recorded them: kernel time from the profiled nsys stages, SYPD from the unprofiled AMIP ones, which is the only number that answers "is the model faster".
The difference between the bands is large enough to change what is worth doing, which is why it mattered that the label on them was wrong.

## Subsystem is the wrong label; kernel shape is the right one

The parameter fold is microphysics work that converted at <!-- calkit value key=microphysics.max_pct -->73<!-- /calkit value -->%, inside the radiation range and nowhere near the <!-- calkit value key=microphysics.min_pct -->31<!-- /calkit value -->% the two fusion experiments gave.
Its two measurements are the third and fourth microphysics points, and between them they widened that band from a cluster to a spread, which is the signal that the band was never measuring a subsystem.

What it was measuring is kernel shape.
The rule of thumb this project carried for months --- about a third of a kernel-time saving shows up end to end --- came from two experiments whose savings were spread over the hundreds of launches per step that the linearization and microphysics-cache paths each make, where whatever else is in flight can absorb them.
The fold's saving is not: it is concentrated in one kernel that runs <!-- calkit value key=fold_quadrature.launches -->120<!-- /calkit value --> times averaging <!-- calkit value key=fold_quadrature.mean_ms -->39<!-- /calkit value --> ms, a few long dense launches with nothing to overlap against, which is exactly the shape every radiation treatment had.

So the earlier hypothesis here --- that radiation converts better because it is a few long kernels rather than one of thousands --- is no longer just plausible.
This is a test of it that did not have to come out this way, on a treatment in the subsystem that was supposed to convert badly.

## Cautions

The record holds <!-- calkit value key=n_runs -->9<!-- /calkit value --> experiments, <!-- calkit value key=microphysics.n -->4<!-- /calkit value --> microphysics and <!-- calkit value key=radiation.n -->5<!-- /calkit value --> radiation, which is not many; and the ratios are quotients of two measured quantities, so the SYPD noise floor propagates into them --- read them as bands rather than to the point.
The microphysics mean in particular is now an average over two kernel shapes rather than a description of one, which is exactly the reading this document argues against; the <!-- calkit value key=microphysics.min_pct -->31<!-- /calkit value -->% and <!-- calkit value key=microphysics.max_pct -->73<!-- /calkit value -->% ends of it are the figures that mean something.

`pass_through_pct` divides the per-step walltime fraction by the kernel-time change, not the SYPD gain by it.
The two differ by about three points at these magnitudes, so a figure computed the other way does not belong beside these.

What the wall-clock side of the conversion looks like step by step, and how wide the floor under these ratios is, is in [the wall-clock conversion document](wall-clock-conversion.md).
