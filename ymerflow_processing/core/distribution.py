"""Distribution summaries for step reports.

R3 requires that any step with a threshold report *the distribution of the
quantity it thresholded on*, before and after. That is what turns "disabled
4,182 gates" into "here is the spread of the values, here is where you cut,
here is what that removed."

One implementation, used by every step, so that culling and QC describe a
distribution the same way and a UI has one shape to render.

Everything returned here is plain Python — floats, ints, lists — because a
report is stored in the process record and rendered by a web client, so it has
to survive being serialized. numpy scalars do not.
"""

import numpy as np

#: Percentiles reported for every distribution. The extremes matter as much as
#: the middle: a threshold usually sits out in a tail, and the median tells you
#: nothing about how much data is near it.
PERCENTILES = (5, 25, 50, 75, 95)


def summarize(values, bins: int = 20, percentiles=PERCENTILES) -> dict:
    """Summarize ``values`` as counts, moments, percentiles and a histogram.

    Missing values are excluded from every statistic and reported separately as
    ``n_missing`` — a distribution that silently dropped a third of its input
    tells the reader the wrong story about how much data there was.

    An empty input (or one that is entirely missing) returns the same keys with
    ``None`` statistics rather than raising. A step that classified nothing
    still has to be able to report, and a report that cannot be produced is
    worse than one that says "nothing here".
    """
    values = np.asarray(values, dtype=float).ravel()
    finite = values[np.isfinite(values)]

    summary: dict = {
        "n": int(finite.size),
        "n_missing": int(values.size - finite.size),
        "min": None, "max": None, "mean": None, "std": None,
        "percentiles": {},
        "histogram": None,
    }
    if finite.size == 0:
        return summary

    summary["min"] = float(finite.min())
    summary["max"] = float(finite.max())
    summary["mean"] = float(finite.mean())
    # Population standard deviation: this describes the data in hand, it does
    # not estimate a parameter of a wider population.
    summary["std"] = float(finite.std())
    summary["percentiles"] = {
        str(p): float(v) for p, v in zip(percentiles, np.percentile(finite, percentiles))
    }

    # A degenerate range would make numpy pick its own bounds and produce edges
    # that do not contain the value, which reads as a bug in the UI. Widen by a
    # hair instead so a constant series renders as one full bin.
    #
    # Relative to the magnitude, not absolute. Transient amplitudes are of order
    # 1e-9, so a series that is constant to within floating-point noise has a
    # range of ~1e-25 while its values are 1e-9 — numpy then cannot place twenty
    # distinguishable edges and raises "Too many bins for data range". Equality
    # alone does not catch that, because the values are not equal.
    lo, hi = summary["min"], summary["max"]
    scale = max(abs(lo), abs(hi))
    if (hi - lo) <= scale * 1e-9:
        pad = scale * 1e-6 if scale else 0.5
        lo, hi = lo - pad, hi + pad
    counts, edges = np.histogram(finite, bins=bins, range=(lo, hi))
    summary["histogram"] = {
        "edges": [float(e) for e in edges],
        "counts": [int(c) for c in counts],
    }
    return summary
