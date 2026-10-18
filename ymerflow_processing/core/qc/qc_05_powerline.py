"""qc_05 — powerline and cultural noise.

Finds **local peaks** in a powerline monitor — where it rises above what it was
doing either side — and writes that excess as a column on ``flightlines``.

**What it does to the dataset:** adds one numeric column holding how far above
its local background each sounding sat, and a verdict column only if thresholds
were given. It changes no measurement, no uncertainty and no mask (R6).

Reference first, verdict only if asked
--------------------------------------
This is primarily something to **look at**. ``03`` gives every other check a
threshold kind and gives this one none, which is the specification being right
rather than incomplete, and a strip plot of the monitor against its own
background is what a geophysicist actually reads.

So the excess is always computed and written; a verdict appears only when
thresholds are supplied. Measure, and let whoever holds the contract decide
whether a number matters — the same stance the line-level rules take.

The baseline is local, not per line
-----------------------------------
**The ambient level moves within a line**, not merely between surveys. A line
flown out of farmland and into a town rises and stays risen, and a baseline
taken over the whole line would call that entire second half a spike while
diluting a real crossing in the first.

So each sounding is compared to its **neighbours** — a rolling window along the
line — rather than to the line as a whole. What that measures is a departure
from the local background, which is what a crossing looks like and what a
change of region does not.

This makes it a **local peak detector**, and that is worth saying plainly
because it settles what the check can and cannot see. A crossing narrower than
the window stands out. A broad elevated region wider than the window *becomes*
the background and vanishes. The window is therefore the definition of "local",
not a tuning knob: it has to be comfortably wider than the feature being looked
for. The default spans about a kilometre at typical sounding spacing.

Median and MAD, not mean and sigma
----------------------------------
The obvious baseline is the mean and the obvious spread is the standard
deviation, and both are wrong for this: **a spike drags the mean up toward
itself and inflates sigma**, so a strong powerline crossing partly hides in the
statistics meant to reveal it. The same trap qc_04 avoids for the same reason.

The median and the median absolute deviation do not move for a handful of
extreme samples — up to half the line would have to be affected before the
median shifted — so a crossing stands out against them. MAD is scaled by 1.4826
so that a threshold reads in units comparable to standard deviations for
normally distributed data, which is what makes "five" a number anyone can
reason about.

What this cannot see, and what does see it
------------------------------------------
**A line flown along a powerline rather than across one.** Affect enough of the
window and the elevated level is the background, so nothing peaks. That is
inherent to detecting a peak at all, and it is why this complements culling by
proximity to known infrastructure (filter_30) rather than replacing it: one
needs no map and finds crossings, the other needs a map and finds everything
near a feature, including the stretch flown parallel to it.

Not a flight-quality finding
----------------------------
A powerline is a fact about the ground. The contractor did not cause it, flying
the line again will not avoid it, and the instrument is working correctly — it
is reporting exactly what is there. So this reports as an ``environment``
finding: it appears in the reasons a sounding was flagged, and feeds neither the
flight-quality verdict nor the instrument-health one.

Sources
-------
The median absolute deviation as a robust scale estimate, and the 1.4826
consistency constant that makes it comparable to the standard deviation for
normal data: Rousseeuw, P.J. and Croux, C., "Alternatives to the Median Absolute
Deviation", *Journal of the American Statistical Association*, 1993.

On cultural conductors coupling into an airborne receiver and producing
anomalies that are real measurements of the wrong thing, see filter_30 in
``building_plans/01``.

**There are no default thresholds.** Unset, this measures and says nothing —
which is how it is usually read.
"""

import logging

import numpy as np
import pandas as pd

from ..distribution import summarize
from ..pipeline import OPTIONAL, UNJUDGED, StepReport
from . import aggregate
from .thresholds import (
    ASCENDING, GOOD, MARGINAL, OUT_OF_SPEC, classify, describe, is_tier,
    mark_verdicts, tier_counts,
)

log = logging.getLogger(__name__)

STEP = "powerline_noise"

#: Spellings a powerline monitor turns up under. PLNI is what SkyTEM exports
#: call it; the general block of a system description also names the gate and
#: frequency it is derived from.
POWERLINE_COLUMNS = ("plni", "plm", "powerline", "powerline_monitor", "pline",
                     "pl_monitor", "powerlinemonitor")

#: Scales the median absolute deviation so a threshold reads in units
#: comparable to standard deviations for normally distributed data.
MAD_TO_SIGMA = 1.4826


def find_powerline_column(frame) -> str | None:
    """The powerline monitor column, or ``None`` if the survey records none.

    Parameters
    ----------
    frame : pandas.DataFrame
        The ``flightlines`` to search.

    Returns
    -------
    str | None
        The column as spelled in the frame.
    """
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    for spelling in POWERLINE_COLUMNS:
        if spelling in lookup:
            return lookup[spelling]
    return None


def local_excess(values, groups, window: int = 101):
    """How far each sounding sits above its *local* baseline, in robust sigmas.

    ``(value - rolling median) / (1.4826 * rolling MAD)``, computed inside each
    group so a window never spans the turn between two lines.

    Rolling rather than per group because the ambient level drifts along a line;
    see the module docstring. The window is centered, so a sounding is judged
    against what came before *and* after it, which matters at the leading edge
    of a crossing.

    Parameters
    ----------
    values : array_like
        The monitor reading, one per sounding.
    groups : array_like
        Line identity per sounding.
    window : int
        The rolling window, in soundings.

    Returns
    -------
    numpy.ndarray
        One excess per sounding; ``NaN`` where there is no local spread to
        measure against — a perfectly flat stretch gives a MAD of zero, and
        dividing by it would call every departure infinite rather than
        admitting the scale is unknown.
    """
    values = pd.Series(np.asarray(values, dtype=float))
    groups = pd.Series(np.asarray(groups)).astype(str)
    excess = np.full(len(values), np.nan)

    for key in groups.unique():
        rows = (groups == key).to_numpy()
        block = values[rows].reset_index(drop=True)
        span = int(min(window, len(block)))
        if span < 3:
            continue
        # Enough of the window to be a baseline, but not so much that the ends
        # of a line go unjudged.
        least = max(3, span // 4)

        baseline = block.rolling(span, center=True, min_periods=least).median()
        deviation = (block - baseline).abs()
        spread = deviation.rolling(span, center=True,
                                   min_periods=least).median() * MAD_TO_SIGMA

        with np.errstate(divide="ignore", invalid="ignore"):
            local = np.where(spread > 0, (block - baseline) / spread, np.nan)
        excess[rows] = local
    return excess


def powerline_noise(
    dataset,
    good_max: float | None = None,
    marginal_max: float | None = None,
    window: int = 101,
    column: str | None = None,
    output_column: str = "qs_pline",
    excess_column: str = "pl_excess",
    line_column: str | None = None,
    gap_tolerance: float = 0.0,
):
    """Measure how far the powerline monitor rose above its local background.

    Parameters
    ----------
    dataset : Dataset
        The dataset to grade; the excess and a verdict column are written, no
        measurement is changed.
    good_max : float | None
        Excess above the local baseline, in robust sigmas, at or below which a
        sounding is unremarkable. **Left unset, no verdict is produced at all**
        and only the measurement is written — which is the usual way to read
        this check.
    marginal_max : float | None
        Excess above which the sounding is Out of Spec. Required if
        ``good_max`` is given.
    window : int
        How many soundings the rolling baseline spans. Must be comfortably
        wider than the feature being detected, or a broad anomaly becomes its
        own baseline; the default is about a kilometre at typical spacing.
    column : str | None
        Which ``flightlines`` column holds the monitor. Discovered if unset.
    output_column : str
        Where the per-sounding verdict is written.
    excess_column : str
        Where the numeric excess is written. Kept because the verdict says a
        sounding was noisy without saying how noisy, and because a later
        culling step may want to threshold it differently.
    line_column : str | None
        Which column identifies the line — the grain the baseline is derived
        over. Discovered if unset.
    gap_tolerance : float
        In-spec stretches shorter than this, in CRS units, do not break an
        out-of-spec run when the contiguous-run length is measured.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if frame is None or len(frame) == 0:
        report.notes.append("No soundings; nothing to classify.")
        return dataset, report

    if column is not None and column not in frame.columns:
        raise KeyError(
            f"No column {column!r} in flightlines. "
            f"Present: {', '.join(map(str, frame.columns))}"
        )
    monitor = column or find_powerline_column(frame)
    if monitor is None:
        report.notes.append(
            "No powerline monitor column found. Not every system carries one, "
            "so this is an absent instrument rather than a missing "
            "column: the check is skipped and the run is not marked "
            "ungraded. Name one with "
            "'column:' if it is present under an unusual name."
        )
        report.ungraded = OPTIONAL
        return dataset, report

    xyz = dataset.survey.xyz
    line_column = line_column or xyz.line_id_column
    has_lines = line_column is not None and line_column in frame.columns
    lines = frame[line_column].to_numpy() if has_lines else np.zeros(len(frame))
    if not has_lines:
        report.notes.append(
            "No line identifier; the baseline is derived over the whole "
            "delivery rather than per line, which is coarser than intended."
        )

    values = frame[monitor].to_numpy(dtype=float)
    excess = local_excess(values, lines, window)

    flat = int(np.sum(np.isnan(excess) & np.isfinite(values)))
    if flat:
        report.notes.append(
            f"{flat} sounding(s) sit on a line with no variation in the monitor, "
            "so there is no scale to measure a spike against and they carry no "
            "verdict."
        )

    if excess_column in frame.columns:
        report.notes.append(f"Overwrote existing column {excess_column!r}.")
    frame[excess_column] = excess

    verdicts = None
    if good_max is not None:
        if marginal_max is None:
            raise ValueError(
                "good_max was given without marginal_max; a threshold needs "
                "both bounds."
            )
        spec = {"kind": ASCENDING, "good_max": good_max,
                "marginal_max": marginal_max}
        verdicts = classify(excess, spec)
        if output_column in frame.columns:
            report.notes.append(f"Overwrote existing column {output_column!r}.")
        frame[output_column] = verdicts
        mark_verdicts(dataset, output_column)

        report.verdict_column = output_column
        # A powerline is a fact about the ground: not the contractor's doing,
        # not avoidable by reflying, and not the instrument misbehaving. It
        # belongs to neither the flight-quality nor the instrument-health
        # summary, so it reports as a third kind.
        report.finding = "environment"
        counts = tier_counts(verdicts)
        report.soundings_affected = counts[OUT_OF_SPEC]
    else:
        report.notes.append(
            "No thresholds given, so the excess is measured and left unjudged. "
            "Supply good_max and marginal_max to classify it."
        )
        report.ungraded = report.ungraded or UNJUDGED

    kept = (~is_tier(verdicts, OUT_OF_SPEC) if verdicts is not None
            else np.ones(len(frame), dtype=bool))
    report.distributions = {
        "excess": {
            "column": monitor,
            "unit": "robust sigmas above the line baseline",
            "before": summarize(excess),
            "after": summarize(excess[kept]),
        },
        # The monitor's own levels, which the excess deliberately discards. Two
        # surveys with identical excesses may sit decades apart in absolute
        # terms, and that is worth seeing.
        "monitor": summarize(values),
        "window": window,
    }
    if verdicts is not None:
        report.distributions["verdicts"] = counts
        report.distributions["thresholds"] = describe(spec, "robust sigmas")

    if not has_lines or verdicts is None:
        return dataset, report

    easting = frame[xyz.x_column].to_numpy() if xyz.x_column in frame.columns else None
    northing = frame[xyz.y_column].to_numpy() if xyz.y_column in frame.columns else None

    flight_column = aggregate.find_flight_column(frame)
    if flight_column is not None:
        report.distributions["per_flight"] = aggregate.per_flight(
            verdicts, frame[flight_column].to_numpy(), lines, easting, northing,
            gap_tolerance=gap_tolerance)
    report.distributions["per_line"] = aggregate.per_line(
        verdicts, lines, easting, northing, gap_tolerance=gap_tolerance)

    log.info("%s: %d soundings, %d Good / %d Marginal / %d Out of Spec",
             STEP, len(frame), counts[GOOD], counts[MARGINAL], counts[OUT_OF_SPEC])
    return dataset, report
