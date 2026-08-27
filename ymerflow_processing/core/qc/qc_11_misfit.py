"""qc_11 — how well the model explained each sounding.

**What it does to the dataset:** adds a verdict column only if thresholds were
given. It changes no measurement, no uncertainty and no mask — the misfit it
grades was written by ``compute_misfit`` (R6).

``03``: *"Included when an inversion exists. Absent otherwise, with no error."*
So a dataset with no misfit column is skipped with a note rather than failing.
Every other check reads something the instrument recorded; this reads something
a *model* produced, and a delivery that has not been inverted has simply not got
to that point yet.

It grades what is already there
-------------------------------
``01``'s resolution of filter_31 made misfit a column precisely so it would be
*"plottable, exportable in the vector output, usable by QC"*. This is the QC
half. It computes nothing: ``compute_misfit`` writes ``resdata`` — the
root-mean-square gatewise misfit over gates in use — and this classifies it.

That keeps one definition of misfit on the dataset. A check that recomputed it
could disagree with the column beside it, which is the defect R9 exists to
prevent, one group over.

A poor fit is not necessarily bad data
---------------------------------------
This is the reading that matters, and it is why the finding is neither ``flight``
nor ``instrument``. ``01`` says it plainly about culling on misfit: *"a poor fit
may indicate genuine geology the model cannot represent."*

A sounding over a steeply dipping contact, a conductive body the 1D assumption
cannot describe, or genuine three-dimensionality will misfit — and the data may
be perfectly good. What a high misfit says is that **the model did not explain
this sounding**, which is a statement about the pair, not about the measurement.

So it reports as a ``model`` finding: it appears among the reasons a sounding is
flagged and feeds neither the flight-quality verdict nor the instrument-health
one. Nobody should reflight a line because a 1D inversion could not fit a fault.

**There are no default thresholds.** What misfit is acceptable depends on the
inversion, the earth, and how the uncertainties were assigned — and since misfit
is measured in standard deviations, a threshold here inherits whatever
``assign_uncertainty`` decided. A house number would grade the uncertainty model
as much as the fit.
"""

from __future__ import annotations

import logging

import numpy as np

from ..distribution import summarize
from ..misfit import RMS_COLUMN
from ..pipeline import OPTIONAL, UNJUDGED, StepReport
from . import aggregate
from .thresholds import (
    ASCENDING, GOOD, MARGINAL, OUT_OF_SPEC, classify, describe, is_tier,
    mark_thresholds, mark_verdicts, tier_counts,
)

log = logging.getLogger(__name__)

STEP = "data_misfit"


def data_misfit(
    dataset,
    good_max: float | None = None,
    marginal_max: float | None = None,
    column: str = RMS_COLUMN,
    output_column: str = "qs_misfit",
    line_column: str | None = None,
    gap_tolerance: float = 0.0,
):
    """Classify how well a previous inversion explained each sounding.

    :param good_max: Root-mean-square misfit, **in standard deviations**, at or
        below which a sounding is unremarkable. **Left unset, no verdict is
        produced** and the distribution is reported alone.
    :param marginal_max: Misfit above which the sounding is Out of Spec.
        Required if ``good_max`` is given.
    :param column: Which column holds the per-sounding misfit. Written by
        ``compute_misfit``, or by an inversion under the same name.
    :param output_column: Where the per-sounding verdict is written.
    :param line_column: Which column identifies the line. Discovered if unset.
    :param gap_tolerance: In-spec stretches shorter than this, in CRS units, do
        not break an out-of-spec run when contiguous runs are measured.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if frame is None or len(frame) == 0:
        report.notes.append("No soundings; nothing to classify.")
        return dataset, report

    if column not in frame.columns:
        report.notes.append(
            f"No misfit column {column!r}, so this dataset has not been "
            f"inverted and there is nothing to grade. Run compute_misfit "
            f"against an inversion result first. This check is optional by "
            f"design and its absence is not an error."
        )
        report.ungraded = OPTIONAL
        return dataset, report

    misfit = frame[column].to_numpy(dtype=float)
    unmeasured = int((~np.isfinite(misfit)).sum())
    if unmeasured:
        report.notes.append(
            f"{unmeasured} sounding(s) carry no misfit and so no verdict. Why "
            f"depends on who wrote the column: computed here it means nothing "
            f"of theirs was in use when the fit was measured; written by an "
            f"inversion it more likely means the sounding was never inverted. "
            f"This cannot tell which."
        )

    verdicts = None
    if good_max is not None:
        if marginal_max is None:
            raise ValueError(
                "good_max was given without marginal_max; a threshold needs "
                "both bounds."
            )
        spec = {"kind": ASCENDING, "good_max": good_max,
                "marginal_max": marginal_max}
        verdicts = classify(misfit, spec)
        if output_column in frame.columns:
            report.notes.append(f"Overwrote existing column {output_column!r}.")
        frame[output_column] = verdicts
        mark_verdicts(dataset, output_column)
        mark_thresholds(dataset, STEP, spec)

        report.verdict_column = output_column
        # Neither the contractor's doing nor the instrument's. A poor fit may be
        # genuine geology a 1D model cannot represent, so it names itself among
        # the reasons and feeds neither summary. Nobody should reflight a line
        # because an inversion could not fit a fault.
        report.finding = "model"
        counts = tier_counts(verdicts)
        report.soundings_affected = counts[OUT_OF_SPEC]
    else:
        report.notes.append(
            "No thresholds given, so the misfit is reported and left unjudged. "
            "Supply good_max and marginal_max to classify it."
        )
        report.ungraded = report.ungraded or UNJUDGED

    kept = (~is_tier(verdicts, OUT_OF_SPEC) if verdicts is not None
            else np.ones(len(frame), dtype=bool))
    report.distributions = {
        "column": column,
        "unit": "standard deviations",
        "misfit": {"before": summarize(misfit), "after": summarize(misfit[kept])},
    }
    if verdicts is not None:
        report.distributions["verdicts"] = counts
        report.distributions["thresholds"] = describe(spec, "sigma")

    xyz = dataset.survey.xyz
    line_column = line_column or xyz.line_id_column
    if verdicts is not None and line_column and line_column in frame.columns:
        easting = (frame[xyz.x_column].to_numpy()
                   if xyz.x_column in frame.columns else None)
        northing = (frame[xyz.y_column].to_numpy()
                    if xyz.y_column in frame.columns else None)
        report.distributions["per_line"] = aggregate.per_line(
            verdicts, frame[line_column].to_numpy(), easting, northing,
            gap_tolerance=gap_tolerance)

    log.info("%s: median misfit %.3g sigma over %d sounding(s)",
             STEP, float(np.nanmedian(misfit)), len(frame))
    return dataset, report
