"""qc_08 — altimeter agreement.

Compares two or more independent altimeters against each other and writes both
their disagreement and a verdict on it as columns on ``flightlines``.

**What it does to the dataset:** adds one numeric column holding the spread
between the altimeters, one verdict column, and optionally the worst verdict on
each line. It changes no measurement, no uncertainty and no mask (R6).

Not the same question as qc_01
------------------------------
qc_01 asks whether the height was *right* — was the aircraft flown where the
contract said. This asks whether the height is *known*. A laser and a radar
altimeter reading 40 m and 55 m over the same ground are not describing a flying
problem; at least one of them is wrong, and nothing downstream can tell which.

That distinction matters because the two failures need opposite responses. Flown
too high is a conversation with the contractor about the flight. Altimeters
disagreeing is a conversation about the *instrument*, and it invalidates qc_01's
own input — a height verdict computed from a failing sensor is a confident
statement about a number nobody should trust.

Why the spread, and why ascending
---------------------------------
Disagreement is the full spread — the largest reading minus the smallest — which
generalizes to any number of altimeters and needs no view about which is
authoritative. That is the point: if we knew which to trust there would be no
check to run.

Higher is unambiguously worse and zero is the ideal, so the threshold is
``ascending``. Some disagreement is normal and expected: a laser returns from the
canopy while a radar returns from nearer the ground, so vegetation alone
separates them. The contract sets how much is tolerable.

Sources
-------
The comparison is arithmetic; nothing here is model-dependent. On why laser and
radar returns legitimately differ over vegetation, and the consequence for
terrain clearance in airborne survey, see the discussion of altimetry in
Christiansen, A.V., Auken, E. and Sørensen, K., "The transient electromagnetic
method", in *Groundwater Geophysics: A Tool for Hydrogeology*, Springer, 2009.

**Threshold values come from the survey contract**, so they are required
parameters with no defaults.
"""

import logging

import numpy as np

from ..distribution import summarize
from ..pipeline import StepReport
from . import aggregate
from .thresholds import (
    ASCENDING, GOOD, MARGINAL, OUT_OF_SPEC, classify, describe, is_tier,
    mark_thresholds, mark_verdicts, tier_counts,
)

log = logging.getLogger(__name__)

STEP = "altimeter_agreement"


def altimeter_agreement(
    dataset,
    columns: list[str],
    good_max: float,
    marginal_max: float,
    output_column: str = "qs_altim",
    spread_column: str = "alt_spread",
    line_column: str | None = None,
    unit: str = "m",
    gap_tolerance: float = 0.0,
):
    """Classify the disagreement between independent altimeters.

    Parameters
    ----------
    columns
        The altimeter columns to compare — at least two. Required and never
        guessed: which sensors a survey carries, and what they are called, is a
        property of the aircraft rather than something to infer.
    good_max : float
        Spread at or below which the altimeters agree acceptably, in ``unit``.
        A contract term, so no default.
    marginal_max : float
        Spread above which the disagreement is Out of Spec.
    output_column : str
        Where the per-sounding verdict is written.
    spread_column : str
        Where the numeric spread is written. Kept because the verdict alone
        does not say *how far* apart they were, and because a later step may
        want to threshold it differently.
    line_column : str | None
        Which column identifies the line. Discovered if unset.
    unit : str
        Unit of the altimeter columns, for report text.
    gap_tolerance : float
        In-spec stretches shorter than this, in CRS units, do not break an
        out-of-spec run when the contiguous-run length is measured. A contract
        saying "out of spec for 1 km or more" describes a stretch of flying
        that one recovered sounding does not end. ``0`` measures runs raw; the
        raw figure is reported either way.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if frame is None or len(frame) == 0:
        report.notes.append("No soundings; nothing to compare.")
        return dataset, report

    if isinstance(columns, str):
        columns = [columns]
    columns = list(columns)
    if len(columns) < 2:
        raise ValueError(
            f"Altimeter agreement needs at least two columns to compare; got "
            f"{columns}. With one altimeter there is nothing to disagree with."
        )

    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(
            f"No column(s) {', '.join(map(repr, missing))} in flightlines. "
            f"Present: {', '.join(map(str, frame.columns))}"
        )

    readings = np.column_stack([frame[c].to_numpy(dtype=float) for c in columns])

    # A sounding where only one altimeter reported has nothing to compare, so it
    # gets no spread rather than a spread of zero — which would read as perfect
    # agreement and is the opposite of what is known.
    measured = np.sum(np.isfinite(readings), axis=1)
    spread = np.where(
        measured >= 2,
        np.nanmax(readings, axis=1) - np.nanmin(readings, axis=1),
        np.nan,
    )

    partial = int(np.sum((measured == 1)))
    if partial:
        report.notes.append(
            f"{partial} sounding(s) had only one altimeter reporting; they have "
            "no spread and no verdict, since a lone reading cannot disagree."
        )

    spec = {"kind": ASCENDING, "good_max": good_max, "marginal_max": marginal_max}
    verdicts = classify(spread, spec)

    for name in (spread_column, output_column):
        if name in frame.columns:
            report.notes.append(f"Overwrote existing column {name!r}.")
    frame[spread_column] = spread
    frame[output_column] = verdicts
    mark_verdicts(dataset, output_column)
    mark_thresholds(dataset, STEP, spec)

    report.verdict_column = output_column
    report.finding = "instrument"
    counts = tier_counts(verdicts)
    report.soundings_affected = counts[OUT_OF_SPEC]

    kept = ~is_tier(verdicts, OUT_OF_SPEC)
    report.distributions = {
        "spread": {
            "columns": columns,
            "unit": unit,
            "before": summarize(spread),
            "after": summarize(spread[kept]),
        },
        # Each altimeter's own distribution, because a spread says they differ
        # without saying which one moved. One sensor drifting while the others
        # hold shows up here and nowhere else.
        "per_altimeter": {
            column: summarize(frame[column].to_numpy(dtype=float))
            for column in columns
        },
        "verdicts": counts,
        "thresholds": describe(spec, unit),
    }

    line_column = line_column or dataset.survey.xyz.line_id_column
    if line_column is None or line_column not in frame.columns:
        report.notes.append(
            "No line identifier found; per-line summary and line verdict skipped."
        )
        return dataset, report

    xyz = dataset.survey.xyz
    lines = frame[line_column].to_numpy()
    easting = frame[xyz.x_column].to_numpy() if xyz.x_column in frame.columns else None
    northing = frame[xyz.y_column].to_numpy() if xyz.y_column in frame.columns else None

    flight_column = aggregate.find_flight_column(frame)
    if flight_column is not None:
        report.distributions["per_flight"] = aggregate.per_flight(
            verdicts, frame[flight_column].to_numpy(), lines, easting, northing,
            gap_tolerance=gap_tolerance)
    report.distributions["per_line"] = aggregate.per_line(
        verdicts, lines, easting, northing, gap_tolerance=gap_tolerance)


    log.info(
        "%s: %d soundings across %d altimeters, %d Good / %d Marginal / "
        "%d Out of Spec", STEP, len(frame), len(columns),
        counts[GOOD], counts[MARGINAL], counts[OUT_OF_SPEC],
    )
    return dataset, report
