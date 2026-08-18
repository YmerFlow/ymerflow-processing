"""qc_10 — sounding spacing.

Measures how far apart the soundings actually fall along each line, against the
spacing the survey was contracted to deliver, and writes the verdict as a column
on ``flightlines``.

**What it does to the dataset:** adds the along-line spacing to each sounding,
a verdict on the local density, and optionally the worst verdict on each line.
It changes no measurement, no uncertainty and no mask (R6).

Spacing, not density — and the threshold kind follows
-----------------------------------------------------
``03`` describes this check as producing "per-line along-line spacing" with
threshold kind ``descending`` — *less coverage is worse*. Those two do not agree.
Spacing and density are inverses, so only one of them can be ``descending``:
wider **spacing** is worse, which is ``ascending``; lower **density** is worse,
which is ``descending``.

**Spacing carries the verdict**, because that is the quantity a contract states
and the one a geophysicist pictures — "soundings no more than 15 m apart" rather
than "at least 67 per kilometre". The density it implies is computed and
reported alongside, since a contract may equally be written that way and the two
are the same measurement seen from either end.

Deviates from ``03``'s stated threshold kind, deliberately and for that reason.

Why report it separately from speed
-----------------------------------
Density is partly a consequence of qc_03 — fly faster and the soundings spread
out — but it is worth its own check because it is usually what the contract
actually specifies, and because speed is not the only way to lose it. Dropped
soundings, a stacking change, or an export at a coarser interval all reduce
density at unchanged ground speed.

That last one matters for more than coverage: the noise floor is a property of
how much stacking produced a sounding, so a survey exported at a coarser
along-line interval has a proportionally lower floor. Density is the visible
symptom of a change that also moves the error model.

Measured within lines
---------------------
Spacing is the distance from the previous sounding on the *same* line. The step
between the last sounding of one line and the first of the next crosses the
turn, which is a ferry rather than acquisition, and would read as an enormous
gap. The first sounding of each line therefore has no spacing and no verdict.

Sources
-------
Distance is Euclidean in the survey CRS, so this assumes a projected coordinate
system in linear units — true of any survey CRS this library handles, and the
same assumption the contiguous-run distances in :mod:`.aggregate` make.

**Threshold values come from the survey contract**, so they are required
parameters with no defaults.
"""

from __future__ import annotations

import logging

import numpy as np

from ..distribution import summarize
from ..pipeline import StepReport
from . import aggregate
from .thresholds import (
    ASCENDING, GOOD, MARGINAL, OUT_OF_SPEC, classify, describe, is_tier,
    mark_verdicts, tier_counts,
)

log = logging.getLogger(__name__)

STEP = "sounding_spacing"

#: Metres in a kilometre. Density is quoted per kilometre because that is how
#: contracts are written; position arrives in CRS units.
UNITS_PER_KM = 1000.0


def along_line_spacing(easting, northing, lines) -> np.ndarray:
    """Distance from the previous sounding on the same line, in CRS units.

    ``NaN`` for the first sounding of each line: nothing preceded it, and the
    step back to the previous line is the turn rather than a survey leg.
    """
    easting = np.asarray(easting, dtype=float)
    northing = np.asarray(northing, dtype=float)
    lines = np.asarray(lines)

    spacing = np.full(easting.shape, np.nan)
    step = np.hypot(np.diff(easting), np.diff(northing))
    spacing[1:] = np.where(lines[1:] == lines[:-1], step, np.nan)
    return spacing


def sounding_spacing(
    dataset,
    good_max: float,
    marginal_max: float,
    output_column: str = "qs_spacing",
    spacing_column: str = "spacing",
    line_column: str | None = None,
    units_per_km: float = UNITS_PER_KM,
    gap_tolerance: float = 0.0,
):
    """Classify local sounding density against the contracted minimum.

    :param good_max: Spacing at or below which the line meets spec, in CRS
        units. A contract term, so no default.
    :param marginal_max: Spacing above which the line is Out of Spec. Between
        the two is Marginal.
    :param output_column: Where the per-sounding verdict is written.
    :param spacing_column: Where the along-line spacing is written, in CRS
        units. Reported alongside the verdict because spacing is the number a
        geophysicist pictures, even though density is what is classified.
    :param line_column: Which column identifies the line. Discovered if unset.
    :param units_per_km: CRS units in a kilometre — 1000 for a metric CRS.
    :param gap_tolerance: In-spec stretches shorter than this, in CRS units, do
        not break an out-of-spec run when the contiguous-run length is measured.
        A contract saying "out of spec for 1 km or more" describes a stretch of
        flying that one recovered sounding does not end. ``0`` measures runs
        raw; the raw figure is reported either way.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if frame is None or len(frame) == 0:
        report.notes.append("No soundings; nothing to measure.")
        return dataset, report

    xyz = dataset.survey.xyz
    if xyz.x_column not in frame.columns or xyz.y_column not in frame.columns:
        report.notes.append(
            "No position columns; density cannot be measured and the check is "
            "skipped."
        )
        return dataset, report

    line_column = line_column or xyz.line_id_column
    has_lines = line_column is not None and line_column in frame.columns
    lines = frame[line_column].to_numpy() if has_lines else np.zeros(len(frame))
    if not has_lines:
        report.notes.append(
            "No line identifier; spacing may span the turn between two lines, "
            "which is a ferry rather than a survey leg."
        )

    spacing = along_line_spacing(frame[xyz.x_column], frame[xyz.y_column], lines)

    # A repeated position is a duplicate, not spacing of zero. Left as a number
    # it would classify as the best possible coverage, which is the opposite of
    # what it means, so it is dropped from the measurement entirely.
    duplicated = int(np.sum(spacing == 0))
    spacing = np.where(spacing == 0, np.nan, spacing)

    # Density is the inverse, reported because a contract may be written either
    # way round.
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(spacing > 0, units_per_km / spacing, np.nan)

    if duplicated:
        report.notes.append(
            f"{duplicated} sounding(s) sit at the same position as the previous "
            "one; they have zero spacing, which is not a measurement, so they "
            "carry no density and no verdict."
        )

    spec = {"kind": ASCENDING, "good_max": good_max,
            "marginal_max": marginal_max}
    verdicts = classify(spacing, spec)

    for name in (spacing_column, output_column):
        if name in frame.columns:
            report.notes.append(f"Overwrote existing column {name!r}.")
    frame[spacing_column] = spacing
    frame[output_column] = verdicts
    mark_verdicts(dataset, output_column)

    report.verdict_column = output_column
    report.finding = "flight"
    counts = tier_counts(verdicts)
    report.soundings_affected = counts[OUT_OF_SPEC]

    kept = ~is_tier(verdicts, OUT_OF_SPEC)
    report.distributions = {
        "density": {
            "unit": "soundings/km",
            "before": summarize(density),
            "after": summarize(density[kept]),
        },
        # The same measurement the other way up, because a contract may quote
        # either and a reader thinks in one of them.
        "spacing": {
            "unit": "CRS units",
            "before": summarize(spacing),
            "after": summarize(spacing[kept]),
        },
        "verdicts": counts,
        "thresholds": describe(spec, "CRS units"),
    }

    if not has_lines:
        report.notes.append(
            "No line identifier found; per-line summary and line verdict skipped."
        )
        return dataset, report

    easting = frame[xyz.x_column].to_numpy()
    northing = frame[xyz.y_column].to_numpy()

    flight_column = aggregate.find_flight_column(frame)
    if flight_column is not None:
        report.distributions["per_flight"] = aggregate.per_flight(
            verdicts, frame[flight_column].to_numpy(), lines, easting, northing,
            gap_tolerance=gap_tolerance)
    report.distributions["per_line"] = aggregate.per_line(
        verdicts, lines, easting, northing, gap_tolerance=gap_tolerance)


    log.info(
        "%s: %d soundings, %d Good / %d Marginal / %d Out of Spec",
        STEP, len(frame), counts[GOOD], counts[MARGINAL], counts[OUT_OF_SPEC],
    )
    return dataset, report
