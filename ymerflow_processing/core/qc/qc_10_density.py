"""qc_10 — sounding density.

Measures how closely spaced the soundings actually are along each line, against
the density the survey was contracted to deliver, and writes the verdict as a
column on ``flightlines``.

**What it does to the dataset:** adds the along-line spacing to each sounding,
a verdict on the local density, and optionally the worst verdict on each line.
It changes no measurement, no uncertainty and no mask (R6).

Density or spacing — they are inverses and the threshold kind flips
------------------------------------------------------------------
``03`` describes this check as producing "per-line along-line spacing" with
threshold kind ``descending`` — *less coverage is worse*. Those two do not agree:
wider **spacing** is worse, so spacing alone would be ``ascending``. It is
**density** — soundings per kilometre — that gets worse as it falls.

Classification is on density, matching the stated kind, because that is also how
contracts are usually written. The spacing that produced it is reported and
written alongside, since it is the number a geophysicist pictures.

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
    DESCENDING, GOOD, MARGINAL, OUT_OF_SPEC, classify, describe, tier_counts,
)

log = logging.getLogger(__name__)

STEP = "sounding_density"

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


def sounding_density(
    dataset,
    good_min: float,
    marginal_min: float,
    output_column: str = "qc_density",
    spacing_column: str = "along_line_spacing",
    line_verdict_column: str | None = "qc_density_line",
    line_column: str | None = None,
    units_per_km: float = UNITS_PER_KM,
    gap_tolerance: float = 0.0,
):
    """Classify local sounding density against the contracted minimum.

    :param good_min: Soundings per kilometre at or above which the line meets
        spec. A contract term, so no default.
    :param marginal_min: Density below which the line is Out of Spec. Between
        the two is Marginal.
    :param output_column: Where the per-sounding verdict is written.
    :param spacing_column: Where the along-line spacing is written, in CRS
        units. Reported alongside the verdict because spacing is the number a
        geophysicist pictures, even though density is what is classified.
    :param line_verdict_column: Where the worst verdict on each line is written,
        denormalized for export. ``None`` to skip.
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
    # Density is the inverse of spacing. A repeated position gives zero spacing
    # and an infinite density, which is not a measurement — drop it rather than
    # let it dominate every statistic downstream.
    with np.errstate(divide="ignore", invalid="ignore"):
        density = np.where(spacing > 0, units_per_km / spacing, np.nan)

    duplicated = int(np.sum(spacing == 0))
    if duplicated:
        report.notes.append(
            f"{duplicated} sounding(s) sit at the same position as the previous "
            "one; they have no density and no verdict."
        )

    spec = {"kind": DESCENDING, "good_min": good_min, "marginal_min": marginal_min}
    verdicts = classify(density, spec)

    for name in (spacing_column, output_column):
        if name in frame.columns:
            report.notes.append(f"Overwrote existing column {name!r}.")
    frame[spacing_column] = spacing
    frame[output_column] = verdicts

    report.verdict_column = output_column
    report.finding = "flight"
    counts = tier_counts(verdicts)
    report.soundings_affected = counts[OUT_OF_SPEC]

    kept = np.asarray(verdicts != OUT_OF_SPEC)
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
        "thresholds": describe(spec, "soundings/km"),
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

    if line_verdict_column is not None:
        if line_verdict_column in frame.columns:
            report.notes.append(f"Overwrote existing column {line_verdict_column!r}.")
        frame[line_verdict_column] = aggregate.worst_verdict_per_line(
            verdicts, lines).to_numpy()

    log.info(
        "%s: %d soundings, %d Good / %d Marginal / %d Out of Spec",
        STEP, len(frame), counts[GOOD], counts[MARGINAL], counts[OUT_OF_SPEC],
    )
    return dataset, report
