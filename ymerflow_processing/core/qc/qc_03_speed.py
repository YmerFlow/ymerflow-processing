"""qc_03 — ground speed.

Judges how fast the aircraft was flown against the speed the survey was
contracted to hold, and writes the verdict as a column on ``flightlines``.

**What it does to the dataset:** adds one verdict per sounding, and optionally
the worst verdict on each line. It changes no measurement, no uncertainty and no
mask (R6).

Why the threshold has two ends
------------------------------
Speed is the first quantity here that is wrong in *both* directions, which is
what the ``window`` threshold kind exists for:

**Too fast** reduces the number of transients stacked into each sounding and
widens the effective along-line spacing. Both cost data quality — and the first
one bears directly on the noise floor, since ambient noise averages down as the
square root of the stack count, so the floor is a property of how much stacking
produced a sounding rather than of the instrument alone.

**Too slow** wastes flight hours, which is money rather than data quality, but it
is a contract term all the same.

``centered`` cannot express this: the good range is not symmetric about anything,
and zero is not the target. Hence a window with four bounds.

Where the speed comes from
--------------------------
A recorded ground-speed column if the survey has one — SkyTEM Geosoft exports
carry ``GdSpeed``. Failing that it is derived from position and time, which is
what ``03`` specifies as the input, using the along-line distance between
consecutive soundings and the interval between them.

Derivation is per line. Between the last sounding of one line and the first of
the next, the aircraft turned, flew a long way and did not acquire — a "speed"
computed across that boundary is an artifact of the gap, not a measurement, and
would classify as wildly out of spec.

Sources
-------
Ground speed is distance over time; nothing here is model-dependent. The link
between stacking and the noise floor — that ambient noise falls as √N in the
number of stacked transients — is the standard result for averaging independent
samples, and is why speed is a data-quality term and not only a cost one. See
R9 in ``building_plans/01`` for how the floor is treated once assigned.

**Threshold values come from the survey contract**, so they are required
parameters with no defaults. one prior survey used a target of 85 km/h.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..distribution import summarize
from ..pipeline import StepReport
from . import aggregate
from .thresholds import (
    GOOD, MARGINAL, OUT_OF_SPEC, WINDOW, classify, describe, tier_counts,
)

log = logging.getLogger(__name__)

STEP = "ground_speed"

#: Spellings a recorded ground speed turns up under. Continues the idiom
#: libaarhusxyz uses for position, altitude and line identity.
SPEED_COLUMNS = ("ground_speed", "GdSpeed", "gdspeed", "groundspeed",
                 "speed", "Speed")

#: Metres per second in one kilometre per hour. Position is in CRS units —
#: metres for any projected survey CRS — while contracts are written in km/h.
MS_TO_KMH = 3.6


def find_speed_column(frame) -> str | None:
    """A recorded ground-speed column, or ``None`` if the survey has none."""
    for name in SPEED_COLUMNS:
        if name in frame.columns:
            return name
    return None


def derive_speed(easting, northing, seconds, lines) -> np.ndarray:
    """Ground speed per sounding from position and time, in distance-units/second.

    Computed **within each line**. The gap between one line and the next is a
    turn, not a survey leg, so a speed across that boundary measures the ferry
    rather than the acquisition and would read as grossly out of spec.

    Each sounding takes the speed of the step that *reached* it; the first
    sounding of a line has no preceding step and so no speed.
    """
    easting = np.asarray(easting, dtype=float)
    northing = np.asarray(northing, dtype=float)
    seconds = np.asarray(seconds, dtype=float)
    lines = np.asarray(lines)

    speed = np.full(easting.shape, np.nan)
    distance = np.hypot(np.diff(easting), np.diff(northing))
    elapsed = np.diff(seconds)
    same_line = lines[1:] == lines[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        step = np.where(same_line & (elapsed > 0), distance / elapsed, np.nan)
    speed[1:] = step
    return speed


def _seconds(frame, xyz, column: str | None):
    """Acquisition time per sounding in seconds, or ``None``.

    A normalized survey carries a ``timestamp`` counted in days since
    1900-01-01, which is libaarhusxyz's convention; an unnormalized one may
    carry whatever it was delivered with.
    """
    if column is not None:
        return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)

    timestamp = xyz.get_column("timestamp")
    if timestamp is not None and timestamp in frame.columns:
        days = pd.to_numeric(frame[timestamp], errors="coerce")
        return days.to_numpy(dtype=float) * 86400.0
    return None


def ground_speed(
    dataset,
    good_min: float,
    good_max: float,
    marginal_min: float,
    marginal_max: float,
    column: str | None = None,
    time_column: str | None = None,
    output_column: str = "qs_speed",
    line_column: str | None = None,
    unit: str = "km/h",
    scale: float = MS_TO_KMH,
    gap_tolerance: float = 0.0,
):
    """Classify each sounding's ground speed against the contracted window.

    :param good_min: Slowest speed meeting spec, in ``unit``. A contract term,
        so no default.
    :param good_max: Fastest speed meeting spec.
    :param marginal_min: Below this is Out of Spec; between it and ``good_min``
        is Marginal.
    :param marginal_max: Above this is Out of Spec.
    :param column: Which ``flightlines`` column holds ground speed. Discovered
        if unset; derived from position and time if the survey records none.
    :param time_column: Which column holds acquisition time, used only when
        deriving. Discovered if unset.
    :param output_column: Where the per-sounding verdict is written.
    :param line_column: Which column identifies the line. Discovered if unset.
    :param unit: Unit the thresholds are expressed in, for report text.
    :param scale: Multiplier from derived units — distance-units per second, so
        m/s for a projected CRS in metres — to ``unit``. Only applied to derived
        speed; a recorded column is assumed to be in ``unit`` already.
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
        report.notes.append("No soundings; nothing to classify.")
        return dataset, report

    xyz = dataset.survey.xyz
    line_column = line_column or xyz.line_id_column
    has_lines = line_column is not None and line_column in frame.columns
    lines = frame[line_column].to_numpy() if has_lines else np.zeros(len(frame))

    # A recorded column is preferred: it is what the instrument measured, and
    # deriving would substitute our arithmetic for its measurement.
    if column is not None and column not in frame.columns:
        raise KeyError(
            f"No column {column!r} in flightlines. "
            f"Present: {', '.join(map(str, frame.columns))}"
        )
    speed_column = column or find_speed_column(frame)

    if speed_column is not None:
        speeds = frame[speed_column].to_numpy(dtype=float)
        source = f"recorded ({speed_column!r})"
    else:
        seconds = _seconds(frame, xyz, time_column)
        has_xy = xyz.x_column in frame.columns and xyz.y_column in frame.columns
        if seconds is None or not has_xy:
            report.notes.append(
                "No ground-speed column, and it cannot be derived without "
                "position and time; check skipped. Name one with 'column:' if "
                "it is present under an unusual name."
            )
            return dataset, report
        speeds = derive_speed(frame[xyz.x_column], frame[xyz.y_column],
                              seconds, lines) * scale
        source = "derived from position and time"
        if not has_lines:
            report.notes.append(
                "No line identifier; derived speed may span the turn between "
                "two lines, which is a ferry rather than a survey leg."
            )
        report.notes.append(
            "Ground speed derived from position and time; the first sounding "
            "of each line has no preceding step and so no verdict."
        )

    spec = {"kind": WINDOW, "good_min": good_min, "good_max": good_max,
            "marginal_min": marginal_min, "marginal_max": marginal_max}
    verdicts = classify(speeds, spec)

    if output_column in frame.columns:
        report.notes.append(f"Overwrote existing column {output_column!r}.")
    frame[output_column] = verdicts

    report.verdict_column = output_column
    report.finding = "flight"
    counts = tier_counts(verdicts)
    report.soundings_affected = counts[OUT_OF_SPEC]

    kept = np.asarray(verdicts != OUT_OF_SPEC)
    report.distributions = {
        "speed": {
            "source": source,
            "unit": unit,
            "before": summarize(speeds),
            "after": summarize(speeds[kept]),
        },
        "verdicts": counts,
        "thresholds": describe(spec, unit),
    }
    # Which end of the window a breach was at. "Thirty soundings out of spec"
    # means opposite things for a contractor depending on the answer.
    finite = np.isfinite(speeds)
    report.distributions["out_of_spec_by_end"] = {
        "too_slow": int(np.sum(finite & (speeds < marginal_min))),
        "too_fast": int(np.sum(finite & (speeds > marginal_max))),
    }

    if not has_lines:
        report.notes.append(
            "No line identifier found; per-line summary and line verdict skipped."
        )
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


    log.info(
        "%s: %d soundings %s, %d Good / %d Marginal / %d Out of Spec",
        STEP, len(frame), source, counts[GOOD], counts[MARGINAL],
        counts[OUT_OF_SPEC],
    )
    return dataset, report
