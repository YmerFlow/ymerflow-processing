"""qc_02 — system attitude.

Judges how level the instrument was flown, against the attitude the survey was
contracted to hold, and writes the verdict as columns on ``flightlines``.

**What it does to the dataset:** adds a verdict per sounding for pitch, one for
roll, and one combining them, plus optionally the worst verdict on each line. It
changes no measurement, no uncertainty and no mask (R6).

Why pitch and roll are judged separately
----------------------------------------
They are the same quantity geometrically and different quantities operationally,
and the contract is written per axis. Sustained **roll** means a turn or a crab
into a crosswind; sustained **pitch** means climbing or descending on terrain.
Same tilt, different conversation with the contractor — so collapsing them to one
number would discard the part that says what to do about it.

The combined verdict is there for what comes after: a culling step wants one
column to threshold on and should not have to know there were two angles.

Which axis is which
-------------------
``tilt_x`` is **pitch** and ``tilt_y`` is **roll**. The x axis lies along the
flight direction, so the angle measured in that vertical plane is nose-up/down.
Roll is the perpendicular one. ``libaarhusxyz`` resolves the columns this way
(``tilt_pitch_column`` → ``tilt_x``/``TxPitch``/``AngleX``).

Worth stating because the two were briefly swapped downstream — harmlessly, since
that code only used them in the symmetric product ``cos(roll)·cos(pitch)``, and
since corrected. A check that reports the axes *separately*, as this one does,
would not have been so forgiving: it would have blamed a crosswind turn on
terrain following.

Why it matters that the geometry broke
--------------------------------------
The 1D forward model assumes a horizontal transmitter loop. Tilt it and the loop
develops a horizontal dipole component, which couples to a layered earth
differently from a vertical one — so the response is not merely attenuated, it
contains something the model cannot represent. That part is *not* removed by the
tilt correction, which rescales amplitude only. Beyond the spec limit the data is
wrong rather than noisy.

The total tilt of the loop normal from vertical follows from composing the two
rotations:

.. math:: \\cos\\theta = \\cos\\theta_{pitch}\\,\\cos\\theta_{roll}

which is exact, and reduces to the familiar :math:`\\sqrt{p^2 + r^2}` for small
angles. It is **reported but not classified** — the contract specifies per-axis
limits, and inventing a combined threshold the contract did not set would put a
number with no authority into a client deliverable. It is reported because two
axes each just inside their limit combine to more tilt than either suggests: 5°
and 5° is 7.07° in total.

Sources
-------
Composition of the two tilts is plane rotation geometry, derived above. For the
consequence of a non-horizontal loop for a 1D interpretation: Christiansen, A.V.,
Auken, E. and Sørensen, K., "The transient electromagnetic method", in
*Groundwater Geophysics: A Tool for Hydrogeology*, Springer, 2009.

**Threshold values come from the survey contract**, which is why they are
required parameters with no defaults. The classification model is carried over
from our own prior flight QC work, where a representative attitude spec was ±5° good, ±7.5°
marginal.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ..distribution import summarize
from ..pipeline import StepReport
from . import aggregate
from .thresholds import (
    CENTERED, GOOD, MARGINAL, OUT_OF_SPEC, classify, describe, is_tier,
    mark_thresholds, mark_verdicts, tier_counts, worst_of,
)

log = logging.getLogger(__name__)

STEP = "system_attitude"


def _worst(*verdicts) -> np.ndarray:
    """The most severe of several verdicts, per sounding.

    Missing where *any* contributing axis is missing: a sounding whose roll was
    not measured has an unknown attitude, and reporting the pitch verdict as the
    attitude verdict would claim more than was measured. Axes the instrument does
    not record at all are simply not passed in — that is a different thing from a
    sounding whose value is absent.
    """
    return worst_of(verdicts, require_all=True)


def _total_tilt(pitch, roll) -> np.ndarray:
    """Angle of the loop normal from vertical, in degrees.

    ``cos(total) = cos(pitch) * cos(roll)`` — exact, not a small-angle form.
    """
    cosines = np.cos(np.radians(pitch)) * np.cos(np.radians(roll))
    return np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))


def _looks_zeroed(values) -> bool:
    """Whether a series is identically zero wherever it is present.

    Attitude is sometimes forced to zero when it was unreliable or never
    recorded, so that a downstream tilt correction becomes a no-op rather than
    applying garbage. A check that took that at face value would report a
    perfectly level flight — the opposite of the truth. Real attitude is never
    exactly zero for every sounding.
    """
    finite = values[np.isfinite(values)]
    return finite.size > 0 and bool(np.all(finite == 0.0))


def system_attitude(
    dataset,
    good: float,
    marginal: float,
    pitch_column: str | None = None,
    roll_column: str | None = None,
    pitch_output: str = "qs_pitch",
    roll_output: str = "qs_roll",
    output_column: str = "qs_tilt",
    line_column: str | None = None,
    unit: str = "deg",
    gap_tolerance: float = 0.0,
):
    """Classify each sounding's pitch and roll against the contracted limits.

    :param good: Attitude within ±this, in ``unit``, meets spec. A contract term,
        so it has no default.
    :param marginal: Beyond ±this is Out of Spec. Between the two is Marginal.
    :param pitch_column: Which ``flightlines`` column holds pitch. Discovered
        from the survey if unset.
    :param roll_column: Likewise for roll.
    :param pitch_output: Where the per-sounding pitch verdict is written.
    :param roll_output: Likewise for roll.
    :param output_column: Where the combined verdict — the worse of the two —
        is written, so a later culling step has one column to threshold on.
    :param line_column: Which column identifies the line. Discovered if unset.
    :param unit: Unit of the attitude columns, used in report text only.
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
    axes = {}
    for label, named, discovered, output in (
        ("pitch", pitch_column, xyz.tilt_pitch_column, pitch_output),
        ("roll", roll_column, xyz.tilt_roll_column, roll_output),
    ):
        # A column the caller *named* and that is absent is a mistake worth
        # failing on; one we went looking for and did not find is an absent
        # input, which building_plans/03 requires be a note.
        if named is not None:
            if named not in frame.columns:
                raise KeyError(
                    f"No column {named!r} in flightlines. "
                    f"Present: {', '.join(map(str, frame.columns))}"
                )
            axes[label] = (named, output)
        elif discovered is not None and discovered in frame.columns:
            axes[label] = (discovered, output)

    if not axes:
        report.notes.append(
            "No attitude columns found; check skipped. Name them with "
            "'pitch_column:' / 'roll_column:' if they are present under "
            "unusual names."
        )
        return dataset, report
    if len(axes) == 1:
        report.notes.append(
            f"Only {next(iter(axes))} was found; the combined attitude verdict "
            "covers that axis alone."
        )

    spec = {"kind": CENTERED, "good": good, "marginal": marginal}
    verdicts, values = {}, {}
    for label, (column, output) in axes.items():
        values[label] = frame[column].to_numpy(dtype=float)
        verdicts[label] = classify(values[label], spec)
        if output in frame.columns:
            report.notes.append(f"Overwrote existing column {output!r}.")
        frame[output] = verdicts[label]

        if _looks_zeroed(values[label]):
            report.notes.append(
                f"{label} ({column!r}) is exactly zero for every sounding. That "
                "is a zeroed or absent sensor rather than a perfectly level "
                "flight, and this check cannot judge attitude that was not "
                "measured."
            )

    combined = _worst(*verdicts.values())
    if output_column in frame.columns:
        report.notes.append(f"Overwrote existing column {output_column!r}.")
    frame[output_column] = combined
    mark_verdicts(dataset, pitch_output, roll_output, output_column)
    mark_thresholds(dataset, STEP, spec)

    report.verdict_column = output_column
    report.finding = "flight"
    counts = tier_counts(combined)
    report.soundings_affected = counts[OUT_OF_SPEC]

    # R3's distribution, before and after. A QC check removes nothing, so
    # "after" is read as what would remain if the verdict were acted on.
    kept = ~is_tier(combined, OUT_OF_SPEC)
    report.distributions = {
        "verdicts": counts,
        "thresholds": describe(spec, unit),
        "per_axis": {
            label: {
                "column": axes[label][0],
                "unit": unit,
                "counts": tier_counts(verdicts[label]),
                "before": summarize(values[label]),
                "after": summarize(values[label][kept]),
            }
            for label in axes
        },
    }

    # Total tilt: reported, never classified. Two axes each just inside their
    # limit combine to more than either suggests.
    if len(axes) == 2:
        total = _total_tilt(values["pitch"], values["roll"])
        report.distributions["total_tilt"] = {
            "unit": unit,
            "note": "loop normal from vertical, cos(total)=cos(pitch)cos(roll); "
                    "reported, not classified — the contract sets per-axis limits",
            "before": summarize(total),
            "after": summarize(total[kept]),
        }

    line_column = line_column or xyz.line_id_column
    if line_column is None or line_column not in frame.columns:
        report.notes.append(
            "No line identifier found; per-line summary and line verdict skipped."
        )
        return dataset, report

    lines = frame[line_column].to_numpy()
    easting = frame[xyz.x_column].to_numpy() if xyz.x_column in frame.columns else None
    northing = frame[xyz.y_column].to_numpy() if xyz.y_column in frame.columns else None

    flight_column = aggregate.find_flight_column(frame)
    if flight_column is not None:
        report.distributions["per_flight"] = aggregate.per_flight(
            combined, frame[flight_column].to_numpy(), lines, easting, northing,
            gap_tolerance=gap_tolerance)
    report.distributions["per_line"] = aggregate.per_line(
        combined, lines, easting, northing, gap_tolerance=gap_tolerance)


    log.info(
        "%s: %d soundings, %d Good / %d Marginal / %d Out of Spec across %d line(s)",
        STEP, len(frame), counts[GOOD], counts[MARGINAL], counts[OUT_OF_SPEC],
        len(report.distributions["per_line"]),
    )
    return dataset, report
