"""qc_01 — height above ground.

Judges each sounding's height above the terrain against the height the survey
was contracted to fly, and writes the verdict as a column on ``flightlines``.
The single most common spec violation, and the one most directly tied to depth
of investigation.

**What it does to the dataset:** adds one column of Good / Marginal / Out of Spec
per sounding, and optionally a second column carrying the worst verdict on each
line. It changes no measurement, no uncertainty and no mask — run it and the
data going into processing is byte-identical (R6). Culling on the verdict is a
separate, later decision, and deliberately so: a contractor flying at 70 m
against a 65 m spec is Out of Spec — the client cares, it may be a payment
conversation — while the data is very likely still fine to invert.

Why height is the quantity
--------------------------
An airborne EM system's response falls off steeply with sensor altitude, and its
footprint widens with it, so flying high costs signal at depth and blurs
laterally at the same time. Both effects are monotonic in height, which is what
makes a one-sided ``ascending`` threshold the honest shape: there is no such
thing as too low for data quality, only for safety.

Sources
-------
The height–footprint relationship, and its consequence for AEM resolution:
Reid, J.E. and Vrbancich, J., "A comparison of the inductive-limit footprints of
airborne electromagnetic configurations", *Geophysics*, 2004. General treatment
of TEM depth of investigation and its dependence on system geometry:
Christiansen, A.V., Auken, E. and Sørensen, K., "The transient electromagnetic
method", in *Groundwater Geophysics: A Tool for Hydrogeology*, Springer, 2009.

The **threshold values themselves come from the survey contract, not from the
literature** — they are a commercial term, which is exactly why they are
required parameters here with no defaults.

The classification model is carried over from our own prior flight QC work; see
``ymerflow_processing/core/qc/thresholds.py``.
"""

from __future__ import annotations

import logging

import numpy as np

from ..distribution import summarize
from ..pipeline import StepReport
from . import aggregate
from .thresholds import (
    ASCENDING, GOOD, MARGINAL, OUT_OF_SPEC, classify, describe, tier_counts,
)

log = logging.getLogger(__name__)

STEP = "height_above_ground"


def height_above_ground(
    dataset,
    good_max: float,
    marginal_max: float,
    column: str | None = None,
    output_column: str = "qs_height",
    line_column: str | None = None,
    unit: str = "m",
    gap_tolerance: float = 0.0,
):
    """Classify each sounding's height above ground against the contracted spec.

    :param good_max: Height at or below which a sounding meets spec, in ``unit``.
        A contract term — no default, because inventing one would put a number
        with no authority into a client deliverable.
    :param marginal_max: Height above which a sounding is Out of Spec. Between
        the two is Marginal.
    :param column: Which ``flightlines`` column holds height above ground. Left
        unset, it is discovered from the survey's own altitude column.
    :param output_column: Where the per-sounding verdict is written.
    :param line_column: Which column identifies the line. Discovered if unset.
    :param unit: Unit of the height column, used in report text only.
    :param gap_tolerance: In-spec stretches shorter than this, in CRS units, do
        not break an out-of-spec run when the contiguous-run length is measured.
        A contract saying "out of spec for 1 km or more" describes a stretch of
        flying that one recovered sounding does not end. ``0`` measures runs
        raw; the raw figure is reported either way.

    Returns ``(dataset, StepReport)``. The dataset is returned rather than
    mutated silently, per ``building_plans/02``.

    **On terrain models.** ``building_plans/03`` lists a terrain model among this
    check's inputs. It is not taken here, and that is deliberate: producing a
    correct height-above-ground column from a DTM is filter_05's job in the
    processing chain, and duplicating it here would put the same physical
    statement in two places with two chances to disagree — precisely the defect
    R9 exists to prevent. This check reads the column; filter_05 is responsible
    for it being right.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if frame is None or len(frame) == 0:
        report.notes.append("No soundings; nothing to classify.")
        return dataset, report

    # A column the caller *named* and that is absent is a mistake worth failing
    # on. A column we went looking for and did not find is an absent input,
    # which building_plans/03 requires be a note rather than an error — that is
    # what lets one configuration run against a flight with no altimeter.
    if column is not None:
        if column not in frame.columns:
            raise KeyError(
                f"No column {column!r} in flightlines. "
                f"Present: {', '.join(map(str, frame.columns))}"
            )
        height_column = column
    else:
        height_column = dataset.survey.xyz.alt_column
        if height_column is None or height_column not in frame.columns:
            report.notes.append(
                "No height-above-ground column found; check skipped. "
                "Name one with 'column:' if it is present under an unusual name."
            )
            return dataset, report

    spec = {"kind": ASCENDING, "good_max": good_max, "marginal_max": marginal_max}
    heights = frame[height_column].to_numpy(dtype=float)
    verdicts = classify(heights, spec)

    if output_column in frame.columns:
        report.notes.append(f"Overwrote existing column {output_column!r}.")
    # Assigned through the frame the dataset holds, not a copy: R5 rules out
    # inplace= and chained assignment, not ordinary column assignment.
    frame[output_column] = verdicts

    report.verdict_column = output_column
    report.finding = "flight"
    counts = tier_counts(verdicts)
    # "Affected" is read here as "flagged": how many soundings this check has
    # something to say about. The full breakdown is in the report body.
    report.soundings_affected = counts[OUT_OF_SPEC]

    # R3's distribution, before and after. A QC check removes nothing, so
    # "after" is read as *what would remain if the verdict were acted on* —
    # which is the comparison that makes a threshold judgeable. Stated because
    # it is an interpretation, not an obvious reading.
    kept = np.asarray(verdicts != OUT_OF_SPEC)
    report.distributions = {
        "height": {
            "column": height_column,
            "unit": unit,
            "before": summarize(heights),
            "after": summarize(heights[kept]),
        },
        "verdicts": counts,
        "thresholds": describe(spec, unit),
    }

    xyz = dataset.survey.xyz
    easting = frame[xyz.x_column].to_numpy() if xyz.x_column in frame.columns else None
    northing = frame[xyz.y_column].to_numpy() if xyz.y_column in frame.columns else None
    if easting is None or northing is None:
        report.notes.append(
            "No position columns; contiguous runs reported in soundings only."
        )

    # Per flight, where the data says which flight each sounding belongs to. A
    # delivery commonly holds three or four flights, so this is what turns one
    # run over one delivered file into the per-flight grain that drift is a
    # series over, and the per-day rollup at the same time.
    flight_column = aggregate.find_flight_column(frame)
    line_column = line_column or dataset.survey.xyz.line_id_column
    has_lines = line_column is not None and line_column in frame.columns
    lines = frame[line_column].to_numpy() if has_lines else None

    if flight_column is None:
        report.notes.append(
            "No flight identifier found; per-flight summary skipped."
        )
    else:
        report.distributions["per_flight"] = aggregate.per_flight(
            verdicts, frame[flight_column].to_numpy(), lines, easting, northing,
            gap_tolerance=gap_tolerance
        )
        if not has_lines:
            report.notes.append(
                "No line identifier; per-flight contiguous runs may span line "
                "changes and so overstate the worst stretch."
            )

    if not has_lines:
        report.notes.append(
            "No line identifier found; per-line summary and line verdict skipped."
        )
        return dataset, report

    report.distributions["per_line"] = aggregate.per_line(
        verdicts, lines, easting, northing, gap_tolerance=gap_tolerance
    )

    log.info(
        "%s: %d soundings, %d Good / %d Marginal / %d Out of Spec across %d line(s)",
        STEP, len(frame), counts[GOOD], counts[MARGINAL], counts[OUT_OF_SPEC],
        len(report.distributions["per_line"]),
    )
    return dataset, report
