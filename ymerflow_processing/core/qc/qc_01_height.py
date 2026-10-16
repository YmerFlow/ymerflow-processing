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

import logging

import numpy as np

from ..dataset import HEIGHT_COLUMNS, find_column
from ..distribution import summarize
from ..pipeline import MISSING, StepReport
from . import aggregate
from .thresholds import (
    ASCENDING, GOOD, MARGINAL, OUT_OF_SPEC, classify, describe, is_tier,
    mark_thresholds, mark_verdicts, tier_counts,
)

log = logging.getLogger(__name__)

STEP = "height_above_ground"


def _check_it_is_a_height(heights, height_column, frame, dataset, report):
    """Say so when the column being graded looks like elevation, not height.

    The two are different quantities, and one real delivery carries both —
    ``Height`` at tens of metres above ground and ``Alt`` at hundreds above the
    geoid. Nothing in a *name* reliably separates them, so this compares values.

    The signature is specific: subtract the terrain column and what is left is a
    plausible flying height, small against the original. Elevation minus ground
    is height; height minus ground is negative or nonsense. So it fires on the
    one case it is meant to and stays quiet otherwise, which matters — a warning
    that fires on correct use is worse than none.
    """
    z_column = getattr(dataset.survey.xyz, "z_column", None)
    if not z_column or z_column not in frame.columns:
        return
    ground = frame[z_column].to_numpy(dtype=float)
    # Checked before the medians rather than after: nanmedian of an all-NaN
    # slice warns, and a delivery with no heights at all is a legitimate input
    # rather than a mistake worth a warning.
    usable = np.isfinite(heights) & np.isfinite(ground)
    if not usable.any():
        return
    above = float(np.median(heights[usable] - ground[usable]))
    level = float(np.median(heights[usable]))
    if level <= 0:
        return
    if 0 < above < 0.5 * level:
        report.notes.append(
            f"{height_column!r} looks like ELEVATION rather than height above "
            f"ground: subtracting {z_column!r} leaves a median of {above:.1f}, "
            f"which is a plausible flying height, against {level:.1f} for the "
            f"column itself. Graded as it stands every sounding reads as flown "
            f"far too high. Name the right column with 'column:', or run "
            f"apply_terrain_model to derive one."
        )


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

    Parameters
    ----------
    good_max : float
        Height at or below which a sounding meets spec, in ``unit``. A contract
        term — no default, because inventing one would put a number with no
        authority into a client deliverable.
    marginal_max : float
        Height above which a sounding is Out of Spec. Between the two is
        Marginal.
    column : str | None
        Which ``flightlines`` column holds height above ground. Left unset, it
        is discovered from the survey's own altitude column.
    output_column : str
        Where the per-sounding verdict is written.
    line_column : str | None
        Which column identifies the line. Discovered if unset.
    unit : str
        Unit of the height column, used in report text only.
    gap_tolerance : float
        In-spec stretches shorter than this, in CRS units, do not break an
        out-of-spec run when the contiguous-run length is measured. A contract
        saying "out of spec for 1 km or more" describes a stretch of flying
        that one recovered sounding does not end. ``0`` measures runs raw; the
        raw figure is reported either way.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``. The dataset is returned rather than mutated
        silently, per ``building_plans/02``.

    **On terrain models.** ``building_plans/03`` lists a terrain model among
    this check's inputs. It is not taken here, and that is deliberate:
    producing a correct height-above-ground column from a DTM is filter_05's
    job in the processing chain, and duplicating it here would put the same
    physical statement in two places with two chances to disagree — precisely
    the defect R9 exists to prevent. This check reads the column; filter_05 is
    responsible for it being right.
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
        # Discovered case-insensitively and preferring the more specific
        # spelling. libaarhusxyz's own lookup is consulted first but is not
        # trusted alone: on a delivery carrying both `Height` (tens of metres
        # above ground) and `Alt` (hundreds, above the geoid) it returns `Alt`,
        # and grading height-above-ground against elevation puts every sounding
        # out of spec while looking like a survey flown far too high.
        height_column = (find_column(frame, HEIGHT_COLUMNS)
                         or dataset.survey.xyz.alt_column)
        if height_column is None or height_column not in frame.columns:
            report.notes.append(
                "No height-above-ground column found; check skipped. "
                "Name one with 'column:' if it is present under an unusual name."
            )
            report.ungraded = MISSING
            return dataset, report

    spec = {"kind": ASCENDING, "good_max": good_max, "marginal_max": marginal_max}
    heights = frame[height_column].to_numpy(dtype=float)
    verdicts = classify(heights, spec)
    _check_it_is_a_height(heights, height_column, frame, dataset, report)

    if output_column in frame.columns:
        report.notes.append(f"Overwrote existing column {output_column!r}.")
    # Assigned through the frame the dataset holds, not a copy: R5 rules out
    # inplace= and chained assignment, not ordinary column assignment.
    frame[output_column] = verdicts
    mark_verdicts(dataset, output_column)
    mark_thresholds(dataset, STEP, spec)

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
    kept = ~is_tier(verdicts, OUT_OF_SPEC)
    # R3 asks for the parameters actually used, including the ones nobody
    # supplied. A discovered column is exactly that, and not recording it is why
    # reading the wrong one took three steps to diagnose rather than one.
    report.distributions = {
        "column": height_column,
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
