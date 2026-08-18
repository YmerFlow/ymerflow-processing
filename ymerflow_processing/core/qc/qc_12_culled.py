"""qc_12 — how much of each sounding survived processing.

**What it does to the dataset:** adds one numeric column holding the fraction of
each sounding's gates that culling disabled, and a verdict column only if
thresholds were given. It changes no measurement, no uncertainty and no mask —
this reads the mask that culling wrote (R6).

``03`` describes it as *"How much of the line survived processing. Included when
a processing history exists."* — optional in exactly that sense: it measures the
work the culling group did, so on a delivery nothing has culled it has nothing to
report and says so rather than reporting zero.

Measured against what arrived, not against the array
----------------------------------------------------
**The denominator is the mask as it was found, not the gate count.** A delivery
can arrive already culled by the contractor's own processing, and that is
preserved rather than reset. Measuring against the full array would report the
contractor's culling as ours, which flatters or damns the wrong party depending
on which way you read it.

So this reports what *this* pipeline removed. Where the mask arrived partly
disabled, the arriving state is reported alongside so both are visible.

This makes qc_12 unusual among the checks: it is the only one whose answer
depends on a step having run earlier in the same session, which is why it reads
the provenance array rather than only the mask.

Per sounding, across channels
-----------------------------
The fraction is over every channel's gates together, because the question is how
much of *this sounding* survived, and a sounding is not one channel. A dual-moment
sounding that lost its low moment entirely and kept its high moment lost about
half its data, and that is the honest number.

The per-channel split is in the report, since "half gone" reads very differently
if it is half of each moment rather than all of one.

Sources
-------
The mask this reads is described in ``core/mask.py``; the per-datum provenance
codes it attributes removals to are written by the culling filters and explained
on the dataset itself.

**There are no default thresholds.** Unset, this measures and says nothing —
what fraction of a sounding may be culled before the result is untrustworthy
depends on the inversion, not on this library.
"""

from __future__ import annotations

import logging

import numpy as np

from ..distribution import summarize
from ..mask import CULLED, PROVENANCE_PREFIX, provenance_key, step_codes
from ..pipeline import StepReport
from . import aggregate
from .thresholds import (
    ASCENDING, GOOD, MARGINAL, OUT_OF_SPEC, classify, describe, is_tier,
    mark_thresholds, mark_verdicts, tier_counts,
)

log = logging.getLogger(__name__)

STEP = "culled_fraction"


def culled_fraction(
    dataset,
    good_max: float | None = None,
    marginal_max: float | None = None,
    output_column: str = "qs_culled",
    fraction_column: str = "cull_frac",
    line_column: str | None = None,
    gap_tolerance: float = 0.0,
):
    """Measure how much of each sounding culling disabled.

    :param good_max: Culled fraction at or below which a sounding is
        unremarkable. **Left unset, no verdict is produced** and only the
        measurement is written.
    :param marginal_max: Fraction above which the sounding is Out of Spec.
        Required if ``good_max`` is given.
    :param output_column: Where the per-sounding verdict is written.
    :param fraction_column: Where the numeric fraction is written.
    :param line_column: Which column identifies the line. Discovered if unset.
    :param gap_tolerance: In-spec stretches shorter than this, in CRS units, do
        not break an out-of-spec run when contiguous runs are measured.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if frame is None or len(frame) == 0:
        report.notes.append("No soundings; nothing to measure.")
        return dataset, report

    channels = dataset.channels
    masked = [c for c in channels
              if c.inuse_key and c.inuse_key in dataset.layer_data]
    if not masked:
        report.notes.append(
            "No in-use mask on this dataset, so nothing has been culled and "
            "there is nothing to measure. This check reports on the culling "
            "group's work and is skipped when none has run."
        )
        return dataset, report

    total = np.zeros(len(frame), dtype=float)
    removed = np.zeros(len(frame), dtype=float)
    arrived = np.zeros(len(frame), dtype=float)
    per_channel = {}

    for channel in masked:
        mask = dataset.layer_data[channel.inuse_key].to_numpy()
        gates = mask.shape[1]

        # Attributed to a step in this session, so the contractor's own culling
        # is not reported as ours. A gate off with no provenance arrived that
        # way; the provenance array only exists once something here wrote it.
        why_key = provenance_key(channel.inuse_key)
        if why_key in dataset.layer_data:
            why = dataset.layer_data[why_key].to_numpy()
            ours = (mask == CULLED) & (why != 0)
        else:
            ours = np.zeros_like(mask, dtype=bool)
        theirs = (mask == CULLED) & ~ours

        total += gates
        removed += ours.sum(axis=1)
        arrived += theirs.sum(axis=1)
        per_channel[str(channel)] = {
            "gates": int(gates),
            "culled_here": int(ours.sum()),
            "arrived_culled": int(theirs.sum()),
        }

    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.where(total > 0, removed / total, np.nan)

    if fraction_column in frame.columns:
        report.notes.append(f"Overwrote existing column {fraction_column!r}.")
    frame[fraction_column] = fraction

    verdicts = None
    if good_max is not None:
        if marginal_max is None:
            raise ValueError(
                "good_max was given without marginal_max; a threshold needs "
                "both bounds."
            )
        spec = {"kind": ASCENDING, "good_max": good_max,
                "marginal_max": marginal_max}
        verdicts = classify(fraction, spec)
        if output_column in frame.columns:
            report.notes.append(f"Overwrote existing column {output_column!r}.")
        frame[output_column] = verdicts
        mark_verdicts(dataset, output_column)
        mark_thresholds(dataset, STEP, spec)

        report.verdict_column = output_column
        # Culling is this library's doing, not the contractor's and not the
        # instrument's. It is neither a flight-quality nor an instrument-health
        # finding, so it names itself as a reason and feeds neither summary.
        report.finding = "processing"
        counts = tier_counts(verdicts)
        report.soundings_affected = counts[OUT_OF_SPEC]
    else:
        report.notes.append(
            "No thresholds given, so the culled fraction is measured and left "
            "unjudged. Supply good_max and marginal_max to classify it."
        )

    if arrived.any():
        report.notes.append(
            f"{int(arrived.sum())} gate-sample(s) arrived already disabled and "
            f"are excluded from the fraction — that is the contractor's "
            f"culling, not this pipeline's."
        )

    report.distributions = {
        "culled_fraction": summarize(fraction),
        "per_channel": per_channel,
        "culled_here": int(removed.sum()),
        "arrived_culled": int(arrived.sum()),
        "by_step": step_codes(dataset) or None,
    }
    if verdicts is not None:
        report.distributions["verdicts"] = counts
        report.distributions["thresholds"] = describe(spec, "of gates")

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

    log.info("%s: %d gate-sample(s) culled here, %d arrived culled",
             STEP, int(removed.sum()), int(arrived.sum()))
    return dataset, report
