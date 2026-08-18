"""qc_04 — transmitter output stability.

Judges whether the transmitter held its output, and whether the system
description describes the instrument that actually flew. Writes both verdicts as
columns on ``flightlines`` and the jump events into the report.

**What it does to the dataset:** adds a stability verdict per sounding per
moment, a combined stability verdict, the measured offset from nominal per
moment, and a nominal-agreement verdict only if that comparison was given
thresholds. It changes no measurement, no uncertainty and no mask (R6).

Stability is the question; agreement with nominal is a second, weaker one
--------------------------------------------------------------------------
**A steady current that differs from nominal is not a data problem.** The current
actually flown is recorded per sounding and carried into the normalization, so a
transmitter holding 105 A against a 110 A nominal produces perfectly
self-consistent data. What costs data quality is the current *moving*, because
then the excitation differs between soundings that are later stacked and compared.

So stability carries the verdict, **alone**. A large offset from nominal earns
its own separate column and never joins the combined verdict — not because the
data is wrong, but because the system description and the hardware disagree, and
everything downstream trusts the description. Folding it into the combined
verdict would penalize a survey for a paperwork discrepancy, which is exactly
what "noted, not penalized" rules out.

Like qc_05, the comparison is a reference measurement by default: the offset is
always computed and written, and a verdict appears only when thresholds are
supplied. What counts as too far from nominal is an operational judgement, and
guessing it here would put a house number into a client deliverable. A survey has been seen flying some 26 A away from its declared
nominal; steady throughout, and still worth knowing before anyone computes a
dipole moment from the declared figure.

Both directions count. Measured current can sit above nominal as readily as
below, and either says the same thing about the description.

Why the window is a fraction and not a multiple of sigma
--------------------------------------------------------
The obvious window is a multiple of the standard deviation, and it is wrong here:
**sigma is inflated by the very instability it is meant to detect.** A transmitter
swinging wildly has a large sigma, so every sounding sits comfortably within two
of them and nothing is flagged. A fraction of the reference current does not
self-inflate.

That still satisfies ``03``'s portability requirement — the reference is derived
from the data rather than hardcoded — which matters because current levels differ
completely between systems, from a few amps to a few hundred.

Two failure modes, measured separately
--------------------------------------
**Slow drift** shows up as soundings departing from the survey-wide mean, which
is what the verdict measures.

**Sudden steps** do not: a step part-way through leaves both halves near the mean
while the transition itself is the fault. Those are found by comparing each
sounding to the one before it, and reported as discrete events rather than as a
verdict, since the event is the transition and not the soundings either side.

Jump detection **excludes the first sample of each flight**, where the current
legitimately steps as the transmitter comes up. That is operational knowledge
carried over from our own prior flight QC work rather than something derivable.

Sources
-------
The dipole moment a transmitter produces is proportional to its current
(``m = N I A``; see ``em/tem/method.py``), so a fractional change in current is a
fractional change in the moment driving the earth. Everything downstream that
normalizes by moment inherits it.

**Threshold values are contract or operational parameters, so none of them have
defaults.** The stability window must be given for the check to classify
anything; the nominal window is optional, and unset leaves the offset measured
and unjudged.
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
    mark_verdicts, tier_counts, worst_of,
)

log = logging.getLogger(__name__)

STEP = "transmitter_stability"

#: How a per-moment current column is spelled, given a moment named e.g. "LM".
#: Continues the idiom libaarhusxyz uses for position, altitude and line.
MOMENT_CURRENT_PATTERNS = (
    "{name}current", "{name}_current", "current_{name}", "tx{name}current",
    "{name}txcurrent", "i{name}",
)

#: What a single-moment system, or one that names its current generically, calls
#: it. Only consulted when there is exactly one moment to attach it to.
GENERIC_CURRENT_COLUMNS = ("current", "txcurrent", "tx_current", "tx_curr",
                           "transmitter_current")


def find_current_columns(frame, channels) -> dict[str, str]:
    """Which ``flightlines`` column carries each moment's transmitter current.

    R1 applied to columns rather than arrays: the number of moments is data, so
    the number of current columns is too, and neither may be assumed. A
    dual-moment system carries two; a single-moment system carries one, often
    named without any moment in it at all.

    Matched case-insensitively, since a round trip through the xyz format
    lowercases column names.
    """
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    found: dict[str, str] = {}

    for channel in channels:
        if channel.name:
            name = channel.name.strip().lower()
            for pattern in MOMENT_CURRENT_PATTERNS:
                candidate = lookup.get(pattern.format(name=name))
                if candidate is not None:
                    found[channel.key] = candidate
                    break

    # A generic name can only be attributed when there is one moment it could
    # belong to. With two, "current" is ambiguous and guessing would be worse
    # than reporting nothing.
    if not found and len(channels) == 1:
        for spelling in GENERIC_CURRENT_COLUMNS:
            candidate = lookup.get(spelling)
            if candidate is not None:
                found[channels[0].key] = candidate
                break
    return found


def detect_jumps(values, groups, fraction: float = 0.01) -> np.ndarray:
    """Soundings where the current stepped from the one before it.

    A step part-way through a flight leaves both halves close to the survey mean,
    so a threshold on deviation does not see it. Comparing consecutive soundings
    does.

    **The first sample of each group is never flagged.** Current legitimately
    steps there as the transmitter comes up, and treating that as a fault would
    flag every flight exactly once.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)

    jumps = np.zeros(values.shape, dtype=bool)
    if values.size < 2:
        return jumps

    previous = values[:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        change = np.abs(values[1:] - previous) / np.abs(previous)
    jumps[1:] = (change > fraction) & np.isfinite(change) & (groups[1:] == groups[:-1])
    return jumps


def _fractional_offset(values, reference: float) -> np.ndarray:
    """Signed departure from ``reference``, as a fraction of it."""
    if reference is None or not np.isfinite(reference) or reference == 0:
        return np.full(np.shape(values), np.nan)
    return (np.asarray(values, dtype=float) - reference) / abs(reference)


def _worst(frames) -> np.ndarray:
    """Worst verdict across several, per sounding; missing only if all are."""
    return worst_of(frames)


def transmitter_stability(
    dataset,
    good_fraction: float,
    marginal_fraction: float,
    nominal_good_fraction: float | None = None,
    nominal_marginal_fraction: float | None = None,
    jump_fraction: float = 0.01,
    columns: dict | None = None,
    output_column: str = "qs_cur",
    nominal_column: str | None = "qs_nom",
    nominal_offset_prefix: str | None = "nom_",
    per_moment_prefix: str | None = "qs_cur_",
    line_column: str | None = None,
    flight_column: str | None = None,
    gap_tolerance: float = 0.0,
):
    """Classify transmitter output stability, and agreement with nominal.

    :param good_fraction: Departure from the moment's own mean current, as a
        fraction of it, within which a sounding meets spec — e.g. ``0.01`` for
        1%. No default: what counts as steady is an operational judgement.
    :param marginal_fraction: Departure beyond which a sounding is Out of Spec.
    :param nominal_good_fraction: Departure of the *mean* current from the
        system description's declared nominal within which the description is
        considered to describe this instrument. Both directions. **Left unset,
        the offset is measured and no verdict is produced** — what counts as too
        far from nominal is an operational judgement, not something to guess.
    :param nominal_marginal_fraction: Beyond which the disagreement is Out of
        Spec. Required if ``nominal_good_fraction`` is given.
    :param jump_fraction: Sounding-to-sounding change, as a fraction of the
        previous value, counted as a sudden step.
    :param columns: ``{channel key or moment name: column}``, when the current
        columns cannot be discovered. Discovered from the moment names otherwise.
    :param output_column: Where the combined verdict is written — the worst of
        stability and nominal agreement, across every moment.
    :param nominal_column: Where the nominal-agreement verdict alone is written.
        ``None`` to skip.
    :param per_moment_prefix: Prefix for the per-moment stability verdicts,
        suffixed with the channel key so the name stays inside the ten
        characters a shapefile field allows. ``None`` to skip them.
    :param line_column: Which column identifies the line. Discovered if unset.
    :param flight_column: Which column identifies the flight, used to keep jump
        detection from flagging each flight's first sample. Discovered if unset.
    :param gap_tolerance: In-spec stretches shorter than this, in CRS units, do
        not break an out-of-spec run when the contiguous-run length is measured.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if frame is None or len(frame) == 0:
        report.notes.append("No soundings; nothing to classify.")
        return dataset, report

    channels = dataset.channels
    if not channels:
        report.notes.append(
            "No moments found, so no current to attribute to one; check skipped."
        )
        return dataset, report

    if columns:
        by_key = {}
        for wanted, column in columns.items():
            if column not in frame.columns:
                raise KeyError(
                    f"No column {column!r} in flightlines. "
                    f"Present: {', '.join(map(str, frame.columns))}"
                )
            by_key[dataset.channel(str(wanted)).key] = column
    else:
        by_key = find_current_columns(frame, channels)

    if not by_key:
        report.notes.append(
            "No transmitter-current column found for any moment; check skipped. "
            "Name them with 'columns:' if present under unusual names."
        )
        return dataset, report

    missing = [str(c) for c in channels if c.key not in by_key]
    if missing:
        report.notes.append(
            f"No current column for {', '.join(missing)}; "
            "those moments are not judged."
        )

    flight_column = flight_column or aggregate.find_flight_column(frame)
    if flight_column is None:
        report.notes.append(
            "No flight identifier; jump detection cannot exclude each flight's "
            "first sample, where the current legitimately steps."
        )
        flights = np.zeros(len(frame))
    else:
        flights = frame[flight_column].to_numpy()

    stability_spec = {"kind": CENTERED, "good": good_fraction,
                      "marginal": marginal_fraction}
    if nominal_good_fraction is None and nominal_marginal_fraction is not None:
        raise ValueError(
            "nominal_marginal_fraction was given without nominal_good_fraction; "
            "a threshold needs both bounds."
        )
    if nominal_good_fraction is not None and nominal_marginal_fraction is None:
        raise ValueError(
            "nominal_good_fraction was given without nominal_marginal_fraction; "
            "a threshold needs both bounds."
        )
    nominal_spec = (
        {"kind": CENTERED, "good": nominal_good_fraction,
         "marginal": nominal_marginal_fraction}
        if nominal_good_fraction is not None else None
    )

    per_moment, nominal_verdicts, moments_report = [], [], {}
    for channel in channels:
        column = by_key.get(channel.key)
        if column is None:
            continue
        current = frame[column].to_numpy(dtype=float)
        finite = current[np.isfinite(current)]
        reference = float(finite.mean()) if finite.size else np.nan

        # Stability: departure from this moment's own mean.
        stability = classify(_fractional_offset(current, reference),
                             stability_spec)
        per_moment.append(stability)
        if per_moment_prefix:
            name = f"{per_moment_prefix}{channel.key}"
            if name in frame.columns:
                report.notes.append(f"Overwrote existing column {name!r}.")
            frame[name] = stability
            mark_verdicts(dataset, name)

        # Nominal agreement: the *mean* against what the system description
        # declares. A property of the survey, broadcast so it exports.
        nominal = _nominal_current(dataset, channel)
        offset = (float((reference - nominal) / abs(nominal))
                  if nominal and np.isfinite(reference) else None)
        if offset is not None:
            # Written whether or not it is judged, so the comparison is legible
            # in an export rather than only in the report. Constant across the
            # survey by construction: it is a property of the mean, not of any
            # one sounding.
            if nominal_offset_prefix:
                name = f"{nominal_offset_prefix}{channel.key}"
                if name in frame.columns:
                    report.notes.append(f"Overwrote existing column {name!r}.")
                frame[name] = offset
            if nominal_spec is not None:
                nominal_verdicts.append(
                    classify(np.full(len(frame), offset), nominal_spec))

        jumps = detect_jumps(current, flights, jump_fraction)
        moments_report[str(channel)] = {
            "column": column,
            "reference": reference if np.isfinite(reference) else None,
            "nominal": nominal,
            "nominal_offset_fraction": offset,
            "counts": tier_counts(stability),
            "jumps": int(jumps.sum()),
            "jump_indices": [int(i) for i in np.flatnonzero(jumps)[:50]],
            "current": summarize(current),
        }

    if not per_moment:
        report.notes.append("No moment had a usable current column.")
        return dataset, report

    stability_verdict = _worst(per_moment)
    nominal_verdict = _worst(nominal_verdicts) if nominal_verdicts else None

    if nominal_column and nominal_verdict is not None:
        if nominal_column in frame.columns:
            report.notes.append(f"Overwrote existing column {nominal_column!r}.")
        frame[nominal_column] = nominal_verdict
        mark_verdicts(dataset, nominal_column)
    elif nominal_spec is None:
        report.notes.append(
            "No nominal thresholds given, so agreement with the declared nominal "
            "is measured and left unjudged. Supply nominal_good_fraction and "
            "nominal_marginal_fraction to classify it."
        )
    else:
        report.notes.append(
            "The system description declares no nominal current; agreement with "
            "it is not judged."
        )

    # Stability alone. A steady transmitter that sits away from its declared
    # nominal produces self-consistent data, and combining the two here would
    # let a description error fail a survey that flew correctly.
    combined = stability_verdict
    if output_column in frame.columns:
        report.notes.append(f"Overwrote existing column {output_column!r}.")
    frame[output_column] = combined
    mark_verdicts(dataset, output_column)

    report.verdict_column = output_column
    # Equipment, not flying: an unstable transmitter is a maintenance
    # conversation, and a nominal that disagrees is a description problem.
    # Neither is a reason to reflight a line.
    report.finding = "instrument"

    counts = tier_counts(combined)
    report.soundings_affected = counts[OUT_OF_SPEC]
    report.distributions = {
        "verdicts": counts,
        "stability": tier_counts(stability_verdict),
        "thresholds": {
            "stability": describe(stability_spec, "fraction of mean"),
            # Absent rather than empty when unset, so a report says plainly that
            # nominal was measured and not judged.
            "nominal": (describe(nominal_spec, "fraction of nominal")
                        if nominal_spec is not None else None),
        },
        "per_moment": moments_report,
    }
    if nominal_verdict is not None:
        report.distributions["nominal_agreement"] = tier_counts(nominal_verdict)

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

    if flight_column is not None:
        report.distributions["per_flight"] = aggregate.per_flight(
            combined, flights, lines, easting, northing,
            gap_tolerance=gap_tolerance)
    report.distributions["per_line"] = aggregate.per_line(
        combined, lines, easting, northing, gap_tolerance=gap_tolerance)


    log.info("%s: %d soundings across %d moment(s), %d Good / %d Marginal / "
             "%d Out of Spec", STEP, len(frame), len(per_moment),
             counts[GOOD], counts[MARGINAL], counts[OUT_OF_SPEC])
    return dataset, report


def _nominal_current(dataset, channel) -> float | None:
    """The current the system description says this moment runs at, in amps.

    ``TxApproximateCurrent``, declared per channel in every system description
    to hand — and varying from a few amps to a few hundred between instruments,
    which is exactly why nothing here may hardcode a level.
    """
    system = dataset.system
    if system is None:
        return None
    block = getattr(system, "gex_dict", {}).get(f"Channel{channel.index}", {})
    value = block.get("TxApproximateCurrent")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
