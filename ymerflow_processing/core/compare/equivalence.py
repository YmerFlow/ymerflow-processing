"""Two implementations over one input, compared per datum.

``02``'s fourth use for the comparison shape, and the one it calls *"precisely the
acceptance criterion for the rebuild itself"*:

    Running the old and new implementations over the same input and diffing the
    outputs is legitimate — comparing behavior is not copying expression — and
    having a runner for it makes that systematic rather than ad hoc.

``consumes: datasets``, per ``02``'s table. No report contains per-datum arrays,
and per-datum is the whole question.

Bit-for-bit is the wrong bar
-----------------------------
This is the design decision the rest follows from. A predecessor has defects; an
implementation that agrees with it everywhere has **reproduced its bugs**, which
is a failure dressed as a pass.

So the bar is *equivalent where the baseline is right, better where it is not,
and documented wherever they deliberately differ* — and that third clause needs
somewhere to live. A report whose vocabulary is only *matches* and *does not
match* has nowhere to put a deliberate improvement, so someone widens a tolerance
until the improvement disappears into it, and the comparison stops meaning
anything.

Four outcomes, then, not two:

``agrees``
    Within tolerance.
``expected``
    Differs, and a declared expectation covers it. **A pass.**
``differs``
    Differs with nothing declared. The finding.
``absent``
    One side has no such quantity — not a difference, a gap in coverage.

A declared difference can still fail
-------------------------------------
An expectation may name a ``direction``, and that is what makes "better" testable
rather than assertable. Declaring that a new uncertainty should be *smaller* than
a baseline that clipped its denominator means a candidate that comes out larger
is a finding — even though the difference was declared. Without the direction the
declaration would excuse a regression as readily as an improvement.

An expectation nothing exercised is also reported. It means the declaration is
stale, or the comparison never reached the quantity it describes, and both are
worth knowing before someone reads the absence as agreement.

Alignment cannot assume a shared schema
----------------------------------------
Two implementations are two schemas, so nothing here matches by column name —
see :mod:`~ymerflow_processing.core.compare.align`, which every dataset-consuming
comparison shares.

**Values are compared only where both sides kept the datum.** Underneath a mask
an implementation may leave anything at all, so comparing there measures nothing.
The masks themselves are compared as their own quantity, which is where "this one
culled more" belongs.
"""

from __future__ import annotations

import logging

import numpy as np

from ..pipeline import StepReport
from .align import (
    DATA, GATE_RTOL, MASK, POSITION_TOLERANCE, QUANTITIES, UNCERTAINTY,
    align_channels, align_soundings, axis_of, block, gate_overlap,
)

log = logging.getLogger(__name__)

STEP = "implementation_equivalence"

#: What this check needs. Per ``02``: reports are the default and the cheap path;
#: this one cannot use them, because no report carries per-datum arrays.
CONSUMES = "datasets"

#: The four outcomes. Two would not be enough — see the module docstring.
AGREES = "agrees"
EXPECTED = "expected"
DIFFERS = "differs"
ABSENT = "absent"

# ── Comparing one quantity ───────────────────────────────────────────────────

def _within(candidate, baseline, relative, absolute):
    """Agreement per datum. Two NaNs agree; a NaN against a number does not."""
    both_absent = ~np.isfinite(candidate) & ~np.isfinite(baseline)
    with np.errstate(invalid="ignore"):
        close = np.abs(candidate - baseline) <= (
            absolute + relative * np.abs(baseline))
    return both_absent | (close & np.isfinite(candidate) & np.isfinite(baseline))


def _metrics(candidate, baseline, agreement):
    """How far apart, and where the worst of it is."""
    with np.errstate(invalid="ignore", divide="ignore"):
        absolute = np.abs(candidate - baseline)
        relative = np.where(np.abs(baseline) > 0, absolute / np.abs(baseline),
                            np.nan)

    comparable = np.isfinite(absolute)
    result = {
        "n": int(candidate.size),
        "n_comparable": int(comparable.sum()),
        "n_differing": int((~agreement).sum()),
        "fraction_differing": (float((~agreement).sum() / candidate.size)
                               if candidate.size else 0.0),
        "max_absolute": (float(np.nanmax(absolute)) if comparable.any()
                         else None),
        "max_relative": (float(np.nanmax(relative))
                         if np.isfinite(relative).any() else None),
    }

    # A datum present on one side and absent on the other differs by no
    # *amount* — the difference is one of presence, and it has no magnitude to
    # be the worst of. Counted separately rather than folded in, because "one
    # side has nothing here" is a different finding from "these are 3% apart",
    # and because taking a maximum over a set with no finite member is how this
    # crashed on exactly the input it most needed to survive.
    presence = ~agreement & ~comparable
    if presence.any():
        result["n_present_on_one_side_only"] = int(presence.sum())

    measurable = ~agreement & comparable
    if measurable.any():
        worst = np.unravel_index(
            int(np.argmax(np.where(measurable, absolute, -np.inf))),
            absolute.shape)
        result["worst_at"] = {"sounding": int(worst[0]), "gate": int(worst[1]),
                              "candidate": float(candidate[worst]),
                              "baseline": float(baseline[worst])}
    return result


def _direction_holds(candidate, baseline, agreement, direction):
    """Whether a declared improvement actually went the declared way.

    This is what makes "better where the baseline is not right" testable rather
    than assertable. A declaration with no direction excuses a regression as
    readily as an improvement.
    """
    if direction in (None, "any"):
        return True, None
    differing = ~agreement & np.isfinite(candidate) & np.isfinite(baseline)
    if not differing.any():
        return True, None

    if direction == "smaller":
        wrong = differing & (candidate > baseline)
    elif direction == "larger":
        wrong = differing & (candidate < baseline)
    else:
        raise ValueError(
            f"direction must be 'smaller', 'larger' or 'any'; got "
            f"{direction!r}.")
    return not wrong.any(), int(wrong.sum())


def _expectation_for(expectations, channel, quantity):
    """The declaration covering this quantity, if one does.

    Matched on the semantic quantity and optionally the channel — never on a
    column name, since the two sides may not share one.
    """
    for expectation in expectations:
        if expectation.get("quantity") != quantity:
            continue
        named = expectation.get("channel")
        if named in (None, "", channel.key, channel.name, str(channel)):
            return expectation
    return None


# ── The check ────────────────────────────────────────────────────────────────

def implementation_equivalence(
    items,
    reference=None,
    relative: float = 1e-6,
    absolute: float = 0.0,
    justification: str | None = None,
    expected_differences=(),
    quantities=QUANTITIES,
    position_tolerance: float = POSITION_TOLERANCE,
):
    """Compare a candidate implementation against a baseline, per datum.

    :param items: The candidate's output — one dataset, or a one-item sequence.
        Per ``02``'s table, the candidate is ``items`` and the baseline is
        ``reference``.
    :param reference: The baseline's output over the same input. Required.
    :param relative: Fractional agreement required, against the baseline value.
    :param absolute: Absolute agreement required, added to the relative part.
        Needed where the baseline can be zero, since a fraction of zero is zero.
    :param justification: Why that tolerance is the right one. Carried into the
        report because a tolerance without a reason is a number someone tuned
        until the comparison passed.
    :param expected_differences: Declarations of where the two are *meant* to
        differ. Each is ``{quantity, channel?, reason, direction?}``, where
        ``direction`` is ``"smaller"``, ``"larger"`` or ``"any"``. A covered
        difference is a pass; one whose direction is violated is not.
    :param quantities: Which of ``data``, ``uncertainty``, ``mask`` to compare.
    :param position_tolerance: How close two soundings must be, in CRS units, to
        be the same sounding when there is no timestamp.

    Returns a :class:`StepReport`.
    """
    report = StepReport(step=STEP)

    if reference is None:
        raise ValueError(
            "implementation_equivalence needs a reference: the baseline "
            "implementation's output over the same input. Without one there is "
            "nothing to be equivalent to.")

    candidates = list(items) if isinstance(items, (list, tuple)) else [items]
    if len(candidates) != 1:
        raise ValueError(
            f"implementation_equivalence compares one candidate against one "
            f"baseline; got {len(candidates)} candidates. Run it once per pair "
            f"— the register in the migration record is per entry point.")
    candidate, baseline = candidates[0], reference

    for side, dataset in (("candidate", candidate), ("baseline", baseline)):
        if not hasattr(dataset, "layer_data"):
            raise TypeError(
                f"The {side} is a {type(dataset).__name__}, not a dataset. This "
                f"check declares consumes: datasets — no report carries the "
                f"per-datum arrays it compares.")

    expectations = [dict(e) for e in expected_differences]
    for expectation in expectations:
        if expectation.get("quantity") not in QUANTITIES:
            raise ValueError(
                f"An expected difference must name a quantity from "
                f"{QUANTITIES}; got {expectation.get('quantity')!r}. Quantities "
                f"are semantic rather than column names, because the two sides "
                f"may not share one.")
        if not expectation.get("reason"):
            raise ValueError(
                "An expected difference must carry a reason. The point of "
                "declaring one is that somebody reading the migration record "
                "learns why the two differ.")
        expectation["exercised"] = False

    if not len(candidate.flightlines) or not len(baseline.flightlines):
        report.notes.append(
            "One side has no soundings, so there is nothing to compare. That is "
            "worth recording rather than passing: an empty result agrees with "
            "everything.")
        report.distributions = {"verdict": ABSENT, "quantities": {}}
        return report

    rows_c, rows_b, matched_by = align_soundings(
        candidate, baseline, position_tolerance, report)
    pairs = align_channels(candidate, baseline, report)

    results, outcomes = {}, []
    for channel_c, channel_b in pairs:
        width_c = candidate.layer_data[channel_c.data_key].shape[1]
        width_b = baseline.layer_data[channel_b.data_key].shape[1]

        overlap = gate_overlap(axis_of(channel_c), axis_of(channel_b))
        if overlap is None:
            width = min(width_c, width_b)
            cols_c = cols_b = np.arange(width)
        else:
            # A system description can name more gates than the arrays hold —
            # the method layer warns when it does — so the axis is a claim about
            # the columns rather than a count of them, and a gate it names past
            # the end of the data cannot be compared.
            cols_c, cols_b = overlap
            fits = (cols_c < width_c) & (cols_b < width_b)
            if not fits.all():
                report.notes.append(
                    f"{channel_c}: the system description names "
                    f"{int((~fits).sum())} gate(s) the arrays do not hold, so "
                    f"they are not compared.")
            cols_c, cols_b = cols_c[fits], cols_b[fits]
            if not cols_c.size:
                cols_c = cols_b = np.arange(min(width_c, width_b))

        label = str(channel_c)
        entry = {"baseline_channel": str(channel_b),
                 "gates_compared": int(len(cols_c))}

        # Underneath a mask an implementation may leave anything at all, so a
        # value comparison there measures nothing. The masks are compared as
        # their own quantity, which is where "this one culled more" belongs.
        live = None
        for side, dataset, channel, rows, cols in (
                ("c", candidate, channel_c, rows_c, cols_c),
                ("b", baseline, channel_b, rows_b, cols_b)):
            mask = block(dataset, channel, MASK, rows, cols)
            if mask is not None:
                kept = np.abs(mask) > 0
                live = kept if live is None else (live & kept)

        for quantity in quantities:
            block_c = block(candidate, channel_c, quantity, rows_c, cols_c)
            block_b = block(baseline, channel_b, quantity, rows_b, cols_b)
            if block_c is None or block_b is None:
                entry[quantity] = {
                    "outcome": ABSENT,
                    "note": (f"Only the "
                             f"{'baseline' if block_c is None else 'candidate'} "
                             f"carries this quantity."),
                }
                outcomes.append(ABSENT)
                continue

            if quantity != MASK and live is not None:
                # Compare only where both sides kept the datum.
                block_c = np.where(live, block_c, np.nan)
                block_b = np.where(live, block_b, np.nan)
            if quantity == MASK:
                # The mask's magnitude is provenance, not value: what is being
                # compared is which data each side kept.
                block_c, block_b = (np.abs(block_c) > 0).astype(float), \
                                   (np.abs(block_b) > 0).astype(float)

            agreement = _within(block_c, block_b, relative, absolute)
            measured = _metrics(block_c, block_b, agreement)

            if agreement.all():
                outcome = AGREES
            else:
                expectation = _expectation_for(expectations, channel_c, quantity)
                if expectation is None:
                    outcome = DIFFERS
                else:
                    expectation["exercised"] = True
                    held, wrong = _direction_holds(
                        block_c, block_b, agreement,
                        expectation.get("direction"))
                    outcome = EXPECTED if held else DIFFERS
                    measured["expectation"] = expectation["reason"]
                    if not held:
                        measured["direction_violated"] = wrong
                        report.notes.append(
                            f"{label} {quantity}: the difference was declared, "
                            f"but {wrong} datum(s) went the wrong way — "
                            f"'{expectation.get('direction')}' was expected. A "
                            f"declaration excuses a difference, not a "
                            f"regression.")
            measured["outcome"] = outcome
            entry[quantity] = measured
            outcomes.append(outcome)

        results[label] = entry

    stale = [e for e in expectations if not e["exercised"]]
    if stale:
        report.notes.append(
            f"{len(stale)} declared difference(s) were never exercised: "
            f"{'; '.join(e['reason'] for e in stale)}. Either the declaration "
            f"is stale or the comparison never reached what it describes — "
            f"read neither as agreement.")

    if DIFFERS in outcomes:
        verdict = "differs"
    elif EXPECTED in outcomes:
        verdict = "equivalent_with_expected_differences"
    elif AGREES in outcomes:
        verdict = "equivalent"
    else:
        verdict = ABSENT

    report.soundings_total = len(candidate.flightlines)
    report.soundings_affected = sum(
        1 for entry in results.values()
        for quantity in QUANTITIES
        if isinstance(entry.get(quantity), dict)
        and entry[quantity].get("outcome") == DIFFERS)
    report.distributions = {
        "verdict": verdict,
        "tolerance": {"relative": relative, "absolute": absolute,
                      "justification": justification},
        "alignment": {
            "soundings_matched_by": matched_by,
            "soundings_compared": int(len(rows_c)),
            "channels_compared": len(pairs),
        },
        "quantities": results,
        "expected_differences": expectations,
    }

    if justification is None:
        report.notes.append(
            "No justification was given for the tolerance. A tolerance without "
            "a reason is a number somebody tuned until the comparison passed, "
            "and the migration record is where the reason belongs.")
    if verdict == "equivalent_with_expected_differences":
        report.notes.append(
            "Equivalent where the two are meant to agree, and different only "
            "where a difference was declared. That is a pass: bit-for-bit "
            "agreement with a defective baseline would mean reproducing its "
            "defects.")

    log.info("%s: %s, %d sounding(s) matched by %s across %d channel(s)",
             STEP, verdict, len(rows_c), matched_by, len(pairs))
    return report
