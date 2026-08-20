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
Two implementations are two schemas. They may name columns differently, order
channels differently, keep different gates, and cull different soundings. So
nothing here matches by column name:

**Soundings** by timestamp where there is one, else by position on the ground,
else — only if both sides have the same count — by order, which is said out loud
because it is an assumption rather than a match.

**Channels** by their gate times, which is what a channel physically *is*. An
instrument's second moment is its second moment whether the file calls it
``ch2``, ``HM`` or something else.

**Gates** by center time within a tolerance, so an implementation that dropped
three gates is compared over the ones it kept rather than shifted against the
other side and declared wrong everywhere.

**Values are compared only where both sides kept the datum.** Underneath a mask
an implementation may leave anything at all, so comparing there measures nothing.
The masks themselves are compared as their own quantity, which is where "this one
culled more" belongs.
"""

from __future__ import annotations

import logging

import numpy as np

from ..pipeline import StepReport

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

#: The quantities compared, named semantically rather than by column, because the
#: two sides may not agree on a column name.
DATA = "data"
UNCERTAINTY = "uncertainty"
MASK = "mask"
QUANTITIES = (DATA, UNCERTAINTY, MASK)

#: How closely gate center times must agree to be the same gate.
GATE_RTOL = 1e-3

#: How close two soundings must be on the ground to be the same sounding, in CRS
#: units, when there is no timestamp to match on.
POSITION_TOLERANCE = 0.5


# ── Aligning the two sides ───────────────────────────────────────────────────

def _sounding_keys(dataset, position_tolerance):
    """What identifies each sounding, most trustworthy form first.

    A timestamp is an identity: the instrument recorded it, and it survives
    re-export, renaming and reordering. Position is a near-identity, good to
    whatever tolerance the two sides round coordinates to. Order is neither, so
    it is used only as a last resort and always said out loud.
    """
    frame = dataset.flightlines
    xyz = dataset.survey.xyz

    stamp = xyz.get_column("timestamp")
    if stamp and stamp in frame.columns:
        values = frame[stamp].to_numpy(dtype=float)
        if np.isfinite(values).all():
            # Microseconds, from days. Two implementations that both round trip
            # through a text format should agree to well within that.
            return [round(float(v) * 86_400e6) for v in values], "timestamp"

    if xyz.x_column in frame.columns and xyz.y_column in frame.columns:
        x = frame[xyz.x_column].to_numpy(dtype=float)
        y = frame[xyz.y_column].to_numpy(dtype=float)
        if np.isfinite(x).all() and np.isfinite(y).all():
            grid = float(position_tolerance)
            return [(round(a / grid), round(b / grid)) for a, b in zip(x, y)], \
                "position"

    return None, None


def _align_soundings(candidate, baseline, position_tolerance, report):
    """Row indices into each side that refer to the same soundings."""
    keys_c, how = _sounding_keys(candidate, position_tolerance)
    keys_b, how_b = _sounding_keys(baseline, position_tolerance)

    if keys_c is None or keys_b is None or how != how_b:
        n_c, n_b = len(candidate.flightlines), len(baseline.flightlines)
        if n_c != n_b:
            raise ValueError(
                f"The two sides share no timestamp or position to match "
                f"soundings on, and hold different numbers of them "
                f"({n_c} and {n_b}). There is no alignment to fall back to."
            )
        report.notes.append(
            "Neither side carries a timestamp or coordinates both could be "
            "matched on, so soundings are paired BY ORDER. That is an "
            "assumption, not a match: if either implementation reordered or "
            "dropped a sounding, every comparison below is against the wrong "
            "row and nothing in the numbers would say so."
        )
        rows = np.arange(n_c)
        return rows, rows, "order"

    index_b = {}
    for row, key in enumerate(keys_b):
        index_b.setdefault(key, []).append(row)

    pairs, unmatched_c = [], 0
    for row, key in enumerate(keys_c):
        rows = index_b.get(key)
        if rows:
            pairs.append((row, rows.pop(0)))
        else:
            unmatched_c += 1
    unmatched_b = sum(len(rows) for rows in index_b.values())

    if not pairs:
        raise ValueError(
            f"No sounding in the candidate matched one in the baseline by "
            f"{how}. The two sides are not the same survey, or the tolerance "
            f"is too tight."
        )
    if unmatched_c or unmatched_b:
        report.notes.append(
            f"{unmatched_c} sounding(s) exist only in the candidate and "
            f"{unmatched_b} only in the baseline, matched by {how}. Those are "
            f"not compared — a sounding one side dropped is a difference in "
            f"what was kept, which the mask comparison measures, not a "
            f"difference in a value."
        )

    rows_c = np.array([p[0] for p in pairs])
    rows_b = np.array([p[1] for p in pairs])
    return rows_c, rows_b, how


def _align_channels(candidate, baseline, report):
    """Channel pairs, matched on gate times rather than on names.

    A channel *is* its gate times. Two implementations may call the second
    moment ``ch2``, ``HM`` or nothing at all, and the times settle it.
    """
    remaining = list(baseline.channels)
    pairs, unmatched = [], []

    for channel in candidate.channels:
        times = _axis(channel)
        best, score = None, 0
        for other in remaining:
            overlap = _gate_overlap(times, _axis(other))
            if overlap is not None and len(overlap[0]) > score:
                best, score = other, len(overlap[0])
        if best is None:
            # No axis on either side, or nothing in common. Fall back to the
            # index the instrument numbers them by.
            best = next((o for o in remaining if o.index == channel.index), None)
            if best is not None and times is None and _axis(best) is None:
                report.notes.append(
                    f"{channel} was matched to {best} by channel index, since "
                    f"neither side carries gate times to match on.")
        if best is None:
            unmatched.append(str(channel))
            continue
        remaining.remove(best)
        pairs.append((channel, best))

    if unmatched:
        report.notes.append(
            f"The baseline has no channel matching {', '.join(unmatched)}, so "
            f"they are not compared.")
    if remaining:
        report.notes.append(
            f"The candidate has no channel matching "
            f"{', '.join(str(c) for c in remaining)} in the baseline.")
    return pairs


def _axis(channel):
    values = getattr(channel.axis, "values", None)
    if values is None:
        return None
    array = np.asarray(values, dtype=float).ravel()
    return array if array.size else None


def _gate_overlap(times_c, times_b, rtol=GATE_RTOL):
    """Column indices into each side naming the same gates."""
    if times_c is None or times_b is None:
        return None
    cols_c, cols_b = [], []
    used = set()
    for i, t in enumerate(times_c):
        if not np.isfinite(t):
            continue
        close = np.nonzero(np.isclose(times_b, t, rtol=rtol, atol=0.0))[0]
        for j in close:
            if int(j) not in used:
                cols_c.append(i)
                cols_b.append(int(j))
                used.add(int(j))
                break
    if not cols_c:
        return None
    return np.array(cols_c), np.array(cols_b)


# ── Comparing one quantity ───────────────────────────────────────────────────

def _key_for(channel, quantity):
    return {DATA: channel.data_key, UNCERTAINTY: channel.std_key,
            MASK: channel.inuse_key}[quantity]


def _block(dataset, channel, quantity, rows, cols):
    key = _key_for(channel, quantity)
    if not key or key not in dataset.layer_data:
        return None
    array = dataset.layer_data[key].to_numpy(dtype=float)
    return array[np.ix_(rows, cols)]


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

    rows_c, rows_b, matched_by = _align_soundings(
        candidate, baseline, position_tolerance, report)
    pairs = _align_channels(candidate, baseline, report)

    results, outcomes = {}, []
    for channel_c, channel_b in pairs:
        width_c = candidate.layer_data[channel_c.data_key].shape[1]
        width_b = baseline.layer_data[channel_b.data_key].shape[1]

        overlap = _gate_overlap(_axis(channel_c), _axis(channel_b))
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
            block = _block(dataset, channel, MASK, rows, cols)
            if block is not None:
                kept = np.abs(block) > 0
                live = kept if live is None else (live & kept)

        for quantity in quantities:
            block_c = _block(candidate, channel_c, quantity, rows_c, cols_c)
            block_b = _block(baseline, channel_b, quantity, rows_b, cols_b)
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
