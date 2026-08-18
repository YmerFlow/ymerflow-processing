"""Bridging per-sounding verdicts to the summaries a decision is made from.

Classification happens per sounding. Decisions are made at two coarser grains, and
``03`` asks for both:

**Per line** — the reflight decision. A line is the unit that gets reflown.

**Per flight** — the acquisition decision, and the unit cross-flight drift is a
series over. This grain matters more than it first appears because **a delivery
usually contains three or four flights**, so one QC run over one delivered file
produces several flights' worth of results. That makes ``03``'s two cadences —
per flight, and the per-day rollup — fall out of a single run rather than needing
separate invocations.

*Which* rule turns a per-sounding verdict into a pass or fail is an **open
decision** in ``03`` — fraction Out of Spec, longest contiguous run, or both, and
whether Marginal counts fractionally. The numbers come from the survey contract.

So this deliberately **measures and does not decide**. It computes both rules and
reports them; nothing here fails a line or a flight. That keeps the open decision
open, and it is what the specification recommends, for a good reason: three
hundred scattered bad soundings and one continuous 3 km stretch are very
different problems, and a fraction alone cannot tell them apart.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..dataset import find_flight_column  # noqa: F401  (re-exported for callers)
from .thresholds import (
    GOOD, MARGINAL, OUT_OF_SPEC, TIER_POINTS, TIERS, is_tier,
)

#: The measure each tier's fraction is written under, appended to a grain prefix
#: to give ``ql_good``, ``qf_marg``, ``qd_oos`` and so on. Abbreviated so that
#: every one of them clears the shapefile's ten-character field-name limit with
#: the longest prefix attached.
MEASURE_TIER = {"good": GOOD, "marg": MARGINAL, "oos": OUT_OF_SPEC}


# ── Contiguous runs ──────────────────────────────────────────────────────────

def longest_run(flags) -> int:
    """Length of the longest unbroken stretch of ``True`` in ``flags``.

    Soundings are taken in the order they sit in ``flightlines``, which is
    acquisition order — so a run here is a contiguous stretch of flying, which is
    what makes it the useful complement to a plain fraction.
    """
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return 0
    # Walk the boundaries rather than the samples: run lengths are the gaps
    # between rising and falling edges of the flag.
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return int((edges[1::2] - edges[::2]).max())


def _run_distance(flags, easting, northing) -> float | None:
    """Ground length of the longest contiguous ``True`` run, in CRS units.

    Reported alongside the count because a run of soundings means nothing to a
    client without a distance — "1.8 km out of spec" is a conversation, "412
    soundings" is not. Returns ``None`` when there is no position to measure
    against.
    """
    if easting is None or northing is None:
        return None
    flags = np.asarray(flags, dtype=bool)
    if not flags.any():
        return 0.0

    x = np.asarray(easting, dtype=float)
    y = np.asarray(northing, dtype=float)
    padded = np.concatenate(([False], flags, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])

    longest = 0.0
    for start, stop in zip(edges[::2], edges[1::2]):
        if stop - start < 2:
            continue    # a single sounding has no length
        step = np.hypot(np.diff(x[start:stop]), np.diff(y[start:stop]))
        longest = max(longest, float(np.nansum(step)))
    return longest


def close_gaps(flags, easting, northing, tolerance: float):
    """Bridge short in-spec stretches sitting inside a longer out-of-spec one.

    A contract that says "out of spec for 1 km or more" is describing a stretch
    of flying, and a single sounding that happens to dip back into spec does not
    end it. Without this, one such sounding halves the measured run: a 1,190 m
    breach with one good sounding in the middle measures as two runs of 590 m,
    and against a 1 km rule the breach disappears — while the *fraction* out of
    spec barely moves, which is why both rules are reported.

    Morphological closing: an interior gap is filled when the ground distance
    from the last flagged sounding to the next one is shorter than ``tolerance``.
    Distance rather than sounding count, because contracts are written in
    metres and sounding spacing varies with speed.

    Leading and trailing unflagged stretches are never filled — there is no run
    on the far side of them to join.
    """
    flags = np.asarray(flags, dtype=bool)
    if tolerance <= 0 or not flags.any():
        return flags

    x = np.asarray(easting, dtype=float)
    y = np.asarray(northing, dtype=float)
    # Distance travelled along the line, so a gap's length is a difference.
    travelled = np.concatenate(([0.0], np.nancumsum(np.hypot(np.diff(x),
                                                             np.diff(y)))))

    closed = flags.copy()
    flagged = np.flatnonzero(flags)
    for before, after in zip(flagged[:-1], flagged[1:]):
        if after == before + 1:
            continue                      # no gap
        if travelled[after] - travelled[before] < tolerance:
            closed[before:after] = True
    return closed


def _longest_run_within(rows, segments, gap_tolerance: float = 0.0):
    """Longest Out of Spec run, forbidden from crossing a segment boundary.

    A run is a stretch of *continuous flying*, so it must not span a change of
    line. Within one line that is automatic. Across a flight it is not: a flight
    contains many lines, the aircraft turns between them, and two bad stretches
    on either side of a turn are two problems rather than one long one. Left
    unsegmented, a flight-level run would silently join them and overstate the
    worst stretch.

    ``gap_tolerance`` bridges short in-spec stretches before measuring; see
    :func:`close_gaps`. The raw measurement is always returned alongside, so
    closing never hides what was actually flown.

    Returns ``(soundings, distance, closed_soundings, closed_distance)``.
    """
    has_xy = "x" in rows
    parts = ([rows] if segments is None
             else [p for _, p in rows.groupby(segments, sort=False, observed=True)])

    worst = worst_distance = 0
    worst_closed, worst_closed_distance = 0, 0.0
    if not has_xy:
        worst_distance = worst_closed_distance = None

    for part in parts:
        flags = is_tier(part["verdict"], OUT_OF_SPEC)
        x = part["x"].to_numpy() if has_xy else None
        y = part["y"].to_numpy() if has_xy else None

        worst = max(worst, longest_run(flags))
        if has_xy:
            worst_distance = max(worst_distance, _run_distance(flags, x, y))

        # Closing needs position to measure a gap, so without it the closed
        # measurement is simply the raw one.
        bridged = (close_gaps(flags, x, y, gap_tolerance)
                   if has_xy and gap_tolerance > 0 else flags)
        worst_closed = max(worst_closed, longest_run(bridged))
        if has_xy:
            worst_closed_distance = max(worst_closed_distance,
                                        _run_distance(bridged, x, y))
    return worst, worst_distance, worst_closed, worst_closed_distance


# ── Rollups ──────────────────────────────────────────────────────────────────

def _frame(verdicts, groups, easting, northing, segments=None):
    frame = pd.DataFrame({
        "group": np.asarray(groups),
        "verdict": np.asarray(verdicts, dtype=float),
    })
    if easting is not None and northing is not None:
        frame["x"] = np.asarray(easting, dtype=float)
        frame["y"] = np.asarray(northing, dtype=float)
    if segments is not None:
        frame["segment"] = np.asarray(segments)
    return frame


def _rollup(verdicts, groups, easting=None, northing=None,
            segments=None, gap_tolerance: float = 0.0) -> dict[str, dict]:
    """Both rules, per group, with no pass/fail applied."""
    frame = _frame(verdicts, groups, easting, northing, segments)

    summary: dict[str, dict] = {}
    for group, rows in frame.groupby("group", sort=True, observed=True):
        classified = int(rows["verdict"].notna().sum())
        run_soundings, run_distance, closed_soundings, closed_distance = (
            _longest_run_within(rows,
                                rows["segment"] if segments is not None else None,
                                gap_tolerance))

        summary[str(group)] = {
            "n_soundings": int(len(rows)),
            "n_classified": classified,
            "counts": {tier: int(is_tier(rows["verdict"], tier).sum())
                       for tier in TIERS},
            # Fractions are of *classified* soundings: a line half of which was
            # never measured should not read as half in spec.
            "fraction_out_of_spec": (
                float(is_tier(rows["verdict"], OUT_OF_SPEC).sum() / classified)
                if classified else None
            ),
            "fraction_not_good": (
                float((~is_tier(rows["verdict"], GOOD)
                       & rows["verdict"].notna()).sum() / classified)
                if classified else None
            ),
            "longest_out_of_spec_run": run_soundings,
            "longest_out_of_spec_run_distance": run_distance,
            # The same run with short in-spec stretches bridged. Equal to the
            # raw figures when no tolerance was given.
            "longest_out_of_spec_run_closed": closed_soundings,
            "longest_out_of_spec_run_closed_distance": closed_distance,
            "gap_tolerance": gap_tolerance,
        }
    return summary


def per_line(verdicts, lines, easting=None, northing=None,
             gap_tolerance: float = 0.0) -> dict[str, dict]:
    """Both line-level rules, per line, with no pass/fail applied.

    For each line: how many soundings fell in each tier, the fraction Out of
    Spec, the fraction that is not Good — Marginal counted whole, since counting
    it fractionally needs a weight the contract has not supplied — and the
    longest contiguous Out of Spec run in soundings and on the ground.
    """
    return _rollup(verdicts, lines, easting, northing,
                   gap_tolerance=gap_tolerance)


def per_flight(verdicts, flights, lines=None, easting=None,
               northing=None, gap_tolerance: float = 0.0) -> dict[str, dict]:
    """The same measurements, grouped by flight.

    The grain cross-flight drift is a series over, and the one a delivery
    containing several flights has to be broken down into.

    ``lines`` should be supplied where it is known. It does not group anything —
    it bounds the contiguous-run measurement, so a run cannot span the turn
    between two lines. Without it the flight is treated as one continuous
    sequence, which overstates the worst stretch.
    """
    return _rollup(verdicts, flights, easting, northing, segments=lines,
                   gap_tolerance=gap_tolerance)


def grain_measures(verdicts, groups, segments=None, easting=None, northing=None,
                   gap_tolerance: float = 0.0):
    """Each group's tier fractions and worst run, broadcast onto its soundings.

    The measures ``03`` asks for, at whatever grain the caller groups by — line,
    flight, or the delivery as a whole — put where a client can see them. They
    live in the report as well, but a report is not what someone opens in a GIS,
    and the export is the artifact that leaves the building.

    **A fraction per tier, not just the failing one.** An earlier form wrote only
    the out-of-spec fraction, which answers "how bad" but not "how good" — a line
    reading 0.02 could be 98% Good or 98% Marginal, and those are different
    deliveries. All three are written so the split is visible without going back
    to the report.

    They are fractions of *classified* soundings, so the three sum to 1 wherever
    anything was classified at all. That makes them mutually derivable, which is
    a fair objection to writing all three — but the arithmetic has to happen
    somewhere, and a GIS is the worst place to ask for it.

    Broadcast rather than aggregated away because a vector export has one row
    per sounding: repeating the group's figure on each of its soundings is what
    lets a map be colored by it.

    ``segments`` bounds the contiguous run, and should be the line identity at
    every grain coarser than a line — a run is a stretch of continuous flying,
    so it must not span the turn between two lines.

    **Deliberately not a verdict.** An earlier form wrote the worst verdict found
    anywhere on the line, which made one bad sounding in five hundred read
    exactly like a line flown out of spec throughout — the crudest possible rule,
    wearing the same vocabulary as the per-sounding verdict and so certain to be
    read as a pass or fail. These are measurements; turning them into a verdict
    needs thresholds, and those come from the contract.

    Returns a mapping of measure name to an array the length of ``verdicts``:
    one entry per tier in :data:`MEASURE_TIER`, plus ``run`` and ``n``. A mapping
    rather than a tuple because the set of measures is now long enough that
    positional unpacking would silently reorder if a tier were ever added.
    """
    summary = _rollup(verdicts, groups, easting, northing, segments=segments,
                      gap_tolerance=gap_tolerance)
    keys = pd.Series(np.asarray(groups)).astype(str)

    def pick(getter):
        values = keys.map(lambda k: getter(summary.get(k, {})))
        return pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

    def tier_fraction(tier):
        def get(group):
            classified = group.get("n_classified")
            if not classified:
                return None
            return group["counts"][tier] / classified
        return get

    measures = {name: pick(tier_fraction(tier))
                for name, tier in MEASURE_TIER.items()}
    # The denominator, so three fractions summing to 1 cannot be mistaken for a
    # fully classified line when half of it was never measured.
    measures["n"] = pick(lambda g: g.get("n_classified"))
    measures["run"] = pick(
        lambda g: g.get("longest_out_of_spec_run_closed_distance"))
    return measures


def apply_rule(fraction, run, max_fraction=None, max_run=None):
    """Turn the two measures into a verdict, where a contract supplies one.

    Out of Spec when **either** measure trips: they catch different problems —
    degradation scattered across a line, versus one continuous stretch — so
    requiring both would miss whichever case the contract was written for. A
    single long breach on an otherwise clean line is usually the reflight case,
    and a fraction alone would not see it.

    Two states rather than three. A contract rule says whether the line is
    acceptable; grading it Marginal as well would need a second pair of
    thresholds nobody has stated.

    Returns ``None`` when neither threshold is given — no rule, no verdict.
    """
    if max_fraction is None and max_run is None:
        return None

    fraction = np.asarray(fraction, dtype=float)
    run = np.asarray(run, dtype=float)
    failed = np.zeros(fraction.shape, dtype=bool)
    if max_fraction is not None:
        failed |= np.nan_to_num(fraction, nan=0.0) > max_fraction
    if max_run is not None:
        failed |= np.nan_to_num(run, nan=0.0) > max_run

    return np.where(failed, TIER_POINTS[OUT_OF_SPEC],
                    TIER_POINTS[GOOD]).astype(np.float32)
