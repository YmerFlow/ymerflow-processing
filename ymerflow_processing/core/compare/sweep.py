"""One parameter varied across a series of runs.

``02``'s third use: *"One parameter choice against another — the averaging-width
comparison, also by hand."*

The question is not which run is best. It is **how the output responds to the
parameter**, because the choice is nearly always a trade — a wider averaging
window suppresses more noise and resolves less, and the number worth having is
where one stops buying the other.

It discovers which parameter varied
------------------------------------
``dataset.history`` records every applied step with its parameters, so the sweep
reads the runs rather than being told about them. A caller who has just produced
five datasets should not have to restate the five numbers that produced them —
restating is how a report comes to describe a sweep that was never run.

Discovery also catches the mistake the check exists around: if *two* parameters
differ across the series, the response cannot be attributed to either, and that
is said rather than silently averaged over. Same for a series where nothing
varies at all.

The response is measured against the control
---------------------------------------------
Each run is compared to a ``reference`` — the control, per ``02``'s table — and
what is reported per run is how far the data moved from it, how much more or
less was culled, and any per-sounding column the caller names.

**Divergence from the control is the primary response** because it needs no
agreement about what "better" means. A noise estimate or a misfit is a better
answer where one exists, which is why any column can be named; but the divergence
curve exists for every sweep and every parameter.

Where the response flattens
----------------------------
The one derived number worth computing. A response curve that rises and then
levels says the parameter stopped doing anything past that point, which is
exactly the averaging-width answer: **beyond this width, nothing changes, so
take the cheapest.**

Reported as the first run whose step from its predecessor falls below a fraction
of the largest step in the series — relative rather than absolute, since the
units differ per response and an absolute threshold would need retuning for each.
Reported as an observation, never as a recommendation: the trade is the user's to
make, and a knee in one response is not a knee in the one they care about.
"""

from __future__ import annotations

import logging

import numpy as np

from ..pipeline import StepReport
from .align import (
    DATA, MASK, POSITION_TOLERANCE, UNCERTAINTY, align_channels,
    align_soundings, axis_of, block, gate_overlap,
)

log = logging.getLogger(__name__)

STEP = "parameter_sweep"

#: Per ``02``'s table: the response lives in the arrays, not in a report.
CONSUMES = "datasets"

#: A step below this fraction of the largest step counts as flat.
FLAT_FRACTION = 0.1


# ── Finding out what varied ──────────────────────────────────────────────────

def _parameters(dataset):
    """Every ``(step, position, name) -> value`` a run recorded."""
    found = {}
    for position, entry in enumerate(getattr(dataset, "history", []) or []):
        for name, value in (entry.get("params") or {}).items():
            found[(entry["step"], position, name)] = value
    return found


def _hashable(value):
    """Something comparable, for a parameter that may be a list or a mapping."""
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set)):
        return tuple(_hashable(v) for v in value)
    return value


def _varied(runs, report):
    """Which parameter differs across the series, discovered rather than told.

    Returns ``(key, values)`` or ``(None, None)``. More than one varying
    parameter is refused: a response that cannot be attributed to a parameter is
    not a sweep, and averaging over the ambiguity would produce a curve that
    looks exactly like a real one.
    """
    seen = [_parameters(run) for run in runs]
    keys = set().union(*seen) if seen else set()

    varying = {}
    for key in keys:
        values = [entry.get(key) for entry in seen]
        distinct = {_hashable(v) for v in values}
        if len(distinct) > 1:
            varying[key] = values

    if not varying:
        report.notes.append(
            "No parameter differs across the series, so there is nothing to "
            "sweep. Either the runs are identical or their pipelines did not "
            "record parameters — check that each was produced by the runner "
            "rather than by calling the step directly.")
        return None, None

    if len(varying) > 1:
        named = ", ".join(f"{step}.{name}" for step, _, name in sorted(varying))
        raise ValueError(
            f"More than one parameter varies across the series ({named}). A "
            f"response cannot be attributed to any of them, and a curve drawn "
            f"over the ambiguity would look exactly like a real one. Sweep one "
            f"parameter at a time."
        )

    key, values = next(iter(varying.items()))
    return key, values


# ── Measuring one run against the control ────────────────────────────────────

def _response(run, control, position_tolerance, quiet):
    """How far one run's data moved from the control, and what it kept."""
    rows_r, rows_c, matched_by = align_soundings(
        run, control, position_tolerance, quiet)
    pairs = align_channels(run, control, quiet)

    divergences, kept_run, kept_control, total = [], 0, 0, 0
    for channel_r, channel_c in pairs:
        width_r = run.layer_data[channel_r.data_key].shape[1]
        width_c = control.layer_data[channel_c.data_key].shape[1]
        overlap = gate_overlap(axis_of(channel_r), axis_of(channel_c))
        if overlap is None:
            cols_r = cols_c = np.arange(min(width_r, width_c))
        else:
            cols_r, cols_c = overlap
            fits = (cols_r < width_r) & (cols_c < width_c)
            cols_r, cols_c = cols_r[fits], cols_c[fits]
            if not cols_r.size:
                cols_r = cols_c = np.arange(min(width_r, width_c))

        live = None
        for dataset, channel, rows, cols in ((run, channel_r, rows_r, cols_r),
                                             (control, channel_c, rows_c,
                                              cols_c)):
            mask = block(dataset, channel, MASK, rows, cols)
            if mask is not None:
                here = np.abs(mask) > 0
                live = here if live is None else (live & here)
                if dataset is run:
                    kept_run += int(here.sum())
                else:
                    kept_control += int(here.sum())
                total += int(here.size)

        values_r = block(run, channel_r, DATA, rows_r, cols_r)
        values_c = block(control, channel_c, DATA, rows_c, cols_c)
        if values_r is None or values_c is None:
            continue
        if live is not None:
            values_r = np.where(live, values_r, np.nan)
            values_c = np.where(live, values_c, np.nan)

        # Median absolute fractional change, over data both runs kept. Median
        # rather than mean because one wild sounding should not decide the shape
        # of a response curve; fractional because a transient spans orders of
        # magnitude and an absolute change is dominated by the earliest gate.
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.abs(values_r - values_c) / np.abs(values_c)
        usable = fraction[np.isfinite(fraction)]
        if usable.size:
            divergences.append(float(np.median(usable)))

    return {
        "divergence_from_control": (float(np.mean(divergences))
                                    if divergences else None),
        "fraction_in_use": (kept_run / (total / 2) if total else None),
        "fraction_in_use_control": (kept_control / (total / 2) if total
                                    else None),
        "soundings_compared": int(len(rows_r)),
        "soundings_matched_by": matched_by,
    }


def _knee(values, responses, flat_fraction):
    """The first point past which the response stops moving.

    An observation, not a recommendation. A knee in one response is not a knee
    in the one someone cares about, and the trade — noise against resolution —
    is theirs to make.
    """
    usable = [(v, r) for v, r in zip(values, responses)
              if r is not None and np.isfinite(r)]
    if len(usable) < 3:
        return None

    steps = [abs(usable[i][1] - usable[i - 1][1])
             for i in range(1, len(usable))]
    largest = max(steps)
    if largest <= 0:
        return None

    for index, step in enumerate(steps):
        if step < flat_fraction * largest:
            return {
                "at": usable[index][0],
                "response_there": usable[index][1],
                "remaining_change": float(sum(steps[index:]) / largest),
            }
    return None


# ── The check ────────────────────────────────────────────────────────────────

def parameter_sweep(items, reference=None, response_column: str | None = None,
                    flat_fraction: float = FLAT_FRACTION,
                    position_tolerance: float = POSITION_TOLERANCE):
    """Trace how the output responds to one varied parameter.

    Parameters
    ----------
    items
        The runs, **ordered**. ``02`` makes ``items`` ordered and says a check
        may rely on it; a response curve is read in order.
    reference
        The control every run is measured against. Without one the first run is
        used, which is the conventional sweep — a series starting from the
        current setting.
    response_column : str | None
        A per-sounding column to trace alongside the divergence, such as a
        misfit or a noise estimate. Optional: divergence from the control
        exists for every sweep, while a better response measure exists only
        sometimes.
    flat_fraction : float
        A step below this fraction of the largest counts as flat, when looking
        for where the response levels off.
    position_tolerance : float
        How close two soundings must be, in CRS units, to be the same sounding
        when there is no timestamp.

    Returns
    -------
    tuple
        a :class:`StepReport`.

    **Nothing here is a recommendation** — the trade a sweep explores is the
    user's to make.
    """
    report = StepReport(step=STEP)

    runs = list(items)
    if len(runs) < 2:
        raise ValueError(
            f"A sweep needs at least two runs to have a response; got "
            f"{len(runs)}. For one run against one other, use version_diff.")

    for index, run in enumerate(runs):
        if not hasattr(run, "layer_data"):
            raise TypeError(
                f"Run {index} is a {type(run).__name__}, not a dataset. This "
                f"check declares consumes: datasets — the response is measured "
                f"in the arrays, which no report carries.")

    control = reference
    if control is None:
        control = runs[0]
        report.notes.append(
            "No control was given, so the first run is the baseline — the "
            "conventional sweep, starting from the current setting.")

    key, values = _varied(runs + ([control] if control not in runs else []),
                          report)
    if key is not None:
        step_name, position, parameter = key
        # Only the runs' own values describe the curve; the control may sit
        # outside the series.
        values = [_parameters(run).get(key) for run in runs]

    # The alignment notes belong to the sweep as a whole rather than to each
    # run, so they are collected once and deduplicated.
    quiet = StepReport(step=STEP)
    series = []
    for index, run in enumerate(runs):
        entry = {"value": values[index] if key is not None else None,
                 **_response(run, control, position_tolerance, quiet)}
        if response_column and response_column in run.flightlines.columns:
            column = run.flightlines[response_column].to_numpy(dtype=float)
            finite = column[np.isfinite(column)]
            entry["response"] = float(np.median(finite)) if finite.size else None
        series.append(entry)

    for note in dict.fromkeys(quiet.notes):
        report.notes.append(note)

    if response_column and not any("response" in entry for entry in series):
        report.notes.append(
            f"No run carries a {response_column!r} column, so only the "
            f"divergence from the control is traced.")

    knee = None
    if key is not None:
        # Ordering the curve by the parameter rather than by position: a series
        # handed over out of order would otherwise trace a zigzag and report a
        # knee wherever it happened to double back.
        try:
            order = sorted(range(len(series)),
                           key=lambda i: float(series[i]["value"]))
        except (TypeError, ValueError):
            order = list(range(len(series)))
            report.notes.append(
                "The varied parameter is not numeric, so the series stays in "
                "the order given and no flattening point is looked for.")
        else:
            if order != list(range(len(series))):
                report.notes.append(
                    "The runs were given out of order by parameter value; the "
                    "response curve below is sorted by the parameter.")
            series = [series[i] for i in order]
            curve = [entry.get("response",
                               entry["divergence_from_control"])
                     for entry in series]
            knee = _knee([entry["value"] for entry in series], curve,
                         flat_fraction)
            if knee is not None:
                report.notes.append(
                    f"The response stops moving past {parameter}="
                    f"{knee['at']}: every later step is under "
                    f"{flat_fraction:.0%} of the largest. That is an "
                    f"observation, not a recommendation — the trade a sweep "
                    f"explores is yours to make, and a knee in this response "
                    f"is not a knee in one you care about more.")

    report.soundings_total = len(runs[0].flightlines)
    report.distributions = {
        "parameter": (None if key is None
                      else {"step": step_name, "at": position,
                            "name": parameter}),
        "control_is_a_member_of_the_series": control in runs,
        "response_column": response_column,
        "series": series,
        "levels_off_at": knee,
    }
    report.notes.append(
        "A sweep measures a trade rather than settling one: nothing here "
        "passes, fails, or recommends a value.")

    log.info("%s: %d runs over %s", STEP, len(runs),
             "nothing" if key is None else f"{step_name}.{parameter}")
    return report
