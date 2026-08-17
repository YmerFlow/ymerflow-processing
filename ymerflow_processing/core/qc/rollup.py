"""Gathering several checks' verdicts into one answer per sounding.

Each check writes its own column, so nothing is ever lost when a sounding fails
for more than one reason — a sounding flown too high, too fast and out of
attitude carries an Out of Spec in all three. What is missing without this is a
single place saying *"this one failed, and here is why"*, which is exactly what
the vector export needs: ``03`` calls that export "the one that leaves the
building", and a client opening it should not have to read across five columns
to learn why a point is red.

``03`` names the runner as the thing that does this:

    **The runner aggregates.** Individual checks report per sounding; the runner
    assembles those into the per-line and per-flight summary that a reflight
    decision is made from.

which is also the only place that *can*: a check cannot see what the other
checks found, and the runner holds every :class:`StepReport`.

Flight quality and instrument health are kept apart
---------------------------------------------------
Not every Out of Spec means the same thing. Height, attitude, speed and coverage
say the aircraft was not flown as contracted — that is the reflight
conversation. Altimeters disagreeing says the *equipment* is misbehaving, and
the data may be perfectly usable; what needs fixing is the kit, before the next
flight.

Rolled into one column those become indistinguishable, and a dead altimeter
reads as a badly flown line. So they get separate verdicts, and every reason is
listed regardless of which kind it was.

**Nothing here decides anything.** The reflight call belongs to whoever holds
the contract; this makes the information they need visible in one place.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..dataset import find_flight_column
from .aggregate import apply_rule, grain_measures
from .thresholds import OUT_OF_SPEC, TIERS

#: The grains a verdict is summarized at, and the prefix each writes under.
#: One letter per grain, so the scheme reads itself:
#:
#:     qs_  this sounding          qf_  its flight
#:     ql_  its line               qd_  the delivery it arrived in
#:
#: A bare name with no prefix is a measurement rather than a verdict.
GRAINS = ("line", "flight", "dataset")
GRAIN_PREFIX = {"line": "ql_", "flight": "qf_", "dataset": "qd_"}

#: Which finding kinds roll into which column. A check declaring some other kind
#: still contributes its reasons, it simply does not feed a summary verdict.
FLIGHT = "flight"
INSTRUMENT = "instrument"


def _worst_of(frame, columns) -> pd.Categorical:
    """The most severe verdict across ``columns``, per sounding.

    Missing only where *every* contributing column is missing. Unlike the two
    axes of one attitude measurement, these are independent questions: a
    sounding whose height was never measured but whose speed was grossly out of
    spec is out of spec, and waiting for the unmeasured one would suppress a
    finding that is already certain.
    """
    if not columns:
        return pd.Categorical([None] * len(frame), categories=TIERS, ordered=True)

    codes = np.vstack([
        pd.Categorical(frame[c], categories=TIERS, ordered=True).codes
        for c in columns
    ])
    worst = codes.max(axis=0)               # -1 only if every column is -1
    return pd.Categorical.from_codes(worst, categories=TIERS, ordered=True)


def _reasons(frame, named_columns) -> list[str]:
    """Which checks found each sounding Out of Spec, joined for one field.

    Out of Spec only. Marginal is visible in the per-check columns for anyone
    who wants it, and listing it here would make the common case — a line that
    is merely near a limit throughout — unreadable.

    Named by step, so a reason matches what the pipeline document called it.
    """
    flags = {
        name: (pd.Categorical(frame[column], categories=TIERS, ordered=True)
               == OUT_OF_SPEC)
        for name, column in named_columns
    }
    if not flags:
        return [""] * len(frame)
    return [
        ";".join(name for name, failed in flags.items() if failed[i])
        for i in range(len(frame))
    ]


def _grain_groups(dataset, frame):
    """What each coarser grain groups soundings by, and what bounds its runs.

    Runs must never span the turn between two lines, so every grain coarser
    than a line is segmented by line identity.
    """
    xyz = dataset.survey.xyz
    line_column = xyz.line_id_column
    lines = (frame[line_column].to_numpy()
             if line_column and line_column in frame.columns else None)

    flight_column = find_flight_column(frame)
    flights = (frame[flight_column].to_numpy()
               if flight_column is not None else None)

    groups = {"line": lines, "flight": flights,
              # One group covering everything: "how did this delivery do".
              "dataset": np.zeros(len(frame), dtype=int)}
    return groups, lines


def summarize_run(
    dataset,
    step_reports,
    rollup: dict | None = None,
    verdict_column: str = "qs_verdict",
    reasons_column: str = "qs_reasons",
    sensor_column: str = "qs_sensor",
) -> dict:
    """Write the cross-check rollup onto ``flightlines``.

    Reads each step's declared ``verdict_column`` and ``finding`` rather than
    inspecting column dtypes, so one check's verdict is never mistaken for
    another's.

    Three columns, all within the ten characters a shapefile allows for a field
    name, because these are what the export carries:

    ``qs_verdict``
        Worst flight-quality verdict — what a map is colored by.
    ``qs_reasons``
        Which checks flagged this sounding, e.g. ``height_above_ground;ground_speed``.
    ``qs_sensor``
        Worst instrument-health verdict, kept apart so a failing altimeter does
        not read as a badly flown line.

    Returns a summary of what it wrote, for the run report.
    """
    frame = dataset.flightlines
    if frame is None or len(frame) == 0:
        return {"columns": {}, "checks": {}}

    # Only steps that actually wrote a verdict, and only where the column
    # survived — a check that skipped for want of an input declares nothing.
    contributing = [
        (r.step, r.verdict_column, r.finding)
        for r in step_reports
        if r.verdict_column is not None and r.verdict_column in frame.columns
    ]
    if not contributing:
        return {"columns": {}, "checks": {}}

    flight = [c for _, c, f in contributing if f == FLIGHT]
    instrument = [c for _, c, f in contributing if f == INSTRUMENT]

    frame[verdict_column] = _worst_of(frame, flight)
    frame[sensor_column] = _worst_of(frame, instrument)
    frame[reasons_column] = _reasons(frame, [(s, c) for s, c, _ in contributing])

    # Each coarser grain carries the two measures 03 asks for, and a verdict
    # only where the contract supplied a rule to produce one.
    settings = dict(rollup or {})
    gap_tolerance = float(settings.get("gap_tolerance", 0.0) or 0.0)
    groups, lines = _grain_groups(dataset, frame)
    xyz = dataset.survey.xyz
    has_xy = xyz.x_column in frame.columns and xyz.y_column in frame.columns
    easting = frame[xyz.x_column].to_numpy() if has_xy else None
    northing = frame[xyz.y_column].to_numpy() if has_xy else None

    grains: dict = {}
    for grain in GRAINS:
        by = groups.get(grain)
        if by is None:
            continue
        prefix = GRAIN_PREFIX[grain]
        fraction, run = grain_measures(
            frame[verdict_column], by,
            segments=None if grain == "line" else lines,
            easting=easting, northing=northing, gap_tolerance=gap_tolerance)
        frame[f"{prefix}frac"] = fraction
        frame[f"{prefix}run"] = run

        rule = settings.get(grain) or {}
        verdict = apply_rule(fraction, run, rule.get("max_fraction"),
                             rule.get("max_run"))
        written = {"fraction": f"{prefix}frac", "run": f"{prefix}run"}
        if verdict is not None:
            frame[f"{prefix}verdict"] = verdict
            written["verdict"] = f"{prefix}verdict"
        grains[grain] = {"columns": written, "rule": rule or None}

    return {
        "columns": {
            "verdict": verdict_column,
            "reasons": reasons_column,
            "sensor": sensor_column,
        },
        "grains": grains,
        "gap_tolerance": gap_tolerance,
        # What went into each, so a reader of the report can tell why a rollup
        # says what it says without re-running anything.
        "checks": {step: {"column": column, "finding": finding}
                   for step, column, finding in contributing},
    }
