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

import numpy as np
import pandas as pd

from ..dataset import find_flight_column
from .aggregate import apply_rule, grain_measures
from .thresholds import (
    MAX_POINTS, OUT_OF_SPEC, is_tier, mark_verdicts, score, worst_of,
)

#: The grains a verdict is summarized at, and the prefix each writes under.
#: One letter per grain, so the scheme reads itself:
#:
#:     qs_  this sounding          qf_  its flight
#:     ql_  its line               qd_  the delivery it arrived in
#:
#: A bare name with no prefix is a measurement rather than a verdict.
GRAINS = ("line", "flight", "dataset")
GRAIN_PREFIX = {"line": "ql_", "flight": "qf_", "dataset": "qd_"}

#: What a sounding with nothing wrong is given, since an empty string cannot be
#: written to a whitespace-delimited format.
NO_REASON = "NONE"

#: Which finding kinds roll into which column. A check declaring some other kind
#: still contributes its reasons, it simply does not feed a summary verdict.
FLIGHT = "flight"
INSTRUMENT = "instrument"

#: How much each check counts toward the grade, by default. Override any of them
#: under ``rollup: weights:`` in the pipeline document.
#:
#: These are **operational judgements about what makes data unusable**, not
#: contract terms, which is why they carry defaults where thresholds do not: a
#: threshold decides whether a sounding passes and goes to a client as a
#: deliverable, while a weight only orders which problems matter more when
#: ranking. Getting a weight wrong sorts a list badly; getting a threshold wrong
#: mis-states a deliverable.
#:
#: The shape of the scale: **height and attitude dominate**, because they change
#: what the measurement means — a sounding flown too high or at a bad attitude is
#: coupling to the ground differently, and no amount of processing recovers it.
#: Speed, spacing and transmitter stability are quarter-weight: they degrade
#: coverage or repeatability rather than invalidating the physics.
#:
#: Two are zero, and zero is not the same as absent. A weighted-out check still
#: runs, still writes its own verdict column, still names itself in
#: ``qs_reasons`` and still feeds ``qs_sensor`` where it is an instrument
#: finding. It simply does not move the grade:
#:
#: ``altimeter_agreement``
#:     Flagged, not graded. Disagreeing altimeters are a sensor fault to raise
#:     with the operator, and one bad altimeter should not drag down the grade of
#:     data that was flown correctly.
#: ``powerline_noise``
#:     Excluded. A powerline is a fact about the ground — not the contractor's
#:     doing, not avoidable by reflying — so grading a survey down for one would
#:     penalize it for where it was asked to fly.
#: ``data_misfit``
#:     Excluded, for the same reason one step further out. A poor fit may be
#:     genuine geology a 1D model cannot represent, so a high misfit says the
#:     *model* did not explain the sounding — a statement about the pair, not
#:     about the measurement. Grading the data down for it would penalize the
#:     survey for the earth being complicated. It is named in ``qs_reasons``,
#:     which is where "look at this one" belongs.
DEFAULT_WEIGHTS = {
    "height_above_ground": 2.0,
    "system_attitude": 2.0,
    "ground_speed": 0.25,
    "sounding_spacing": 0.25,
    "transmitter_stability": 0.25,
    "altimeter_agreement": 0.0,
    "powerline_noise": 0.0,
    "data_misfit": 0.0,
}

#: What a check absent from :data:`DEFAULT_WEIGHTS` counts for. Deliberately
#: neutral rather than zero: a new check should have to be weighted *down*
#: explicitly, not vanish from the grade because nobody remembered it. Every
#: effective weight is recorded in the run report, so an unintended 1.0 is
#: visible rather than silent.
UNWEIGHTED = 1.0


def _worst_of(frame, columns) -> np.ndarray:
    """The most severe verdict across ``columns``, per sounding.

    Missing only where *every* contributing column is missing. Unlike the two
    axes of one attitude measurement, these are independent questions: a
    sounding whose height was never measured but whose speed was grossly out of
    spec is out of spec, and waiting for the unmeasured one would suppress a
    finding that is already certain.
    """
    if not columns:
        return np.full(len(frame), np.nan)
    return worst_of([frame[c] for c in columns])


def _score_of(frame, named_columns, weights) -> np.ndarray:
    """A weighted grade across checks, per sounding, on the 0-2 point scale.

    Each check's verdict becomes points, and the sounding's grade is their
    weighted mean over the checks that actually classified it. ``NaN`` where no
    contributing check reached a verdict, so an unmeasured sounding is blank
    rather than failing.

    Weights come from :data:`DEFAULT_WEIGHTS`, overridable per pipeline. A
    weight of 0 excludes a check from the grade while leaving its own verdict
    column, its reasons and its sensor rollup intact.

    Graded across *every* check that reached a verdict, not just the
    flight-quality ones. The weight decides what counts, which is a finer
    instrument than the finding kind: transmitter stability is an instrument
    finding and still belongs in the grade, while a powerline is weighted out
    entirely rather than being excluded by category.

    **This does not decide anything, and must not be read as if it did.** A mean
    dilutes: a sounding flown far too high but otherwise perfect still grades
    well, because the checks that passed outvote the one that did not. That is
    the point of an average and the reason it cannot be the gate. ``qs_verdict``
    stays the worst verdict across the same checks, so a single fatal breach
    still reads as Out of Spec no matter what the grade says. The grade is for
    ranking — which lines are worst overall, when everything failing ties at
    Out of Spec — and the verdict is for passing and failing.
    """
    usable = [(name, column) for name, column in named_columns
              if weights[name] > 0]
    if not usable:
        return np.full(len(frame), np.nan)

    points = np.vstack([score(frame[column]) for _, column in usable])
    weight = np.array([[weights[name]] for name, _ in usable])

    measured = ~np.isnan(points)
    total = np.nansum(np.where(measured, points, 0.0) * weight, axis=0)
    divisor = np.sum(np.where(measured, 1.0, 0.0) * weight, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(divisor > 0, total / divisor, np.nan)


def _grain_score(scores, groups) -> np.ndarray:
    """Each group's mean grade, broadcast back onto every sounding of it.

    The average of an average, which is exactly what composes here: a line is
    the mean of its soundings, a flight the mean of its soundings, the delivery
    the mean of all of them. Taken over soundings rather than over lines, so a
    two-sounding line does not weigh as much as a thousand-sounding one.
    """
    frame = pd.DataFrame({"score": np.asarray(scores, dtype=float),
                          "group": pd.Series(np.asarray(groups)).astype(str)})
    means = frame.groupby("group", sort=False)["score"].transform("mean")
    return means.to_numpy(dtype=float)


def _reasons(frame, named_columns) -> list[str]:
    """Which checks found each sounding Out of Spec, joined for one field.

    Out of Spec only. Marginal is visible in the per-check columns for anyone
    who wants it, and listing it here would make the common case — a line that
    is merely near a limit throughout — unreadable.

    Named by step, so a reason matches what the pipeline document called it.

    A sounding with no reasons gets ``NONE`` rather than an empty string. Empty
    is not writable in a whitespace-delimited format — it occupies no field at
    all, so the row comes back one column short and the file will not parse.
    """
    flags = {
        name: is_tier(frame[column], OUT_OF_SPEC)
        for name, column in named_columns
    }
    if not flags:
        return [NO_REASON] * len(frame)
    return [
        ";".join(name for name, failed in flags.items() if failed[i]) or NO_REASON
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
    score_column: str = "qs_score",
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

    Parameters
    ----------
    dataset : Dataset
        Whose ``flightlines`` receive the rollup columns.
    step_reports : sequence of StepReport
        The checks that ran, each declaring its verdict column and finding.
    rollup : dict | None
        The pipeline document's ``rollup`` block, if it has one.
    verdict_column : str
        Where the worst flight-quality verdict goes.
    score_column : str
        Where its numeric score goes.
    reasons_column : str
        Where the names of the checks that flagged each sounding go.
    sensor_column : str
        Where the worst instrument-health verdict goes.

    Returns
    -------
    dict
        A summary of what it wrote, for the run report.
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
    mark_verdicts(dataset, verdict_column, sensor_column)
    frame[reasons_column] = _reasons(frame, [(s, c) for s, c, _ in contributing])

    settings = dict(rollup or {})
    gap_tolerance = float(settings.get("gap_tolerance", 0.0) or 0.0)
    overrides = dict(settings.get("weights") or {})

    ran = {s for s, _, _ in contributing}
    unknown = sorted(set(overrides) - ran)
    if unknown:
        raise KeyError(
            "rollup weights name step(s) that produced no verdict: "
            f"{', '.join(unknown)}. Steps that ran and classified: "
            f"{', '.join(sorted(ran))}."
        )

    # Resolved for every step that ran, so the report records the weight each
    # check actually carried rather than only the ones someone overrode.
    weights = {
        step: float(overrides.get(step, DEFAULT_WEIGHTS.get(step, UNWEIGHTED)))
        for step in ran
    }
    negative = sorted(s for s, w in weights.items() if w < 0)
    if negative:
        raise ValueError(
            f"rollup weights must not be negative: {', '.join(negative)}. "
            "Use 0 to exclude a check from the grade."
        )

    frame[score_column] = _score_of(
        frame, [(s, c) for s, c, _ in contributing], weights)

    # Each coarser grain carries the measures 03 asks for, its own mean grade,
    # and a verdict only where the contract supplied a rule to produce one.
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
        measures = grain_measures(
            frame[verdict_column], by,
            segments=None if grain == "line" else lines,
            easting=easting, northing=northing, gap_tolerance=gap_tolerance)
        written = {}
        for name, values in measures.items():
            frame[f"{prefix}{name}"] = values
            written[name] = f"{prefix}{name}"

        frame[f"{prefix}score"] = _grain_score(frame[score_column], by)
        written["score"] = f"{prefix}score"

        rule = settings.get(grain) or {}
        # The rule reads the out-of-spec fraction: a contract states what share
        # of a line may fail, not what share must pass, and 'max_fraction' is
        # the key it was written under before the other tiers were reported.
        verdict = apply_rule(measures["oos"], measures["run"],
                             rule.get("max_fraction"), rule.get("max_run"))
        if verdict is not None:
            frame[f"{prefix}verdict"] = verdict
            mark_verdicts(dataset, f"{prefix}verdict")
            written["verdict"] = f"{prefix}verdict"
        grains[grain] = {"columns": written, "rule": rule or None}

    return {
        "weights": weights,
        "max_score": MAX_POINTS,
        "columns": {
            "verdict": verdict_column,
            "score": score_column,
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
