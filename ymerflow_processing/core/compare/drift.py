"""qc_09 — the noise floor across a survey, as a series.

The check ``03`` specifies and deliberately cannot implement as a QC check:

    The floor is a property of the environment and the instrument together. A
    trend across flights means something is changing — a failing component, a
    seasonal shift, or a region genuinely noisier than the rest. **Only visible
    cumulatively**, which is why per-flight results have to accumulate.

That last clause is why this is a comparison rather than a check. One delivery's
floor is a number; twenty deliveries' floors are a diagnosis, and no per-run
check can see the second.

It invents no state
-------------------
Every input already exists. ``estimate_noise_model`` records the floor, the
exponent and what was measured versus supplied in its step report, and R3 means
those reports are stored. So this reads prior runs rather than carrying anything
between them — which is ``02``'s argument for why the comparison shape resolves
cumulative QC without persistent state.

What it distinguishes, and why both numbers are needed
------------------------------------------------------
A rising floor and a *jumping* floor are different faults. A component degrading
gives a trend; a change of survey area or a swapped receiver gives a step. So
the report carries both:

**The trend** — floors fitted against position in the series, in log space, so
the slope reads as a fractional change per delivery. Log space because a floor
is a scale: going from 5e-10 to 1e-9 is the same event as going from 1e-9 to
2e-9, and a linear fit would call the second twenty times larger.

**The largest single step** — the biggest ratio between consecutive deliveries.
A trend can be small while one step is enormous, and averaging hides exactly the
delivery someone needs to look at.

Neither is judged unless asked
------------------------------
No thresholds by default: what counts as too much drift depends on the
instrument and the survey, and a house number would flag a seasonal change as a
fault. Supplied, ``max_ratio`` compares each delivery against the baseline — the
``reference`` where one was given, otherwise the series median, which is robust
to the outlier being looked for.
"""

import logging

import numpy as np

from ..pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "noise_floor_drift"

#: The step whose report carries the model. Named rather than assumed, so a
#: pipeline that estimated under a different name can still be compared.
SOURCE_STEP = "estimate_noise_model"


def _model_of(report: dict, source: str) -> dict | None:
    """The noise model a stored run recorded, if it recorded one."""
    for step in report.get("steps") or []:
        if step.get("step") != source:
            continue
        return (step.get("distributions") or {}).get("model")
    return None


def _label(report: dict, index: int) -> str:
    """What to call a delivery in the series."""
    return str(report.get("identity") or f"item {index}")


def _flight_key(flights):
    """The earliest flight in a delivery, ordered numerically where it can be.

    **Numeric and increasing is the safe bet.** Flight *numbering schemes* vary
    by contractor — SkyTEM writes a date with an index on the end, like
    ``26051201``; another may number sequentially from one — but whatever the
    scheme, flights are numbered in the order they were flown. So the value is
    ordered as a number rather than as text.

    That distinction is not cosmetic. Sorted as text, ``"10"`` comes before
    ``"9"``, so a sequentially numbered survey would have its tenth flight
    trended before its ninth and every delivery after it shifted. Nothing in the
    numbers would give that away.

    Falls back to text for a scheme that is not numeric at all, kept in a
    separate rank so the two never compare against each other.
    """
    values = [f for f in flights if f is not None]
    if not values:
        return None
    try:
        return (0, min(float(f) for f in values))
    except (TypeError, ValueError):
        return (1, min(str(f) for f in values))


def _check_flight_assumption(found, report):
    """Test that flight numbers behave the way ordering by them assumes.

    "Numeric and increasing" is a safe bet and not a guarantee, and the
    difference matters because a series trended in the wrong order produces a
    plausible slope with nothing to give it away. So rather than asserting the
    assumption in a docstring, this checks it and says when it does not hold.

    Three things can be wrong, and each means something different:

    **Not numeric.** A scheme that does not parse as a number is ordered as
    text, where ``"A10"`` precedes ``"A9"``. Sound only if the scheme is
    zero-padded.

    **Duplicated.** Two deliveries claiming the same flight is either a
    re-export entered twice or a contractor restarting their numbering — and a
    series containing the same flight twice trends it twice.

    **Disagreeing with the order given.** ``items`` is ordered by contract, so
    if the caller passed deliveries chronologically and the flight numbers
    disagree, the numbering is not chronological. That is the case this exists
    to catch.
    """
    keys = [entry["flight_key"] for entry in found]
    if any(key is None for key in keys):
        return

    if any(kind == 1 for kind, _ in keys):
        report.notes.append(
            "Some flight numbers do not parse as numbers and were ordered as "
            "text, where 'A10' precedes 'A9'. Sound only if the scheme is "
            "zero-padded — check the order before reading a trend from it.")

    values = [value for _, value in keys]
    seen = {}
    for entry, value in zip(found, values):
        seen.setdefault(value, []).append(entry["delivery"])
    repeated = {v: names for v, names in seen.items() if len(names) > 1}
    if repeated:
        detail = "; ".join(f"{v}: {', '.join(names)}"
                           for v, names in repeated.items())
        report.notes.append(
            f"The same flight number appears in more than one delivery "
            f"({detail}). That is a re-export entered twice or a contractor "
            f"restarting their numbering, and either way the series trends one "
            f"flight more than once.")

    given = [entry["given_at"] for entry in found]
    if given != sorted(given):
        report.notes.append(
            "Ordering by flight number disagrees with the order the reports "
            "were given in. Flight numbering is chronological by convention "
            "rather than by construction, so one of the two is wrong — and if "
            "the reports were passed chronologically, it is the numbering.")


def _ordering(report: dict, index: int):
    """What to order the series by, most trustworthy first.

    **The time range.** Read from the soundings' own timestamps, so it is
    chronological for any instrument and survives a file being renamed or
    re-exported.

    **Then the flight number**, ordered numerically — see :func:`_flight_key`.
    Second rather than first because a time is chronological by construction
    while a flight number is chronological by convention, and where both exist
    they agree anyway.

    **Then position in the sequence.** ``02`` makes ``items`` ordered and says a
    check may rely on it, so this is a fallback rather than a failure.

    Never the identity. It is usually a filename, delivery names routinely embed
    an export date, and ordering by one would enter a single delivery into the
    series twice after a re-export.
    """
    span = report.get("time_range")
    if isinstance(span, (list, tuple)) and span and span[0]:
        return (0, str(span[0]))

    flights = report.get("flights")
    if isinstance(flights, (list, tuple)) and flights:
        key = _flight_key(flights)
        if key is not None:
            return (1,) + key
    return (2, index)


def noise_floor_drift(items, reference=None, source_step: str = SOURCE_STEP,
                      max_ratio: float | None = None):
    """Trend the estimated noise floor across a sequence of deliveries.

    Parameters
    ----------
    items
        Stored run reports, ordered.
    reference
        A run to measure everything against. Without one the baseline is the
        series median, which is robust to the outlier being looked for.
    source_step : str
        Which step's report carries the model.
    max_ratio : float | None
        Largest acceptable factor between a delivery's floor and the baseline.
        **Unset, this measures and says nothing** — what counts as too much
        drift depends on the instrument and the survey.

    Returns
    -------
    StepReport
        The comparison's report; the series and its trend are in
        ``distributions``.
    """
    report = StepReport(step=STEP)
    if not items:
        report.notes.append("No reports given; nothing to trend.")
        return report

    found, without = [], []
    for index, item in enumerate(items):
        model = _model_of(item, source_step)
        level = None if model is None else model.get("level_1ms")
        if level is None or not np.isfinite(float(level)):
            without.append(_label(item, index))
            continue
        found.append({
            "delivery": _label(item, index),
            "order": _ordering(item, index),
            "given_at": index,
            "flight_key": _flight_key(item.get("flights") or []),
            "level_1ms": float(level),
            "exponent": model.get("exponent"),
            "flights": item.get("flights"),
            "time_range": item.get("time_range"),
            # An estimate and a number someone typed are different evidence, and
            # a series that silently mixes them trends the typing.
            "supplied": list(model.get("supplied") or ()),
        })

    if without:
        report.notes.append(
            f"{len(without)} report(s) carry no {source_step!r} model and are "
            f"not in the series: {', '.join(without)}."
        )
    if not found:
        report.notes.append(
            f"No report carries a {source_step!r} model, so there is no series "
            f"to trend.")
        return report

    found.sort(key=lambda entry: entry["order"])
    # Labelled by the weakest key any entry fell back to. The first entry has
    # the strongest, so reading the label from it called a series "ordered by
    # time range" when one delivery carried one and the rest were placed by
    # flight number.
    priorities = {entry["order"][0] for entry in found}
    ordered_by = {0: "time_range", 1: "flight number",
                  2: "the order given"}[max(priorities)]
    if len(priorities) > 1:
        report.notes.append(
            "The deliveries do not all carry the same ordering key, so the "
            "series mixes them: those with a time range first, then those "
            "placed by flight number, then the rest in the order given. A "
            "trend across that boundary is not chronological."
        )
    if ordered_by == "flight number":
        _check_flight_assumption(found, report)
    for entry in found:
        entry.pop("order")
        entry.pop("given_at")
        entry.pop("flight_key")
    if ordered_by == "flight number":
        report.notes.append(
            "The series is ordered by flight number, since no delivery carries "
            "a time range. Flights are numbered in the order they were flown "
            "whatever the contractor's scheme, so this is sound — but it is a "
            "convention rather than a measurement."
        )
    elif ordered_by == "the order given":
        report.notes.append(
            "The series carries neither a time range nor flight numbers, so it "
            "is in the order the reports were given. A trend read from it is a "
            "trend in that order and nothing more."
        )
    levels = np.array([entry["level_1ms"] for entry in found], dtype=float)

    baseline_model = (_model_of(reference, source_step)
                      if reference is not None else None)
    baseline_level = (None if baseline_model is None
                      else baseline_model.get("level_1ms"))
    if baseline_level:
        baseline = float(baseline_level)
        source = "reference"
    else:
        baseline = float(np.median(levels))
        source = "series median"
        if reference is not None:
            report.notes.append(
                f"The reference carries no {source_step!r} model, so the "
                f"baseline is the series median instead.")

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = levels / baseline
    for entry, value in zip(found, ratio):
        entry["ratio_to_baseline"] = float(value)

    # Log space, because a floor is a scale: 5e-10 to 1e-9 is the same event as
    # 1e-9 to 2e-9, and a linear fit would call the second twenty times larger.
    trend = None
    if levels.size >= 2 and np.all(levels > 0):
        slope, _ = np.polyfit(np.arange(levels.size), np.log(levels), 1)
        trend = {"log_slope_per_delivery": float(slope),
                 "fractional_change_per_delivery": float(np.expm1(slope)),
                 "total_change": float(levels[-1] / levels[0])}

    # A trend can be small while one step is enormous, and averaging hides
    # exactly the delivery someone needs to look at.
    steps = None
    if levels.size >= 2:
        with np.errstate(divide="ignore", invalid="ignore"):
            jumps = levels[1:] / levels[:-1]
        worst = int(np.nanargmax(np.abs(np.log(jumps))))
        steps = {
            "largest_ratio": float(jumps[worst]),
            "between": [found[worst]["delivery"], found[worst + 1]["delivery"]],
        }

    exceeded = []
    if max_ratio is not None:
        if max_ratio <= 1:
            raise ValueError(
                f"max_ratio is a factor away from the baseline in either "
                f"direction and must exceed 1; got {max_ratio:g}.")
        for entry in found:
            value = entry["ratio_to_baseline"]
            if value > max_ratio or (value > 0 and 1 / value > max_ratio):
                exceeded.append(entry["delivery"])
        if exceeded:
            report.notes.append(
                f"{len(exceeded)} delivery(s) differ from the {source} by more "
                f"than {max_ratio:g}x: {', '.join(exceeded)}. That is a "
                f"failing component, a change of survey area, or ground that is "
                f"genuinely noisier — this cannot tell which."
            )

    typed = [entry["delivery"] for entry in found if "level_1ms" in entry["supplied"]]
    if typed:
        report.notes.append(
            f"{len(typed)} delivery(s) had their floor supplied rather than "
            f"estimated: {', '.join(typed)}. A series mixing the two trends the "
            f"typing as much as the data."
        )

    report.distributions = {
        "ordered_by": ordered_by,
        "series": found,
        "baseline": {"level_1ms": baseline, "from": source},
        "trend": trend,
        "largest_step": steps,
        "max_ratio": max_ratio,
        "exceeded": exceeded,
    }
    if max_ratio is None:
        report.notes.append(
            "No max_ratio given, so the series is measured and left unjudged. "
            "What counts as too much drift depends on the instrument and the "
            "survey.")
    log.info("%s: %d deliveries, baseline %.3g from the %s",
             STEP, len(found), baseline, source)
    return report
