"""The Good / Marginal / Out of Spec classification every QC check shares.

Carried over from that prior flight QC panel, which is our own prior work and has
flown a real survey. Four threshold kinds, distinguished by the shape of the
quantity being judged rather than by which check is asking:

``centered``
    Should sit near zero and can deviate either way. Attitude.
``ascending``
    One-sided, higher is worse. Height above ground.
``descending``
    One-sided, lower is worse. Coverage, sounding density.
``window``
    Both bounds meaningful and possibly asymmetric, which ``centered`` cannot
    express. Ground speed.

A metric with no threshold entry gets **no classification at all** rather than a
default. That graceful absence is deliberate: a check that cannot say whether
something met spec should say nothing, not guess "Good". The prior panel behaves this way
and it is the right behavior to preserve.

Boundaries are inclusive at the Good end
----------------------------------------
A value exactly on a limit is the *better* class: with ``good_max: 45``, a
sounding at precisely 45.000 m is Good. "The spec is a maximum of 45 m" reads,
in every survey contract, as 45 m being acceptable.

This is stated because it is a real decision, not an accident. That panel's own
implementation is inconsistent between kinds — ``centered`` inclusive,
``ascending`` exclusive — so one of the two had to change to make the rule
sayable in a sentence. Since verdicts go to a client as a contract deliverable,
the rule they are computed by needs to be one anybody can state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GOOD = "Good"
MARGINAL = "Marginal"
OUT_OF_SPEC = "Out of Spec"

#: Ordered worst-last, so a Categorical built from it sorts and compares the way
#: severity reads.
TIERS = (GOOD, MARGINAL, OUT_OF_SPEC)

CENTERED = "centered"
ASCENDING = "ascending"
DESCENDING = "descending"
WINDOW = "window"

#: Which parameters each kind requires. Also what the error message quotes back
#: when one is missing, so a mistyped pipeline says what it should have said.
REQUIRED = {
    CENTERED: ("good", "marginal"),
    ASCENDING: ("good_max", "marginal_max"),
    DESCENDING: ("good_min", "marginal_min"),
    WINDOW: ("good_min", "good_max", "marginal_min", "marginal_max"),
}


def validate_spec(spec: dict) -> str:
    """Check a threshold specification and return its kind.

    Fails loudly and early: a threshold that is wrong produces verdicts that are
    quietly wrong, and a QC verdict is what a client is shown.
    """
    if not isinstance(spec, dict):
        raise TypeError(f"A threshold specification must be a mapping, got {spec!r}")
    kind = spec.get("kind")
    if kind not in REQUIRED:
        raise ValueError(
            f"Unknown threshold kind {kind!r}. Available: {', '.join(sorted(REQUIRED))}"
        )
    missing = [p for p in REQUIRED[kind] if spec.get(p) is None]
    if missing:
        raise ValueError(
            f"A {kind!r} threshold needs {', '.join(REQUIRED[kind])}; "
            f"missing {', '.join(missing)}"
        )

    # An ordering mistake is silent otherwise: marginal tighter than good would
    # classify nothing as Marginal and nobody would notice.
    if kind == CENTERED and spec["marginal"] < spec["good"]:
        raise ValueError("A 'centered' threshold needs marginal >= good")
    if kind == ASCENDING and spec["marginal_max"] < spec["good_max"]:
        raise ValueError("An 'ascending' threshold needs marginal_max >= good_max")
    if kind == DESCENDING and spec["marginal_min"] > spec["good_min"]:
        raise ValueError("A 'descending' threshold needs marginal_min <= good_min")
    if kind == WINDOW:
        if spec["good_min"] > spec["good_max"]:
            raise ValueError("A 'window' threshold needs good_min <= good_max")
        if spec["marginal_min"] > spec["good_min"] or spec["marginal_max"] < spec["good_max"]:
            raise ValueError(
                "A 'window' threshold needs the marginal window to contain the good one"
            )
    return kind


def classify(values, spec: dict) -> pd.Categorical:
    """Classify ``values`` as Good / Marginal / Out of Spec against ``spec``.

    Missing values stay missing — a sounding with no measurement has no verdict,
    which is different from failing one. Returns an ordered
    :class:`pandas.Categorical` so that "worse than Marginal" is expressible
    directly and the column sorts by severity.
    """
    kind = validate_spec(spec)
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, None, dtype=object)
    valid = np.isfinite(values)

    if kind == CENTERED:
        # Distance from zero is the whole quantity of interest; sign is not.
        deviation = np.abs(values)
        good = deviation <= spec["good"]
        marginal = deviation <= spec["marginal"]
    elif kind == ASCENDING:
        good = values <= spec["good_max"]
        marginal = values <= spec["marginal_max"]
    elif kind == DESCENDING:
        good = values >= spec["good_min"]
        marginal = values >= spec["marginal_min"]
    else:  # WINDOW
        good = (values >= spec["good_min"]) & (values <= spec["good_max"])
        marginal = (values >= spec["marginal_min"]) & (values <= spec["marginal_max"])

    # Assigned worst-first so each better class overwrites: every valid value
    # ends up in exactly one tier, and Marginal is by construction the band
    # between the two, never a separate test that could disagree with them.
    out[valid] = OUT_OF_SPEC
    out[valid & marginal] = MARGINAL
    out[valid & good] = GOOD

    return pd.Categorical(out, categories=TIERS, ordered=True)


def describe(spec: dict, unit: str = "") -> dict[str, str]:
    """Legend text naming each tier's actual limits, e.g. ``"Good (<=45 m)"``.

    Built from the specification rather than from anything axis-derived, so the
    numbers quoted are the ones classification actually used.
    """
    kind = validate_spec(spec)
    suffix = f" {unit}" if unit else ""

    if kind == CENTERED:
        g, m = spec["good"], spec["marginal"]
        return {GOOD: f"Good (<=±{g:g}{suffix})",
                MARGINAL: f"Marginal (±{g:g}-{m:g}{suffix})",
                OUT_OF_SPEC: f"Out of Spec (>±{m:g}{suffix})"}
    if kind == ASCENDING:
        g, m = spec["good_max"], spec["marginal_max"]
        return {GOOD: f"Good (<={g:g}{suffix})",
                MARGINAL: f"Marginal ({g:g}-{m:g}{suffix})",
                OUT_OF_SPEC: f"Out of Spec (>{m:g}{suffix})"}
    if kind == DESCENDING:
        g, m = spec["good_min"], spec["marginal_min"]
        return {GOOD: f"Good (>={g:g}{suffix})",
                MARGINAL: f"Marginal ({m:g}-{g:g}{suffix})",
                OUT_OF_SPEC: f"Out of Spec (<{m:g}{suffix})"}
    gl, gh = spec["good_min"], spec["good_max"]
    ml, mh = spec["marginal_min"], spec["marginal_max"]
    return {GOOD: f"Good ({gl:g}-{gh:g}{suffix})",
            MARGINAL: f"Marginal ({ml:g}-{gl:g}, {gh:g}-{mh:g}{suffix})",
            OUT_OF_SPEC: f"Out of Spec (<{ml:g} or >{mh:g}{suffix})"}


def verdict_columns(frame) -> list[str]:
    """Every column holding a Good / Marginal / Out of Spec verdict.

    Recognized by dtype rather than by name, so a check contributed by another
    package is found without registering its column anywhere. Any check writing
    a verdict through :func:`classify` produces exactly this dtype.
    """
    if frame is None:
        return []
    return [
        name for name in frame.columns
        if isinstance(frame[name].dtype, pd.CategoricalDtype)
        and tuple(frame[name].dtype.categories) == TIERS
    ]


def worst_verdict(frame) -> str | None:
    """The most severe verdict anywhere in ``frame``, or ``None`` if unjudged.

    What a caller needs to answer "did anything fail?" across every check that
    ran, without knowing which checks those were.
    """
    worst = None
    for name in verdict_columns(frame):
        column = frame[name].dropna()
        if column.empty:
            continue
        # Categories are ordered worst-last, so max is severity.
        candidate = column.max()
        if worst is None or TIERS.index(candidate) > TIERS.index(worst):
            worst = candidate
    return worst


def tier_counts(verdicts) -> dict[str, int]:
    """How many soundings landed in each tier, plus how many got no verdict.

    Every tier appears even at zero — a reader should see "Out of Spec: 0"
    rather than have to infer it from an absence.
    """
    series = pd.Series(pd.Categorical(verdicts, categories=TIERS, ordered=True))
    counts = {tier: int((series == tier).sum()) for tier in TIERS}
    counts["unclassified"] = int(series.isna().sum())
    return counts
