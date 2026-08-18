"""The Good / Marginal / OOS classification every QC check shares.

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

import warnings

import numpy as np
import pandas as pd

GOOD = "Good"
MARGINAL = "Marginal"
#: Abbreviated, and not for brevity's sake. The survey xyz format is
#: whitespace-delimited with no quoting, so a value containing a space becomes
#: three fields and the file it was written to will not parse back. "Out of
#: Spec" made every enriched export unreadable. OOS is what the domain says
#: aloud anyway, and it survives every format this library writes.
OUT_OF_SPEC = "OOS"

#: Ordered worst-last, so a Categorical built from it sorts and compares the way
#: severity reads.
TIERS = (GOOD, MARGINAL, OUT_OF_SPEC)

#: Points per tier, so a grain can be averaged into a single grade. Two for
#: in-spec, one for Marginal, none for Out of Spec.
#:
#: **This treats an ordinal scale as an interval one.** Scoring 0/1/2 asserts
#: that the step from Out of Spec to Marginal is worth exactly as much as the
#: step from Marginal to Good, and that is a choice rather than a measurement —
#: arguably the first step is the one that matters, since it is where data
#: becomes acceptable at all, while the second is comfort margin. The scale is
#: kept even because no contract states a different exchange rate, and inventing
#: one would be the only unstated number in the QC path.
#:
#: The consequence to keep in view: a mean aliases. A line averaging 1.0 is
#: either uniformly Marginal or half Good and half Out of Spec, which are very
#: different deliveries. That is why the score is reported *alongside* the
#: per-tier fractions and the contiguous run rather than in place of them —
#: see :func:`ymerflow_processing.core.qc.aggregate.grain_measures`.
TIER_POINTS = {OUT_OF_SPEC: 0.0, MARGINAL: 1.0, GOOD: 2.0}

#: The best score obtainable, so a reader can normalize without knowing the
#: scale and a change of scale does not need chasing through call sites.
MAX_POINTS = max(TIER_POINTS.values())


def score(verdicts) -> np.ndarray:
    """Each verdict as points, ``NaN`` where nothing was classified.

    Now that verdicts are stored as codes on this very scale, this is the
    identity for anything :func:`classify` produced, and kept only so callers
    read as what they mean and so a label-valued input still works.

    NaN rather than zero for the unmeasured: a sounding no check looked at has
    no grade, and scoring it zero would make an unmeasured line indistinguishable
    from an entirely failing one — the more alarming of the two readings, for
    the case that carries the least evidence.
    """
    values = pd.Series(np.asarray(verdicts, dtype=object))
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(numeric.notna(),
                         values.map(TIER_POINTS)).to_numpy(dtype=float)

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


def classify(values, spec: dict) -> np.ndarray:
    """Classify ``values`` against ``spec``, as verdict codes.

    Returns ``float32`` codes on the :data:`TIER_POINTS` scale — 0 Out of Spec,
    1 Marginal, 2 Good — with ``NaN`` where nothing was measured. A sounding
    with no measurement has no verdict, which is different from failing one.

    **Numeric rather than labelled, because a string column is invisible
    downstream.** YmerFlow's plot layer enumerates what it can draw with
    ``Object.keys(fl).filter(k => ArrayBuffer.isView(fl[k]))`` and then coerces
    through ``toFloat32Array``. A column of labels is not a typed array after the
    msgpack round-trip, so it fails that filter and never appears at all — not
    awkward to plot, absent. Any numeric column is available to every plot the
    moment it is written, with nothing to register anywhere.

    ``float32`` rather than an integer type because integers cannot carry
    ``NaN``, and "not evaluated" has to be representable. Float32 holds integers
    exactly to 2^24, so the codes are never approximate.

    The labels are not lost: :func:`describe` puts the legend in the step report,
    and the vector exports translate back on the way out, since a GIS is read by
    people rather than by a plot axis.
    """
    kind = validate_spec(spec)
    values = np.asarray(values, dtype=float)
    out = np.full(values.shape, np.nan, dtype=np.float32)
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
    out[valid] = TIER_POINTS[OUT_OF_SPEC]
    out[valid & marginal] = TIER_POINTS[MARGINAL]
    out[valid & good] = TIER_POINTS[GOOD]

    return out


def describe(spec: dict, unit: str = "") -> dict[str, str]:
    """Legend text naming each tier's actual limits, e.g. ``"Good (<=45 m)"``.

    Built from the specification rather than from anything axis-derived, so the
    numbers quoted are the ones classification actually used.
    """
    kind = validate_spec(spec)
    suffix = f" {unit}" if unit else ""

    if kind == CENTERED:
        g, m = spec["good"], spec["marginal"]
        return {GOOD: f"{GOOD} (<=±{g:g}{suffix})",
                MARGINAL: f"{MARGINAL} (±{g:g}-{m:g}{suffix})",
                OUT_OF_SPEC: f"{OUT_OF_SPEC} (>±{m:g}{suffix})"}
    if kind == ASCENDING:
        g, m = spec["good_max"], spec["marginal_max"]
        return {GOOD: f"{GOOD} (<={g:g}{suffix})",
                MARGINAL: f"{MARGINAL} ({g:g}-{m:g}{suffix})",
                OUT_OF_SPEC: f"{OUT_OF_SPEC} (>{m:g}{suffix})"}
    if kind == DESCENDING:
        g, m = spec["good_min"], spec["marginal_min"]
        return {GOOD: f"{GOOD} (>={g:g}{suffix})",
                MARGINAL: f"{MARGINAL} ({m:g}-{g:g}{suffix})",
                OUT_OF_SPEC: f"{OUT_OF_SPEC} (<{m:g}{suffix})"}
    gl, gh = spec["good_min"], spec["good_max"]
    ml, mh = spec["marginal_min"], spec["marginal_max"]
    return {GOOD: f"{GOOD} ({gl:g}-{gh:g}{suffix})",
            MARGINAL: f"{MARGINAL} ({ml:g}-{gl:g}, {gh:g}-{mh:g}{suffix})",
            OUT_OF_SPEC: f"{OUT_OF_SPEC} (<{ml:g} or >{mh:g}{suffix})"}


def is_tier(codes, tier: str) -> np.ndarray:
    """Which entries of ``codes`` are in ``tier``. Unclassified is never in one.

    Replaces comparing a column to a label. ``NaN == anything`` is false, which
    is the wanted answer: a sounding nobody classified is not Out of Spec, and
    is not Good either.
    """
    return np.asarray(codes, dtype=float) == TIER_POINTS[tier]


def worst_of(arrays, require_all: bool = False) -> np.ndarray:
    """The most severe code per position, across several verdict columns.

    Severity is *lowest*, since the scale runs worst-to-best.

    By default missing only where *every* column is missing, because separate
    checks ask independent questions: a sounding whose height was never measured
    but whose speed was grossly out of spec is out of spec, and waiting on the
    unmeasured one would suppress a finding already certain.

    ``require_all`` inverts that, giving missing where *any* column is missing.
    That is right when the columns are components of one measurement rather than
    separate questions — a sounding whose roll was not recorded has an unknown
    attitude, and reporting its pitch verdict as its attitude verdict would
    claim more than was measured.
    """
    stack = np.vstack([np.asarray(a, dtype=float) for a in arrays])
    with warnings.catch_warnings():
        # An all-NaN column is the ordinary "nothing classified this" case, not
        # something to warn about.
        warnings.simplefilter("ignore", RuntimeWarning)
        worst = np.nanmin(stack, axis=0)
    if require_all:
        worst[np.isnan(stack).any(axis=0)] = np.nan
    return worst


def labels(codes):
    """Codes back to their tier names, for anything read by a person.

    ``None`` where unclassified, which every vector driver can write; the
    exports replace it with an explicit label of their own, since shapefile
    cannot distinguish a null from an empty string.
    """
    lookup = {v: k for k, v in TIER_POINTS.items()}
    return [lookup.get(v) for v in np.asarray(codes, dtype=float)]


#: Where the set of verdict columns is recorded, so a file loaded back from disk
#: can say which of its numeric columns are verdicts rather than measurements.
#: ``ql_verdict`` is one and ``ql_score`` is not, and no naming rule separates
#: them — encoding verdicts as numbers is what took the dtype away as an answer.
#: ``model_info`` round-trips as text, so this is stored space-joined; column
#: names cannot contain spaces in this format, which makes that unambiguous.
VERDICT_REGISTRY = "qc_verdict_columns"

#: Where the thresholds a dataset was judged against are recorded.
#:
#: Culling and contract compliance are different questions, so a culling filter
#: restates its own limits rather than reusing a QC check's — a sounding flown at
#: 70 m against a 65 m contract is out of spec and still perfectly invertible.
#: But the two are *related*, and a sensible starting point for "where has the
#: geometry assumption broken down" is "where did QC stop calling it acceptable".
#: So a filter can take a QC threshold as its default, which needs the number to
#: outlive the run that chose it: QC and culling are separate pipeline documents,
#: often separate invocations.
#:
#: Encoded rather than serialized because ``model_info`` round-trips as flat
#: text — a dict arrives back as an unparseable repr. One space-separated entry
#: per step, colon-separated fields, ``name=value`` for each parameter.
THRESHOLD_REGISTRY = "qc_thresholds"


def mark_thresholds(dataset, step: str, spec: dict) -> None:
    """Record what ``step`` judged this dataset against."""
    known = recorded_thresholds(dataset)
    known[step] = dict(spec)
    entries = []
    for name, values in sorted(known.items()):
        kind = values.get("kind", "")
        params = ":".join(f"{k}={v:g}" for k, v in sorted(values.items())
                          if k != "kind" and isinstance(v, (int, float)))
        entries.append(f"{name}:{kind}:{params}")
    dataset.survey.xyz.model_info[THRESHOLD_REGISTRY] = " ".join(entries)


def recorded_thresholds(dataset) -> dict[str, dict]:
    """Every step's threshold specification, as recorded on the dataset."""
    raw = dataset.survey.xyz.model_info.get(THRESHOLD_REGISTRY) or ""
    if not isinstance(raw, str):
        raw = " ".join(map(str, raw))

    out: dict[str, dict] = {}
    for entry in raw.split():
        step, _, rest = entry.partition(":")
        kind, _, params = rest.partition(":")
        if not step:
            continue
        spec: dict = {"kind": kind} if kind else {}
        for pair in params.split(":"):
            name, _, value = pair.partition("=")
            if name and value:
                try:
                    spec[name] = float(value)
                except ValueError:
                    continue
        out[step] = spec
    return out


def mark_verdicts(dataset, *names: str) -> None:
    """Record that these ``flightlines`` columns hold verdict codes."""
    info = dataset.survey.xyz.model_info
    known = _registered(info)
    for name in names:
        if name and name not in known:
            known.append(name)
    info[VERDICT_REGISTRY] = " ".join(known)


def _registered(info) -> list[str]:
    raw = info.get(VERDICT_REGISTRY) or ""
    if isinstance(raw, str):
        return raw.split()
    return [str(name) for name in raw]


def verdict_columns(dataset) -> list[str]:
    """The recorded verdict columns still present in ``flightlines``."""
    frame = dataset.flightlines
    if frame is None:
        return []
    return [n for n in _registered(dataset.survey.xyz.model_info)
            if n in frame.columns]


def worst_verdict(codes) -> str | None:
    """The most severe tier present, or ``None`` if nothing was classified."""
    values = np.asarray(codes, dtype=float)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return {v: k for k, v in TIER_POINTS.items()}.get(float(values.min()))


def tier_counts(codes) -> dict[str, int]:
    """How many soundings landed in each tier, plus how many got no verdict.

    Every tier appears even at zero — a reader should see "OOS: 0"
    rather than have to infer it from an absence.
    """
    values = np.asarray(codes, dtype=float)
    counts = {tier: int(is_tier(values, tier).sum()) for tier in TIERS}
    counts["unclassified"] = int(np.isnan(values).sum())
    return counts
