"""How verdicts are stored, and why it is not negotiable.

The plot layer this library feeds enumerates what it can draw with

    Object.keys(fl).filter(k => ArrayBuffer.isView(fl[k]))

and then coerces through ``toFloat32Array``. A column of labels is not a typed
array after the msgpack round-trip, so it fails that filter and **never appears
at all** — not awkward to plot, absent. Verdicts are therefore numeric codes.

None of that is visible from inside this library, which is exactly why it is
pinned here: every one of these assertions protects a contract whose other half
lives in another repository, and the failure mode is a QC map that silently has
no QC on it.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset, step, tier_names
from ymerflow_processing.core.dataset import load
from ymerflow_processing.core.pipeline import Pipeline, QCRunner
from ymerflow_processing.core.qc.thresholds import (
    ASCENDING, GOOD, MARGINAL, MAX_POINTS, OUT_OF_SPEC, TIER_POINTS, classify,
    is_tier, labels, mark_verdicts, verdict_columns, worst_of,
)

SPEC = {"kind": ASCENDING, "good_max": 45, "marginal_max": 65}
PIPELINE = """
kind: qc
steps:
  - height_above_ground: {good_max: 45, marginal_max: 65}
"""


def run(heights, **kw):
    dataset = make_dataset(n=len(heights), n_lines=2, height=list(heights), **kw)
    return QCRunner(Pipeline.parse(PIPELINE)).run(dataset)


# ── The storage contract ─────────────────────────────────────────────────────

def test_a_verdict_column_is_a_float_array_not_labels():
    """The whole point. A label column is invisible downstream."""
    dataset, _ = run(np.full(20, 30.0))
    column = dataset.flightlines["qs_height"]

    assert np.issubdtype(column.dtype, np.floating)
    assert not isinstance(column.dtype, pd.CategoricalDtype)
    # And a typed array is what survives msgpack as one.
    assert column.to_numpy().dtype.kind == "f"


def test_float32_rather_than_an_integer_type():
    """Because "not evaluated" has to be representable, and int cannot hold NaN."""
    codes = classify([10.0, np.nan, 80.0], SPEC)

    assert codes.dtype == np.float32
    assert np.isnan(codes[1])
    # Exact, not approximate: float32 holds integers exactly well past 2.
    assert codes[0] == 2.0 and codes[2] == 0.0


def test_the_scale_runs_worst_to_best_like_the_grade():
    """One numeric direction per file. Two pointing opposite ways is the bug
    this ordering exists to prevent — a colorscale inverted against the grade
    renders out-of-spec soundings as good, which is worse than no map."""
    assert TIER_POINTS[OUT_OF_SPEC] == 0.0
    assert TIER_POINTS[MARGINAL] == 1.0
    assert TIER_POINTS[GOOD] == 2.0
    assert MAX_POINTS == 2.0


def test_the_domain_is_fixed_rather_than_derived_from_the_data():
    """A colorscale inferring its range from the data rescales when a flight
    happens to contain no out-of-spec soundings, and Marginal then renders in
    the out-of-spec color. The range is a property of the scheme, not of any
    one delivery, so it is stated here for whoever reads it."""
    clean, _ = run(np.full(20, 30.0))
    mixed, _ = run(np.concatenate([np.full(10, 30.0), np.full(10, 90.0)]))

    assert clean.flightlines["qs_height"].max() == MAX_POINTS
    assert mixed.flightlines["qs_height"].min() == TIER_POINTS[OUT_OF_SPEC]
    # The clean delivery's *observed* range is a single value. Anything reading
    # that as the domain would paint Good and OOS identically.
    assert clean.flightlines["qs_height"].nunique() == 1


# ── Not evaluated ────────────────────────────────────────────────────────────

def test_unevaluated_is_nan_and_not_a_sentinel_code():
    """A sentinel like -1 invites arithmetic mistakes and has to be known about
    by every reader. NaN is distinguishable from every valid code by anything
    that already handles floats."""
    codes = classify([np.nan, np.inf, 10.0], SPEC)

    assert np.isnan(codes[0]) and np.isnan(codes[1])
    assert not (codes[:2] < 0).any()
    assert tier_names(codes) == [None, None, GOOD]


def test_unevaluated_is_in_no_tier_at_all():
    """Not Good, and not Out of Spec either."""
    codes = classify([np.nan], SPEC)
    for tier in (GOOD, MARGINAL, OUT_OF_SPEC):
        assert not is_tier(codes, tier).any()


def test_nan_survives_the_xyz_round_trip(tmp_path):
    """The bug class this library has already been bitten by once: a value that
    cannot be written to a whitespace-delimited format makes the whole file
    unparseable, and the header still looks right."""
    dataset, _ = run(np.concatenate([np.full(10, 30.0), np.full(10, 90.0)]))
    dataset.flightlines.loc[0, "qs_height"] = np.nan
    xyz, gex = tmp_path / "r.xyz", tmp_path / "r.gex"
    dataset.survey.dump(xyzfile=str(xyz), gexfile=str(gex))

    rows = [ln for ln in xyz.read_text().splitlines()
            if ln.strip() and not ln.lstrip().startswith("/")]
    assert len({len(ln.split()) for ln in rows}) == 1, "NaN broke the field count"

    back = load(xyz, gex).flightlines["qs_height"]
    assert np.isnan(back[0])
    assert tier_names(back)[1] == GOOD
    assert tier_names(back)[19] == OUT_OF_SPEC


# ── The registry ─────────────────────────────────────────────────────────────

def test_the_dataset_records_which_columns_are_verdicts():
    """Encoding verdicts as numbers took away the dtype that used to answer
    this. Nothing in the values separates ql_verdict from ql_score."""
    dataset, _ = run(np.full(20, 30.0))

    recorded = verdict_columns(dataset)
    assert "qs_height" in recorded and "qs_verdict" in recorded
    # Measures and grades sit right beside them and are not verdicts.
    for measure in ("qs_score", "ql_oos", "ql_score", "ql_n", "qs_reasons"):
        assert measure not in recorded, measure


def test_the_registry_survives_a_round_trip(tmp_path):
    """Because the export runs on a dataset loaded back from disk, with no
    report to consult."""
    dataset, _ = run(np.full(20, 30.0))
    before = verdict_columns(dataset)
    xyz, gex = tmp_path / "r.xyz", tmp_path / "r.gex"
    dataset.survey.dump(xyzfile=str(xyz), gexfile=str(gex))

    assert verdict_columns(load(xyz, gex)) == before


def test_registering_is_idempotent():
    """R4. Re-running a check must not grow the list."""
    dataset = make_dataset(n=4, n_lines=1)
    mark_verdicts(dataset, "qs_height", "qs_height")
    mark_verdicts(dataset, "qs_height")
    assert verdict_columns(dataset) == []      # not written to the frame yet

    dataset.flightlines["qs_height"] = classify(np.full(4, 30.0), SPEC)
    assert verdict_columns(dataset) == ["qs_height"]


def test_a_registered_column_that_was_dropped_is_not_reported():
    dataset, _ = run(np.full(20, 30.0))
    dataset.survey.xyz.flightlines = dataset.flightlines.drop(columns=["qs_height"])

    assert "qs_height" not in verdict_columns(dataset)
    assert "qs_verdict" in verdict_columns(dataset)      # the rest still stand


# ── Translating back ─────────────────────────────────────────────────────────

def test_labels_are_recoverable_for_anything_read_by_a_person():
    codes = classify([10.0, 50.0, 80.0, np.nan], SPEC)
    assert labels(codes) == [GOOD, MARGINAL, OUT_OF_SPEC, None]


def test_the_report_carries_the_legend_rather_than_the_column():
    """A column of 0/1/2 with no legend is worse than no column. R3 already
    gives every step a structured record to carry it, so a fourth tier would
    never become a frontend change."""
    _, report = run(np.full(20, 30.0))
    thresholds = step(report, "height_above_ground").distributions["thresholds"]

    assert set(thresholds) == {GOOD, MARGINAL, OUT_OF_SPEC}
    assert thresholds[GOOD].startswith(GOOD)


# ── Combining ────────────────────────────────────────────────────────────────

def test_worst_of_is_a_minimum_now_that_the_scale_is_inverted():
    combined = worst_of([[2.0, 2.0], [1.0, 0.0]])
    assert list(combined) == [1.0, 0.0]


def test_worst_of_ignores_the_unmeasured_by_default():
    """Independent questions: an unmeasured height must not suppress a speed
    breach that is already certain."""
    combined = worst_of([[np.nan, np.nan], [0.0, np.nan]])
    assert combined[0] == 0.0
    assert np.isnan(combined[1])


def test_worst_of_can_require_every_component():
    """Right when the columns are two axes of one measurement rather than two
    questions — a sounding missing roll has an unknown attitude."""
    combined = worst_of([[2.0, np.nan], [1.0, 0.0]], require_all=True)
    assert combined[0] == 1.0
    assert np.isnan(combined[1])
