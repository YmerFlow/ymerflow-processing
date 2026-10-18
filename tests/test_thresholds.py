"""The Good / Marginal / Out of Spec model shared by every QC check."""

import numpy as np
import pandas as pd
import pytest

from conftest import tier_names

from ymerflow_processing.qc.thresholds import (
    GOOD, MARGINAL, OUT_OF_SPEC, TIER_POINTS, TIERS,
    classify, describe, tier_counts, validate_spec, worst_verdict,
)

ASCENDING_SPEC = {"kind": "ascending", "good_max": 45, "marginal_max": 65}
CENTERED_SPEC = {"kind": "centered", "good": 5, "marginal": 7.5}
DESCENDING_SPEC = {"kind": "descending", "good_min": 20, "marginal_min": 10}
WINDOW_SPEC = {"kind": "window", "good_min": 70, "good_max": 100,
               "marginal_min": 60, "marginal_max": 120}


# ── The four kinds ───────────────────────────────────────────────────────────

def test_ascending_higher_is_worse():
    verdicts = classify([10, 45, 50, 65, 80], ASCENDING_SPEC)
    assert tier_names(verdicts) == [GOOD, GOOD, MARGINAL, MARGINAL, OUT_OF_SPEC]


def test_centered_deviation_either_way_is_equal():
    verdicts = classify([0, 5, -5, 6, -6, 9, -9], CENTERED_SPEC)
    assert tier_names(verdicts) == [GOOD, GOOD, GOOD, MARGINAL, MARGINAL,
                              OUT_OF_SPEC, OUT_OF_SPEC]


def test_descending_lower_is_worse():
    verdicts = classify([30, 20, 15, 10, 5], DESCENDING_SPEC)
    assert tier_names(verdicts) == [GOOD, GOOD, MARGINAL, MARGINAL, OUT_OF_SPEC]


def test_window_bounds_may_be_asymmetric():
    """What ``centered`` cannot express: ground speed, 70-100 km/h."""
    verdicts = classify([55, 60, 85, 100, 110, 130], WINDOW_SPEC)
    assert tier_names(verdicts) == [OUT_OF_SPEC, MARGINAL, GOOD, GOOD,
                              MARGINAL, OUT_OF_SPEC]


# ── Boundaries ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("spec,value", [
    (ASCENDING_SPEC, 45),
    (CENTERED_SPEC, 5),
    (CENTERED_SPEC, -5),
    (DESCENDING_SPEC, 20),
    (WINDOW_SPEC, 70),
    (WINDOW_SPEC, 100),
])
def test_a_value_exactly_on_the_good_limit_is_good(spec, value):
    """Decided deliberately: 'a maximum of 45 m' means 45 m meets spec.

    The prior implementation is inconsistent between kinds — centered inclusive,
    ascending exclusive. Verdicts reach a client as a contract deliverable, so
    the rule they are computed by has to be sayable in one sentence.
    """
    assert tier_names(classify([value], spec))[0] == GOOD


@pytest.mark.parametrize("spec,value", [
    (ASCENDING_SPEC, 65),
    (CENTERED_SPEC, 7.5),
    (DESCENDING_SPEC, 10),
])
def test_a_value_exactly_on_the_marginal_limit_is_marginal(spec, value):
    assert tier_names(classify([value], spec))[0] == MARGINAL


def test_tiers_partition_the_line_with_no_gap_or_overlap():
    """Every finite value lands in exactly one tier."""
    values = np.linspace(-200, 200, 4001)
    for spec in (ASCENDING_SPEC, CENTERED_SPEC, DESCENDING_SPEC, WINDOW_SPEC):
        verdicts = classify(values, spec)
        assert pd.Series(verdicts).notna().all(), spec["kind"]


# ── Absent and unmeasurable values ───────────────────────────────────────────

def test_missing_values_get_no_verdict():
    """No measurement is different from failing one."""
    verdicts = classify([10.0, np.nan, 80.0, np.inf, -np.inf], ASCENDING_SPEC)
    assert tier_names(verdicts) == [GOOD, None, OUT_OF_SPEC, None, None]
    # NaN rather than a sentinel code, so nothing arithmetic can mistake it for
    # a tier and no reader has to know which number means "not evaluated".
    assert np.isnan(verdicts[1]) and np.isnan(verdicts[3])


def test_an_empty_input_classifies_to_nothing():
    assert len(classify([], ASCENDING_SPEC)) == 0


def test_verdicts_are_numeric_and_ordered_worst_to_best():
    """The encoding the frontend requires, and the direction it runs.

    A label column fails the plot layer's ``ArrayBuffer.isView`` filter and is
    invisible, so verdicts are codes. The direction is the same one the grade
    uses — higher is better — so a file never carries two numeric scales
    pointing opposite ways.
    """
    verdicts = classify([10, 50, 80], ASCENDING_SPEC)

    assert verdicts.dtype == np.float32
    assert list(verdicts) == [2.0, 1.0, 0.0]
    assert TIER_POINTS[OUT_OF_SPEC] < TIER_POINTS[MARGINAL] < TIER_POINTS[GOOD]
    # Severity is lowest, which is what makes the worst verdict a min().
    assert worst_verdict(verdicts) == OUT_OF_SPEC


# ── Validation ───────────────────────────────────────────────────────────────

def test_an_unknown_kind_is_rejected():
    with pytest.raises(ValueError, match="Unknown threshold kind"):
        classify([1], {"kind": "sideways", "good": 1, "marginal": 2})


def test_a_missing_parameter_names_what_was_needed():
    with pytest.raises(ValueError, match="good_max"):
        classify([1], {"kind": "ascending", "marginal_max": 65})


@pytest.mark.parametrize("spec", [
    {"kind": "ascending", "good_max": 65, "marginal_max": 45},
    {"kind": "centered", "good": 8, "marginal": 5},
    {"kind": "descending", "good_min": 10, "marginal_min": 20},
    {"kind": "window", "good_min": 60, "good_max": 120,
     "marginal_min": 70, "marginal_max": 100},
])
def test_inverted_thresholds_are_rejected(spec):
    """Marginal tighter than Good would classify nothing Marginal, silently."""
    with pytest.raises(ValueError):
        validate_spec(spec)


# ── Reporting helpers ────────────────────────────────────────────────────────

def test_describe_quotes_the_actual_limits():
    text = describe(ASCENDING_SPEC, "m")
    assert text[GOOD] == "Good (<=45 m)"
    assert text[MARGINAL] == "Marginal (45-65 m)"
    assert text[OUT_OF_SPEC] == "OOS (>65 m)"


def test_tier_counts_reports_every_tier_even_at_zero():
    counts = tier_counts(classify([10, 20, np.nan], ASCENDING_SPEC))
    assert counts == {GOOD: 2, MARGINAL: 0, OUT_OF_SPEC: 0, "unclassified": 1}
