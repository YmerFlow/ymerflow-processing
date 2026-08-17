"""qc_03 — ground speed, end to end."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.pipeline import QC_GROUP, Pipeline, QCRunner, load_steps
from ymerflow_processing.core.qc.qc_03_speed import (
    derive_speed, find_speed_column, ground_speed,
)
from ymerflow_processing.core.qc.thresholds import GOOD, MARGINAL, OUT_OF_SPEC

# A contract window: 70-100 km/h good, 60-120 tolerated.
SPEC = {"good_min": 70, "good_max": 100, "marginal_min": 60, "marginal_max": 120}


def speeds(*values, **kw):
    return make_dataset(n=len(values), n_lines=1, speed=list(values), **kw)


# ── Both ends of the window ──────────────────────────────────────────────────

def test_too_slow_and_too_fast_are_both_out_of_spec():
    """The reason this is a window and not an ascending threshold."""
    dataset, _ = ground_speed(speeds(50, 65, 85, 110, 130), **SPEC)
    assert list(dataset.flightlines["qc_speed"]) == [
        OUT_OF_SPEC, MARGINAL, GOOD, MARGINAL, OUT_OF_SPEC]


def test_the_window_may_be_asymmetric():
    """What centered cannot express: the good band is not about zero."""
    dataset, _ = ground_speed(speeds(70, 100), **SPEC)
    assert list(dataset.flightlines["qc_speed"]) == [GOOD, GOOD]


def test_which_end_the_breach_was_at_is_reported():
    """'Thirty out of spec' means opposite things to a contractor either way."""
    _, report = ground_speed(speeds(40, 45, 130), **SPEC)
    assert report.distributions["out_of_spec_by_end"] == {
        "too_slow": 2, "too_fast": 1}


# ── Where the speed comes from ───────────────────────────────────────────────

def test_a_recorded_column_is_preferred():
    """It is what the instrument measured; deriving would substitute arithmetic."""
    dataset = make_dataset(n=3, n_lines=1, speed=[85, 85, 85],
                           seconds=[0, 0.5, 1.0])
    _, report = ground_speed(dataset, **SPEC)
    assert report.distributions["speed"]["source"] == "recorded ('GdSpeed')"


def test_speed_is_derived_when_the_survey_records_none():
    """10 m apart, half a second apart -> 20 m/s -> 72 km/h."""
    dataset = make_dataset(n=4, n_lines=1, seconds=[0, 0.5, 1.0, 1.5])
    dataset, report = ground_speed(dataset, **SPEC)

    assert report.distributions["speed"]["source"] == "derived from position and time"
    assert report.distributions["speed"]["before"]["max"] == pytest.approx(72.0)
    assert list(dataset.flightlines["qc_speed"])[1:] == [GOOD] * 3


def test_the_first_sounding_of_a_line_has_no_speed():
    """Nothing preceded it, so there is no step to measure."""
    dataset = make_dataset(n=4, n_lines=1, seconds=[0, 0.5, 1.0, 1.5])
    dataset, _ = ground_speed(dataset, **SPEC)
    assert pd.isna(dataset.flightlines["qc_speed"][0])


def test_derivation_does_not_cross_the_turn_between_lines():
    """The single most important thing this must not get wrong.

    The fixture's two lines sit 1000 m apart, so a step taken across the
    boundary covers ~1000 m rather than 10 m. Measured as a speed it is a ferry
    between lines, not acquisition, and would read as grossly out of spec.
    """
    dataset = make_dataset(n=6, n_lines=2, seconds=[0, 0.5, 1.0, 1.5, 2.0, 2.5])
    dataset, report = ground_speed(dataset, **SPEC)

    verdicts = list(dataset.flightlines["qc_speed"])
    assert pd.isna(verdicts[0])          # first of line 100
    assert pd.isna(verdicts[3])          # first of line 101 — the turn
    assert verdicts[1:3] == [GOOD, GOOD]
    assert verdicts[4:] == [GOOD, GOOD]
    # Nothing absurd leaked into the distribution.
    assert report.distributions["speed"]["before"]["max"] == pytest.approx(72.0)


def test_derive_speed_directly():
    """Unit-level: within a line, distance over time; across a line, nothing."""
    speed = derive_speed(easting=[0, 10, 20, 0], northing=[0, 0, 0, 1000],
                         seconds=[0, 1, 2, 3], lines=[1, 1, 1, 2])
    assert np.isnan(speed[0])            # nothing before it
    assert speed[1] == pytest.approx(10.0)
    assert speed[2] == pytest.approx(10.0)
    assert np.isnan(speed[3])            # the turn


def test_a_zero_time_step_yields_no_speed():
    """Duplicate timestamps must not divide by zero."""
    speed = derive_speed([0, 10], [0, 0], [5.0, 5.0], [1, 1])
    assert np.isnan(speed[1])


def test_derived_speed_is_scaled_to_the_contract_unit():
    dataset = make_dataset(n=3, n_lines=1, seconds=[0, 0.5, 1.0])
    _, report = ground_speed(dataset, scale=1.0, **SPEC)   # leave as m/s
    assert report.distributions["speed"]["before"]["max"] == pytest.approx(20.0)


@pytest.mark.parametrize("name", ["ground_speed", "GdSpeed", "speed"])
def test_a_recorded_column_is_found_under_its_usual_spellings(name):
    frame = pd.DataFrame({name: [85.0], "other": [0]})
    assert find_speed_column(frame) == name


# ── What it writes ───────────────────────────────────────────────────────────

def test_changes_no_measurement_uncertainty_or_mask():
    dataset = speeds(85, 130)
    before = {k: v.copy() for k, v in dataset.layer_data.items()}
    dataset, _ = ground_speed(dataset, **SPEC)
    for key, array in before.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


def test_writes_the_denormalized_line_verdict():
    dataset = make_dataset(n=6, n_lines=2, speed=[85] * 5 + [130])
    dataset, _ = ground_speed(dataset, **SPEC)

    verdict = dataset.flightlines["qc_speed_line"]
    assert list(verdict[:3]) == [GOOD] * 3
    assert list(verdict[3:]) == [OUT_OF_SPEC] * 3


def test_re_running_is_idempotent_and_says_so():
    dataset = speeds(85, 130)
    dataset, first = ground_speed(dataset, **SPEC)
    dataset, second = ground_speed(dataset, **SPEC)

    assert list(dataset.flightlines["qc_speed"]) == [GOOD, OUT_OF_SPEC]
    assert not any("Overwrote" in n for n in first.notes)
    assert any("qc_speed" in n and "Overwrote" in n for n in second.notes)


# ── The report (R3) ──────────────────────────────────────────────────────────

def test_reports_the_distribution_before_and_after():
    dataset = speeds(*([85] * 8 + [130] * 2))
    _, report = ground_speed(dataset, **SPEC)

    speed = report.distributions["speed"]
    assert speed["before"]["n"] == 10
    assert speed["before"]["max"] == 130.0
    assert speed["after"]["n"] == 8
    assert speed["after"]["max"] == 85.0


def test_reports_the_thresholds_actually_used():
    _, report = ground_speed(speeds(85), **SPEC)
    assert report.distributions["thresholds"][GOOD] == "Good (70-100 km/h)"


def test_reports_both_grains():
    dataset = make_dataset(n=8, n_lines=4, n_flights=2,
                           speed=[85] * 4 + [130] * 4)
    _, report = ground_speed(dataset, **SPEC)

    assert report.distributions["per_flight"]["1"]["fraction_out_of_spec"] == 0.0
    assert report.distributions["per_flight"]["2"]["fraction_out_of_spec"] == 1.0
    assert len(report.distributions["per_line"]) == 4


def test_the_report_survives_serialization():
    _, report = ground_speed(speeds(85, 130), **SPEC)
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["step"] == "ground_speed"
    assert restored["distributions"]["out_of_spec_by_end"]["too_fast"] == 1


# ── Absent inputs ────────────────────────────────────────────────────────────

def test_neither_recorded_nor_derivable_is_a_note_not_an_error():
    dataset = make_dataset(n=3, n_lines=1)          # no speed, no time
    dataset, report = ground_speed(dataset, **SPEC)

    assert "qc_speed" not in dataset.flightlines.columns
    assert any("cannot be derived" in n for n in report.notes)


def test_a_column_named_by_the_caller_and_absent_is_an_error():
    with pytest.raises(KeyError, match="No column 'knots'"):
        ground_speed(speeds(85), column="knots", **SPEC)


def test_an_explicitly_named_column_is_used():
    dataset = speeds(85)
    dataset.flightlines["gps_speed"] = [130.0]
    dataset, _ = ground_speed(dataset, column="gps_speed", **SPEC)
    assert dataset.flightlines["qc_speed"][0] == OUT_OF_SPEC


def test_an_empty_dataset_reports_rather_than_raises():
    dataset = make_dataset(n=0, n_lines=1, height=[], pitch=[], roll=[], speed=[])
    _, report = ground_speed(dataset, **SPEC)
    assert report.soundings_total == 0


def test_runs_without_any_channels():
    """Speed is a property of the aircraft, not of the method."""
    dataset = speeds(85, 130)
    dataset.survey.xyz.model_dict["layer_data"] = {}

    assert dataset.n_channels == 0
    dataset, _ = ground_speed(dataset, **SPEC)
    assert list(dataset.flightlines["qc_speed"]) == [GOOD, OUT_OF_SPEC]


# ── Registration and the runner ──────────────────────────────────────────────

def test_registered_under_the_qc_entry_point_group():
    assert "ground_speed" in load_steps(QC_GROUP)


def test_all_three_checks_run_from_one_pipeline():
    pipeline = Pipeline.parse("""
        kind: qc
        steps:
          - height_above_ground: {good_max: 45, marginal_max: 65}
          - system_attitude: {good: 5, marginal: 7.5}
          - ground_speed: {good_min: 70, good_max: 100,
                           marginal_min: 60, marginal_max: 120}
    """)
    dataset = make_dataset(n=3, n_lines=1, height=[30.0, 30.0, 80.0],
                           pitch=[0.0, 9.0, 0.0], roll=[0.0] * 3,
                           speed=[85.0, 85.0, 130.0])

    dataset, report = QCRunner(pipeline).run(dataset)

    assert list(dataset.flightlines["qc_height"]) == [GOOD, GOOD, OUT_OF_SPEC]
    assert list(dataset.flightlines["qc_attitude"]) == [GOOD, OUT_OF_SPEC, GOOD]
    assert list(dataset.flightlines["qc_speed"]) == [GOOD, GOOD, OUT_OF_SPEC]
    assert len(report.steps) == 3
