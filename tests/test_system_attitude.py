"""qc_02 — system attitude, end to end."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset, named, step, tier_names
from ymerflow_processing.pipeline import QC_GROUP, Pipeline, QCRunner, load_steps
from ymerflow_processing.qc.attitude import system_attitude
from ymerflow_processing.qc.thresholds import GOOD, MARGINAL, OUT_OF_SPEC

# Representative attitude spec. Real thresholds are contract parameters per survey.
SPEC = {"good": 5, "marginal": 7.5}


def attitude(pitch, roll, **kw):
    return make_dataset(n=len(pitch), n_lines=1, pitch=pitch, roll=roll, **kw)


# ── Classification ───────────────────────────────────────────────────────────

def test_deviation_either_way_is_equal():
    """centered: it is distance from level that matters, not which way."""
    dataset = attitude([0, 5, -5, 6, -6, 9, -9], [0] * 7)
    dataset, _ = system_attitude(dataset, **SPEC)

    assert tier_names(dataset.flightlines["qs_pitch"]) == [
        GOOD, GOOD, GOOD, MARGINAL, MARGINAL, OUT_OF_SPEC, OUT_OF_SPEC]


def test_the_two_axes_are_judged_independently():
    dataset = attitude(pitch=[0, 9], roll=[9, 0])
    dataset, _ = system_attitude(dataset, **SPEC)

    assert tier_names(dataset.flightlines["qs_pitch"]) == [GOOD, OUT_OF_SPEC]
    assert tier_names(dataset.flightlines["qs_roll"]) == [OUT_OF_SPEC, GOOD]


def test_the_combined_verdict_is_the_worse_of_the_two():
    """What a later culling step thresholds on, so it must not be optimistic."""
    dataset = attitude(pitch=[0, 6, 0, 9], roll=[0, 0, 9, 6])
    dataset, _ = system_attitude(dataset, **SPEC)

    assert tier_names(dataset.flightlines["qs_tilt"]) == [
        GOOD, MARGINAL, OUT_OF_SPEC, OUT_OF_SPEC]


def test_a_sounding_missing_one_axis_has_no_attitude_verdict():
    """Half a measurement is not a verdict: an unmeasured roll could be anything."""
    dataset = attitude(pitch=[1.0, 1.0], roll=[1.0, np.nan])
    dataset, _ = system_attitude(dataset, **SPEC)

    assert tier_names(dataset.flightlines["qs_pitch"])[1] == GOOD      # pitch is still judged
    assert pd.isna(dataset.flightlines["qs_roll"][1])
    assert pd.isna(dataset.flightlines["qs_tilt"][1])


# ── Total tilt: reported, never classified ───────────────────────────────────

def test_total_tilt_is_reported():
    _, report = system_attitude(attitude([9.0], [9.0]), **SPEC)
    total = report.distributions["total_tilt"]["before"]
    # cos(total) = cos(9) * cos(9)
    assert total["max"] == pytest.approx(12.70, abs=0.01)


def test_two_axes_inside_spec_can_combine_past_it():
    """The reason total tilt is worth reporting at all: 5 and 5 make 7.07."""
    dataset = attitude([5.0], [5.0])
    dataset, report = system_attitude(dataset, **SPEC)

    assert tier_names(dataset.flightlines["qs_tilt"])[0] == GOOD        # both axes pass
    assert report.distributions["total_tilt"]["before"]["max"] == pytest.approx(
        7.07, abs=0.01)                                         # yet 7.07 total


def test_total_tilt_carries_no_verdict():
    """The contract sets per-axis limits; a combined threshold would be invented."""
    _, report = system_attitude(attitude([5.0], [5.0]), **SPEC)
    assert "counts" not in report.distributions["total_tilt"]
    assert "not classified" in report.distributions["total_tilt"]["note"]


def test_total_tilt_is_absent_with_only_one_axis():
    dataset = attitude([1.0], [1.0])
    dataset.flightlines.drop(columns=["tilt_y"], inplace=True)
    _, report = system_attitude(dataset, **SPEC)
    assert "total_tilt" not in report.distributions


# ── The zeroed-sensor trap ───────────────────────────────────────────────────

def test_an_identically_zero_axis_is_flagged():
    """Attitude is zeroed when unreliable, so all-zero is a failed sensor —
    reporting a flawless flight would be exactly backwards."""
    dataset, report = system_attitude(attitude([0.0] * 5, [1.0] * 5), **SPEC)

    assert any("exactly zero for every sounding" in n and "pitch" in n
               for n in report.notes)
    assert not any("roll" in n and "exactly zero" in n for n in report.notes)


def test_a_genuinely_level_flight_is_not_flagged():
    """Small non-zero values are a real level flight, not a dead sensor."""
    _, report = system_attitude(attitude([0.1, -0.2, 0.05], [0.0, 0.1, -0.1]),
                                **SPEC)
    assert not any("exactly zero" in n for n in report.notes)


def test_a_zeroed_axis_is_still_classified():
    """Flagged, not skipped — the verdict is what it is, with a caveat attached."""
    dataset, _ = system_attitude(attitude([0.0] * 3, [1.0] * 3), **SPEC)
    assert tier_names(dataset.flightlines["qs_pitch"]) == [GOOD] * 3


# ── What it writes ───────────────────────────────────────────────────────────

def test_changes_no_measurement_uncertainty_or_mask():
    """R6: QC writes attributes only."""
    dataset = attitude([1.0, 9.0], [1.0, 1.0])
    before = {k: v.copy() for k, v in dataset.layer_data.items()}

    dataset, _ = system_attitude(dataset, **SPEC)

    for key, array in before.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)



def test_the_output_columns_are_nameable():
    dataset = attitude([1.0], [1.0])
    dataset, _ = system_attitude(dataset, pitch_output="p", roll_output="r",
                                 output_column="att",
                                 **SPEC)
    assert {"p", "r", "att"} <= set(dataset.flightlines.columns)


def test_re_running_is_idempotent_and_says_so():
    dataset = attitude([1.0, 9.0], [1.0, 1.0])
    dataset, first = system_attitude(dataset, **SPEC)
    dataset, second = system_attitude(dataset, **SPEC)

    assert tier_names(dataset.flightlines["qs_pitch"]) == [GOOD, OUT_OF_SPEC]
    assert not any("Overwrote" in n for n in first.notes)
    assert any("qs_pitch" in n and "Overwrote" in n for n in second.notes)


# ── The report (R3) ──────────────────────────────────────────────────────────

def test_reports_each_axis_separately():
    dataset = attitude([0, 0, 9], [0, 9, 0])
    _, report = system_attitude(dataset, **SPEC)

    per_axis = report.distributions["per_axis"]
    assert per_axis["pitch"]["column"] == "tilt_x"
    assert per_axis["roll"]["column"] == "tilt_y"
    assert per_axis["pitch"]["counts"][OUT_OF_SPEC] == 1
    assert per_axis["roll"]["counts"][OUT_OF_SPEC] == 1


def test_reports_the_distribution_before_and_after():
    dataset = attitude([1.0] * 8 + [20.0] * 2, [0.0] * 10)
    _, report = system_attitude(dataset, **SPEC)

    pitch = report.distributions["per_axis"]["pitch"]
    assert pitch["before"]["n"] == 10
    assert pitch["before"]["max"] == 20.0
    assert pitch["after"]["n"] == 8          # what survives acting on the verdict
    assert pitch["after"]["max"] == 1.0


def test_reports_the_thresholds_actually_used():
    _, report = system_attitude(attitude([1.0], [1.0]), **SPEC)
    assert report.distributions["thresholds"][GOOD] == "Good (<=±5 deg)"


def test_reports_both_grains_when_the_data_supports_them():
    dataset = make_dataset(n=8, n_lines=4, n_flights=2,
                           pitch=[0] * 4 + [9] * 4, roll=[0] * 8)
    _, report = system_attitude(dataset, **SPEC)

    assert report.distributions["per_flight"]["1"]["fraction_out_of_spec"] == 0.0
    assert report.distributions["per_flight"]["2"]["fraction_out_of_spec"] == 1.0
    assert len(report.distributions["per_line"]) == 4


def test_the_report_survives_serialization():
    _, report = system_attitude(attitude([1.0, 9.0], [1.0, 1.0]), **SPEC)
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["step"] == "system_attitude"
    assert restored["distributions"]["per_axis"]["pitch"]["before"]["n"] == 2


# ── Absent inputs ────────────────────────────────────────────────────────────

def test_missing_attitude_columns_are_a_note_not_an_error():
    dataset = attitude([1.0], [1.0])
    dataset.flightlines.drop(columns=["tilt_x", "tilt_y"], inplace=True)

    dataset, report = system_attitude(dataset, **SPEC)

    assert "qs_tilt" not in dataset.flightlines.columns
    assert any("no attitude columns" in n.lower() for n in report.notes)


def test_one_axis_present_is_a_note_and_still_judged():
    dataset = attitude([9.0], [1.0])
    dataset.flightlines.drop(columns=["tilt_y"], inplace=True)

    dataset, report = system_attitude(dataset, **SPEC)

    assert tier_names(dataset.flightlines["qs_tilt"])[0] == OUT_OF_SPEC
    assert any("that axis alone" in n for n in report.notes)


def test_a_column_named_by_the_caller_and_absent_is_an_error():
    with pytest.raises(KeyError, match="No column 'roll_deg'"):
        system_attitude(attitude([1.0], [1.0]), roll_column="roll_deg", **SPEC)


def test_an_explicitly_named_column_is_used():
    dataset = attitude([1.0], [1.0])
    dataset.flightlines["boom_pitch"] = [30.0]
    dataset, _ = system_attitude(dataset, pitch_column="boom_pitch", **SPEC)
    assert tier_names(dataset.flightlines["qs_pitch"])[0] == OUT_OF_SPEC


def test_an_empty_dataset_reports_rather_than_raises():
    dataset = make_dataset(n=0, n_lines=1, height=[], pitch=[], roll=[])
    _, report = system_attitude(dataset, **SPEC)
    assert report.soundings_total == 0


# ── Method-agnostic ──────────────────────────────────────────────────────────

def test_runs_without_any_channels():
    """Attitude is a property of the aircraft, not of the method."""
    dataset = attitude([1.0, 9.0], [1.0, 1.0])
    dataset.survey.xyz.model_dict["layer_data"] = {}

    assert dataset.n_channels == 0
    dataset, _ = system_attitude(dataset, **SPEC)
    assert tier_names(dataset.flightlines["qs_tilt"]) == [GOOD, OUT_OF_SPEC]


# ── Registration and the runner ──────────────────────────────────────────────

def test_registered_under_the_qc_entry_point_group():
    assert "system_attitude" in load_steps(QC_GROUP)


def test_runs_alongside_qc_01_from_one_pipeline():
    """One QC document, several checks, each writing its own column."""
    pipeline = Pipeline.parse("""
        kind: qc
        steps:
          - height_above_ground: {good_max: 45, marginal_max: 65}
          - system_attitude: {good: 5, marginal: 7.5}
    """)
    dataset = make_dataset(n=3, n_lines=1, height=[30.0, 30.0, 80.0],
                           pitch=[0.0, 9.0, 0.0], roll=[0.0, 0.0, 0.0])

    dataset, report = QCRunner(pipeline).run(dataset)

    assert tier_names(dataset.flightlines["qs_height"]) == [GOOD, GOOD, OUT_OF_SPEC]
    assert tier_names(dataset.flightlines["qs_tilt"]) == [GOOD, OUT_OF_SPEC, GOOD]
    assert named(report) == ["inventory", "height_above_ground",
                             "system_attitude"]
