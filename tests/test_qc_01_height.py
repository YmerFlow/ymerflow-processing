"""qc_01 — height above ground, end to end."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset, tier_names
from ymerflow_processing.core.pipeline import QC_GROUP, Pipeline, QCRunner, load_steps
from ymerflow_processing.core.qc.qc_01_height import height_above_ground
from ymerflow_processing.core.qc.thresholds import GOOD, MARGINAL, OUT_OF_SPEC

# Representative survey-spec values. Real thresholds are contract parameters per survey.
SPEC = {"good_max": 45, "marginal_max": 65}


def heights(*values):
    return make_dataset(n=len(values), n_lines=1, height=list(values))


# ── What it writes ───────────────────────────────────────────────────────────

def test_writes_a_verdict_column_on_flightlines():
    dataset = heights(30.0, 50.0, 80.0)
    dataset, report = height_above_ground(dataset, **SPEC)

    assert tier_names(dataset.flightlines["qs_height"]) == [GOOD, MARGINAL, OUT_OF_SPEC]
    assert report.step == "height_above_ground"


def test_changes_no_measurement_uncertainty_or_mask():
    """R6: QC writes attributes. Running it must leave the data byte-identical."""
    dataset = heights(30.0, 80.0)
    before = {k: v.copy() for k, v in dataset.layer_data.items()}

    dataset, _ = height_above_ground(dataset, **SPEC)

    assert set(dataset.layer_data) == set(before)
    for key, array in before.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)




def test_the_output_column_is_nameable():
    dataset = heights(30.0)
    dataset, _ = height_above_ground(dataset, output_column="qc_agl", **SPEC)
    assert "qc_agl" in dataset.flightlines.columns


# ── Re-running ───────────────────────────────────────────────────────────────

def test_re_running_is_idempotent_and_says_so():
    """R4: safe to re-run, and detectable that it happened."""
    dataset = heights(30.0, 80.0)
    dataset, first = height_above_ground(dataset, **SPEC)
    dataset, second = height_above_ground(dataset, **SPEC)

    assert tier_names(dataset.flightlines["qs_height"]) == [GOOD, OUT_OF_SPEC]
    assert not any("Overwrote" in n for n in first.notes)
    assert any("Overwrote existing column 'qs_height'" in n for n in second.notes)


def test_a_changed_threshold_replaces_the_earlier_verdict():
    """The verdict is a fact; the threshold is what is legitimately revisable."""
    dataset = heights(50.0)
    dataset, _ = height_above_ground(dataset, good_max=45, marginal_max=65)
    assert tier_names(dataset.flightlines["qs_height"])[0] == MARGINAL

    dataset, _ = height_above_ground(dataset, good_max=55, marginal_max=65)
    assert tier_names(dataset.flightlines["qs_height"])[0] == GOOD


# ── The report (R3) ──────────────────────────────────────────────────────────

def test_reports_the_distribution_before_and_after():
    dataset = heights(*([30.0] * 8 + [80.0] * 2))
    _, report = height_above_ground(dataset, **SPEC)

    height = report.distributions["height"]
    assert height["column"] == "tx_alt"
    assert height["before"]["n"] == 10
    assert height["before"]["max"] == 80.0
    # "After" is what remains if the verdict were acted on.
    assert height["after"]["n"] == 8
    assert height["after"]["max"] == 30.0


def test_reports_the_thresholds_actually_used():
    """R3: including defaults that were not supplied."""
    _, report = height_above_ground(heights(30.0), **SPEC)
    assert report.distributions["thresholds"][GOOD] == "Good (<=45 m)"


def test_reports_counts_and_totals():
    dataset = heights(30.0, 50.0, 80.0, 90.0)
    _, report = height_above_ground(dataset, **SPEC)

    assert report.soundings_total == 4
    assert report.soundings_affected == 2       # the Out of Spec ones
    assert report.distributions["verdicts"] == {
        GOOD: 1, MARGINAL: 1, OUT_OF_SPEC: 2, "unclassified": 0}


def test_reports_both_line_rules():
    dataset = make_dataset(n=8, n_lines=1,
                           height=[30.0, 80.0, 80.0, 80.0, 30.0, 30.0, 80.0, 30.0])
    _, report = height_above_ground(dataset, **SPEC)

    line = report.distributions["per_line"]["100"]
    assert line["fraction_out_of_spec"] == 0.5
    assert line["longest_out_of_spec_run"] == 3          # not 4, though 4 are bad
    assert line["longest_out_of_spec_run_distance"] == 20.0


def test_reports_per_flight_when_the_data_says_which_flight():
    """A delivery holds several flights; one run must break down by both grains."""
    dataset = make_dataset(n=8, n_lines=4, n_flights=2,
                           height=[30.0] * 4 + [80.0] * 4)
    _, report = height_above_ground(dataset, **SPEC)

    per_flight = report.distributions["per_flight"]
    assert set(per_flight) == {"1", "2"}
    assert per_flight["1"]["fraction_out_of_spec"] == 0.0
    assert per_flight["2"]["fraction_out_of_spec"] == 1.0
    # Both grains are present in the same report.
    assert len(report.distributions["per_line"]) == 4


def test_a_missing_flight_column_is_a_note_not_an_error():
    dataset = make_dataset(n=4, n_lines=1, height=[30.0] * 4)   # no Flight column
    _, report = height_above_ground(dataset, **SPEC)

    assert "per_flight" not in report.distributions
    assert any("no flight identifier" in n.lower() for n in report.notes)
    assert "per_line" in report.distributions


def test_the_report_survives_serialization():
    """It is stored in the process record and rendered by a web client."""
    _, report = height_above_ground(heights(30.0, 80.0), **SPEC)
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["step"] == "height_above_ground"
    assert restored["distributions"]["height"]["before"]["n"] == 2


# ── Absent inputs ────────────────────────────────────────────────────────────

def test_a_missing_height_column_is_a_note_not_an_error():
    """building_plans/03: one configuration, four contexts, no separate paths."""
    dataset = heights(30.0)
    dataset.flightlines.drop(columns=["tx_alt"], inplace=True)

    dataset, report = height_above_ground(dataset, **SPEC)

    assert "qs_height" not in dataset.flightlines.columns
    assert any("no height-above-ground column" in n.lower() for n in report.notes)


def test_a_column_named_by_the_caller_and_absent_is_an_error():
    """Auto-discovery finding nothing is absence; a named column missing is a typo."""
    with pytest.raises(KeyError, match="No column 'height_agl'"):
        height_above_ground(heights(30.0), column="height_agl", **SPEC)


def test_an_explicitly_named_column_is_used():
    dataset = heights(30.0)
    dataset.flightlines["laser_agl"] = [90.0]
    dataset, _ = height_above_ground(dataset, column="laser_agl", **SPEC)
    assert tier_names(dataset.flightlines["qs_height"])[0] == OUT_OF_SPEC


def test_soundings_with_no_height_get_no_verdict():
    dataset = heights(30.0, np.nan, 80.0)
    dataset, report = height_above_ground(dataset, **SPEC)

    assert pd.isna(dataset.flightlines["qs_height"][1])
    assert report.distributions["verdicts"]["unclassified"] == 1


def test_an_empty_dataset_reports_rather_than_raises():
    dataset = make_dataset(n=0, n_lines=1, height=[])
    dataset, report = height_above_ground(dataset, **SPEC)
    assert report.soundings_total == 0
    assert any("no soundings" in n.lower() for n in report.notes)


# ── Method-agnostic ──────────────────────────────────────────────────────────

def test_runs_without_any_channels_at_all():
    """core/ QC must not need EM data. A magnetometer survey has a height too."""
    dataset = heights(30.0, 80.0)
    dataset.survey.xyz.model_dict["layer_data"] = {}

    assert dataset.n_channels == 0
    dataset, report = height_above_ground(dataset, **SPEC)
    assert tier_names(dataset.flightlines["qs_height"]) == [GOOD, OUT_OF_SPEC]


@pytest.mark.parametrize("moments", [("LM",), ("LM", "HM"), ("LM", "MM", "HM")])
def test_moment_count_is_irrelevant_to_this_check(moments):
    dataset = make_dataset(moments=moments, n=2, n_lines=1, height=[30.0, 80.0])
    dataset, report = height_above_ground(dataset, **SPEC)
    assert report.distributions["verdicts"][OUT_OF_SPEC] == 1


# ── Registration and the runner ──────────────────────────────────────────────

def test_registered_under_the_qc_entry_point_group():
    assert "height_above_ground" in load_steps(QC_GROUP)


def test_runs_from_a_yaml_pipeline_with_no_platform():
    """The standalone claim: a YAML document, a runner, and nothing else."""
    pipeline = Pipeline.parse("""
        steps:
          - height_above_ground:
              good_max: 45
              marginal_max: 65
    """)
    dataset, report = QCRunner(pipeline).run(heights(30.0, 50.0, 80.0))

    assert tier_names(dataset.flightlines["qs_height"]) == [GOOD, MARGINAL, OUT_OF_SPEC]
    assert len(report.steps) == 1
    assert report.steps[0].params == {"good_max": 45, "marginal_max": 65}


def test_an_unknown_check_fails_before_running_anything():
    pipeline = Pipeline.parse("steps: [not_a_real_check]")
    with pytest.raises(KeyError, match="Unknown step"):
        QCRunner(pipeline)
