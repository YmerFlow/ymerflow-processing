"""qc_10 — sounding density, end to end."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.pipeline import QC_GROUP, Pipeline, QCRunner, load_steps
from ymerflow_processing.core.qc.qc_10_density import (
    along_line_spacing, sounding_density,
)
from ymerflow_processing.core.qc.thresholds import GOOD, MARGINAL, OUT_OF_SPEC

# The fixture lays soundings 10 m apart, which is 100 per kilometre.
SPEC = {"good_min": 80, "marginal_min": 50}


# ── The measurement ──────────────────────────────────────────────────────────

def test_spacing_is_measured_within_lines():
    dataset = make_dataset(n=6, n_lines=2)
    dataset, _ = sounding_density(dataset, **SPEC)

    spacing = list(dataset.flightlines["along_line_spacing"])
    assert pd.isna(spacing[0])          # first of line 100
    assert pd.isna(spacing[3])          # first of line 101 — across the turn
    assert spacing[1:3] == [10.0, 10.0]
    assert spacing[4:] == [10.0, 10.0]


def test_the_turn_between_lines_is_not_a_gap_in_coverage():
    """The fixture's lines sit 1000 m apart. Measured as spacing that is an
    enormous hole; it is the ferry between lines, not missing acquisition."""
    dataset = make_dataset(n=6, n_lines=2)
    dataset, report = sounding_density(dataset, **SPEC)

    assert report.distributions["spacing"]["before"]["max"] == 10.0
    assert list(dataset.flightlines["qc_density"])[3] is None or pd.isna(
        dataset.flightlines["qc_density"][3])


def test_density_is_the_inverse_of_spacing():
    dataset = make_dataset(n=4, n_lines=1)
    _, report = sounding_density(dataset, **SPEC)
    assert report.distributions["density"]["before"]["mean"] == 100.0


def test_lower_density_is_worse():
    """descending — the kind nothing else in the library uses."""
    dataset = make_dataset(n=4, n_lines=1)
    # 10 m, 20 m, 40 m steps -> 100, 50, 25 soundings/km
    dataset.flightlines["x"] = [0.0, 10.0, 30.0, 70.0]
    dataset.flightlines["y"] = [0.0] * 4

    dataset, _ = sounding_density(dataset, **SPEC)
    assert list(dataset.flightlines["qc_density"])[1:] == [
        GOOD, MARGINAL, OUT_OF_SPEC]


def test_along_line_spacing_directly():
    spacing = along_line_spacing([0, 10, 20, 0], [0, 0, 0, 1000], [1, 1, 1, 2])
    assert np.isnan(spacing[0])
    assert spacing[1] == pytest.approx(10.0)
    assert np.isnan(spacing[3])         # the turn


def test_a_repeated_position_has_no_density():
    """Zero spacing is an infinite density, which is not a measurement."""
    dataset = make_dataset(n=3, n_lines=1)
    dataset.flightlines["x"] = [0.0, 0.0, 10.0]
    dataset.flightlines["y"] = [0.0] * 3

    dataset, report = sounding_density(dataset, **SPEC)

    assert pd.isna(dataset.flightlines["qc_density"][1])
    assert any("same position" in n for n in report.notes)


def test_the_crs_unit_is_configurable():
    dataset = make_dataset(n=3, n_lines=1)
    _, report = sounding_density(dataset, units_per_km=3280.84, **SPEC)   # feet
    assert report.distributions["density"]["before"]["mean"] == pytest.approx(
        328.084)


# ── What it writes and reports ───────────────────────────────────────────────

def test_writes_both_the_spacing_and_the_verdict():
    dataset = make_dataset(n=4, n_lines=1)
    dataset, _ = sounding_density(dataset, **SPEC)
    assert {"along_line_spacing", "qc_density", "qc_density_line"} <= set(
        dataset.flightlines.columns)


def test_reports_the_measurement_both_ways_up():
    """A contract may quote either, and a reader thinks in one of them."""
    dataset = make_dataset(n=4, n_lines=1)
    _, report = sounding_density(dataset, **SPEC)

    assert report.distributions["density"]["unit"] == "soundings/km"
    assert report.distributions["spacing"]["before"]["mean"] == 10.0


def test_reports_both_grains():
    dataset = make_dataset(n=8, n_lines=4, n_flights=2)
    dataset.flightlines["x"] = [0.0, 10.0] * 2 + [0.0, 100.0] * 2
    _, report = sounding_density(dataset, **SPEC)

    assert report.distributions["per_flight"]["1"]["fraction_out_of_spec"] == 0.0
    assert report.distributions["per_flight"]["2"]["fraction_out_of_spec"] == 1.0


def test_the_report_survives_serialization():
    _, report = sounding_density(make_dataset(n=4, n_lines=1), **SPEC)
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["step"] == "sounding_density"


def test_changes_no_measurement_uncertainty_or_mask():
    dataset = make_dataset(n=4, n_lines=1)
    before = {k: v.copy() for k, v in dataset.layer_data.items()}
    dataset, _ = sounding_density(dataset, **SPEC)
    for key, array in before.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


# ── Absent inputs ────────────────────────────────────────────────────────────

def test_no_position_is_a_note_not_an_error():
    dataset = make_dataset(n=4, n_lines=1)
    dataset.flightlines.drop(columns=["x", "y"], inplace=True)

    dataset, report = sounding_density(dataset, **SPEC)

    assert "qc_density" not in dataset.flightlines.columns
    assert any("no position columns" in n.lower() for n in report.notes)


def test_an_empty_dataset_reports_rather_than_raises():
    dataset = make_dataset(n=0, n_lines=1, height=[], pitch=[], roll=[])
    _, report = sounding_density(dataset, **SPEC)
    assert report.soundings_total == 0


# ── Registration ─────────────────────────────────────────────────────────────

def test_registered_and_runs_from_a_pipeline():
    assert "sounding_density" in load_steps(QC_GROUP)

    pipeline = Pipeline.parse("""
        kind: qc
        steps:
          - sounding_density: {good_min: 80, marginal_min: 50}
    """)
    dataset, _ = QCRunner(pipeline).run(make_dataset(n=4, n_lines=1))
    assert list(dataset.flightlines["qc_density"])[1:] == [GOOD] * 3
