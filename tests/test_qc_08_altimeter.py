"""qc_08 — altimeter agreement, end to end."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.pipeline import QC_GROUP, Pipeline, QCRunner, load_steps
from ymerflow_processing.core.qc.qc_01_height import height_above_ground
from ymerflow_processing.core.qc.qc_08_altimeter import altimeter_agreement
from ymerflow_processing.core.qc.thresholds import GOOD, MARGINAL, OUT_OF_SPEC

SPEC = {"good_max": 1.0, "marginal_max": 3.0}


def altimeters(**readings):
    n = len(next(iter(readings.values())))
    dataset = make_dataset(n=n, n_lines=1)
    for name, values in readings.items():
        dataset.flightlines[name] = np.asarray(values, dtype=float)
    return dataset


# ── The measurement ──────────────────────────────────────────────────────────

def test_disagreement_is_the_spread_and_higher_is_worse():
    dataset = altimeters(laser=[40.0, 40.0, 40.0], radar=[40.5, 42.0, 55.0])
    dataset, _ = altimeter_agreement(dataset, ["laser", "radar"], **SPEC)

    assert list(dataset.flightlines["alt_spread"]) == [0.5, 2.0, 15.0]
    assert list(dataset.flightlines["qs_altim"]) == [
        GOOD, MARGINAL, OUT_OF_SPEC]


def test_the_sign_of_the_disagreement_does_not_matter():
    """Which sensor read higher is not the question; that they differ is."""
    dataset = altimeters(laser=[40.0, 45.0], radar=[45.0, 40.0])
    dataset, _ = altimeter_agreement(dataset, ["laser", "radar"], **SPEC)
    assert list(dataset.flightlines["alt_spread"]) == [5.0, 5.0]


def test_three_altimeters_use_the_full_spread():
    """Generalizes without a view about which sensor is authoritative — which
    is the point, since knowing that would remove the need for the check."""
    dataset = altimeters(laser=[40.0], radar=[41.0], baro=[48.0])
    dataset, _ = altimeter_agreement(dataset, ["laser", "radar", "baro"], **SPEC)
    assert dataset.flightlines["alt_spread"][0] == 8.0


def test_a_lone_reading_has_no_verdict():
    """One altimeter cannot disagree; a spread of zero would read as perfect
    agreement, which is the opposite of what is known."""
    dataset = altimeters(laser=[40.0, np.nan], radar=[40.2, 40.0])
    dataset, report = altimeter_agreement(dataset, ["laser", "radar"], **SPEC)

    assert pd.isna(dataset.flightlines["alt_spread"][1])
    assert pd.isna(dataset.flightlines["qs_altim"][1])
    assert any("only one altimeter" in n for n in report.notes)


# ── It is a different question from qc_01 ────────────────────────────────────

def test_height_can_be_in_spec_while_the_altimeters_disagree():
    """qc_01 asks whether the height was right; this asks whether it is known.
    A height verdict computed from a failing sensor is confidently wrong."""
    dataset = altimeters(laser=[30.0], radar=[55.0])
    dataset.flightlines["tx_alt"] = [30.0]

    dataset, _ = height_above_ground(dataset, good_max=45, marginal_max=65)
    dataset, _ = altimeter_agreement(dataset, ["laser", "radar"], **SPEC)

    assert dataset.flightlines["qs_height"][0] == GOOD           # looks fine
    assert dataset.flightlines["qs_altim"][0] == OUT_OF_SPEC  # but unknowable


# ── Refusing to do the wrong thing ───────────────────────────────────────────

def test_one_column_is_an_error():
    with pytest.raises(ValueError, match="at least two columns"):
        altimeter_agreement(altimeters(laser=[40.0]), ["laser"], **SPEC)


def test_a_missing_column_is_an_error_naming_it():
    with pytest.raises(KeyError, match="'radar'"):
        altimeter_agreement(altimeters(laser=[40.0, 41.0]),
                            ["laser", "radar"], **SPEC)


# ── The report ───────────────────────────────────────────────────────────────

def test_each_altimeter_gets_its_own_distribution():
    """A spread says they differ without saying which one moved."""
    dataset = altimeters(laser=[40.0, 40.0], radar=[40.0, 60.0])
    _, report = altimeter_agreement(dataset, ["laser", "radar"], **SPEC)

    per = report.distributions["per_altimeter"]
    assert per["laser"]["max"] == 40.0
    assert per["radar"]["max"] == 60.0          # the one that drifted


def test_reports_the_distribution_before_and_after():
    dataset = altimeters(laser=[40.0] * 4, radar=[40.1, 40.2, 40.1, 55.0])
    _, report = altimeter_agreement(dataset, ["laser", "radar"], **SPEC)

    spread = report.distributions["spread"]
    assert spread["before"]["n"] == 4
    assert spread["after"]["n"] == 3
    assert spread["columns"] == ["laser", "radar"]


def test_the_report_survives_serialization():
    dataset = altimeters(laser=[40.0, 40.0], radar=[40.1, 55.0])
    _, report = altimeter_agreement(dataset, ["laser", "radar"], **SPEC)
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["step"] == "altimeter_agreement"


def test_changes_no_measurement_uncertainty_or_mask():
    dataset = altimeters(laser=[40.0, 40.0], radar=[40.1, 55.0])
    before = {k: v.copy() for k, v in dataset.layer_data.items()}
    dataset, _ = altimeter_agreement(dataset, ["laser", "radar"], **SPEC)
    for key, array in before.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


# ── Registration ─────────────────────────────────────────────────────────────

def test_registered_and_runs_from_a_pipeline():
    assert "altimeter_agreement" in load_steps(QC_GROUP)

    dataset = altimeters(laser=[40.0, 40.0], radar=[40.1, 55.0])
    pipeline = Pipeline.parse("""
        kind: qc
        steps:
          - altimeter_agreement:
              columns: [laser, radar]
              good_max: 1.0
              marginal_max: 3.0
    """)
    dataset, report = QCRunner(pipeline).run(dataset)
    assert list(dataset.flightlines["qs_altim"]) == [GOOD, OUT_OF_SPEC]
