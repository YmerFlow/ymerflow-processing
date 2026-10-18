"""The cross-check rollup: one answer per sounding, and why.

Each check owns a column, so nothing is lost when a sounding fails several
ways. What this adds is the single place the vector export needs — a client
opening it should not read across five columns to learn why a point is red.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset, tier_names, step
from ymerflow_processing.pipeline import Pipeline, QCRunner
from ymerflow_processing.qc.rollup import NO_REASON
from ymerflow_processing.qc.thresholds import GOOD, MARGINAL, OUT_OF_SPEC

ALL_THREE = """
kind: qc
steps:
  - height_above_ground: {good_max: 45, marginal_max: 65}
  - system_attitude: {good: 5, marginal: 7.5}
  - ground_speed: {good_min: 70, good_max: 100,
                   marginal_min: 60, marginal_max: 120}
"""


def run(document, dataset):
    return QCRunner(Pipeline.parse(document)).run(dataset)


# ── Several reasons at once ──────────────────────────────────────────────────

def test_a_sounding_failing_three_checks_names_all_three():
    """The question that prompted this: what if it fails for 2 or 3 reasons?"""
    dataset = make_dataset(n=3, n_lines=1, height=[30, 30, 80],
                           pitch=[0, 0, 12], roll=[0, 0, 0],
                           speed=[85, 85, 140])
    dataset, _ = run(ALL_THREE, dataset)

    reasons = list(dataset.flightlines["qs_reasons"])
    assert reasons[:2] == [NO_REASON, NO_REASON]
    assert set(reasons[2].split(";")) == {
        "height_above_ground", "system_attitude", "ground_speed"}


def test_the_per_check_columns_are_untouched():
    """The rollup adds; it does not replace. Every reason stays addressable."""
    dataset = make_dataset(n=2, n_lines=1, height=[30, 80], pitch=[0, 12],
                           roll=[0, 0], speed=[85, 140])
    dataset, _ = run(ALL_THREE, dataset)

    assert tier_names(dataset.flightlines["qs_height"]) == [GOOD, OUT_OF_SPEC]
    assert tier_names(dataset.flightlines["qs_tilt"]) == [GOOD, OUT_OF_SPEC]
    assert tier_names(dataset.flightlines["qs_speed"]) == [GOOD, OUT_OF_SPEC]


def test_one_reason_lists_only_that_one():
    dataset = make_dataset(n=2, n_lines=1, height=[30, 80], pitch=[0, 0],
                           roll=[0, 0], speed=[85, 85])
    dataset, _ = run(ALL_THREE, dataset)
    assert list(dataset.flightlines["qs_reasons"]) == [NO_REASON, "height_above_ground"]


def test_marginal_is_not_a_reason():
    """qs_reasons answers 'why did this fail'. Marginal stays in its own column."""
    dataset = make_dataset(n=1, n_lines=1, height=[50], pitch=[0], roll=[0],
                           speed=[85])
    dataset, _ = run(ALL_THREE, dataset)

    assert tier_names(dataset.flightlines["qs_height"])[0] == MARGINAL
    assert dataset.flightlines["qs_reasons"][0] == NO_REASON
    assert tier_names(dataset.flightlines["qs_verdict"])[0] == MARGINAL


# ── Flight quality and instrument health stay apart ──────────────────────────

WITH_ALTIMETERS = """
kind: qc
steps:
  - height_above_ground: {good_max: 45, marginal_max: 65}
  - altimeter_agreement: {columns: [laser, radar], good_max: 1.0, marginal_max: 3.0}
"""


def altimeter_dataset(heights, laser, radar):
    dataset = make_dataset(n=len(heights), n_lines=1, height=heights)
    dataset.flightlines["laser"] = np.asarray(laser, dtype=float)
    dataset.flightlines["radar"] = np.asarray(radar, dtype=float)
    return dataset


def test_a_dead_sensor_does_not_read_as_a_bad_flight():
    """The case that started this: flown perfectly, radar altimeter failed."""
    dataset = altimeter_dataset([30, 30, 30], [30, 30, 30], [30.1, 55, 55])
    dataset, _ = run(WITH_ALTIMETERS, dataset)

    assert tier_names(dataset.flightlines["qs_verdict"]) == [GOOD] * 3
    assert tier_names(dataset.flightlines["qs_sensor"]) == [
        GOOD, OUT_OF_SPEC, OUT_OF_SPEC]


def test_the_sensor_fault_is_still_named_as_a_reason():
    """Highlighted, not hidden — it is just not a flying problem."""
    dataset = altimeter_dataset([30, 30], [30, 30], [30.1, 55])
    dataset, _ = run(WITH_ALTIMETERS, dataset)
    assert list(dataset.flightlines["qs_reasons"]) == [NO_REASON, "altimeter_agreement"]


def test_a_bad_flight_and_a_bad_sensor_are_both_visible():
    dataset = altimeter_dataset([30, 80], [30, 30], [30.1, 55])
    dataset, _ = run(WITH_ALTIMETERS, dataset)

    assert tier_names(dataset.flightlines["qs_verdict"]) == [GOOD, OUT_OF_SPEC]
    assert tier_names(dataset.flightlines["qs_sensor"]) == [GOOD, OUT_OF_SPEC]
    assert set(dataset.flightlines["qs_reasons"][1].split(";")) == {
        "height_above_ground", "altimeter_agreement"}


# ── How the verdict combines ─────────────────────────────────────────────────

def test_the_verdict_is_the_worst_across_flight_checks():
    dataset = make_dataset(n=3, n_lines=1, height=[30, 50, 80], pitch=[0] * 3,
                           roll=[0] * 3, speed=[85] * 3)
    dataset, _ = run(ALL_THREE, dataset)
    assert tier_names(dataset.flightlines["qs_verdict"]) == [
        GOOD, MARGINAL, OUT_OF_SPEC]


def test_an_unmeasured_check_does_not_suppress_a_certain_finding():
    """Unlike the two axes of one attitude measurement, these are independent
    questions: a missing height does not make a grossly bad speed unknown."""
    dataset = make_dataset(n=1, n_lines=1, height=[np.nan], pitch=[0], roll=[0],
                           speed=[140])
    dataset, _ = run(ALL_THREE, dataset)

    assert pd.isna(dataset.flightlines["qs_height"][0])
    assert tier_names(dataset.flightlines["qs_verdict"])[0] == OUT_OF_SPEC


def test_a_sounding_no_check_could_judge_has_no_verdict():
    dataset = make_dataset(n=1, n_lines=1, height=[np.nan], pitch=[np.nan],
                           roll=[np.nan], speed=[np.nan])
    dataset, _ = run(ALL_THREE, dataset)
    assert pd.isna(dataset.flightlines["qs_verdict"][0])


# ── What the report records ──────────────────────────────────────────────────

def test_the_report_says_which_checks_fed_the_rollup():
    dataset = altimeter_dataset([30, 80], [30, 30], [30.1, 55])
    _, report = run(WITH_ALTIMETERS, dataset)

    checks = report.rollup["checks"]
    assert checks["height_above_ground"] == {
        "column": "qs_height", "finding": "flight"}
    assert checks["altimeter_agreement"] == {
        "column": "qs_altim", "finding": "instrument"}


def test_each_step_declares_its_own_verdict_column():
    dataset = make_dataset(n=1, n_lines=1, height=[30], pitch=[0], roll=[0],
                           speed=[85])
    _, report = run(ALL_THREE, dataset)

    declared = {s.step: (s.verdict_column, s.finding) for s in report.steps}
    assert declared["height_above_ground"] == ("qs_height", "flight")
    assert declared["system_attitude"] == ("qs_tilt", "flight")


def test_a_check_that_skipped_contributes_nothing():
    """No altimeter columns, so nothing declared and nothing rolled up from it."""
    dataset = make_dataset(n=2, n_lines=1, height=[30, 80])
    dataset.flightlines.drop(columns=["tx_alt"], inplace=True)

    dataset, report = run("kind: qc\nsteps: [{height_above_ground: "
                          "{good_max: 45, marginal_max: 65}}]", dataset)
    assert report.rollup == {"columns": {}, "checks": {}}
    assert "qs_verdict" not in dataset.flightlines.columns


def test_the_rollup_survives_serialization():
    import json
    dataset = altimeter_dataset([30, 80], [30, 30], [30.1, 55])
    _, report = run(WITH_ALTIMETERS, dataset)
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["rollup"]["columns"]["verdict"] == "qs_verdict"
    assert step(restored, "altimeter_agreement")["finding"] == "instrument"


def test_the_column_names_fit_a_shapefile_field():
    """DBF allows ten characters, and these are what the export carries."""
    dataset = altimeter_dataset([30], [30], [30.1])
    _, report = run(WITH_ALTIMETERS, dataset)
    for name in report.rollup["columns"].values():
        assert len(name) <= 10, name
