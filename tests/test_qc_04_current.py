"""qc_04 — transmitter output stability, end to end."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.pipeline import QC_GROUP, Pipeline, QCRunner, load_steps
from ymerflow_processing.core.qc.qc_04_current import (
    detect_jumps, find_current_columns, transmitter_stability,
)
from ymerflow_processing.core.qc.thresholds import GOOD, MARGINAL, OUT_OF_SPEC

# 1% steady is good, beyond 3% is out of spec. Operational judgements, not
# measured values.
SPEC = {"good_fraction": 0.01, "marginal_fraction": 0.03}

# The fixture's declared nominals, from TxApproximateCurrent.
LM_NOMINAL, HM_NOMINAL = 9.0, 110.0


def currents(lm, hm=None, n_flights=1, **kw):
    values = np.asarray(lm, dtype=float)
    dataset = make_dataset(n=len(values), n_lines=1, n_flights=n_flights, **kw)
    dataset.flightlines["LMcurrent"] = values
    dataset.flightlines["HMcurrent"] = (
        np.full(len(values), HM_NOMINAL) if hm is None
        else np.asarray(hm, dtype=float))
    return dataset


# ── Stability carries the verdict ────────────────────────────────────────────

def test_a_steady_transmitter_passes():
    dataset, _ = transmitter_stability(currents([9.0] * 10), **SPEC)
    assert list(dataset.flightlines["qc_current"]) == [GOOD] * 10


def test_drift_away_from_the_mean_is_flagged():
    # Mean is 9.0; the ends sit 5% away.
    dataset, _ = transmitter_stability(
        currents([8.55, 9.0, 9.0, 9.0, 9.45]), **SPEC)
    assert list(dataset.flightlines["qc_cur_ch1"]) == [
        OUT_OF_SPEC, GOOD, GOOD, GOOD, OUT_OF_SPEC]


def test_departure_either_way_counts_equally():
    dataset, _ = transmitter_stability(currents([8.82, 9.0, 9.18]), **SPEC)
    verdicts = list(dataset.flightlines["qc_cur_ch1"])
    assert verdicts[0] == verdicts[2] == MARGINAL


def test_a_wildly_unstable_transmitter_is_caught():
    """The reason the window is a fraction and not a multiple of sigma: sigma is
    inflated by the instability, so a two-sigma rule would pass all of this."""
    swinging = [9.0, 9.9, 8.1, 9.9, 8.1, 9.9, 8.1]        # +/-10%
    dataset, _ = transmitter_stability(currents(swinging), **SPEC)

    verdicts = list(dataset.flightlines["qc_cur_ch1"])
    assert verdicts.count(OUT_OF_SPEC) == 6
    assert np.std(swinging) > 0.8      # two sigma spans the whole range


# ── Agreement with nominal is separate and does not penalize steadiness ──────

def test_steady_but_off_nominal_is_stable_and_flagged_separately():
    """The case that shaped this check: a transmitter holding a current well
    away from what the system description declares. The data is self-consistent;
    the description is not describing this instrument."""
    dataset, report = transmitter_stability(
        currents([9.0] * 10, hm=[136.0] * 10), **SPEC)

    assert report.distributions["stability"][GOOD] == 10          # rock steady
    assert list(dataset.flightlines["qc_nominal"]) == [OUT_OF_SPEC] * 10
    offset = report.distributions["per_moment"]["ch2 (HM/z)"][
        "nominal_offset_fraction"]
    assert offset == pytest.approx(0.236, abs=0.001)


def test_measured_above_nominal_counts_as_much_as_below():
    high, _ = transmitter_stability(currents([9.0] * 5, hm=[143.0] * 5), **SPEC)
    low, _ = transmitter_stability(currents([9.0] * 5, hm=[77.0] * 5), **SPEC)
    assert list(high.flightlines["qc_nominal"]) == [OUT_OF_SPEC] * 5
    assert list(low.flightlines["qc_nominal"]) == [OUT_OF_SPEC] * 5


def test_a_small_offset_is_tolerated():
    dataset, _ = transmitter_stability(
        currents([9.0] * 5, hm=[115.0] * 5), **SPEC)      # 4.5% high
    assert list(dataset.flightlines["qc_nominal"]) == [GOOD] * 5


def test_the_nominal_window_is_exposed():
    """A round default, not a measured one, so it has to be adjustable."""
    steady_but_off = currents([9.0] * 5, hm=[121.0] * 5)          # 10% high
    strict, _ = transmitter_stability(steady_but_off, nominal_good_fraction=0.05,
                                      nominal_marginal_fraction=0.08, **SPEC)
    assert list(strict.flightlines["qc_nominal"]) == [OUT_OF_SPEC] * 5

    lenient, _ = transmitter_stability(currents([9.0] * 5, hm=[121.0] * 5),
                                       nominal_good_fraction=0.25,
                                       nominal_marginal_fraction=0.5, **SPEC)
    assert list(lenient.flightlines["qc_nominal"]) == [GOOD] * 5


def test_the_combined_verdict_takes_the_worse_of_the_two():
    dataset, _ = transmitter_stability(
        currents([9.0] * 5, hm=[143.0] * 5), **SPEC)
    assert list(dataset.flightlines["qc_current"]) == [OUT_OF_SPEC] * 5


def test_no_nominal_declared_means_no_nominal_verdict():
    dataset = currents([9.0] * 5)
    for channel in (1, 2):
        dataset.system.gex_dict[f"Channel{channel}"].pop("TxApproximateCurrent")

    dataset, report = transmitter_stability(dataset, **SPEC)
    assert "qc_nominal" not in dataset.flightlines.columns
    assert any("declares no nominal current" in n for n in report.notes)


# ── Sudden steps ─────────────────────────────────────────────────────────────

def test_a_step_is_found_where_a_mean_test_would_miss_it():
    """Both halves sit near the mean; the transition is the fault."""
    stepped = [9.0] * 5 + [9.5] * 5
    _, report = transmitter_stability(currents(stepped), **SPEC)
    assert report.distributions["per_moment"]["ch1 (LM/z)"]["jumps"] == 1


def test_the_first_sample_of_each_flight_is_never_a_jump():
    """Current legitimately steps as the transmitter comes up."""
    values = np.array([0.1, 9.0, 9.0, 0.1, 9.0, 9.0])
    flights = np.array([1, 1, 1, 2, 2, 2])
    jumps = detect_jumps(values, flights, fraction=0.01)

    assert not jumps[0] and not jumps[3]        # each flight's first sample
    assert jumps[1] and jumps[4]                # the real steps within a flight


def test_jump_detection_directly():
    jumps = detect_jumps([10.0, 10.0, 12.0, 12.0], [1, 1, 1, 1], fraction=0.05)
    assert list(jumps) == [False, False, True, False]


def test_the_jump_threshold_is_settable():
    stepped = [9.0] * 5 + [9.05] * 5            # a 0.6% step
    _, tight = transmitter_stability(currents(stepped), jump_fraction=0.001, **SPEC)
    _, loose = transmitter_stability(currents(stepped), jump_fraction=0.05, **SPEC)
    assert tight.distributions["per_moment"]["ch1 (LM/z)"]["jumps"] == 1
    assert loose.distributions["per_moment"]["ch1 (LM/z)"]["jumps"] == 0


# ── Finding the per-moment columns — R1 applied to columns ───────────────────

def test_each_moment_gets_its_own_current_column():
    dataset = currents([9.0] * 3)
    found = find_current_columns(dataset.flightlines, dataset.channels)
    assert found == {"ch1": "LMcurrent", "ch2": "HMcurrent"}


@pytest.mark.parametrize("spelling", ["LMcurrent", "LM_current", "current_LM",
                                      "TxLMcurrent"])
def test_a_moment_column_is_found_under_its_usual_spellings(spelling):
    dataset = make_dataset(moments=("LM",), n=3, n_lines=1)
    dataset.flightlines[spelling] = [9.0] * 3
    found = find_current_columns(dataset.flightlines, dataset.channels)
    assert found == {"ch1": spelling}


def test_a_single_moment_system_may_name_it_generically():
    dataset = make_dataset(moments=("LM",), n=3, n_lines=1)
    dataset.flightlines["TxCurrent"] = [9.0] * 3
    assert find_current_columns(dataset.flightlines, dataset.channels) == {
        "ch1": "TxCurrent"}


def test_a_generic_name_is_not_guessed_at_with_two_moments():
    """'current' cannot be attributed when two moments could own it."""
    dataset = make_dataset(moments=("LM", "HM"), n=3, n_lines=1)
    dataset.flightlines["current"] = [9.0] * 3
    assert find_current_columns(dataset.flightlines, dataset.channels) == {}


def test_columns_may_be_named_explicitly():
    dataset = make_dataset(moments=("LM", "HM"), n=3, n_lines=1)
    dataset.flightlines["i_low"] = [9.0] * 3
    dataset, _ = transmitter_stability(dataset, columns={"LM": "i_low"}, **SPEC)
    assert "qc_cur_ch1" in dataset.flightlines.columns


def test_a_named_column_that_is_absent_is_an_error():
    with pytest.raises(KeyError, match="'nope'"):
        transmitter_stability(currents([9.0] * 3), columns={"LM": "nope"}, **SPEC)


def test_a_moment_with_no_current_column_is_noted():
    dataset = currents([9.0] * 3)
    dataset.flightlines.drop(columns=["HMcurrent"], inplace=True)
    dataset, report = transmitter_stability(dataset, **SPEC)
    assert any("HM" in n and "not judged" in n for n in report.notes)


# ── What it is, and what it writes ───────────────────────────────────────────

def test_it_reports_as_an_instrument_finding():
    """An unstable transmitter is a maintenance conversation, not a reflight."""
    _, report = transmitter_stability(currents([9.0] * 3), **SPEC)
    assert report.finding == "instrument"
    assert report.verdict_column == "qc_current"


def test_changes_no_measurement_uncertainty_or_mask():
    dataset = currents([9.0, 12.0])
    before = {k: v.copy() for k, v in dataset.layer_data.items()}
    dataset, _ = transmitter_stability(dataset, **SPEC)
    for key, array in before.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


def test_the_per_sounding_verdict_columns_fit_a_shapefile_field():
    """The columns a client reads. DBF allows ten characters for a field name.

    Deliberately not asserting this of *every* qc_ column: the denormalized
    per-line verdicts exceed ten across several checks and collide on
    truncation (qc_attitude and qc_attitude_line both become qc_attitud). That
    is a naming problem for the vector export to resolve centrally, not one for
    each check to work around.
    """
    dataset, _ = transmitter_stability(currents([9.0] * 3), **SPEC)
    for name in ("qc_current", "qc_nominal", "qc_cur_ch1"):
        assert name in dataset.flightlines.columns
        assert len(name) <= 10


def test_the_report_survives_serialization():
    _, report = transmitter_stability(currents([9.0, 12.0]), **SPEC)
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["step"] == "transmitter_stability"
    assert restored["finding"] == "instrument"


def test_no_moments_is_a_note_not_an_error():
    dataset = currents([9.0] * 3)
    dataset.survey.xyz.model_dict["layer_data"] = {}
    dataset, report = transmitter_stability(dataset, **SPEC)
    assert any("no moments" in n.lower() for n in report.notes)


# ── Registration ─────────────────────────────────────────────────────────────

def test_registered_and_runs_from_a_pipeline():
    assert "transmitter_stability" in load_steps(QC_GROUP)

    dataset = currents([9.0] * 4)
    pipeline = Pipeline.parse("""
        kind: qc
        steps:
          - transmitter_stability: {good_fraction: 0.01, marginal_fraction: 0.03}
    """)
    dataset, report = QCRunner(pipeline).run(dataset)

    assert list(dataset.flightlines["qc_current"]) == [GOOD] * 4
    # An instrument finding, so it feeds qc_sensor rather than qc_verdict.
    assert list(dataset.flightlines["qc_sensor"]) == [GOOD] * 4
