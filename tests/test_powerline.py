"""qc_05 — powerline and cultural noise, end to end."""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset, tier_names
from ymerflow_processing.core.pipeline import QC_GROUP, Pipeline, QCRunner, load_steps
from ymerflow_processing.core.qc.powerline import (
    find_powerline_column, local_excess, powerline_noise,
)
from ymerflow_processing.core.qc.thresholds import GOOD, MARGINAL, OUT_OF_SPEC

# Detection sensitivity, not a contract term.
SPEC = {"good_max": 5.0, "marginal_max": 10.0}


def monitor(values, n_lines=1, **kw):
    values = np.asarray(values, dtype=float)
    dataset = make_dataset(n=len(values), n_lines=n_lines, **kw)
    dataset.flightlines["PLNI"] = values
    return dataset


def quiet(n, level=50.0, spread=3.0, seed=5):
    return np.random.default_rng(seed).normal(level, spread, n)


def test_the_monitor_is_found_under_geotech_s_spelling_too():
    """A VTEM delivery calls it `PLM`. Without that spelling the check reported
    the survey as recording no monitor at all."""
    dataset = monitor(quiet(6))
    dataset.flightlines.rename(columns={"PLNI": "PLM"}, inplace=True)
    assert find_powerline_column(dataset.flightlines) == "PLM"


# ── Reference by default ─────────────────────────────────────────────────────

def test_without_thresholds_it_measures_and_says_nothing():
    """The usual way to read this check: a trace to look at."""
    dataset, report = powerline_noise(monitor(quiet(200)))

    assert "pl_excess" in dataset.flightlines.columns
    assert "qs_pline" not in dataset.flightlines.columns
    assert report.verdict_column is None
    assert report.finding is None
    assert any("left unjudged" in n for n in report.notes)


def test_with_thresholds_it_classifies():
    values = quiet(200)
    values[100:104] += 400
    dataset, report = powerline_noise(monitor(values), **SPEC)

    assert tier_names(dataset.flightlines["qs_pline"])[101] == OUT_OF_SPEC
    assert report.verdict_column == "qs_pline"
    assert report.finding == "environment"


def test_half_a_threshold_is_an_error():
    with pytest.raises(ValueError, match="both bounds"):
        powerline_noise(monitor(quiet(50)), good_max=5.0)


# ── It finds local peaks, whatever the absolute level ────────────────────────

def test_a_crossing_is_found_at_any_baseline():
    """A quiet region and one an order of magnitude noisier, each with a
    crossing. Absolute levels differ hugely; both peaks show."""
    values = np.concatenate([quiet(200, 50, 3), quiet(200, 900, 40, seed=6)])
    values[80:84] += 400          # crossing in the quiet half
    values[300:304] += 2500       # crossing in the noisy half

    dataset, _ = powerline_noise(monitor(values), **SPEC)
    verdicts = dataset.flightlines["qs_pline"]

    assert tier_names(verdicts)[81] == OUT_OF_SPEC
    assert tier_names(verdicts)[301] == OUT_OF_SPEC


def test_a_change_of_region_is_not_a_peak():
    """The reason the baseline is local rather than per line: flying from
    farmland into a town raises the background without being a crossing."""
    values = np.concatenate([quiet(200, 50, 3), quiet(200, 900, 40, seed=6)])
    values[80:84] += 400

    dataset, _ = powerline_noise(monitor(values))
    excess = dataset.flightlines["pl_excess"].to_numpy()

    # A sharp step does register at the boundary — half the centered window
    # sits either side of it — but at ~16 sigma against the crossing's ~142.
    # Nearly an order of magnitude, which is what lets one threshold separate
    # "there is a powerline here" from "the countryside changed".
    crossing = np.nanmax(excess[78:86])
    step = np.nanmax(np.abs(excess[195:205]))
    assert crossing > 100
    assert step < 20
    assert crossing > 5 * step


def test_a_whole_line_baseline_would_have_drowned_the_quiet_crossing():
    """What the local window buys, stated as a comparison."""
    values = np.concatenate([quiet(200, 50, 3), quiet(200, 900, 40, seed=6)])
    values[80:84] += 400

    local = local_excess(values, np.zeros(len(values)), window=101)
    # Against a single median and MAD for the whole line, the quiet half's
    # crossing is smaller than the routine scatter of the noisy half.
    whole = (values - np.median(values)) / (
        1.4826 * np.median(np.abs(values - np.median(values))))

    assert local[81] > 50            # obvious locally
    assert whole[81] < whole[300]    # invisible against the line as a whole


def test_the_window_defines_what_local_means():
    """An anomaly wider than the window becomes its own background."""
    values = quiet(400)
    values[150:250] += 300           # 100 soundings wide

    narrow = local_excess(values, np.zeros(400), window=51)   # narrower than it
    wide = local_excess(values, np.zeros(400), window=301)    # wider than it

    # Swallowed: inside the anomaly only the ordinary scatter shows, far below
    # any sensible threshold. Seen: against a wider view it is unmistakable.
    assert np.nanmax(np.abs(narrow[190:210])) < 3.0
    assert np.nanmax(wide[150:250]) > 20.0


# ── The baseline respects line boundaries ────────────────────────────────────

def test_the_window_does_not_span_the_turn_between_lines():
    """Two lines at very different levels must not contaminate each other."""
    values = np.concatenate([quiet(100, 50, 3), quiet(100, 900, 40, seed=6)])
    dataset = monitor(values, n_lines=2)

    dataset, _ = powerline_noise(dataset, **SPEC)
    verdicts = dataset.flightlines["qs_pline"]

    # Nothing near the join is flagged; each line is judged against itself.
    assert not (verdicts[95:105] == OUT_OF_SPEC).any()


def test_a_flat_line_has_no_scale_to_measure_against():
    dataset, report = powerline_noise(monitor(np.full(200, 50.0)), **SPEC)

    assert dataset.flightlines["pl_excess"].isna().all()
    assert any("no variation" in n for n in report.notes)


# ── Finding the column ───────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["PLNI", "plni", "powerline", "pline"])
def test_the_monitor_is_found_under_its_usual_spellings(name):
    frame = pd.DataFrame({name: [1.0], "other": [0]})
    assert find_powerline_column(frame) == name


def test_no_monitor_is_a_note_not_an_error():
    dataset = make_dataset(n=10, n_lines=1)
    dataset, report = powerline_noise(dataset, **SPEC)

    assert "pl_excess" not in dataset.flightlines.columns
    assert any("no powerline monitor" in n.lower() for n in report.notes)


def test_a_named_column_that_is_absent_is_an_error():
    with pytest.raises(KeyError, match="'emi'"):
        powerline_noise(monitor(quiet(50)), column="emi", **SPEC)


# ── How it lands in the rollup ───────────────────────────────────────────────

def test_it_feeds_neither_the_flight_nor_the_sensor_verdict():
    """A powerline is neither the contractor's doing nor the instrument's
    fault, so it is named as a reason and summarized as neither."""
    values = quiet(200)
    values[100:104] += 400
    dataset = monitor(values, height=np.full(200, 30.0))

    dataset, _ = QCRunner(Pipeline.parse("""
        kind: qc
        steps:
          - height_above_ground: {good_max: 45, marginal_max: 65}
          - powerline_noise: {good_max: 5.0, marginal_max: 10.0}
    """)).run(dataset)

    row = dataset.flightlines.iloc[101]
    assert tier_names([row["qs_pline"]])[0] == OUT_OF_SPEC
    assert tier_names([row["qs_verdict"]])[0] == GOOD          # the flying was fine
    assert pd.isna(row["qs_sensor"])          # no instrument check ran
    assert "powerline_noise" in row["qs_reasons"]


def test_the_report_survives_serialization():
    values = quiet(200); values[100:104] += 400
    _, report = powerline_noise(monitor(values), **SPEC)
    restored = json.loads(json.dumps(report.to_dict()))
    assert restored["step"] == "powerline_noise"
    assert restored["distributions"]["window"] == 101


def test_changes_no_measurement_uncertainty_or_mask():
    dataset = monitor(quiet(50))
    before = {k: v.copy() for k, v in dataset.layer_data.items()}
    dataset, _ = powerline_noise(dataset, **SPEC)
    for key, array in before.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


def test_registered():
    assert "powerline_noise" in load_steps(QC_GROUP)
