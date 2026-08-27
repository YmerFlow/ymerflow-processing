"""qc_11 — grading how well the model explained each sounding.

The QC half of ``01``'s resolution of filter_31: once misfit is a column it is
gradeable, and this grades it without recomputing it.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.misfit import RMS_COLUMN, compute_misfit
from ymerflow_processing.core.pipeline import (
    QC_GROUP, Pipeline, QCRunner, load_steps,
)
from ymerflow_processing.core.qc.qc_11_misfit import STEP, data_misfit
from ymerflow_processing.core.qc.rollup import DEFAULT_WEIGHTS
from ymerflow_processing.core.qc.thresholds import (
    GOOD, MARGINAL, OUT_OF_SPEC, TIER_POINTS, recorded_thresholds,
)

N = 12
GATES = 6


def graded(misfits=None, **kw):
    dataset = make_dataset(n=N, n_lines=2, n_gates=GATES, **kw)
    if misfits is not None:
        dataset.flightlines[RMS_COLUMN] = np.asarray(misfits, dtype=float)
    return dataset


# ── Optional by design ───────────────────────────────────────────────────────

def test_a_dataset_that_was_never_inverted_is_skipped_not_failed():
    """03: included when an inversion exists, absent otherwise with no error."""
    dataset, report = data_misfit(graded(), good_max=1.5, marginal_max=3.0)

    assert report.verdict_column is None
    assert any("has not been inverted" in note for note in report.notes)
    assert any("not an error" in note for note in report.notes)


def test_the_note_says_how_to_make_the_column_appear():
    _, report = data_misfit(graded())
    assert any("compute_misfit" in note for note in report.notes)


def test_no_thresholds_measures_and_says_nothing():
    """Misfit is in sigmas, so a threshold here inherits whatever the
    uncertainty model decided. A house number would grade that."""
    dataset, report = data_misfit(graded(np.full(N, 1.0)))

    assert report.verdict_column is None
    assert "qs_misfit" not in dataset.flightlines.columns
    assert report.distributions["misfit"]["before"]["percentiles"]["50"] == 1.0
    assert any("left unjudged" in note for note in report.notes)


def test_half_a_threshold_is_refused():
    with pytest.raises(ValueError, match="both bounds"):
        data_misfit(graded(np.full(N, 1.0)), good_max=1.5)


# ── The classification ───────────────────────────────────────────────────────

def test_ascending_a_larger_misfit_is_worse():
    values = np.array([0.5, 2.0, 5.0] * 4)
    dataset, report = data_misfit(graded(values), good_max=1.5, marginal_max=3.0)

    verdicts = dataset.flightlines["qs_misfit"].to_numpy()
    assert verdicts[0] == TIER_POINTS[GOOD]
    assert verdicts[1] == TIER_POINTS[MARGINAL]
    assert verdicts[2] == TIER_POINTS[OUT_OF_SPEC]
    assert report.soundings_affected == 4


def test_a_sounding_with_no_misfit_carries_no_verdict():
    """Nothing of theirs was in use when the fit was measured."""
    values = np.full(N, 1.0)
    values[3] = np.nan
    dataset, report = data_misfit(graded(values), good_max=1.5, marginal_max=3.0)

    assert np.isnan(dataset.flightlines["qs_misfit"].to_numpy()[3])
    assert any("no misfit" in note for note in report.notes)


def test_the_verdict_is_numeric_so_the_frontend_can_see_it():
    dataset, _ = data_misfit(graded(np.full(N, 1.0)), good_max=1.5,
                             marginal_max=3.0)
    assert dataset.flightlines["qs_misfit"].dtype == np.float32


def test_the_thresholds_are_recorded_with_their_unit():
    dataset, report = data_misfit(graded(np.full(N, 1.0)), good_max=1.5,
                                  marginal_max=3.0)

    assert recorded_thresholds(dataset)[STEP]["good_max"] == 1.5
    assert "sigma" in json.dumps(report.distributions["thresholds"])


# ── A poor fit is not necessarily bad data ───────────────────────────────────

def test_it_is_a_model_finding_not_a_flight_or_instrument_one():
    """01, on culling by misfit: a poor fit may indicate genuine geology the
    model cannot represent. Nobody should reflight a line because a 1D inversion
    could not fit a fault."""
    _, report = data_misfit(graded(np.full(N, 5.0)), good_max=1.5,
                            marginal_max=3.0)
    assert report.finding == "model"


def test_it_does_not_move_the_grade():
    """Zero rather than absent — an unlisted check would count 1.0, and this one
    would then grade a survey down for the earth being complicated."""
    assert DEFAULT_WEIGHTS[STEP] == 0.0


def test_it_still_names_itself_among_the_reasons():
    """Weighted out is not silent: 'look at this one' is exactly what a high
    misfit is for. Run beside a check that *is* weighted, so there is a grade
    for it to fail to move."""
    dataset = make_dataset(n=2, n_lines=1, height=[30, 30])
    dataset.flightlines[RMS_COLUMN] = np.array([0.5, 9.0])
    dataset, _ = QCRunner(Pipeline.parse("""
        kind: qc
        steps:
          - height_above_ground: {good_max: 45, marginal_max: 65}
          - data_misfit: {good_max: 1.5, marginal_max: 3.0}
    """)).run(dataset)

    reasons = dataset.flightlines["qs_reasons"].to_numpy()
    assert STEP in reasons[1] and STEP not in reasons[0]

    # Named, and the grade is identical — both were flown equally well.
    scores = dataset.flightlines["qs_score"].to_numpy()
    assert scores[1] == pytest.approx(scores[0])


# ── It grades what is already there ──────────────────────────────────────────

def test_it_reads_the_column_rather_than_recomputing_it():
    """One definition of misfit on the dataset. A check that recomputed it could
    disagree with the column beside it."""
    dataset = graded(np.full(N, 42.0))
    before = {k: v.copy() for k, v in dataset.layer_data.items()}

    dataset, _ = data_misfit(dataset, good_max=1.5, marginal_max=3.0)
    assert np.allclose(dataset.flightlines[RMS_COLUMN], 42.0)
    for key, array in before.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


def test_it_grades_what_compute_misfit_wrote():
    """The two halves meet: compute_misfit writes resdata, this classifies it,
    with no agreement needed beyond the column name."""
    from libaarhusxyz.xyz import XYZ
    dataset = make_dataset(n=N, n_lines=1, n_gates=GATES)
    for channel in dataset.channels:
        dataset.layer_data[channel.std_key].iloc[:, :] = 0.10

    layers = {}
    for channel in dataset.channels:
        predicted = dataset.layer_data[channel.data_key].to_numpy() * 0.8
        layers[channel.data_key] = pd.DataFrame(predicted)
    synthetic = XYZ({"flightlines": dataset.flightlines.copy(),
                     "layer_data": layers, "model_info": {}, "file_meta": {}})

    dataset, _ = compute_misfit(dataset, synthetic)          # two sigma at every gate
    dataset, report = data_misfit(dataset, good_max=1.5, marginal_max=3.0)

    assert report.soundings_affected == 0                    # 2.0 is Marginal
    assert (dataset.flightlines["qs_misfit"] == TIER_POINTS[MARGINAL]).all()


# ── Housekeeping ─────────────────────────────────────────────────────────────

def test_it_is_idempotent():
    dataset = graded(np.array([0.5, 2.0, 5.0] * 4))
    once, _ = data_misfit(dataset, good_max=1.5, marginal_max=3.0)
    first = once.flightlines["qs_misfit"].to_numpy().copy()
    twice, report = data_misfit(once, good_max=1.5, marginal_max=3.0)

    assert np.array_equal(twice.flightlines["qs_misfit"].to_numpy(), first,
                          equal_nan=True)
    assert any("Overwrote" in note for note in report.notes)


def test_it_rolls_up_per_line():
    dataset, report = data_misfit(graded(np.array([0.5] * 6 + [9.0] * 6)),
                                  good_max=1.5, marginal_max=3.0)
    per_line = report.distributions["per_line"]
    assert len(per_line) == 2
    assert sorted(v["fraction_out_of_spec"] for v in per_line.values()) == [0.0, 1.0]


def test_an_empty_dataset_is_handled():
    dataset = graded(np.full(N, 1.0))
    dataset.survey.xyz.flightlines = dataset.flightlines.iloc[:0]
    _, report = data_misfit(dataset, good_max=1.5, marginal_max=3.0)
    assert report.soundings_total == 0


def test_the_report_survives_serialization():
    _, report = data_misfit(graded(np.full(N, 1.0)), good_max=1.5,
                            marginal_max=3.0)
    restored = json.loads(json.dumps(report.to_dict(), default=str))
    assert restored["verdict_column"] == "qs_misfit"


def test_registered():
    assert STEP in load_steps(QC_GROUP)


def test_a_missing_misfit_does_not_claim_to_know_why():
    """The column may have been written here or by an inversion, and the two
    mean different things. Asserting one on a real inversion result — where 37
    of 568 soundings carried no misfit — would have been a guess stated as a
    fact."""
    values = np.full(N, 1.0)
    values[3] = np.nan
    _, report = data_misfit(graded(values), good_max=1.5, marginal_max=3.0)

    note = next(n for n in report.notes if "no misfit" in n)
    assert "cannot tell which" in note
