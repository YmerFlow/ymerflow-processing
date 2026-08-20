"""Misfit as a column — 01's resolution of filter_31's oddity.

Once misfit is a column it is plottable, exportable, gradeable by QC, and
thresholdable by an ordinary culling filter. The loop becomes explicit as steps
rather than one filter reaching backwards.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.mask import CULLED
from ymerflow_processing.core.misfit import (
    RMS_COLUMN, compute_misfit, ensure_misfit, find_misfit, gatewise_misfit,
    misfit_key,
)
from ymerflow_processing.core.pipeline import (
    FILTER_GROUP, Pipeline, ProcessingRunner, load_steps,
)

N = 8
GATES = 6


def delivery(sigma=0.10, **kw):
    dataset = make_dataset(n=N, n_lines=1, n_gates=GATES, **kw)
    for channel in dataset.channels:
        dataset.layer_data[channel.std_key].iloc[:, :] = sigma
    return dataset


def forward(dataset, scale=1.0, gates=None):
    """A forward synthetic: the data, missed by ``scale`` at ``gates``.

    ``gates=None`` means every gate — an earlier version silently ignored
    ``scale`` in that case, so a test asking for a uniform 20% shortfall got an
    identical synthetic and a misfit of zero.
    """
    from libaarhusxyz.xyz import XYZ
    layers = {}
    for channel in dataset.channels:
        predicted = dataset.layer_data[channel.data_key].to_numpy().copy()
        chosen = slice(None) if gates is None else gates
        predicted[:, chosen] *= scale
        layers[channel.data_key] = pd.DataFrame(predicted)
    return XYZ({"flightlines": dataset.flightlines.copy(), "layer_data": layers,
                "model_info": {}, "file_meta": {}})


# ── The quantity ─────────────────────────────────────────────────────────────

def test_misfit_is_measured_minus_synthetic_over_uncertainty():
    """A 20% shortfall against a 10% sigma is two standard deviations."""
    dataset = delivery(sigma=0.10)
    misfit = gatewise_misfit(dataset, forward(dataset, scale=0.8, gates=[3]),
                             dataset.channels[0])

    assert misfit[0, 3] == pytest.approx(2.0, rel=1e-6)
    assert misfit[0, 0] == pytest.approx(0.0)


def test_the_same_shortfall_reads_the_same_at_every_gate():
    """The whole point of dividing. A transient spans orders of magnitude, so a
    raw difference of fixed size is enormous at late time and negligible at
    early time — no single threshold on it is wrong in neither direction."""
    dataset = delivery(sigma=0.10)
    times = np.asarray(dataset.channels[0].axis.values, dtype=float)
    for channel in dataset.channels:
        dataset.layer_data[channel.data_key].iloc[:, :] = np.tile(
            (times ** -2.0)[None, :], (N, 1))

    misfit = gatewise_misfit(dataset, forward(dataset, scale=0.8,
                                              gates=[0, GATES - 1]),
                             dataset.channels[0])
    assert misfit[0, 0] == pytest.approx(misfit[0, -1], rel=1e-6)


def test_a_stated_residual_is_preferred_over_recomputing():
    """An inversion writes (dobs - dpred) * W.diagonal() itself."""
    from libaarhusxyz.xyz import XYZ
    dataset = delivery()
    result = XYZ({"flightlines": dataset.flightlines.copy(),
                  "layer_data": {"dbdt_err_ch1gt": pd.DataFrame(
                      np.full((N, GATES), 7.0))},
                  "model_info": {}, "file_meta": {}})

    assert np.allclose(gatewise_misfit(dataset, result, dataset.channels[0]),
                       7.0)


def test_predictions_with_no_uncertainty_are_refused():
    dataset = delivery()
    synthetic = forward(dataset)
    del dataset.layer_data[dataset.channels[0].std_key]

    with pytest.raises(ValueError, match="raw difference cannot be"):
        gatewise_misfit(dataset, synthetic, dataset.channels[0])


# ── What it writes ───────────────────────────────────────────────────────────

def test_it_writes_the_gatewise_column_under_the_inversion_s_own_name():
    """So a dataset whose misfit came from an inversion and one computed here
    are indistinguishable, and anything reading misfit finds it the same way."""
    dataset, _ = compute_misfit(delivery(), forward(delivery()))

    assert misfit_key(dataset.channels[0]) == "dbdt_err_ch1gt"
    assert "dbdt_err_ch1gt" in dataset.layer_data


def test_it_writes_a_per_sounding_rms():
    dataset = delivery(sigma=0.10)
    dataset, report = compute_misfit(dataset, forward(dataset, scale=0.8))

    # Every gate off by two sigma, so the RMS is two.
    assert np.allclose(dataset.flightlines[RMS_COLUMN], 2.0)
    assert report.distributions["rms"]["max"] == pytest.approx(2.0)


def test_the_rms_ignores_culled_gates():
    """A culled gate is not evidence about the fit: a sounding whose bad gates
    were removed before inverting should not carry their residual."""
    dataset = delivery(sigma=0.10)
    synthetic = forward(dataset, scale=0.2, gates=[5])       # 8 sigma at gate 5
    dataset.layer_data[dataset.channels[0].inuse_key].iloc[:, 5] = CULLED
    dataset.layer_data[dataset.channels[1].inuse_key].iloc[:, 5] = CULLED

    dataset, _ = compute_misfit(dataset, synthetic)
    assert np.allclose(dataset.flightlines[RMS_COLUMN], 0.0)


def test_the_rms_column_fits_a_shapefile_field():
    assert len(RMS_COLUMN) <= 10


def test_it_changes_no_measurement_uncertainty_or_mask():
    dataset = delivery()
    keep = {k: v.copy() for k, v in dataset.layer_data.items()}

    dataset, _ = compute_misfit(dataset, forward(dataset, scale=0.8))
    for key, array in keep.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


def test_a_channel_the_result_says_nothing_about_is_named():
    from libaarhusxyz.xyz import XYZ
    dataset = delivery()
    partial = XYZ({"flightlines": dataset.flightlines.copy(),
                   "layer_data": {"dbdt_err_ch1gt": pd.DataFrame(
                       np.zeros((N, GATES)))},
                   "model_info": {}, "file_meta": {}})

    dataset, report = compute_misfit(dataset, partial)
    assert any("ch2" in n and "no misfit column" in n for n in report.notes)


# ── Creating one on demand ───────────────────────────────────────────────────

def test_ensure_returns_an_existing_column_without_a_result():
    dataset = delivery(sigma=0.10)
    dataset, _ = compute_misfit(dataset, forward(dataset, scale=0.8))

    found = ensure_misfit(dataset, dataset.channels[0])
    assert np.allclose(found, 2.0)


def test_ensure_creates_and_writes_one_from_a_result():
    dataset = delivery(sigma=0.10)
    key = misfit_key(dataset.channels[0])
    assert key not in dataset.layer_data

    found = ensure_misfit(dataset, dataset.channels[0],
                          forward(dataset, scale=0.8))
    assert np.allclose(found, 2.0)
    assert key in dataset.layer_data


def test_ensure_returns_nothing_when_it_has_neither():
    """The caller's cue to say what it needs rather than to guess."""
    dataset = delivery()
    assert ensure_misfit(dataset, dataset.channels[0]) is None


def test_find_matches_on_a_captured_channel_index():
    """Not by substring — channel 1 must not match ch10."""
    from libaarhusxyz.xyz import XYZ
    dataset = delivery()
    result = XYZ({"flightlines": dataset.flightlines.copy(),
                  "layer_data": {"dbdt_err_ch10gt": pd.DataFrame(
                      np.zeros((N, GATES)))},
                  "model_info": {}, "file_meta": {}})

    assert find_misfit(result, dataset.channels[0]) is None


# ── The loop, explicit as steps ──────────────────────────────────────────────

def test_the_whole_chain_runs_from_one_pipeline(tmp_path):
    """01: that makes the loop explicit as steps rather than one filter
    reaching backwards."""
    dataset = delivery(sigma=0.10)
    synthetic = forward(dataset, scale=0.4, gates=[4, 5])
    path = tmp_path / "result.xyz"
    synthetic.dump(str(path))

    dataset, report = ProcessingRunner(Pipeline.parse(f"""
        kind: filters
        steps:
          - compute_misfit: {{result: {path}}}
          - cull_by_external_evidence: {{criterion: misfit, max_misfit: 3.0}}
    """)).run(dataset)

    ran = [s.step for s in report.steps]
    assert ran[-2:] == ["compute_misfit", "cull_by_external_evidence"]
    assert (dataset.layer_data[dataset.channels[0].inuse_key].to_numpy()[:, 4:]
            == CULLED).all()


def test_misfit_is_generally_available_once_it_is_a_column():
    """01: plottable, exportable in the vector output, usable by QC — instead
    of locked inside one filter."""
    dataset = delivery(sigma=0.10)
    dataset, _ = compute_misfit(dataset, forward(dataset, scale=0.8))

    # A numeric flightlines column, which is what the plot layer can draw.
    assert RMS_COLUMN in dataset.flightlines.columns
    assert np.issubdtype(dataset.flightlines[RMS_COLUMN].dtype, np.floating)


def test_the_report_survives_serialization():
    dataset = delivery()
    _, report = compute_misfit(dataset, forward(dataset, scale=0.8))
    restored = json.loads(json.dumps(report.to_dict(), default=str))
    assert restored["step"] == "compute_misfit"
    assert restored["distributions"]["rms_column"] == RMS_COLUMN


def test_registered():
    assert "compute_misfit" in load_steps(FILTER_GROUP)
