"""filter_11 and filter_13 — adjusting uncertainty, and drawing noise from it.

filter_11 adjusts what is there; filter_12 asserts a model. filter_13 turns a
noise-free forward model into something an inversion can be fairly tested on,
and most of its substance is making a wrong ordering detectable rather than
silently plausible.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.filter_11_inflate import inflate_uncertainty
from ymerflow_processing.core.filter_13_perturb import (
    STASH_SUFFIX, applied, perturb_from_uncertainty,
)
from ymerflow_processing.core.filter_12_uncertainty import assign_uncertainty
from ymerflow_processing.core.noise import estimate_noise_model, REFERENCE_TIME
from ymerflow_processing.core.pipeline import (
    FILTER_GROUP, Pipeline, ProcessingRunner, load_steps,
)

N = 40
GATES = 8


def delivery(**kw):
    return make_dataset(n=N, n_lines=2, n_gates=GATES, **kw)


def realistic(seed=3):
    """A decaying transient over ambient noise, so a model can be estimated."""
    dataset = delivery()
    rng = np.random.default_rng(seed)
    times = np.asarray(dataset.channels[0].axis.values, dtype=float)
    for channel in dataset.channels:
        response = 1e-6 * (times / times[0]) ** -2.0
        ambient = rng.normal(0.0, 5e-10 * (times / REFERENCE_TIME) ** -0.5,
                             (N, GATES))
        dataset.layer_data[channel.data_key].iloc[:, :] = response[None, :] + ambient
    return dataset


def std(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].std_key].to_numpy()


# ── filter_11 ────────────────────────────────────────────────────────────────

def test_it_adds_a_fraction_arithmetically():
    """"Add two percent" moves 3% to 5%, not to 3.6%. A number someone chose is
    not an independent error, so it does not combine like one."""
    dataset, _ = inflate_uncertainty(delivery(), fraction=0.02)
    assert std(dataset)[0, 0] == pytest.approx(0.05)


def test_a_negative_fraction_reduces():
    dataset, _ = inflate_uncertainty(delivery(), fraction=-0.01)
    assert std(dataset)[0, 0] == pytest.approx(0.02)


def test_it_cannot_produce_a_negative_uncertainty():
    """One travelling downstream would be squared by something and become a
    plausible positive again, which is the worst kind of wrong."""
    dataset, report = inflate_uncertainty(delivery(), fraction=-0.5)

    assert np.allclose(std(dataset), 0.0)
    assert any("clamped there" in n for n in report.notes)
    assert report.distributions["per_channel"]["ch1 (LM/z)"][
        "clamped_at_zero"] == N * GATES


def test_a_gate_range_limits_where_it_applies():
    dataset, _ = inflate_uncertainty(delivery(), fraction=0.02, gates="0-2")

    assert np.allclose(std(dataset)[:, :3], 0.05)
    assert np.allclose(std(dataset)[:, 3:], 0.03)


def test_the_gate_idiom_is_filter_21s():
    """Learned once, works everywhere (R7)."""
    dataset, _ = inflate_uncertainty(delivery(), fraction=0.02, gates=[0, 7])

    assert std(dataset)[0, 0] == pytest.approx(0.05)
    assert std(dataset)[0, 7] == pytest.approx(0.05)
    assert std(dataset)[0, 3] == pytest.approx(0.03)


def test_it_can_apply_per_channel():
    dataset, _ = inflate_uncertainty(delivery(), fraction={"ch1": 0.02})

    assert std(dataset, 0)[0, 0] == pytest.approx(0.05)
    assert std(dataset, 1)[0, 0] == pytest.approx(0.03)


def test_it_changes_no_measurement_or_mask():
    dataset = delivery()
    keep = {k: v.copy() for k, v in dataset.layer_data.items()
            if "std" not in k.lower()}

    dataset, _ = inflate_uncertainty(dataset, fraction=0.02)

    for key, array in keep.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


def test_no_fraction_is_an_error():
    with pytest.raises(ValueError, match="needs a fraction"):
        inflate_uncertainty(delivery(), fraction=None)


def test_it_adjusts_rather_than_asserting():
    """The difference from filter_12, which replaces with a physical model.
    Running 11 twice compounds; that is what "adjust" means."""
    dataset, _ = inflate_uncertainty(delivery(), fraction=0.01)
    dataset, _ = inflate_uncertainty(dataset, fraction=0.01)
    assert std(dataset)[0, 0] == pytest.approx(0.05)


# ── filter_13 ────────────────────────────────────────────────────────────────

def test_it_perturbs_the_values():
    dataset = delivery()
    before = dataset.layer_data[dataset.channels[0].data_key].to_numpy().copy()

    dataset, _ = perturb_from_uncertainty(dataset, seed=1)
    after = dataset.layer_data[dataset.channels[0].data_key].to_numpy()

    # atol=0: the default 1e-8 swamps a response of order 1e-9, so every
    # comparison would read as close and the test would pass on a no-op.
    assert not np.allclose(before, after, atol=0)


def test_the_spread_scales_with_the_value_not_the_relative_figure():
    """The stored uncertainty is relative, so drawing from it directly would put
    noise of order one on every datum whatever its size."""
    dataset = delivery()
    key = dataset.channels[0].data_key
    dataset.layer_data[key].iloc[:, :] = 1e-6            # a strong response
    dataset, _ = perturb_from_uncertainty(dataset, seed=1)

    residual = dataset.layer_data[key].to_numpy() - 1e-6
    # 3% of 1e-6 is 3e-8; the scatter should be of that order, not of order 0.03.
    assert np.std(residual) == pytest.approx(3e-8, rel=0.3)


def test_the_same_seed_repeats_exactly():
    """"A fair test" is not a fair test if it cannot be repeated."""
    one, _ = perturb_from_uncertainty(delivery(), seed=7)
    two, _ = perturb_from_uncertainty(delivery(), seed=7)

    np.testing.assert_array_equal(
        one.layer_data[one.channels[0].data_key].to_numpy(),
        two.layer_data[two.channels[0].data_key].to_numpy())


def test_a_different_seed_gives_different_noise():
    one, _ = perturb_from_uncertainty(delivery(), seed=7)
    two, _ = perturb_from_uncertainty(delivery(), seed=8)

    assert not np.allclose(
        one.layer_data[one.channels[0].data_key].to_numpy(),
        two.layer_data[two.channels[0].data_key].to_numpy(), atol=0)


def test_the_noise_free_values_are_stashed():
    dataset = delivery()
    key = dataset.channels[0].data_key
    before = dataset.layer_data[key].copy()

    dataset, _ = perturb_from_uncertainty(dataset, seed=1)

    pd.testing.assert_frame_equal(
        dataset.layer_data[f"{key}{STASH_SUFFIX}"], before)


def test_it_changes_no_uncertainty_or_mask():
    dataset = delivery()
    keep = {k: v.copy() for k, v in dataset.layer_data.items()
            if "std" in k.lower() or "inuse" in k.lower()}

    dataset, _ = perturb_from_uncertainty(dataset, seed=1)

    for key, array in keep.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


# ── Ordering, which is the hazard ────────────────────────────────────────────

def test_it_refuses_to_run_twice():
    """The classic silent error: the result is noisier by sqrt(2) and looks
    exactly like data."""
    dataset, _ = perturb_from_uncertainty(delivery(), seed=1)

    with pytest.raises(ValueError, match="already carries synthetic noise"):
        perturb_from_uncertainty(dataset, seed=2)


def test_a_second_pass_can_be_asked_for_explicitly():
    dataset, _ = perturb_from_uncertainty(delivery(), seed=1)
    dataset, report = perturb_from_uncertainty(dataset, seed=2, again=True)

    assert report.distributions["repeated"] is True


def test_the_application_is_recorded_with_its_seed():
    dataset, _ = perturb_from_uncertainty(delivery(), seed=42)
    assert applied(dataset) == "seed=42"


def test_uniform_uncertainties_are_called_out():
    """The signature of a default from the system description rather than an
    assigned model — noise drawn from it is flat in relative terms, so late
    gates come out as clean as early ones."""
    _, report = perturb_from_uncertainty(delivery(), seed=1)

    assert any("uniform default" in n for n in report.notes)
    assert report.distributions["per_channel"]["ch1 (LM/z)"][
        "uncertainty_uniform"] is True


def test_an_assigned_model_is_not_called_out():
    """A warning that fires on correct use is worse than none."""
    dataset, _ = estimate_noise_model(realistic())
    dataset, _ = assign_uncertainty(dataset)
    _, report = perturb_from_uncertainty(dataset, seed=1)

    assert not any("uniform default" in n for n in report.notes)


def test_a_missing_seed_is_an_error():
    with pytest.raises(ValueError, match="needs a seed"):
        perturb_from_uncertainty(delivery(), seed=None)


# ── The chain they belong to ─────────────────────────────────────────────────

def test_the_whole_synthetic_chain_runs_in_order():
    """estimate -> assign -> perturb, which is the order 01 says matters."""
    dataset, report = ProcessingRunner(Pipeline.parse("""
        kind: filters
        steps:
          - estimate_noise_model: {}
          - inflate_uncertainty: {fraction: 0.01}
          - perturb_from_uncertainty: {seed: 11}
    """)).run(realistic())

    ran = [s.step for s in report.steps]
    assert ran[-3:] == ["estimate_noise_model", "inflate_uncertainty",
                        "perturb_from_uncertainty"]
    assert applied(dataset) == "seed=11"


def test_late_gates_get_noise_larger_than_the_datum():
    """Correct, and what makes the synthetic look like real late-time data: the
    floor has overtaken the signal, so the assigned uncertainty passed one."""
    dataset, _ = estimate_noise_model(realistic())
    dataset, _ = assign_uncertainty(dataset)
    key = dataset.channels[0].data_key
    clean = dataset.layer_data[key].to_numpy().copy()

    dataset, _ = perturb_from_uncertainty(dataset, seed=5)
    residual = np.abs(dataset.layer_data[key].to_numpy() - clean)

    assert np.median(residual[:, -1] / np.abs(clean[:, -1])) > 1.0
    assert np.median(residual[:, 0] / np.abs(clean[:, 0])) < 0.5


def test_the_reports_survive_serialization():
    dataset, inflated = inflate_uncertainty(delivery(), fraction=0.02)
    dataset, perturbed = perturb_from_uncertainty(dataset, seed=1)

    for report in (inflated, perturbed):
        restored = json.loads(json.dumps(report.to_dict(), default=str))
        assert restored["distributions"]["per_channel"]


@pytest.mark.parametrize("name", ["inflate_uncertainty",
                                  "perturb_from_uncertainty"])
def test_registered(name):
    assert name in load_steps(FILTER_GROUP)
