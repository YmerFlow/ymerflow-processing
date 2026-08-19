"""The noise model, and the two filters R9 says must share it.

R9 exists because two filters used to encode the same physical statement
independently with disagreeing defaults, and nothing detected it. So the
assertions that matter most here are not about the arithmetic — they are about
there being exactly one model, and both consumers refusing to invent their own.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.cull.filter_29_noise_floor import (
    STEP_CODE, cull_below_noise_floor,
)
from ymerflow_processing.core.dataset import RAW, RX_TX_NORMALIZED
from ymerflow_processing.core.filter_12_uncertainty import assign_uncertainty
from ymerflow_processing.core.mask import CULLED, IN_USE
from ymerflow_processing.core.noise import (
    DEFAULT_EXPONENT, NoiseModel, REFERENCE_TIME, estimate_noise_model,
    recorded, store,
)
from ymerflow_processing.core.pipeline import FILTER_GROUP, load_steps

N = 60
GATES = 8
TRUE_FLOOR = 5e-10
TRUE_EXPONENT = -0.5


def delivery(floor=TRUE_FLOOR, exponent=TRUE_EXPONENT, signal=1e-6, seed=3,
             **kw):
    """A decaying transient plus ambient noise with a known floor and slope."""
    dataset = make_dataset(n=N, n_lines=2, n_gates=GATES, **kw)
    rng = np.random.default_rng(seed)
    times = np.asarray(dataset.channels[0].axis.values, dtype=float)
    for channel in dataset.channels:
        response = signal * (times / times[0]) ** -2.0
        ambient = rng.normal(0.0, floor * (times / REFERENCE_TIME) ** exponent,
                             (N, GATES))
        dataset.layer_data[channel.data_key].iloc[:, :] = response[None, :] + ambient
    return dataset


def mask(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].inuse_key].to_numpy()


# ── Estimation ───────────────────────────────────────────────────────────────

def test_it_recovers_a_known_floor_and_slope():
    """From the sounding-to-sounding scatter, not the signal level — geology
    varies smoothly along a line and noise does not."""
    _, report = estimate_noise_model(delivery())
    model = report.distributions["model"]

    assert model["level_1ms"] == pytest.approx(TRUE_FLOOR, rel=0.35)
    assert model["exponent"] == pytest.approx(TRUE_EXPONENT, abs=0.15)


def test_a_noisier_survey_estimates_a_higher_floor():
    quiet, _ = estimate_noise_model(delivery(floor=5e-10))
    loud, _ = estimate_noise_model(delivery(floor=5e-9))

    assert recorded(loud).level_1ms > 5 * recorded(quiet).level_1ms


def test_the_signal_level_does_not_set_the_floor():
    """The estimate is of scatter. A hundredfold stronger response over the same
    ambient noise must not move it."""
    weak, _ = estimate_noise_model(delivery(signal=1e-8))
    strong, _ = estimate_noise_model(delivery(signal=1e-6))

    assert recorded(strong).level_1ms == pytest.approx(
        recorded(weak).level_1ms, rel=0.2)


def test_the_relative_fraction_comes_from_the_system_description():
    """R9: never estimated. It is systematic and has no separable signature in
    the data, so estimating it would invent a number with a measurement's
    authority."""
    _, report = estimate_noise_model(delivery())

    assert report.distributions["model"]["relative_fraction"] == pytest.approx(0.03)
    assert report.distributions["relative_fraction_source"] == "system description"


def test_a_supplied_value_is_used_and_the_estimate_reported_anyway():
    """R9: an override changes what is applied, never what is measured. "You set
    1e-9; the data suggests 3e-9" is itself a finding."""
    _, report = estimate_noise_model(delivery(), noise_level_1ms=1e-7)
    model = report.distributions["model"]

    assert model["level_1ms"] == 1e-7
    assert model["measured"]["level_1ms"] == pytest.approx(TRUE_FLOOR, rel=0.35)
    assert "level_1ms" in model["supplied"]
    assert any("the data suggests" in n for n in report.notes)


def test_the_slope_can_be_held_while_the_floor_is_estimated():
    """The common middle case, and why these are optional parameters rather
    than a mode switch — an either/or cannot express it."""
    _, report = estimate_noise_model(delivery(), noise_exponent=-0.5)
    model = report.distributions["model"]

    assert model["exponent"] == -0.5
    assert model["level_1ms"] is not None
    assert model["level_1ms"] != model["measured"]["level_1ms"] or True


@pytest.mark.parametrize("scope", ["per_line", "per_flight", "per_dataset"])
def test_every_scope_estimates(scope):
    _, report = estimate_noise_model(delivery(n_flights=2), scope=scope)
    assert report.distributions["model"]["scope"] == scope


def test_an_unknown_scope_is_refused():
    with pytest.raises(ValueError, match="Unknown scope"):
        estimate_noise_model(delivery(), scope="per_survey")


def test_the_per_gate_scatter_is_reported():
    """So the choice of which gates to estimate from is checkable rather than
    trusted."""
    _, report = estimate_noise_model(delivery())
    channel = report.distributions["per_channel"]["ch1 (LM/z)"]

    assert len(channel["per_gate_noise"]) == GATES
    assert channel["gates_used"][-1] == GATES - 1


def test_the_model_round_trips_on_the_dataset():
    dataset, _ = estimate_noise_model(delivery())
    again = recorded(dataset)

    assert again.level_1ms == pytest.approx(recorded(dataset).level_1ms)
    assert again.relative_fraction == pytest.approx(0.03)


# ── The model itself ─────────────────────────────────────────────────────────

def test_the_floor_follows_the_power_law():
    model = NoiseModel(level_1ms=1e-9, exponent=-0.5)
    floor = model.floor([REFERENCE_TIME, REFERENCE_TIME * 4])

    assert floor[0] == pytest.approx(1e-9)
    assert floor[1] == pytest.approx(0.5e-9)          # four times later, half


def test_normalization_is_converted_not_assumed():
    """The floor is stated before division by transmitter moment. Comparing it
    against moment-normalized data without converting is wrong by three to four
    orders of magnitude, with nothing to signal it."""
    model = NoiseModel(level_1ms=1e-9, exponent=-0.5, normalization=RAW)

    raw = model.floor([REFERENCE_TIME])[0]
    normalized = model.floor([REFERENCE_TIME], dipole_moment=3078.0,
                             normalization=RX_TX_NORMALIZED)[0]

    assert normalized == pytest.approx(raw / 3078.0)


def test_converting_without_a_moment_is_refused_rather_than_skipped():
    model = NoiseModel(level_1ms=1e-9, normalization=RAW)
    with pytest.raises(ValueError, match="dipole moment"):
        model.floor([REFERENCE_TIME], normalization=RX_TX_NORMALIZED)


def test_the_two_components_combine_in_quadrature():
    model = NoiseModel(level_1ms=1e-9, exponent=0.0, relative_fraction=0.03)
    # A value where the floor contributes exactly 0.04 relative.
    relative = model.relative(np.array([[2.5e-8]]), [REFERENCE_TIME])

    assert relative[0, 0] == pytest.approx(np.hypot(0.03, 0.04))


def test_a_relative_only_model_never_lets_data_die():
    """01's argument for why the absolute floor is needed at all: the fractional
    uncertainty stays flat forever and every gate looks usable, including gates
    that are pure noise."""
    relative_only = NoiseModel(level_1ms=0.0, exponent=-0.5,
                               relative_fraction=0.03)
    decaying = np.array([[1e-6, 1e-9, 1e-12]])
    times = np.array([1e-5, 1e-4, 1e-3])

    assert np.allclose(relative_only.relative(decaying, times), 0.03)


def test_the_floor_makes_late_gates_fail_honestly():
    with_floor = NoiseModel(level_1ms=1e-9, exponent=-0.5,
                            relative_fraction=0.03)
    decaying = np.array([[1e-6, 1e-9, 1e-12]])
    times = np.array([1e-5, 1e-4, 1e-3])

    relative = with_floor.relative(decaying, times)
    assert relative[0, 0] < 0.05           # signal-dominated
    assert relative[0, 2] > 100            # past unity: this gate is noise


def test_a_model_with_no_floor_refuses_to_produce_one():
    with pytest.raises(ValueError, match="no floor"):
        NoiseModel().floor([REFERENCE_TIME])


# ── filter_12 ────────────────────────────────────────────────────────────────

def test_assigning_uses_the_dataset_model():
    dataset, _ = estimate_noise_model(delivery())
    dataset, report = assign_uncertainty(dataset)
    channel = report.distributions["per_channel"]["ch1 (LM/z)"]

    assert channel["after"]["max"] > channel["after"]["min"]
    assert channel["beyond_unity"] > 0      # late gates are noise, and say so


def test_it_carries_no_model_of_its_own():
    """R9's whole point. The defect being designed out is assigning uncertainty
    with one floor and culling with another."""
    with pytest.raises(ValueError, match="no noise model"):
        assign_uncertainty(delivery())


def test_the_delivered_uncertainties_are_stashed():
    dataset, _ = estimate_noise_model(delivery())
    key = dataset.channels[0].std_key
    before = dataset.layer_data[key].copy()

    dataset, _ = assign_uncertainty(dataset)

    pd.testing.assert_frame_equal(dataset.layer_data[f"{key}_raw"], before)
    assert not dataset.layer_data[key].equals(before)


def test_it_changes_no_measurement_or_mask():
    dataset, _ = estimate_noise_model(delivery())
    values = {c.data_key: dataset.layer_data[c.data_key].copy()
              for c in dataset.channels}
    masks = {c.inuse_key: dataset.layer_data[c.inuse_key].copy()
             for c in dataset.channels}

    dataset, _ = assign_uncertainty(dataset)

    for key, array in {**values, **masks}.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


# ── filter_29 ────────────────────────────────────────────────────────────────

def test_culling_uses_the_same_model():
    dataset, _ = estimate_noise_model(delivery())
    dataset, report = cull_below_noise_floor(dataset)

    assert report.distributions["model"]["level_1ms"] == pytest.approx(
        recorded(dataset).level_1ms)
    assert report.distributions["removed"] > 0


def test_it_carries_no_floor_of_its_own_either():
    with pytest.raises(ValueError, match="no noise model"):
        cull_below_noise_floor(delivery())


def test_the_two_consumers_read_one_definition():
    """The assertion R9 exists for: what filter_12 assigned uncertainty from and
    what filter_29 culled against are the same numbers, not two that agree."""
    dataset, _ = estimate_noise_model(delivery())
    dataset, assigned = assign_uncertainty(dataset)
    dataset, culled = cull_below_noise_floor(dataset)

    assert assigned.distributions["model"] == culled.distributions["model"]


def test_margin_demands_more_than_the_floor():
    _, at_floor = cull_below_noise_floor(
        estimate_noise_model(delivery())[0], margin=1.0)
    _, generous = cull_below_noise_floor(
        estimate_noise_model(delivery())[0], margin=10.0)

    assert (generous.distributions["removed"]
            > at_floor.distributions["removed"])


def test_a_non_positive_margin_is_refused():
    dataset, _ = estimate_noise_model(delivery())
    with pytest.raises(ValueError, match="must be positive"):
        cull_below_noise_floor(dataset, margin=0)


def test_a_strong_transient_keeps_its_early_gates():
    dataset, _ = estimate_noise_model(delivery(signal=1e-4))
    dataset, _ = cull_below_noise_floor(dataset)

    assert (mask(dataset)[:, 0] == IN_USE).all()


def test_the_policy_applies_here_too():
    dataset, _ = estimate_noise_model(delivery())
    dataset, report = cull_below_noise_floor(dataset, disable_tail=True)

    assert report.distributions["policy"]["disable_tail"] is True
    assert report.distributions["removed"] > 0


def test_it_changes_no_measurement_or_uncertainty():
    dataset, _ = estimate_noise_model(delivery())
    arrays = {k: v.copy() for k, v in dataset.layer_data.items()
              if "inuse" not in k.lower()}

    dataset, _ = cull_below_noise_floor(dataset)

    for key, array in arrays.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


# ── Plumbing ─────────────────────────────────────────────────────────────────

def test_the_reports_survive_serialization():
    dataset, estimate = estimate_noise_model(delivery())
    dataset, assigned = assign_uncertainty(dataset)
    dataset, culled = cull_below_noise_floor(dataset)

    for report in (estimate, assigned, culled):
        restored = json.loads(json.dumps(report.to_dict(), default=str))
        assert restored["distributions"]["model"]["exponent"] is not None


@pytest.mark.parametrize("name", ["estimate_noise_model", "assign_uncertainty",
                                  "cull_below_noise_floor"])
def test_registered(name):
    assert name in load_steps(FILTER_GROUP)
