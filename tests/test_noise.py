"""The noise model, and the two filters R9 says must share it.

R9 exists because two filters used to encode the same physical statement
independently with disagreeing defaults, and nothing detected it. So the
assertions that matter most here are not about the arithmetic — they are about
there being exactly one model, and both consumers refusing to invent their own.
"""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.filters.cull_noise_floor import (
    STEP_CODE, cull_below_noise_floor,
)
from ymerflow_processing.filters.uncertainty import assign_uncertainty
from ymerflow_processing.mask import CULLED, IN_USE
from ymerflow_processing.filters.noise import (
    DEFAULT_EXPONENT, NoiseModel, REFERENCE_TIME, set_noise_floor,
    recorded, store,
)
from ymerflow_processing.pipeline import FILTER_GROUP, load_steps

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
    _, report = set_noise_floor(delivery())
    model = report.distributions["model"]

    assert model["level_1ms"] == pytest.approx(TRUE_FLOOR, rel=0.35)
    assert model["exponent"] == pytest.approx(TRUE_EXPONENT, abs=0.15)


def test_a_noisier_survey_estimates_a_higher_floor():
    quiet, _ = set_noise_floor(delivery(floor=5e-10))
    loud, _ = set_noise_floor(delivery(floor=5e-9))

    assert recorded(loud).level_1ms > 5 * recorded(quiet).level_1ms


def test_the_signal_level_does_not_set_the_floor():
    """The estimate is of scatter. A hundredfold stronger response over the same
    ambient noise must not move it."""
    weak, _ = set_noise_floor(delivery(signal=1e-8))
    strong, _ = set_noise_floor(delivery(signal=1e-6))

    assert recorded(strong).level_1ms == pytest.approx(
        recorded(weak).level_1ms, rel=0.2)


def test_the_floor_carries_no_relative_fraction():
    """The proportional error is not a property of the noise floor. It belongs
    to assign_uncertainty (Ben, 2026-10-21), so the model has no slot for it
    and the step takes no such parameter."""
    _, report = set_noise_floor(delivery())

    assert "relative_fraction" not in report.distributions["model"]
    with pytest.raises(TypeError):
        set_noise_floor(delivery(), relative_fraction=0.03)


def test_a_supplied_value_is_used_and_the_estimate_reported_anyway():
    """R9: an override changes what is applied, never what is measured. "You set
    1e-9; the data suggests 3e-9" is itself a finding."""
    _, report = set_noise_floor(delivery(), noise_level_1ms=1e-7)
    model = report.distributions["model"]

    assert model["level_1ms"] == 1e-7
    assert model["measured"]["level_1ms"] == pytest.approx(TRUE_FLOOR, rel=0.35)
    assert "level_1ms" in model["supplied"]
    assert any("scatter suggests" in n for n in report.notes)


def test_the_estimate_is_put_beside_a_supplied_floor_in_the_same_units():
    """A supplied floor is physical (V/Am²); the estimate is in the arrays' own
    units. On a pV delivery the two as written differ by ~1e16, so the note
    converts the estimate with scalefactor, GateFactor and the moment."""
    dataset = delivery()
    dataset.survey.xyz.model_info["scalefactor"] = 1e-12
    moment = dataset.channels[0].dipole_moment
    _, report = set_noise_floor(dataset, noise_level_1ms=1e-7)

    note = next(n for n in report.notes if "scatter suggests" in n)
    measured = report.distributions["model"]["measured"]["level_1ms"]
    assert f"{measured * 1e-12 * moment:.3g} V/Am²" in note


def test_a_physical_floor_without_a_scalefactor_is_called_out():
    """Right for a synthetic system, silently wrong by 1e12 for a delivery in
    pV that never went through the importer."""
    _, without = set_noise_floor(delivery(), noise_level_1ms=1e-9)
    assert any("carries no scalefactor" in n for n in without.notes)

    dataset = delivery()
    dataset.survey.xyz.model_info["scalefactor"] = 1e-12
    _, with_it = set_noise_floor(dataset, noise_level_1ms=1e-9)
    assert not any("carries no scalefactor" in n for n in with_it.notes)

    _, estimated = set_noise_floor(delivery())
    assert not any("carries no scalefactor" in n for n in estimated.notes)


def test_the_slope_is_held_at_minus_a_half_unless_asked_to_fit():
    """The slope is a property of ambient noise, not of a survey, so it is a
    fixed number with a switch rather than an estimate — a fitted slope far
    from -0.5 means the gates are not noise, and fitting by default would
    bake that mistake into every uncertainty downstream."""
    _, held = set_noise_floor(delivery(exponent=-0.9))
    _, fitted = set_noise_floor(delivery(exponent=-0.9), fit_exponent=True)

    assert held.distributions["model"]["exponent"] == -0.5
    assert "exponent" in held.distributions["model"]["supplied"]
    # Measured either way, so the assumption is checkable.
    assert held.distributions["model"]["measured"]["exponent"] == pytest.approx(
        -0.9, abs=0.2)
    assert fitted.distributions["model"]["exponent"] == pytest.approx(-0.9, abs=0.2)
    assert "exponent" not in fitted.distributions["model"]["supplied"]
    assert any("well away from" in n for n in fitted.notes)


def test_a_different_fixed_slope_may_be_given():
    _, report = set_noise_floor(delivery(), noise_exponent=-0.7)
    assert report.distributions["model"]["exponent"] == -0.7


@pytest.mark.parametrize("scope", ["per_line", "per_flight", "per_dataset"])
def test_every_scope_estimates(scope):
    _, report = set_noise_floor(delivery(n_flights=2), scope=scope)
    assert report.distributions["model"]["scope"] == scope


def test_an_unknown_scope_is_refused():
    with pytest.raises(ValueError, match="Unknown scope"):
        set_noise_floor(delivery(), scope="per_survey")


def test_the_per_gate_scatter_is_reported():
    """So the choice of which gates to estimate from is checkable rather than
    trusted."""
    _, report = set_noise_floor(delivery())
    channel = report.distributions["per_channel"]["ch1 (LM/z)"]

    assert len(channel["per_gate_noise"]) == GATES
    assert channel["gates_used"][-1] == GATES - 1


def test_the_model_round_trips_on_the_dataset():
    dataset, _ = set_noise_floor(delivery())
    again = recorded(dataset)

    assert again.level_1ms == pytest.approx(recorded(dataset).level_1ms)
    assert again.exponent == -0.5


def test_a_model_written_before_the_split_still_reads():
    """A dataset processed when the fraction lived in the model carries
    `relative_fraction=` in model_info. It is read past, not refused."""
    dataset = delivery()
    dataset.survey.xyz.model_info["noise_model"] = (
        "level_1ms=1e-09 exponent=-0.5 relative_fraction=0.03 physical=true "
        "scope=per_dataset")
    model = recorded(dataset)

    assert model.level_1ms == pytest.approx(1e-9)
    assert model.physical is True


# ── The model itself ─────────────────────────────────────────────────────────

def test_the_floor_follows_the_power_law():
    model = NoiseModel(level_1ms=1e-9, exponent=-0.5)
    floor = model.floor([REFERENCE_TIME, REFERENCE_TIME * 4])

    assert floor[0] == pytest.approx(1e-9)
    assert floor[1] == pytest.approx(0.5e-9)          # four times later, half


def test_a_physical_floor_is_divided_by_the_dipole_moment():
    """A supplied floor is physical (V/Am², before moment normalization), so it
    is divided by the transmitter moment — the one amplitude conversion that
    remains, and the one that is silently catastrophic if skipped."""
    model = NoiseModel(level_1ms=1e-9, exponent=-0.5, physical=True)

    assert model.floor([REFERENCE_TIME], dipole_moment=3078.0)[0] == pytest.approx(
        1e-9 / 3078.0)


def test_a_physical_floor_without_a_moment_is_refused_rather_than_skipped():
    model = NoiseModel(level_1ms=1e-9, physical=True)
    with pytest.raises(ValueError, match="dipole moment"):
        model.floor([REFERENCE_TIME])


def test_an_estimated_floor_is_in_the_raw_array_units_and_needs_no_moment():
    """The scatter is measured from the raw arrays, so the floor is already in
    their units: not physical, and `floor()` needs no dipole moment. Marking it
    physical would have the consumers divide it by the moment a second time."""
    dataset, _ = set_noise_floor(delivery())
    model = recorded(dataset)
    channel = dataset.channels[0]
    times = np.asarray(channel.axis.values, dtype=float)

    assert model.physical is False
    assert np.allclose(model.floor(times, dipole_moment=channel.dipole_moment),
                       model.floor(times))


def test_a_supplied_floor_is_physical():
    """The parameter is documented as the floor before moment normalization —
    physical V/Am², the same number SimPEG takes — so the recorded model marks
    it physical and divides it by the moment when applied."""
    dataset, _ = set_noise_floor(delivery(), noise_level_1ms=1e-9)

    assert recorded(dataset).physical is True


def test_the_floor_matches_simpegs_uncert_array():
    """The whole point of the effort: processing culls and weights against the
    same floor SimPEG weights by at inversion. SimPEG's uncert_array adds an
    absolute noise term ``noise = (noise_level_1ms / moment) * (t*1e3)**exponent``
    to the data in physical V/m² — ``dbdt * scalefactor * GateFactor``
    (simpeg static_instrument base.py uncert_array, dual.py:251). Reproduce that
    term and check the noise model's floor contribution to the relative std is
    numerically identical, scalefactor x GateFactor and the moment included."""
    sf, gf, level, exponent = 1e-12, 2.0, 1e-9, -0.5
    dataset = delivery()
    dataset.survey.xyz.model_info["scalefactor"] = sf
    for channel in dataset.channels:
        dataset.survey.gex.gex_dict[f"Channel{channel.index}"]["GateFactor"] = gf

    # No relative fraction (the default) isolates the floor from the
    # proportional component.
    dataset, _ = set_noise_floor(dataset, noise_level_1ms=level,
                                 noise_exponent=exponent)
    model = recorded(dataset)

    for channel in dataset.channels:
        times = np.asarray(channel.axis.values, dtype=float)
        data_raw = dataset.layer_data[channel.data_key].to_numpy(dtype=float)
        moment = channel.dipole_moment

        simpeg_noise = (level / moment) * (times * 1e3) ** exponent    # V/m²
        data_phys = np.abs(data_raw) * sf * gf                          # V/m²
        simpeg_floor_relative = simpeg_noise[None, :] / data_phys

        ymerflow_relative = model.relative(
            data_raw, times, dipole_moment=moment,
            scalefactor=sf, gate_factor=gf)

        assert np.allclose(ymerflow_relative, simpeg_floor_relative)


def test_the_two_components_combine_in_quadrature():
    model = NoiseModel(level_1ms=1e-9, exponent=0.0)
    # A value where the floor contributes exactly 0.04 relative.
    relative = model.relative(np.array([[2.5e-8]]), [REFERENCE_TIME],
                              relative_fraction=0.03)

    assert relative[0, 0] == pytest.approx(np.hypot(0.03, 0.04))


def test_a_relative_only_model_never_lets_data_die():
    """01's argument for why the absolute floor is needed at all: the fractional
    uncertainty stays flat forever and every gate looks usable, including gates
    that are pure noise."""
    no_floor = NoiseModel(level_1ms=0.0, exponent=-0.5)
    decaying = np.array([[1e-6, 1e-9, 1e-12]])
    times = np.array([1e-5, 1e-4, 1e-3])

    assert np.allclose(no_floor.relative(decaying, times, relative_fraction=0.03),
                       0.03)


def test_the_floor_makes_late_gates_fail_honestly():
    with_floor = NoiseModel(level_1ms=1e-9, exponent=-0.5)
    decaying = np.array([[1e-6, 1e-9, 1e-12]])
    times = np.array([1e-5, 1e-4, 1e-3])

    relative = with_floor.relative(decaying, times, relative_fraction=0.03)
    assert relative[0, 0] < 0.05           # signal-dominated
    assert relative[0, 2] > 100            # past unity: this gate is noise


def test_a_model_with_no_floor_refuses_to_produce_one():
    with pytest.raises(ValueError, match="no floor"):
        NoiseModel().floor([REFERENCE_TIME])


# ── filter_12 ────────────────────────────────────────────────────────────────

def test_assigning_uses_the_dataset_model():
    dataset, _ = set_noise_floor(delivery())
    dataset, report = assign_uncertainty(dataset)
    channel = report.distributions["per_channel"]["ch1 (LM/z)"]

    assert channel["after"]["max"] > channel["after"]["min"]
    assert channel["beyond_unity"] > 0      # late gates are noise, and say so


def test_it_carries_no_model_of_its_own():
    """R9's whole point. The defect being designed out is assigning uncertainty
    with one floor and culling with another."""
    with pytest.raises(ValueError, match="no noise floor"):
        assign_uncertainty(delivery())


def test_the_relative_fraction_defaults_to_the_system_description():
    """Never estimated — it is systematic and has no separable signature in the
    data. Unset, each channel's UniformDataSTD is used."""
    dataset, _ = set_noise_floor(delivery())
    _, report = assign_uncertainty(dataset)
    channel = report.distributions["per_channel"]["ch1 (LM/z)"]

    assert channel["relative_fraction"] == pytest.approx(0.03)
    assert channel["relative_fraction_source"] == "system description"


def test_a_given_relative_fraction_wins_over_the_system_description():
    dataset, _ = set_noise_floor(delivery())
    _, report = assign_uncertainty(dataset, relative_fraction=0.05)
    channel = report.distributions["per_channel"]["ch1 (LM/z)"]

    assert channel["relative_fraction"] == 0.05
    assert channel["relative_fraction_source"] == "supplied"


def test_no_fraction_anywhere_is_a_floor_only_assignment_and_says_so():
    dataset, _ = set_noise_floor(delivery())
    for channel in dataset.channels:
        del dataset.survey.gex.gex_dict[f"Channel{channel.index}"]["UniformDataSTD"]
    _, report = assign_uncertainty(dataset)

    for entry in report.distributions["per_channel"].values():
        assert entry["relative_fraction"] == 0.0
        assert entry["relative_fraction_source"] == "floor only"
    assert any("floor alone" in n for n in report.notes)


def test_a_negative_relative_fraction_is_refused():
    dataset, _ = set_noise_floor(delivery())
    with pytest.raises(ValueError, match="cannot be negative"):
        assign_uncertainty(dataset, relative_fraction=-0.03)


def test_the_delivered_uncertainties_are_stashed():
    dataset, _ = set_noise_floor(delivery())
    key = dataset.channels[0].std_key
    before = dataset.layer_data[key].copy()

    dataset, _ = assign_uncertainty(dataset)

    pd.testing.assert_frame_equal(dataset.layer_data[f"{key}_raw"], before)
    assert not dataset.layer_data[key].equals(before)


def test_it_changes_no_measurement_or_mask():
    dataset, _ = set_noise_floor(delivery())
    values = {c.data_key: dataset.layer_data[c.data_key].copy()
              for c in dataset.channels}
    masks = {c.inuse_key: dataset.layer_data[c.inuse_key].copy()
             for c in dataset.channels}

    dataset, _ = assign_uncertainty(dataset)

    for key, array in {**values, **masks}.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


# ── filter_29 ────────────────────────────────────────────────────────────────

def test_culling_uses_the_same_model():
    dataset, _ = set_noise_floor(delivery())
    dataset, report = cull_below_noise_floor(dataset)

    assert report.distributions["model"]["level_1ms"] == pytest.approx(
        recorded(dataset).level_1ms)
    assert report.distributions["removed"] > 0


def test_it_carries_no_floor_of_its_own_either():
    with pytest.raises(ValueError, match="no noise"):
        cull_below_noise_floor(delivery())


def test_the_two_consumers_read_one_definition():
    """The assertion R9 exists for: what filter_12 assigned uncertainty from and
    what filter_29 culled against are the same numbers, not two that agree."""
    dataset, _ = set_noise_floor(delivery())
    dataset, assigned = assign_uncertainty(dataset)
    dataset, culled = cull_below_noise_floor(dataset)

    assert assigned.distributions["model"] == culled.distributions["model"]


def test_margin_demands_more_than_the_floor():
    _, at_floor = cull_below_noise_floor(
        set_noise_floor(delivery())[0], margin=1.0)
    _, generous = cull_below_noise_floor(
        set_noise_floor(delivery())[0], margin=10.0)

    assert (generous.distributions["removed"]
            > at_floor.distributions["removed"])


def test_a_non_positive_margin_is_refused():
    dataset, _ = set_noise_floor(delivery())
    with pytest.raises(ValueError, match="must be positive"):
        cull_below_noise_floor(dataset, margin=0)


def test_a_strong_transient_keeps_its_early_gates():
    dataset, _ = set_noise_floor(delivery(signal=1e-4))
    dataset, _ = cull_below_noise_floor(dataset)

    assert (mask(dataset)[:, 0] == IN_USE).all()


def test_the_policy_applies_here_too():
    dataset, _ = set_noise_floor(delivery())
    dataset, report = cull_below_noise_floor(dataset, disable_tail=True)

    assert report.distributions["policy"]["disable_tail"] is True
    assert report.distributions["removed"] > 0


def test_it_changes_no_measurement_or_uncertainty():
    dataset, _ = set_noise_floor(delivery())
    arrays = {k: v.copy() for k, v in dataset.layer_data.items()
              if "inuse" not in k.lower()}

    dataset, _ = cull_below_noise_floor(dataset)

    for key, array in arrays.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


# ── Plumbing ─────────────────────────────────────────────────────────────────

def test_the_reports_survive_serialization():
    dataset, estimate = set_noise_floor(delivery())
    dataset, assigned = assign_uncertainty(dataset)
    dataset, culled = cull_below_noise_floor(dataset)

    for report in (estimate, assigned, culled):
        restored = json.loads(json.dumps(report.to_dict(), default=str))
        assert restored["distributions"]["model"]["exponent"] is not None


@pytest.mark.parametrize("name", ["set_noise_floor", "assign_uncertainty",
                                  "cull_below_noise_floor"])
def test_registered(name):
    assert name in load_steps(FILTER_GROUP)
