"""filter_06 to filter_09 — geometry, conventions and constants.

These are the first steps in the library that change what a measurement *says*
rather than which measurements count. Every one of them is silently wrong when
misapplied — a doubly tilt-corrected sounding, or one converted from the wrong
convention, is a plausible sounding of a different earth.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.attitude import (
    REGISTRY as TILT_REGISTRY, STASHED_PITCH, STASHED_ROLL, correct_tilt,
    corrected, zero_attitude,
)
from ymerflow_processing.core.dataset import (
    RAW, RX_NORMALIZED, RX_TX_NORMALIZED, load,
)
from ymerflow_processing.core.normalize import (
    apply_nominal_current, convert_normalization, recorded_normalization,
)
from ymerflow_processing.core.pipeline import FILTER_GROUP, load_steps

N = 12
GATES = 6


def delivery(pitch=0.0, roll=0.0, value=1e-9, **kw):
    dataset = make_dataset(n=N, n_lines=1, n_gates=GATES,
                           pitch=np.full(N, pitch), roll=np.full(N, roll), **kw)
    for channel in dataset.channels:
        dataset.layer_data[channel.data_key].iloc[:, :] = value
    return dataset


def data(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].data_key].to_numpy()


# ── filter_06: tilt ──────────────────────────────────────────────────────────

def test_a_level_transmitter_is_unchanged():
    dataset, _ = correct_tilt(delivery(pitch=0.0, roll=0.0))
    assert np.allclose(data(dataset), 1e-9)


def test_tilt_scales_by_one_over_cos_squared():
    """Two cosines: the vertical component of the transmitted moment, and the
    projection of the vertical field onto the tilted receiver's normal."""
    dataset, _ = correct_tilt(delivery(pitch=30.0, roll=0.0))
    expected = 1e-9 / np.cos(np.radians(30.0)) ** 2

    assert np.allclose(data(dataset), expected)


def test_pitch_and_roll_compose_exactly():
    """cos(total) = cos(pitch) * cos(roll), the same relation qc_02 uses — not a
    small-angle sum."""
    dataset, _ = correct_tilt(delivery(pitch=20.0, roll=15.0))
    cosine = np.cos(np.radians(20.0)) * np.cos(np.radians(15.0))

    assert np.allclose(data(dataset), 1e-9 / cosine ** 2)


def test_tilt_either_way_corrects_the_same():
    """A loop leaning left couples like one leaning right."""
    left, _ = correct_tilt(delivery(roll=-20.0))
    right, _ = correct_tilt(delivery(roll=20.0))
    assert np.allclose(data(left), data(right))


def test_the_correction_always_increases_the_amplitude():
    """A tilted loop measures less than a level one would, so recovering the
    horizontal response can only scale up."""
    dataset, _ = correct_tilt(delivery(pitch=10.0, roll=10.0))
    assert (data(dataset) > 1e-9).all()


def test_it_refuses_to_run_twice():
    """01 calls this out: applying it twice would apply the correction twice,
    and a doubly corrected sounding is a plausible sounding of a different
    earth."""
    dataset, _ = correct_tilt(delivery(pitch=10.0))

    with pytest.raises(ValueError, match="already carries a tilt correction"):
        correct_tilt(dataset)


def test_a_second_pass_can_be_asked_for_explicitly():
    dataset, _ = correct_tilt(delivery(pitch=10.0))
    dataset, report = correct_tilt(dataset, again=True)
    assert report.distributions["repeated"] is True


def test_the_application_is_recorded():
    dataset, _ = correct_tilt(delivery(pitch=10.0))
    assert corrected(dataset) is not None
    assert TILT_REGISTRY in dataset.survey.xyz.model_info


def test_the_recorded_attitude_is_left_alone():
    """Clearing it would make a corrected dataset indistinguishable from one
    flown level, and qc_02 would report perfect attitude on salvaged data."""
    dataset, report = correct_tilt(delivery(pitch=10.0, roll=5.0))

    assert (dataset.flightlines["tilt_x"] == 10.0).all()
    assert (dataset.flightlines["tilt_y"] == 5.0).all()
    assert any("still tilted" in n for n in report.notes)


def test_an_extreme_tilt_is_left_uncorrected_and_reported():
    """Past the limit the neglected horizontal mode is no longer small, so
    scaling harder is not a better answer — culling is."""
    dataset, report = correct_tilt(delivery(pitch=70.0), max_correction=4.0)

    assert np.allclose(data(dataset), 1e-9)
    assert report.distributions["left_uncorrected"] == N
    assert any("cull_by_geometry" in n for n in report.notes)


def test_the_originals_are_stashed():
    dataset, _ = correct_tilt(delivery(pitch=10.0))
    key = f"{dataset.channels[0].data_key}_untilted"
    assert np.allclose(dataset.layer_data[key].to_numpy(), 1e-9)


def test_a_missing_attitude_column_is_an_error():
    dataset = delivery()
    dataset.survey.xyz.flightlines = dataset.flightlines.drop(columns=["tilt_y"])
    with pytest.raises(KeyError, match="roll column"):
        correct_tilt(dataset)


# ── filter_07: zeroed attitude ───────────────────────────────────────────────

def test_zeroing_makes_a_later_correction_a_no_op():
    """The whole reason 07 exists: a no-op rather than a correction derived from
    attitude nobody trusts."""
    dataset, _ = zero_attitude(delivery(pitch=25.0, roll=25.0))
    dataset, _ = correct_tilt(dataset)

    assert np.allclose(data(dataset), 1e-9)


def test_zeroing_keeps_the_originals():
    dataset, _ = zero_attitude(delivery(pitch=25.0, roll=-10.0))
    frame = dataset.flightlines

    assert (frame["tilt_x"] == 0.0).all()
    assert (frame[STASHED_PITCH] == 25.0).all()
    assert (frame[STASHED_ROLL] == -10.0).all()


def test_absent_attitude_columns_are_created():
    """01: "If the attitude fields are absent entirely, they are created as
    zero." So a delivery with no attitude can still run a pipeline that
    mentions tilt."""
    dataset = delivery()
    dataset.survey.xyz.flightlines = dataset.flightlines.drop(
        columns=["tilt_x", "tilt_y"])

    dataset, report = zero_attitude(dataset)

    assert (dataset.flightlines["tilt_x"] == 0.0).all()
    assert report.distributions["created"] == ["tilt_x", "tilt_y"]
    assert any("nothing to stash" in n for n in report.notes)


def test_a_delivery_with_no_attitude_can_still_be_tilt_corrected():
    dataset = delivery()
    dataset.survey.xyz.flightlines = dataset.flightlines.drop(
        columns=["tilt_x", "tilt_y"])

    dataset, _ = zero_attitude(dataset)
    dataset, _ = correct_tilt(dataset)
    assert np.allclose(data(dataset), 1e-9)


def test_the_stash_names_fit_a_shapefile_field():
    for name in (STASHED_PITCH, STASHED_ROLL):
        assert len(name) <= 10


# ── filter_08: normalization ─────────────────────────────────────────────────

def test_it_takes_no_claim_about_the_current_convention():
    """01 names that assertion as the defect: a wrong one is undetectable and
    wrong by orders of magnitude."""
    import inspect
    assert "from" not in inspect.signature(convert_normalization).parameters
    assert "current" not in inspect.signature(convert_normalization).parameters


def test_converting_to_moment_normalized_divides_by_the_dipole_moment():
    dataset = delivery()
    dataset.normalization = RX_NORMALIZED
    moment = dataset.channels[0].dipole_moment

    dataset, report = convert_normalization(dataset, to=RX_TX_NORMALIZED)

    assert np.allclose(data(dataset), 1e-9 / moment)
    assert report.distributions["from"] == RX_NORMALIZED
    assert report.distributions["to"] == RX_TX_NORMALIZED


def test_each_channel_uses_its_own_moment():
    """A dual-moment system differs between channels by a factor of tens."""
    dataset = delivery()
    dataset.normalization = RX_NORMALIZED
    moments = [c.dipole_moment for c in dataset.channels]
    assert moments[0] != moments[1]

    dataset, _ = convert_normalization(dataset, to=RX_TX_NORMALIZED)
    assert np.allclose(data(dataset, 0), 1e-9 / moments[0])
    assert np.allclose(data(dataset, 1), 1e-9 / moments[1])


def test_converting_back_recovers_the_values():
    """Each step is a division by a positive constant, so it is invertible."""
    dataset = delivery()
    dataset.normalization = RX_NORMALIZED
    dataset, _ = convert_normalization(dataset, to=RX_TX_NORMALIZED)
    dataset, _ = convert_normalization(dataset, to=RX_NORMALIZED)

    assert np.allclose(data(dataset), 1e-9)


def test_asking_for_the_convention_it_is_already_in_is_a_no_op():
    dataset = delivery()
    dataset.normalization = RX_TX_NORMALIZED
    dataset, report = convert_normalization(dataset, to=RX_TX_NORMALIZED)

    assert np.allclose(data(dataset), 1e-9)
    assert any("nothing was converted" in n for n in report.notes)


def test_the_dataset_declaration_is_updated():
    dataset = delivery()
    dataset.normalization = RX_NORMALIZED
    dataset, _ = convert_normalization(dataset, to=RX_TX_NORMALIZED)

    assert dataset.normalization == RX_TX_NORMALIZED
    assert recorded_normalization(dataset) == RX_TX_NORMALIZED


def test_the_convention_survives_a_round_trip(tmp_path):
    """The half a constructor argument cannot provide: an assertion made once at
    load time does not survive the file."""
    dataset = delivery()
    dataset.normalization = RX_NORMALIZED
    dataset, _ = convert_normalization(dataset, to=RX_TX_NORMALIZED)

    xyz, gex = tmp_path / "d.xyz", tmp_path / "s.gex"
    dataset.survey.dump(xyzfile=str(xyz), gexfile=str(gex))

    assert load(xyz, gex).normalization == RX_TX_NORMALIZED


def test_a_missing_constant_refuses_rather_than_converting_some():
    """Converting the rest would leave one dataset in two conventions, which is
    the failure this filter exists to prevent."""
    dataset = delivery()
    dataset.normalization = RX_NORMALIZED
    del dataset.survey.gex.gex_dict["Channel1"]["ApproxDipoleMoment"]
    dataset.survey.gex.gex_dict["Channel1"]["TxApproximateCurrent"] = 0.0
    dataset.survey.gex.gex_dict["General"]["TxLoopArea"] = 0.0

    with pytest.raises(ValueError, match="does not give the constant"):
        convert_normalization(dataset, to=RX_TX_NORMALIZED)


def test_an_unknown_convention_names_the_available_ones():
    with pytest.raises(ValueError, match="Unknown normalization"):
        convert_normalization(delivery(), to="per_amp")


# ── filter_09: nominal current ───────────────────────────────────────────────

def test_it_writes_the_declared_nominal():
    dataset = delivery()
    dataset.flightlines["LMcurrent"] = 7.5

    dataset, report = apply_nominal_current(dataset)

    assert (dataset.flightlines["LMcurrent"] == 9.0).all()
    assert report.distributions["per_channel"]["ch1 (LM/z)"]["nominal"] == 9.0


def test_the_measured_values_are_stashed():
    dataset = delivery()
    dataset.flightlines["LMcurrent"] = 7.5
    dataset, _ = apply_nominal_current(dataset)

    stashed = [c for c in dataset.flightlines.columns if c.endswith("_raw")]
    assert stashed
    assert (dataset.flightlines[stashed[0]] == 7.5).all()


def test_a_delivery_with_no_current_column_gets_one():
    """The case 01 names: measured current missing entirely."""
    dataset, report = apply_nominal_current(delivery())

    assert any("was created" in n for n in report.notes)
    column = report.distributions["per_channel"]["ch1 (LM/z)"]["column"]
    assert (dataset.flightlines[column] == 9.0).all()


def test_a_channel_with_no_declared_nominal_is_left_alone():
    dataset = delivery()
    dataset.flightlines["LMcurrent"] = 7.5
    del dataset.survey.gex.gex_dict["Channel1"]["TxApproximateCurrent"]

    dataset, report = apply_nominal_current(dataset)

    assert (dataset.flightlines["LMcurrent"] == 7.5).all()
    assert any("declares no nominal current" in n for n in report.notes)


def test_it_says_that_qc_04_is_what_decides():
    _, report = apply_nominal_current(delivery())
    assert any("qc_04" in n for n in report.notes)


# ── Plumbing ─────────────────────────────────────────────────────────────────

def test_the_reports_survive_serialization():
    dataset, tilt = correct_tilt(delivery(pitch=10.0))
    dataset, zeroed = zero_attitude(dataset)
    dataset, current = apply_nominal_current(dataset)

    for report in (tilt, zeroed, current):
        restored = json.loads(json.dumps(report.to_dict(), default=str))
        assert restored["step"]


@pytest.mark.parametrize("name", ["correct_tilt", "zero_attitude",
                                  "convert_normalization",
                                  "apply_nominal_current"])
def test_registered(name):
    assert name in load_steps(FILTER_GROUP)
