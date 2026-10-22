"""filter_06, filter_07 and filter_09 — geometry and the nominal current.

These are the first steps in the library that change what a measurement *says*
rather than which measurements count. Every one of them is silently wrong when
misapplied — a doubly tilt-corrected sounding is a plausible sounding of a
different earth. (filter_08, the amplitude convention, is gone: the data on disk
is fixed in the SimPEG format upstream, so there is nothing left to declare or
convert.)
"""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.dataset import stash_key
from ymerflow_processing.filters.attitude import (
    REGISTRY as TILT_REGISTRY, correct_tilt_1d, corrected, zero_attitude,
)

STASHED_PITCH = stash_key("tilt_x")
STASHED_ROLL = stash_key("tilt_y")
from ymerflow_processing.filters.normalize import apply_nominal_current
from ymerflow_processing.pipeline import FILTER_GROUP, load_steps

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
    dataset, _ = correct_tilt_1d(delivery(pitch=0.0, roll=0.0))
    assert np.allclose(data(dataset), 1e-9)


def test_tilt_scales_by_one_over_cos_squared():
    """Two cosines: the vertical component of the transmitted moment, and the
    projection of the vertical field onto the tilted receiver's normal."""
    dataset, _ = correct_tilt_1d(delivery(pitch=30.0, roll=0.0))
    expected = 1e-9 / np.cos(np.radians(30.0)) ** 2

    assert np.allclose(data(dataset), expected)


def test_pitch_and_roll_compose_exactly():
    """cos(total) = cos(pitch) * cos(roll), the same relation qc_02 uses — not a
    small-angle sum."""
    dataset, _ = correct_tilt_1d(delivery(pitch=20.0, roll=15.0))
    cosine = np.cos(np.radians(20.0)) * np.cos(np.radians(15.0))

    assert np.allclose(data(dataset), 1e-9 / cosine ** 2)


def test_tilt_either_way_corrects_the_same():
    """A loop leaning left couples like one leaning right."""
    left, _ = correct_tilt_1d(delivery(roll=-20.0))
    right, _ = correct_tilt_1d(delivery(roll=20.0))
    assert np.allclose(data(left), data(right))


def test_the_correction_always_increases_the_amplitude():
    """A tilted loop measures less than a level one would, so recovering the
    horizontal response can only scale up."""
    dataset, _ = correct_tilt_1d(delivery(pitch=10.0, roll=10.0))
    assert (data(dataset) > 1e-9).all()


def test_it_refuses_to_run_twice():
    """01 calls this out: applying it twice would apply the correction twice,
    and a doubly corrected sounding is a plausible sounding of a different
    earth."""
    dataset, _ = correct_tilt_1d(delivery(pitch=10.0))

    with pytest.raises(ValueError, match="already carries a tilt correction"):
        correct_tilt_1d(dataset)


def test_a_second_pass_can_be_asked_for_explicitly():
    dataset, _ = correct_tilt_1d(delivery(pitch=10.0))
    dataset, report = correct_tilt_1d(dataset, again=True)
    assert report.distributions["repeated"] is True


def test_the_application_is_recorded():
    dataset, _ = correct_tilt_1d(delivery(pitch=10.0))
    assert corrected(dataset) is not None
    assert TILT_REGISTRY in dataset.survey.xyz.model_info


def test_a_corrected_sounding_presents_zero_attitude_and_keeps_the_measurement():
    """SimPEG applies its own tilt term from the attitude columns at read, so a
    corrected dataset still showing its tilt is corrected twice at inversion.
    The measured attitude moves to orig_ (Ben, 2026-10-21)."""
    dataset, report = correct_tilt_1d(delivery(pitch=10.0, roll=5.0))
    frame = dataset.flightlines

    assert (frame["tilt_x"] == 0.0).all()
    assert (frame["tilt_y"] == 0.0).all()
    assert (frame[STASHED_PITCH] == 10.0).all()
    assert (frame[STASHED_ROLL] == 5.0).all()
    assert any("is now zero" in n for n in report.notes)


def test_a_corrected_dataset_is_a_no_op_for_a_downstream_tilt_term():
    """What the zeroing buys: the 1/cos² of the live attitude is exactly one."""
    dataset, _ = correct_tilt_1d(delivery(pitch=20.0, roll=15.0))
    frame = dataset.flightlines
    downstream = 1.0 / (np.cos(np.radians(frame["tilt_x"]))
                        * np.cos(np.radians(frame["tilt_y"]))) ** 2
    assert np.allclose(downstream, 1.0)


def test_an_extreme_tilt_is_left_uncorrected_and_reported():
    """Past the limit the neglected horizontal mode is no longer small, so
    scaling harder is not a better answer — culling is. Its attitude stays
    live too, so cull_by_altitude_tilt and the inversion still see it."""
    dataset, report = correct_tilt_1d(delivery(pitch=70.0), max_correction=4.0)

    assert np.allclose(data(dataset), 1e-9)
    assert (dataset.flightlines["tilt_x"] == 70.0).all()
    assert report.distributions["left_uncorrected"] == N
    assert any("cull_by_altitude_tilt" in n for n in report.notes)


def test_the_originals_are_stashed_under_the_orig_prefix():
    dataset, _ = correct_tilt_1d(delivery(pitch=10.0))
    key = stash_key(dataset.channels[0].data_key)
    assert key.startswith("orig_")
    assert np.allclose(dataset.layer_data[key].to_numpy(), 1e-9)


def test_the_stash_is_not_mistaken_for_a_second_data_array():
    """The suffix form `dbdt_ch1gt_untilted` matched the anchored data pattern
    `^dbdt_ch(\\d+)` and discovery found two data arrays for one channel; a
    prefix sits outside every anchored pattern."""
    from ymerflow_processing.filters.select_moments import unclaimed_arrays
    from ymerflow_processing.inventory import take

    before = delivery(pitch=10.0)
    keys = [c.data_key for c in before.channels]
    dataset, _ = correct_tilt_1d(before)

    assert [c.data_key for c in dataset.channels] == keys
    assert unclaimed_arrays(dataset) == {}
    assert take(dataset)["arrays"]["unresolved"] == {}
    assert stash_key(keys[0]) in take(dataset)["arrays"]["stashed"]


def test_a_delivery_with_no_attitude_is_skipped_with_a_note():
    """VTEM ships none. A shipped pipeline has to run on it, so the step says
    it did nothing rather than failing the run."""
    dataset = delivery()
    dataset.survey.xyz.flightlines = dataset.flightlines.drop(
        columns=["tilt_x", "tilt_y"])

    dataset, report = correct_tilt_1d(dataset)

    assert np.allclose(data(dataset), 1e-9)
    assert corrected(dataset) is None
    assert any("No pitch or roll column" in n for n in report.notes)


def test_one_missing_axis_is_also_a_skip():
    dataset = delivery()
    dataset.survey.xyz.flightlines = dataset.flightlines.drop(columns=["tilt_y"])
    dataset, report = correct_tilt_1d(dataset)
    assert np.allclose(data(dataset), 1e-9)
    assert any("No roll column" in n for n in report.notes)


def test_a_named_column_that_is_absent_is_still_an_error():
    """Naming a column that is not there is a mistake about this dataset, not
    a dataset without the input."""
    with pytest.raises(KeyError, match="roll column 'TxRoll'"):
        correct_tilt_1d(delivery(), roll_column="TxRoll")


def test_qc_attitude_grades_the_measured_attitude_after_a_correction():
    """A corrected dataset must not grade as perfectly flown: the check reads
    the orig_ columns the correction left behind."""
    from ymerflow_processing.qc.attitude import system_attitude

    dataset, _ = correct_tilt_1d(delivery(pitch=12.0, roll=0.0))
    dataset, report = system_attitude(dataset, good=5.0, marginal=10.0)

    assert report.soundings_affected == N          # 12° is out of spec
    assert any("grades what was flown" in n for n in report.notes)


# ── filter_07: zeroed attitude ───────────────────────────────────────────────

def test_zeroing_makes_a_later_correction_a_no_op():
    """The whole reason 07 exists: a no-op rather than a correction derived from
    attitude nobody trusts."""
    dataset, _ = zero_attitude(delivery(pitch=25.0, roll=25.0))
    dataset, _ = correct_tilt_1d(dataset)

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
    dataset, _ = correct_tilt_1d(dataset)
    assert np.allclose(data(dataset), 1e-9)


def test_the_stash_names_carry_the_orig_prefix():
    """One spelling for every step (Ben, 2026-10-21). `orig_tilt_x` is eleven
    characters, one past a shapefile field; the vector export reports the
    truncation and refuses a collision, which is where that rule lives."""
    assert STASHED_PITCH == "orig_tilt_x"
    assert STASHED_ROLL == "orig_tilt_y"


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

    assert (dataset.flightlines[stash_key("LMcurrent")] == 7.5).all()


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
    dataset, tilt = correct_tilt_1d(delivery(pitch=10.0))
    dataset, zeroed = zero_attitude(dataset)
    dataset, current = apply_nominal_current(dataset)

    for report in (tilt, zeroed, current):
        restored = json.loads(json.dumps(report.to_dict(), default=str))
        assert restored["step"]


@pytest.mark.parametrize("name", ["correct_tilt_1d", "zero_attitude",
                                  "apply_nominal_current"])
def test_registered(name):
    assert name in load_steps(FILTER_GROUP)
