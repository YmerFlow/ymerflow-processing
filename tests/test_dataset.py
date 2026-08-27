"""Channel discovery — R1.

The claim under test is that the number of channels is *data, not structure*: a
single-moment system and a three-moment system both work with no special casing,
nothing assumes a count, and no default names a specific moment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from libaarhusxyz import Survey
from libaarhusxyz.gex import GEX
from libaarhusxyz.xyz import XYZ

from conftest import make_dataset, make_flightlines
from ymerflow_processing.core.dataset import (
    Axis, Channel, ChannelNaming, Dataset, discover_by_naming,
)
from ymerflow_processing.em.tem.method import TEM_NAMING, Moment, TEMMethod


# ── One to N moments ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("moments", [
    ("LM",),                        # HeliTEM, VTEM, Xcite
    ("LM", "HM"),                   # SkyTEM 304/306/312
    ("LM", "MM", "HM"),             # three-moment: no instrument to hand
    ("A", "B", "C", "D"),           # four, to prove nothing caps at three
])
def test_discovers_however_many_moments_exist(moments):
    dataset = make_dataset(moments=moments)
    assert dataset.n_channels == len(moments)
    assert [c.key for c in dataset.channels] == [f"ch{i}" for i in range(1, len(moments) + 1)]
    assert [c.name for c in dataset.channels] == list(moments)


def test_single_and_three_moment_take_the_same_path(single_moment, three_moment):
    """The point of R1: one code path, two answers."""
    for dataset in (single_moment, three_moment):
        assert dataset.method == "tem"
        assert all(isinstance(c, Moment) for c in dataset.channels)
        assert all(c.n_gates == 8 for c in dataset.channels)


def test_moments_are_ordered_by_channel_index():
    """Order must not depend on dict iteration order of layer_data."""
    dataset = make_dataset(moments=("LM", "MM", "HM"))
    # Rebuild layer_data in a deliberately scrambled order.
    scrambled = dict(reversed(list(dataset.layer_data.items())))
    dataset.survey.xyz.model_dict["layer_data"] = scrambled
    assert [c.index for c in dataset.channels] == [1, 2, 3]


@pytest.mark.parametrize("naming", ["dbdt", "gate"])
def test_both_naming_conventions_discover_identically(naming):
    dataset = make_dataset(moments=("LM", "HM"), naming=naming)
    assert [c.key for c in dataset.channels] == ["ch1", "ch2"]
    assert all(c.std_key and c.inuse_key for c in dataset.channels)


# ── Discovery mechanism ──────────────────────────────────────────────────────

def test_index_is_captured_not_substring_matched():
    """``dbdt_ch1`` must not swallow ``dbdt_ch10``.

    Substring matching — what libaarhusxyz's own helpers do — folds channel 10
    into channel 1. Nothing has ten channels today, which is exactly why the
    failure would be invisible when something does.
    """
    keys = ["dbdt_ch1gt", "dbdt_ch10gt", "dbdt_std_ch1gt", "dbdt_std_ch10gt"]
    found = discover_by_naming(keys, TEM_NAMING)
    assert sorted(found) == [1, 10]
    assert found[1]["data_key"] == "dbdt_ch1gt"
    assert found[10]["data_key"] == "dbdt_ch10gt"


def test_roles_are_not_confused_with_each_other():
    """A std or in-use array must never be mistaken for the data array."""
    found = discover_by_naming(
        ["dbdt_ch1gt", "dbdt_std_ch1gt", "dbdt_inuse_ch1gt"], TEM_NAMING)
    assert found[1] == {
        "data_key": "dbdt_ch1gt",
        "std_key": "dbdt_std_ch1gt",
        "inuse_key": "dbdt_inuse_ch1gt",
    }


def test_uncertainty_without_data_is_not_a_channel(caplog):
    """Half a channel is a broken import, not a measurement."""
    found = discover_by_naming(["dbdt_std_ch2gt"], TEM_NAMING)
    assert found == {}


def test_data_without_uncertainty_is_still_a_channel():
    """Uncertainties may legitimately not be assigned yet."""
    found = discover_by_naming(["dbdt_ch1gt"], TEM_NAMING)
    assert found[1]["data_key"] == "dbdt_ch1gt"
    assert "std_key" not in found[1]


def test_case_is_not_significant():
    found = discover_by_naming(["GATE_CH01", "std_ch01"], TEM_NAMING)
    assert found[1]["data_key"] == "GATE_CH01"
    assert found[1]["std_key"] == "std_ch01"


# ── Data that is not this method's ───────────────────────────────────────────

def test_a_dataset_with_no_arrays_has_no_channels():
    """A scalar magnetometer has no array families. That is correct, not an error."""
    xyz = XYZ({"flightlines": make_flightlines(), "layer_data": {}})
    dataset = Dataset(Survey(xyz, GEX({})))
    dataset._adapters = [TEMMethod()]

    assert dataset.channels == []
    assert dataset.n_channels == 0
    assert dataset.method is None
    assert "0 channel(s)" in repr(dataset)


def test_unrecognized_array_names_are_not_claimed():
    xyz = XYZ({
        "flightlines": make_flightlines(),
        "layer_data": {"total_field": pd.DataFrame(np.zeros((40, 1)))},
    })
    dataset = Dataset(Survey(xyz, GEX({})))
    dataset._adapters = [TEMMethod()]
    assert dataset.method is None


def test_moments_refuses_on_non_time_domain_data():
    """``moments`` is TEM's word. It must not silently return something else."""
    class Frequencies:
        name = "fem"

        def channels(self, dataset):
            return [Channel(key="f1", index=1, data_key="inphase_f1")]

    dataset = make_dataset()
    dataset._adapters = [Frequencies()]

    assert dataset.n_channels == 1
    with pytest.raises(TypeError, match="channels, not moments"):
        _ = dataset.moments


# ── Per-channel settings ─────────────────────────────────────────────────────

def test_a_scalar_reaches_every_channel(three_moment):
    """R1: a scalar per-moment setting means 'apply to all'."""
    assert three_moment.resolve_per_channel(15) == {"ch1": 15, "ch2": 15, "ch3": 15}


def test_a_mapping_may_be_addressed_by_key_or_by_name(dual_moment):
    by_name = dual_moment.resolve_per_channel({"LM": 10, "HM": 20})
    by_key = dual_moment.resolve_per_channel({"ch1": 10, "ch2": 20})
    assert by_name == by_key == {"ch1": 10, "ch2": 20}


def test_a_mapping_may_be_partial(dual_moment):
    """Naming one moment must not require naming them all."""
    assert dual_moment.resolve_per_channel({"HM": 20}) == {"ch2": 20}


def test_an_unknown_channel_is_an_error_not_a_silent_drop(dual_moment):
    with pytest.raises(KeyError, match="No channel"):
        dual_moment.resolve_per_channel({"XM": 1})


def test_channel_lookup_by_key_and_name(dual_moment):
    assert dual_moment.channel("LM").key == "ch1"
    assert dual_moment.channel("ch2").name == "HM"


def test_unnamed_moments_remain_addressable_by_key():
    """A system description may leave TransmitterMoment blank; libaarhusxyz
    tolerates that, so identity cannot depend on the name existing."""
    dataset = make_dataset(moments=("LM", "HM"))
    for i in (1, 2):
        dataset.system.gex_dict[f"Channel{i}"]["TransmitterMoment"] = ""

    assert [c.name for c in dataset.channels] == [None, None]
    assert dataset.resolve_per_channel({"ch1": 5}) == {"ch1": 5}


def test_moments_alias_matches_channels(dual_moment):
    assert dual_moment.n_moments == dual_moment.n_channels == 2
    assert dual_moment.resolve_per_moment(3) == dual_moment.resolve_per_channel(3)


# ── The gate-time axis ───────────────────────────────────────────────────────

def test_axis_carries_gate_centers_not_open_or_close(dual_moment):
    moment = dual_moment.channels[0]
    gates = np.asarray(dual_moment.system.gate_times(1), dtype=float)
    np.testing.assert_allclose(moment.gate_times, gates[:, 0])
    # And the center really is between open and close.
    assert np.all(moment.gate_times > gates[:, 1])
    assert np.all(moment.gate_times < gates[:, 2])


def test_axis_is_described_generically(dual_moment):
    """core must be able to read the axis without knowing it is time."""
    axis = dual_moment.channels[0].axis
    assert isinstance(axis, Axis)
    assert (axis.name, axis.unit) == ("gate time", "s")
    assert axis.size == 8


def test_discovery_survives_a_missing_system_description():
    """Channel identity comes from the arrays; the axis is enrichment."""
    dataset = make_dataset(with_gex=False)
    assert [c.key for c in dataset.channels] == ["ch1", "ch2"]
    assert all(c.gate_times is None for c in dataset.channels)
    assert all(c.dipole_moment is None for c in dataset.channels)


def test_gate_count_disagreement_is_reported(caplog):
    """Silent axis misalignment is the expensive failure; warn loudly."""
    dataset = make_dataset(n_gates=8)
    dataset.system.gex_dict["Channel1"]["NoGates"] = 5.0
    with caplog.at_level("WARNING"):
        _ = dataset.channels
    assert any("gates" in r.message for r in caplog.records)


# ── Copying ──────────────────────────────────────────────────────────────────

def test_copy_is_deep_and_independent(dual_moment):
    duplicate = dual_moment.copy()
    duplicate.flightlines.loc[0, "tx_alt"] = 999.0
    assert dual_moment.flightlines.loc[0, "tx_alt"] != 999.0


def test_copy_preserves_the_subclass():
    """type(self), not Dataset: a subclass must not be silently downgraded."""
    class Derived(Dataset):
        pass

    dataset = make_dataset()
    derived = Derived(dataset.survey)
    assert isinstance(derived.copy(), Derived)


# ── The processing history ───────────────────────────────────────────────────
# It lives in model_info rather than on the object, so it survives a round trip
# through a file. Not tidiness: version_diff reads history to say which
# parameter moved, parameter_sweep discovers the swept parameter from it, and
# both read datasets off disk.

def test_history_survives_a_round_trip(tmp_path):
    """The bug this fixes was silent and confident: two files processed
    differently, and a version diff reporting no pipeline difference."""
    from ymerflow_processing.core.dataset import load

    dataset = make_dataset(n=4, n_lines=1, n_gates=4)
    dataset.record_step("cull_negatives", {"start_gate": 3})

    dataset.survey.dump(xyzfile=str(tmp_path / "a.xyz"),
                        gexfile=str(tmp_path / "a.gex"))
    back = load(tmp_path / "a.xyz", tmp_path / "a.gex")

    assert back.history == [{"step": "cull_negatives",
                             "params": {"start_gate": 3}}]


def test_a_processed_delivery_says_how_it_was_processed(tmp_path):
    """Worth having on its own — the same argument mask makes for its
    provenance legend: whoever opens the file will not have the run report."""
    from ymerflow_processing.core.pipeline import Pipeline, ProcessingRunner

    dataset, _ = ProcessingRunner(Pipeline.parse("""
        kind: filters
        steps:
          - disable_initial_gates: {}
    """)).run(make_dataset(n=6, n_lines=1, n_gates=6))

    dataset.survey.dump(xyzfile=str(tmp_path / "b.xyz"),
                        gexfile=str(tmp_path / "b.gex"))
    text = (tmp_path / "b.xyz").read_text()
    assert "disable_initial_gates" in text


def test_a_copy_carries_the_history():
    dataset = make_dataset(n=4, n_lines=1, n_gates=4)
    dataset.record_step("cull_negatives", {})
    assert dataset.copy().history == dataset.history


def test_a_copy_s_history_is_its_own():
    """A step returning a new dataset must not write into the old one's."""
    dataset = make_dataset(n=4, n_lines=1, n_gates=4)
    dataset.record_step("one", {})

    other = dataset.copy()
    other.record_step("two", {})
    assert [e["step"] for e in dataset.history] == ["one"]
    assert [e["step"] for e in other.history] == ["one", "two"]


def test_unreadable_history_is_a_warning_not_a_crash(caplog):
    """model_info is a text field in a file a person can edit."""
    dataset = make_dataset(n=4, n_lines=1, n_gates=4)
    dataset.survey.xyz.model_info["processing_history"] = "{not json"

    assert dataset.history == []
    assert "not readable as processing history" in caplog.text


# ── The coordinate reference system ──────────────────────────────────────────
# Not invented here: the platform's import process takes the projection as a
# required parameter and writes it to model_info["projection"] as an integer
# EPSG code, and every process downstream reads it from there. Using the same
# key is what lets a dataset carry its projection between the two.

def test_the_platform_s_projection_is_read_as_the_crs():
    dataset = make_dataset(n=4, n_lines=1, n_gates=4)
    dataset.survey.xyz.model_info["projection"] = 32610

    assert dataset.crs == "EPSG:32610"


def test_setting_a_crs_writes_the_integer_the_platform_expects():
    dataset = make_dataset(n=4, n_lines=1, n_gates=4)
    dataset.crs = "EPSG:32633"

    assert dataset.survey.xyz.model_info["projection"] == 32633


def test_a_bare_epsg_number_is_accepted_either_way():
    dataset = make_dataset(n=4, n_lines=1, n_gates=4)
    dataset.crs = 32633
    assert dataset.crs == "EPSG:32633"


def test_something_that_is_not_an_epsg_code_survives_unchanged():
    """A WKT or proj string is not ours to reformat."""
    dataset = make_dataset(n=4, n_lines=1, n_gates=4)
    dataset.crs = "+proj=utm +zone=33 +datum=WGS84"

    assert dataset.crs == "+proj=utm +zone=33 +datum=WGS84"


def test_a_delivery_with_no_projection_has_no_crs():
    """A raw delivery that never went through an import carries none, and
    guessing one would put a survey in the wrong hemisphere."""
    assert make_dataset(n=4, n_lines=1, n_gates=4).crs is None


def test_constructing_without_a_crs_does_not_clobber_the_carried_one():
    """Dataset(survey) is called in copy() and in load(); passing None through
    would erase a projection the delivery arrived with."""
    from ymerflow_processing.core.dataset import Dataset

    dataset = make_dataset(n=4, n_lines=1, n_gates=4)
    dataset.crs = "EPSG:32610"

    assert Dataset(dataset.survey).crs == "EPSG:32610"
    assert dataset.copy().crs == "EPSG:32610"


def test_the_crs_survives_a_round_trip(tmp_path):
    from ymerflow_processing.core.dataset import load

    dataset = make_dataset(n=4, n_lines=1, n_gates=4)
    dataset.crs = "EPSG:32610"
    dataset.survey.dump(xyzfile=str(tmp_path / "a.xyz"),
                        gexfile=str(tmp_path / "a.gex"))

    assert load(tmp_path / "a.xyz", tmp_path / "a.gex").crs == "EPSG:32610"


def test_a_float_shaped_code_from_a_round_trip_is_still_recognized():
    """libaarhusxyz writes model_info values as floats, so the integer comes
    back as '32610.0'. A string test would have recognised a projection before
    the file was saved and not after."""
    dataset = make_dataset(n=4, n_lines=1, n_gates=4)
    dataset.survey.xyz.model_info["projection"] = "32610.0"

    assert dataset.crs == "EPSG:32610"


def test_a_non_integer_number_is_not_an_epsg_code():
    dataset = make_dataset(n=4, n_lines=1, n_gates=4)
    dataset.survey.xyz.model_info["projection"] = "32610.5"

    assert dataset.crs == "32610.5"
