"""Reducing a dataset to the channels an inversion will accept.

The case that motivates it: SimPEG takes vertical-component data and fails on X
and Y being present. Ignoring them is not enough — they have to leave the
dataset, because the inversion reads the file, not our discovery.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.pipeline import FILTER_GROUP, Pipeline, ProcessingRunner, load_steps
from ymerflow_processing.core.select_channels import select_channels


def three_component(moments=("LM", "HM")):
    """A 3C instrument: one channel per (moment, component) pair."""
    names, components = [], []
    for moment in moments:
        for component in ("X", "Y", "Z"):
            names.append(moment)
            components.append(component)
    return make_dataset(moments=tuple(names), components=components)


# ── Discovery ────────────────────────────────────────────────────────────────

def test_a_three_component_instrument_is_one_channel_per_component():
    """R1: component and moment are different axes. 2 moments x 3 components = 6."""
    dataset = three_component()
    assert dataset.n_channels == 6
    assert [c.component for c in dataset.channels] == ["x", "y", "z"] * 2
    assert [c.name for c in dataset.channels] == ["LM"] * 3 + ["HM"] * 3


def test_a_single_component_instrument_declares_z_throughout(dual_moment):
    assert [c.component for c in dual_moment.channels] == ["z", "z"]


def test_component_is_absent_without_a_system_description():
    dataset = make_dataset(with_gex=False)
    assert all(c.component is None for c in dataset.channels)


# ── Selecting by component ───────────────────────────────────────────────────

def test_keeping_z_removes_the_other_components():
    dataset = three_component()
    dataset, report = select_channels(dataset, criterion="component", keep=["z"])

    assert [str(c) for c in dataset.channels] == ["ch3 (LM/z)", "ch6 (HM/z)"]
    assert report.points_total == 6
    assert report.points_affected == 4


def test_the_arrays_really_leave_layer_data():
    """The point of the step: ignoring X and Y would not help, they must go."""
    dataset = three_component()
    before = set(dataset.layer_data)

    dataset, report = select_channels(dataset, criterion="component", keep=["z"])

    remaining = set(dataset.layer_data)
    assert len(before) == 18            # 6 channels x data/std/inuse
    assert len(remaining) == 6          # 2 channels x data/std/inuse
    assert set(report.distributions["arrays_removed"]) == before - remaining


def test_kept_channels_are_untouched():
    import pandas as pd
    dataset = three_component()
    keep_key = dataset.channels[2].data_key          # LM/z
    before = dataset.layer_data[keep_key].copy()

    dataset, _ = select_channels(dataset, criterion="component", keep=["z"])

    pd.testing.assert_frame_equal(dataset.layer_data[keep_key], before)


def test_matching_is_case_insensitive():
    for wanted in ("z", "Z"):
        dataset, _ = select_channels(three_component(), criterion="component",
                                     keep=[wanted])
        assert dataset.n_channels == 2


def test_a_bare_string_is_accepted_as_well_as_a_list():
    dataset, _ = select_channels(three_component(), criterion="component", keep="z")
    assert dataset.n_channels == 2


# ── Selecting by moment ──────────────────────────────────────────────────────

def test_keeping_one_moment_keeps_all_its_components():
    """Different axis: moment selection must not care about component."""
    dataset, _ = select_channels(three_component(), criterion="moment", keep=["HM"])
    assert [c.component for c in dataset.channels] == ["x", "y", "z"]
    assert all(c.name == "HM" for c in dataset.channels)


def test_the_two_criteria_compose():
    """Chaining narrows on both axes — the SimPEG case, then a HM inversion."""
    dataset = three_component()
    dataset, _ = select_channels(dataset, criterion="component", keep=["z"])
    dataset, _ = select_channels(dataset, criterion="moment", keep=["HM"])

    assert [str(c) for c in dataset.channels] == ["ch6 (HM/z)"]
    assert len(dataset.layer_data) == 3


def test_selection_by_key_always_works():
    """The fallback when the system description names nothing."""
    dataset = make_dataset(with_gex=False)
    dataset, _ = select_channels(dataset, criterion="key", keep=["ch2"])
    assert [c.key for c in dataset.channels] == ["ch2"]


# ── The system description ───────────────────────────────────────────────────

def test_the_system_description_follows_the_data():
    """Otherwise it goes on declaring channels whose arrays have gone."""
    dataset = three_component()
    assert dataset.system.number_channels == 6

    dataset, _ = select_channels(dataset, criterion="component", keep=["z"])
    assert dataset.system.number_channels == 2


def test_the_system_description_can_be_left_alone():
    dataset = three_component()
    dataset, _ = select_channels(dataset, criterion="component", keep=["z"],
                                 drop_system_blocks=False)
    assert dataset.system.number_channels == 6
    assert len(dataset.layer_data) == 6


# ── Refusing to do the wrong thing ───────────────────────────────────────────

def test_keeping_nothing_is_an_error_naming_what_was_there():
    """Almost always a typo, and an empty dataset fails much later and worse."""
    with pytest.raises(ValueError, match="Present: x, y, z"):
        select_channels(three_component(), criterion="component", keep=["w"])


def test_an_unknown_criterion_is_rejected():
    with pytest.raises(ValueError, match="Unknown criterion"):
        select_channels(three_component(), criterion="polarity", keep=["z"])


def test_keeping_everything_changes_nothing_and_says_so(dual_moment):
    before = set(dual_moment.layer_data)
    dataset, report = select_channels(dual_moment, criterion="component", keep=["z"])

    assert set(dataset.layer_data) == before
    assert any("no channel was removed" in n for n in report.notes)


def test_a_dataset_with_no_channels_is_a_note_not_an_error():
    dataset = make_dataset()
    dataset.survey.xyz.model_dict["layer_data"] = {}
    dataset, report = select_channels(dataset, criterion="component", keep=["z"])
    assert any("no channels" in n.lower() for n in report.notes)


# ── Registration ─────────────────────────────────────────────────────────────

def test_registered_as_a_processing_filter():
    """dataset -> dataset, so it belongs to the filters group, not qc."""
    assert "select_channels" in load_steps(FILTER_GROUP)


def test_runs_from_a_yaml_pipeline():
    pipeline = Pipeline.parse("""
        kind: filters
        steps:
          - select_channels: {criterion: component, keep: [z]}
          - select_channels: {criterion: moment, keep: [HM]}
    """)
    dataset, report = ProcessingRunner(pipeline).run(three_component())

    assert [str(c) for c in dataset.channels] == ["ch6 (HM/z)"]
    assert len(report.steps) == 3      # two steps, plus the inventory


# ── Arrays outside the channel model ─────────────────────────────────────────

def _with_stowaway(**kw):
    """A delivery carrying a gate-shaped array no channel claims — SkyTEM's
    horizontal component, omitted from both the GEX and the ALC by design."""
    dataset = make_dataset(n=6, n_lines=1, n_gates=8, **kw)
    dataset.layer_data["dbdt_x_ch2gt"] = pd.DataFrame(np.full((6, 8), -3e-10))
    return dataset


def test_an_unclaimed_array_is_reported_rather_than_passed_over():
    """The bug this closes: the step said "nothing removed" while a non-Z array
    sat in layer_data. True of channels, false of the file."""
    dataset, report = select_channels(_with_stowaway(), criterion="component",
                                      keep=["z"])

    assert report.distributions["unclaimed_arrays"] == {"dbdt_x_ch2gt": (6, 8)}
    assert any("claimed by no channel" in n for n in report.notes)


def test_it_no_longer_claims_nothing_was_removed():
    """The old note was the misleading part — it described channels while
    reading as a statement about the dataset."""
    _, report = select_channels(_with_stowaway(), criterion="component",
                                keep=["z"])
    assert not any("nothing removed" in n for n in report.notes)
    assert any("no channel was removed" in n for n in report.notes)


def test_an_unclaimed_array_survives_by_default():
    """It is undeclared rather than empty, and at least one such array is real
    coherent signal the incumbent toolchain merely discards. Deleting it unasked
    would be worse than leaving it."""
    dataset, _ = select_channels(_with_stowaway(), criterion="component",
                                 keep=["z"])
    assert "dbdt_x_ch2gt" in dataset.layer_data


def test_it_can_be_dropped_on_request():
    dataset, report = select_channels(_with_stowaway(), criterion="component",
                                      keep=["z"], drop_unclaimed=True)

    assert "dbdt_x_ch2gt" not in dataset.layer_data
    assert "dbdt_x_ch2gt" in report.distributions["arrays_removed"]
    assert any("as asked" in n for n in report.notes)


def test_dropping_unclaimed_leaves_the_kept_channels_alone():
    dataset, _ = select_channels(_with_stowaway(), criterion="component",
                                 keep=["z"], drop_unclaimed=True)

    assert "dbdt_ch1gt" in dataset.layer_data
    assert "dbdt_ch2gt" in dataset.layer_data
    assert len(dataset.channels) == 2


def test_an_unclaimed_array_is_reported_even_when_channels_are_removed():
    """The two are independent: dropping a moment says nothing about an array
    that was never in the model."""
    dataset, report = select_channels(_with_stowaway(), criterion="moment",
                                      keep=["LM"])

    assert "dbdt_x_ch2gt" in report.distributions["unclaimed_arrays"]
    assert "dbdt_x_ch2gt" in dataset.layer_data          # not dropped by default
    assert "dbdt_ch2gt" not in dataset.layer_data        # the HM channel did go


def test_a_clean_delivery_reports_no_unclaimed_arrays():
    dataset = make_dataset(n=6, n_lines=1, n_gates=8)
    _, report = select_channels(dataset, criterion="component", keep=["z"])

    assert "unclaimed_arrays" not in report.distributions
    assert not any("claimed by no channel" in n for n in report.notes)
