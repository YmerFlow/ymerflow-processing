"""filter_14-19 — copy, rename and remove, for attributes and arrays.

They exist to work around import problems, not to do geophysics, which is why
the interesting assertions here are about what they say rather than what they
move: an overwrite reported instead of printed, and a rename that quietly takes
a channel out of the model announcing itself.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.manage import (
    copy_array, copy_attribute, remove_array, remove_attribute, rename_array,
    rename_attribute,
)
from ymerflow_processing.core.pipeline import FILTER_GROUP, load_steps

N = 4
GATES = 8


def delivery(**kw):
    return make_dataset(n=N, n_lines=1, n_gates=GATES, **kw)


# ── Attributes ───────────────────────────────────────────────────────────────

def test_copy_leaves_the_original():
    dataset, _ = copy_attribute(delivery(), "tx_alt", "height_backup")
    frame = dataset.flightlines

    assert "tx_alt" in frame.columns
    pd.testing.assert_series_equal(frame["height_backup"], frame["tx_alt"],
                                   check_names=False)


def test_rename_does_not():
    dataset, _ = rename_attribute(delivery(), "tx_alt", "altitude")
    frame = dataset.flightlines

    assert "altitude" in frame.columns
    assert "tx_alt" not in frame.columns


def test_remove_takes_it_away():
    dataset, _ = remove_attribute(delivery(), "topo")
    assert "topo" not in dataset.flightlines.columns


def test_an_absent_attribute_is_an_error_naming_what_is_present():
    with pytest.raises(KeyError, match="tx_alt"):
        rename_attribute(delivery(), "elevation", "height")


# ── Arrays ───────────────────────────────────────────────────────────────────

def test_copying_an_array_leaves_the_original():
    dataset, report = copy_array(delivery(), "dbdt_ch1gt", "dbdt_raw_ch1gt")

    pd.testing.assert_frame_equal(dataset.layer_data["dbdt_raw_ch1gt"],
                                  dataset.layer_data["dbdt_ch1gt"])
    assert report.distributions["shape"] == (N, GATES)


def test_a_copied_array_is_independent_of_its_source():
    """A view would make a later cull on one silently change the other."""
    dataset, _ = copy_array(delivery(), "dbdt_ch1gt", "dbdt_raw_ch1gt")
    dataset.layer_data["dbdt_ch1gt"].iloc[0, 0] = -99.0

    assert dataset.layer_data["dbdt_raw_ch1gt"].iloc[0, 0] != -99.0


def test_renaming_an_array_moves_it():
    dataset, _ = rename_array(delivery(), "dbdt_std_ch1gt", "uncertainty_lm")

    assert "uncertainty_lm" in dataset.layer_data
    assert "dbdt_std_ch1gt" not in dataset.layer_data


def test_removing_an_array_takes_it_away():
    dataset, _ = remove_array(delivery(), "dbdt_std_ch2gt")
    assert "dbdt_std_ch2gt" not in dataset.layer_data


def test_an_absent_array_is_an_error():
    with pytest.raises(KeyError, match="No array named"):
        remove_array(delivery(), "dbdt_ch9gt")


# ── What they say ────────────────────────────────────────────────────────────

def test_an_overwrite_is_reported_rather_than_printed():
    """01 notes these currently warn; under R3 that belongs in the record."""
    dataset = delivery()
    dataset.flightlines["spare"] = 1.0

    _, report = copy_attribute(dataset, "tx_alt", "spare")
    assert any("Overwrote existing attribute 'spare'" in n for n in report.notes)


def test_an_array_overwrite_is_reported_too():
    _, report = copy_array(delivery(), "dbdt_ch1gt", "dbdt_ch2gt")
    assert any("Overwrote existing array" in n for n in report.notes)


def test_renaming_a_data_array_out_of_the_model_says_so():
    """Channel discovery matches on layer_data keys, so renaming one to
    something that matches no pattern makes the channel vanish. Not an error,
    and not preventable without knowing the intent — but sayable."""
    dataset = delivery()
    assert len(dataset.channels) == 2

    dataset, report = rename_array(dataset, "dbdt_ch2gt", "spare_array")

    assert len(dataset.channels) == 1
    assert any("no longer discovered" in n for n in report.notes)
    assert any("ch2" in n for n in report.notes)


def test_removing_a_data_array_says_so_as_well():
    dataset, report = remove_array(delivery(), "dbdt_ch1gt")

    assert {c.key for c in dataset.channels} == {"ch2"}
    assert any("ch1" in n and "no longer discovered" in n for n in report.notes)


def test_renaming_within_the_convention_keeps_the_channel():
    """The check is on discovery, not on the name — a rename that still matches
    is not a loss and must not be reported as one."""
    dataset, report = rename_array(delivery(), "dbdt_std_ch1gt",
                                   "dbdt_std_ch1gt_v2")

    assert {c.key for c in dataset.channels} == {"ch1", "ch2"}
    assert not any("no longer discovered" in n for n in report.notes)


def test_a_removal_that_touches_no_channel_is_quiet():
    dataset = delivery()
    dataset.layer_data["scratch"] = pd.DataFrame(np.ones((N, GATES)))

    _, report = remove_array(dataset, "scratch")
    assert not any("no longer discovered" in n for n in report.notes)


# ── Plumbing ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", [
    "copy_attribute", "copy_array", "rename_attribute", "rename_array",
    "remove_attribute", "remove_array",
])
def test_registered(name):
    """Six entry points, not one filter with a six-valued mode — which is the
    thing R7 exists to prevent."""
    assert name in load_steps(FILTER_GROUP)
