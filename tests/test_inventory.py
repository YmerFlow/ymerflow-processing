"""The implicit first step: what arrived, and what of it was resolved.

The distinction it exists to make is between data that is **absent** and data
that is **invisible**. Every step works through the channel model, so an array
the model cannot account for is untouched by everything — which looks exactly
like success until someone opens the file.
"""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset, named, step
from ymerflow_processing.core.inventory import inventory, take
from ymerflow_processing.core.pipeline import (
    Pipeline, ProcessingRunner, QCRunner,
)

QC = """
kind: qc
steps:
  - height_above_ground: {good_max: 45, marginal_max: 65}
"""
FILTERS = """
kind: filters
steps:
  - select_channels: {criterion: moment, keep: [LM]}
"""


def with_stowaway(**kw):
    """A delivery carrying a gate-shaped array no channel claims.

    SkyTEM ships a horizontal component omitted from both the GEX and the ALC
    by design, so it matches no naming pattern and has no system-description
    block.
    """
    dataset = make_dataset(n=6, n_lines=1, n_gates=8, **kw)
    dataset.layer_data["dbdt_x_ch2gt"] = pd.DataFrame(np.full((6, 8), -3e-10))
    return dataset


# ── What it accounts for ─────────────────────────────────────────────────────

def test_it_lists_what_arrived():
    account = take(make_dataset(n=6, n_lines=1, n_gates=8))

    assert account["soundings"] == 6
    assert account["method"] == "tem"
    assert "line_id" in account["attributes"] and "tx_alt" in account["attributes"]
    assert account["arrays"]["total"] == 6


def test_it_says_what_each_channel_resolved_to():
    account = take(make_dataset(moments=("LM", "HM"), n=6, n_lines=1, n_gates=8))
    first = account["channels"][0]

    assert first["moment"] == "LM"
    assert first["component"] == "z"
    assert first["gates"] == 8
    assert first["arrays"]["data"] == "dbdt_ch1gt"
    assert first["dipole_moment"] == pytest.approx(3078.0)


def test_it_reads_the_system_description_alongside():
    """So a mismatch between what is declared and what arrived is visible in one
    place rather than inferred from two."""
    account = take(make_dataset(n=6, n_lines=1, n_gates=8))
    block = account["system_description"]["Channel1"]

    assert block["gates"] == 8.0
    assert block["moment"] == "LM"
    assert block["component"] == "Z"
    assert "remove_initial_gates" in block


# ── The point of it ──────────────────────────────────────────────────────────

def test_an_array_no_channel_claims_is_reported():
    dataset, report = inventory(with_stowaway())
    unresolved = report.distributions["arrays"]["unresolved"]

    assert unresolved == {"dbdt_x_ch2gt": (6, 8)}
    assert any("claimed by no channel" in n for n in report.notes)


def test_the_note_says_what_will_happen_to_it():
    """Not merely that it exists — that nothing will touch it, which is the part
    that makes a Z-only request quietly untrue."""
    _, report = inventory(with_stowaway())
    note = next(n for n in report.notes if "claimed by no channel" in n)

    assert "travel through the pipeline untouched" in note
    assert "dbdt_x_ch2gt (6, 8)" in note


def test_undeclared_is_not_treated_as_unusable():
    """At least one such array is real coherent signal that the incumbent
    toolchain merely discards. The recovery path is to describe it."""
    _, report = inventory(with_stowaway())
    note = next(n for n in report.notes if "claimed by no channel" in n)

    assert "system description" in note


def test_arrays_of_another_shape_are_not_called_gate_data():
    """A model or a derived product is not a stowaway; only something shaped
    like gate data can be an unclaimed channel."""
    dataset = make_dataset(n=6, n_lines=1, n_gates=8)
    dataset.layer_data["resistivity"] = pd.DataFrame(np.ones((6, 25)))

    account = take(dataset)
    assert account["arrays"]["unresolved"] == {}
    assert account["arrays"]["other"] == {"resistivity": (6, 25)}


def test_a_clean_delivery_reports_nothing_unresolved():
    _, report = inventory(make_dataset(n=6, n_lines=1, n_gates=8))
    assert report.distributions["arrays"]["unresolved"] == {}
    assert not any("claimed by no channel" in n for n in report.notes)


def test_an_instrument_with_no_channels_says_so_rather_than_failing():
    """A scalar magnetometer correctly resolves none."""
    dataset = make_dataset(n=4, n_lines=1)
    for channel in list(dataset.channels):
        for key in channel.array_keys:
            dataset.layer_data.pop(key, None)

    _, report = inventory(dataset)
    assert report.distributions["channels"] == []
    assert any("No channels resolved" in n for n in report.notes)


# ── It reads and changes nothing ─────────────────────────────────────────────

def test_it_modifies_nothing():
    """Which is what makes it safe as an implicit step in a QC pipeline, where
    R6 promises the data comes out byte-identical."""
    dataset = with_stowaway()
    arrays = {k: v.copy() for k, v in dataset.layer_data.items()}
    flightlines = dataset.flightlines.copy()

    dataset, _ = inventory(dataset)

    assert set(dataset.layer_data) == set(arrays)
    for key, array in arrays.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    pd.testing.assert_frame_equal(dataset.flightlines, flightlines)


def test_it_creates_no_in_use_masks():
    """Materializing the mask is culling's job, not an accounting step's."""
    dataset = make_dataset(n=6, n_lines=1, n_gates=8)
    for channel in dataset.channels:
        del dataset.layer_data[channel.inuse_key]

    before = set(dataset.layer_data)
    dataset, _ = inventory(dataset)
    assert set(dataset.layer_data) == before


# ── Implicit in every run ────────────────────────────────────────────────────

def test_a_qc_run_opens_with_it():
    dataset = make_dataset(n=20, n_lines=2, height=np.full(20, 30.0))
    _, report = QCRunner(Pipeline.parse(QC)).run(dataset)

    assert named(report)[0] == "inventory"
    assert step(report, "inventory").params == {"implicit": True}


def test_a_filter_run_opens_with_it():
    _, report = ProcessingRunner(Pipeline.parse(FILTERS)).run(with_stowaway())

    assert named(report)[0] == "inventory"
    assert step(report, "inventory").distributions["arrays"]["unresolved"]


def test_the_stowaway_is_reported_before_anything_acts_on_the_data():
    """The whole reason it is implicit: the operator learns at the top of the
    run, not from an inversion behaving oddly three steps later."""
    _, report = ProcessingRunner(Pipeline.parse(FILTERS)).run(with_stowaway())

    order = named(report)
    assert order.index("inventory") < order.index("select_channels")


def test_it_can_be_switched_off():
    dataset = make_dataset(n=20, n_lines=2, height=np.full(20, 30.0))
    _, report = QCRunner(Pipeline.parse(QC), inventory=False).run(dataset)

    assert "inventory" not in named(report)


def test_the_account_survives_serialization():
    """It is the top of every report a client or a future reader opens."""
    _, report = inventory(with_stowaway())
    restored = json.loads(json.dumps(report.to_dict(), default=str))

    assert restored["step"] == "inventory"
    assert "dbdt_x_ch2gt" in restored["distributions"]["arrays"]["unresolved"]
