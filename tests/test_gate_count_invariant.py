"""Every step must leave gate times and gate data the same length.

A dual-moment inversion now asserts this before it will run:

    n_times = len(gex.gate_times(channel))
    n_gates = layer_data[key].values.shape[1]
    assert n_times == n_gates

That assertion is new, so nothing in this library has ever been checked against
it. Anything that changed a channel's gate count without updating the system
description would produce a dataset that looks fine here and refuses to invert
downstream — the failure appearing a long way from its cause.

Nothing should do that: QC writes columns on ``flightlines``, culling writes the
in-use mask, and ``select_channels`` removes whole channels rather than gates.
This pins that as a property rather than an assumption, so a future step that
breaks it fails here instead of in an inversion.

Testing the invariant rather than the consumer, deliberately: it needs no
inversion stack installed, and it is the same guarantee.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.pipeline import Pipeline, ProcessingRunner, QCRunner
from ymerflow_processing.core.select_channels import select_channels


def gate_count_mismatches(dataset) -> list[str]:
    """Channels whose gate times and gate data disagree in length.

    Mirrors the downstream assertion exactly, including reading the times
    through ``gex.gate_times`` so that the table-selection behavior is
    exercised rather than assumed.
    """
    problems = []
    for channel in dataset.channels:
        array = dataset.layer_data.get(channel.data_key)
        if array is None:
            continue
        try:
            n_times = len(dataset.system.gate_times(channel.index))
        except Exception as exc:                     # an unusable GEX
            problems.append(f"{channel}: gate times unreadable ({exc})")
            continue
        n_gates = np.asarray(array).shape[1]
        if n_times != n_gates:
            problems.append(
                f"{channel}: {n_gates} gates of data, {n_times} gate times")
    return problems


# ── The invariant holds ──────────────────────────────────────────────────────

def test_a_fresh_dataset_satisfies_it():
    assert gate_count_mismatches(make_dataset()) == []


@pytest.mark.parametrize("moments", [("LM",), ("LM", "HM"), ("LM", "MM", "HM")])
def test_it_holds_for_any_number_of_moments(moments):
    assert gate_count_mismatches(make_dataset(moments=moments)) == []


def test_selecting_channels_preserves_it():
    """Whole channels leave; the survivors keep their gates and their times."""
    dataset = make_dataset(moments=("LM", "LM", "LM", "HM", "HM", "HM"),
                           components=list("XYZXYZ"))
    dataset, _ = select_channels(dataset, criterion="component", keep=["z"])

    assert dataset.n_channels == 2
    assert gate_count_mismatches(dataset) == []


def test_selecting_a_single_moment_preserves_it():
    dataset = make_dataset(moments=("LM", "MM", "HM"))
    dataset, _ = select_channels(dataset, criterion="moment", keep=["HM"])
    assert gate_count_mismatches(dataset) == []


def test_a_qc_pipeline_preserves_it():
    """QC writes columns on flightlines and must not touch layer_data at all."""
    dataset = make_dataset(n=6, n_lines=2, height=[30, 30, 80, 80, 30, 30])
    dataset, _ = QCRunner(Pipeline.parse("""
        kind: qc
        steps:
          - height_above_ground: {good_max: 45, marginal_max: 65}
          - system_attitude: {good: 5, marginal: 7.5}
          - sounding_density: {good_min: 80, marginal_min: 50}
    """)).run(dataset)

    assert gate_count_mismatches(dataset) == []


def test_a_processing_pipeline_preserves_it():
    dataset = make_dataset(moments=("LM", "LM", "LM", "HM", "HM", "HM"),
                           components=list("XYZXYZ"))
    dataset, _ = ProcessingRunner(Pipeline.parse("""
        kind: filters
        steps:
          - select_channels: {criterion: component, keep: [z]}
    """)).run(dataset)

    assert gate_count_mismatches(dataset) == []


# ── The check can actually fail ──────────────────────────────────────────────
# A guard that cannot fire is not a guard.

def test_dropping_a_gate_column_is_caught():
    """The failure mode this exists for: gates removed, GEX left saying otherwise."""
    dataset = make_dataset(n_gates=8)
    key = dataset.channels[0].data_key
    dataset.layer_data[key] = dataset.layer_data[key].iloc[:, :5]

    problems = gate_count_mismatches(dataset)
    assert len(problems) == 1
    assert "5 gates of data, 8 gate times" in problems[0]


def test_a_changed_gate_count_in_the_system_description_is_caught():
    """The mirror image: the GEX edited without the data following."""
    dataset = make_dataset(n_gates=8)
    dataset.system.gex_dict["Channel1"]["NoGates"] = 5.0

    problems = gate_count_mismatches(dataset)
    assert len(problems) == 1
    assert "8 gates of data, 5 gate times" in problems[0]


def test_discovery_already_warns_about_it(caplog):
    """core warns on the same condition; this test asserts the warning fires."""
    dataset = make_dataset(n_gates=8)
    dataset.system.gex_dict["Channel1"]["NoGates"] = 5.0

    with caplog.at_level("WARNING"):
        _ = dataset.channels
    assert any("gates" in record.message for record in caplog.records)
