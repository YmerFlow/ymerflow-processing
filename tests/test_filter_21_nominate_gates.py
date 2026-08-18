"""filter_21 — nominated gates, either direction.

The counterpart to filter_20's mandate, and one of only two steps allowed to
turn data back on. Most of what follows guards that exception: it has to work,
and it has to stay hard to reach by accident.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.cull.filter_20_initial_gates import (
    disable_initial_gates,
)
from ymerflow_processing.core.cull.filter_21_nominate_gates import (
    STEP_CODE, nominate_gates, parse_gates,
)
from ymerflow_processing.core.mask import (
    CULLED, IN_USE, provenance_key, step_codes,
)
from ymerflow_processing.core.pipeline import FILTER_GROUP, load_steps

N = 5
GATES = 8


def delivery(**kw):
    return make_dataset(n=N, n_lines=1, n_gates=GATES, **kw)


def mask(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].inuse_key].to_numpy()


def why(dataset, which=0):
    return dataset.layer_data[
        provenance_key(dataset.channels[which].inuse_key)].to_numpy()


# ── Naming gates ─────────────────────────────────────────────────────────────

def test_a_single_gate():
    assert parse_gates(3, GATES) == [3]


def test_a_list_of_gates():
    assert parse_gates([0, 1, 5], GATES) == [0, 1, 5]


def test_a_range_is_inclusive():
    """A span, not a count. "0-4" names five gates, the same five that
    RemoveInitialGates: 5 removes — they agree on the outcome and differ in
    what they are."""
    assert parse_gates("0-4", GATES) == [0, 1, 2, 3, 4]


def test_ranges_and_singles_mix():
    assert parse_gates(["0-2", 6], GATES) == [0, 1, 2, 6]


def test_duplicates_collapse():
    assert parse_gates(["0-3", 2, 3], GATES) == [0, 1, 2, 3]


def test_a_gate_outside_the_window_is_an_error():
    """Silently doing four fifths of what was asked is how a mistake about the
    data goes unnoticed."""
    with pytest.raises(ValueError, match="outside the delivered window"):
        parse_gates([0, 99], GATES)


def test_a_negative_gate_is_an_error():
    with pytest.raises(ValueError, match="outside the delivered window"):
        parse_gates(-1, GATES)


def test_a_backwards_range_is_an_error():
    with pytest.raises(ValueError, match="runs backwards"):
        parse_gates("5-2", GATES)


def test_an_unreadable_nomination_is_an_error():
    with pytest.raises(ValueError, match="Cannot read"):
        parse_gates("first four", GATES)


# ── Disabling ────────────────────────────────────────────────────────────────

def test_nominated_gates_are_disabled_for_the_whole_channel():
    """A gate is nominated for the channel, not for particular soundings.
    Per-sounding judgement is filter_33's."""
    dataset, report = nominate_gates(delivery(), disable_gates={"ch1": "0-2"})

    assert (mask(dataset)[:, :3] == CULLED).all()
    assert (mask(dataset)[:, 3:] == IN_USE).all()
    assert report.distributions["removed"] == N * 3


def test_only_the_nominated_channel_is_touched():
    dataset, _ = nominate_gates(delivery(), disable_gates={"ch1": 0})
    assert (mask(dataset, 1) == IN_USE).all()


def test_a_channel_may_be_named_by_moment():
    dataset, _ = nominate_gates(delivery(), disable_gates={"HM": 0})
    assert (mask(dataset, 0) == IN_USE).all()
    assert (mask(dataset, 1)[:, 0] == CULLED).all()


def test_one_nomination_can_apply_to_every_channel():
    dataset, _ = nominate_gates(delivery(), disable_gates="0-1")
    for which in (0, 1):
        assert (mask(dataset, which)[:, :2] == CULLED).all()


def test_it_changes_no_measurement_or_attribute():
    dataset = delivery()
    arrays = {k: v.copy() for k, v in dataset.layer_data.items()
              if "inuse" not in k.lower()}
    flightlines = dataset.flightlines.copy()

    dataset, _ = nominate_gates(dataset, disable_gates={"ch1": "0-2"})

    for key, array in arrays.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    pd.testing.assert_frame_equal(dataset.flightlines, flightlines)


# ── Enabling: the exception ──────────────────────────────────────────────────

def test_it_gives_back_the_gates_the_mandate_took():
    """The case the whole design exists for: someone doing full primary-field
    modeling needs the early gates filter_20 disabled on the vendor's word."""
    dataset = delivery()
    for index in (1, 2):
        dataset.survey.gex.gex_dict[f"Channel{index}"]["RemoveInitialGates"] = 4.0
    dataset, _ = disable_initial_gates(dataset)
    assert (mask(dataset)[:, :4] == CULLED).all()

    dataset, report = nominate_gates(dataset, enable_gates={"ch1": "0-3"})

    assert (mask(dataset)[:, :4] == IN_USE).all()
    assert report.distributions["restored"] == N * 4


def test_enabling_a_gate_that_was_never_off_changes_nothing():
    dataset, report = nominate_gates(delivery(), enable_gates={"ch1": "0-2"})

    assert (mask(dataset) == IN_USE).all()
    assert report.distributions["restored"] == 0


def test_restoring_is_recorded_and_distinguishable_from_culling():
    """A non-zero code against a culled entry says why it is off; the same code
    against an in-use entry says it was deliberately restored. The two arrays
    are read together, so no third encoding is needed."""
    dataset = delivery()
    dataset.survey.gex.gex_dict["Channel1"]["RemoveInitialGates"] = 3.0
    dataset, _ = disable_initial_gates(dataset)
    dataset, _ = nominate_gates(dataset, enable_gates={"ch1": "0-1"})

    codes, live = why(dataset), mask(dataset)
    assert codes[0, 0] == STEP_CODE and live[0, 0] == IN_USE     # restored
    assert codes[0, 2] == 20 and live[0, 2] == CULLED            # still culled
    assert codes[0, 5] == 0 and live[0, 5] == IN_USE             # never touched


def test_both_directions_in_one_step():
    dataset, report = nominate_gates(delivery(), disable_gates={"ch1": "0-1"},
                                     enable_gates={"ch2": 0})

    assert (mask(dataset, 0)[:, :2] == CULLED).all()
    assert report.distributions["removed"] == N * 2
    assert report.distributions["restored"] == 0    # ch2 was never off


def test_a_later_cull_can_take_a_restored_gate_again():
    """Restoring is not a permanent exemption — subtractive steps still apply."""
    dataset = delivery()
    dataset.survey.gex.gex_dict["Channel1"]["RemoveInitialGates"] = 2.0
    dataset, _ = disable_initial_gates(dataset)
    dataset, _ = nominate_gates(dataset, enable_gates={"ch1": "0-1"})
    dataset, _ = nominate_gates(dataset, disable_gates={"ch1": 0})

    assert mask(dataset)[0, 0] == CULLED
    assert mask(dataset)[0, 1] == IN_USE


# ── Plumbing ─────────────────────────────────────────────────────────────────

def test_neither_direction_is_an_error():
    with pytest.raises(ValueError, match="needs enable_gates or disable_gates"):
        nominate_gates(delivery())


def test_the_dataset_records_what_the_code_means():
    dataset, _ = nominate_gates(delivery(), disable_gates={"ch1": 0})
    assert step_codes(dataset)[STEP_CODE] == "nominate_gates"
    assert STEP_CODE == 21


def test_it_is_idempotent():
    dataset, _ = nominate_gates(delivery(), disable_gates={"ch1": "0-2"})
    snapshot = {k: v.copy() for k, v in dataset.layer_data.items()}
    dataset, second = nominate_gates(dataset, disable_gates={"ch1": "0-2"})

    for key, array in snapshot.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    assert second.distributions["removed"] == 0


def test_the_report_survives_serialization():
    _, report = nominate_gates(delivery(), disable_gates={"ch1": "0-2"})
    restored = json.loads(json.dumps(report.to_dict(), default=str))

    assert restored["step"] == "nominate_gates"
    assert restored["distributions"]["removed"] == N * 3


def test_registered():
    assert "nominate_gates" in load_steps(FILTER_GROUP)
