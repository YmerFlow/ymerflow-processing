"""filter_32 — negative measurements, and the culling policy they motivate.

The cleanest predicate in the library: no threshold, because zero is not a
number anyone chose. Which makes it the right place to exercise the policy,
since nothing else is varying.
"""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.cull.filter_32_negatives import (
    STEP_CODE, cull_negatives,
)
from ymerflow_processing.core.mask import CULLED, IN_USE, provenance_key
from ymerflow_processing.core.pipeline import FILTER_GROUP, load_steps

N = 5
GATES = 8


def delivery(negatives=None, **kw):
    """A positive transient, with ``negatives`` as ``{(sounding, gate): value}``."""
    dataset = make_dataset(n=N, n_lines=1, n_gates=GATES, **kw)
    values = dataset.layer_data[dataset.channels[0].data_key]
    for (sounding, gate), value in (negatives or {}).items():
        values.iloc[sounding, gate] = value
    return dataset


def mask(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].inuse_key].to_numpy()


def why(dataset, which=0):
    return dataset.layer_data[
        provenance_key(dataset.channels[which].inuse_key)].to_numpy()


# ── The predicate ────────────────────────────────────────────────────────────

def test_a_negative_gate_is_disabled():
    dataset, report = cull_negatives(delivery({(2, 5): -1e-10}))

    assert mask(dataset)[2, 5] == CULLED
    assert mask(dataset)[2, 4] == IN_USE
    assert report.distributions["per_channel"][
        str(dataset.channels[0])]["negative_gates"] == 1


def test_zero_is_not_negative():
    """An exact zero is not a sign change, and usually means a dead channel or
    a fill value rather than a reversal."""
    dataset, _ = cull_negatives(delivery({(1, 3): 0.0}))
    assert mask(dataset)[1, 3] == IN_USE


def test_nan_is_not_negative():
    dataset, _ = cull_negatives(delivery({(1, 3): np.nan}))
    assert mask(dataset)[1, 3] == IN_USE


def test_a_positive_transient_is_left_alone():
    dataset, report = cull_negatives(delivery())
    assert (mask(dataset) == IN_USE).all()
    assert report.distributions["removed"] == 0


def test_it_needs_no_threshold():
    """Zero is not a number anyone chose, which is what makes this the cleanest
    plausibility test in the library."""
    dataset, _ = cull_negatives(delivery({(0, 6): -1.0}))
    assert mask(dataset)[0, 6] == CULLED


def test_it_changes_no_measurement_or_attribute():
    dataset = delivery({(2, 5): -1e-10})
    arrays = {k: v.copy() for k, v in dataset.layer_data.items()
              if "inuse" not in k.lower()}
    flightlines = dataset.flightlines.copy()

    dataset, _ = cull_negatives(dataset)

    for key, array in arrays.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    pd.testing.assert_frame_equal(dataset.flightlines, flightlines)


def test_each_channel_is_judged_on_its_own_transient():
    dataset = delivery({(1, 4): -1e-10})
    dataset, _ = cull_negatives(dataset)

    assert mask(dataset, 0)[1, 4] == CULLED
    assert (mask(dataset, 1) == IN_USE).all()


# ── The policy ───────────────────────────────────────────────────────────────

def test_by_default_only_what_was_observed_is_culled():
    """A step should not widen a cull beyond what it saw unasked."""
    dataset, _ = cull_negatives(delivery({(2, 5): -1e-10}))

    assert mask(dataset)[2, 4] == IN_USE
    assert mask(dataset)[2, 6] == IN_USE


def test_disable_tail_takes_everything_after():
    """Once coupling or noise dominates it dominates everything after — later
    gates are smaller, so the same interference is a larger fraction of them."""
    dataset, _ = cull_negatives(delivery({(2, 5): -1e-10}), disable_tail=True)

    assert (mask(dataset)[2, 5:] == CULLED).all()
    assert (mask(dataset)[2, :5] == IN_USE).all()


def test_spread_earlier_takes_the_gate_before():
    """The sign flip is where the contamination becomes visible, not where it
    started — it was already there in the gate before, and not yet large enough
    to overwhelm the response."""
    dataset, _ = cull_negatives(delivery({(2, 5): -1e-10}), spread_earlier=1)

    assert mask(dataset)[2, 4] == CULLED
    assert mask(dataset)[2, 3] == IN_USE


def test_the_usual_setting_leaves_a_usable_window():
    """spread_earlier: 1 with disable_tail: true — what the docstring
    recommends for negatives."""
    dataset, _ = cull_negatives(delivery({(2, 5): -1e-10}),
                                spread_earlier=1, disable_tail=True)

    assert (mask(dataset)[2, :4] == IN_USE).all()
    assert (mask(dataset)[2, 4:] == CULLED).all()


def test_start_gate_keeps_the_turn_off_window_out_of_it():
    """Early gates legitimately change sign there; evaluating over them would
    flag the vendor's own known-bad window as a discovery."""
    dataset, report = cull_negatives(delivery({(1, 1): -1e-10, (1, 6): -1e-10}),
                                     start_gate=4)

    assert mask(dataset)[1, 1] == IN_USE
    assert mask(dataset)[1, 6] == CULLED
    assert report.distributions["per_channel"][
        str(dataset.channels[0])]["negative_gates"] == 1


def test_a_gate_already_disabled_is_not_claimed_as_a_finding():
    """Counting it would inflate what this step appears to have discovered."""
    dataset = delivery({(1, 5): -1e-10})
    dataset.layer_data[dataset.channels[0].inuse_key].iloc[1, 5] = CULLED

    _, report = cull_negatives(dataset)
    assert report.distributions["per_channel"][
        str(dataset.channels[0])]["negative_gates"] == 0


# ── Provenance ───────────────────────────────────────────────────────────────

def test_observed_and_inferred_are_told_apart():
    """The whole reason the sign convention exists: how much of a cull was
    inference rather than observation."""
    dataset, report = cull_negatives(delivery({(2, 5): -1e-10}),
                                     spread_earlier=1, disable_tail=True)
    codes = why(dataset)

    assert codes[2, 5] == STEP_CODE       # observed negative
    assert codes[2, 4] == -STEP_CODE      # spread
    assert codes[2, 7] == -STEP_CODE      # tail

    outcome = report.distributions["per_channel"][str(dataset.channels[0])]
    assert outcome["removed_direct"] == 1
    assert outcome["removed_collateral"] == 3


def test_the_size_of_the_inference_is_stated_not_judged():
    """A ratio threshold was tried here and removed: with disable_tail on a long
    transient, collateral dominating direct hits is the intended behaviour, so
    any ratio firing on misuse fires on correct use too. The number goes in
    front of an operator instead."""
    _, report = cull_negatives(delivery({(0, 7): -1e-10}),
                               spread_earlier=6, disable_tail=True)

    assert any("6 of 7 disabled gate(s) were collateral" in n
               for n in report.notes)


def test_no_collateral_means_no_note_about_it():
    _, report = cull_negatives(delivery({(0, 7): -1e-10}))
    assert not any("collateral" in n for n in report.notes)


def test_the_policy_that_applied_is_recorded():
    _, report = cull_negatives(delivery({(2, 5): -1e-10}), spread_earlier=1,
                               disable_tail=True, start_gate=2)
    assert report.distributions["policy"] == {
        "start_gate": 2, "spread_earlier": 1, "disable_tail": True}


# ── Plumbing ─────────────────────────────────────────────────────────────────

def test_it_is_idempotent():
    dataset, _ = cull_negatives(delivery({(2, 5): -1e-10}), disable_tail=True)
    snapshot = {k: v.copy() for k, v in dataset.layer_data.items()}
    dataset, second = cull_negatives(dataset, disable_tail=True)

    for key, array in snapshot.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    assert second.distributions["removed"] == 0


def test_the_report_survives_serialization():
    _, report = cull_negatives(delivery({(2, 5): -1e-10}))
    restored = json.loads(json.dumps(report.to_dict(), default=str))
    assert restored["step"] == "cull_negatives"
    assert restored["distributions"]["provenance_code"] == STEP_CODE


def test_registered():
    assert "cull_negatives" in load_steps(FILTER_GROUP)
