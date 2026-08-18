"""filter_23 — soundings reduced to too few gates to constrain a model.

The first step whose *answer* depends on pipeline order rather than merely its
efficiency, so a good deal of this pins that: run first it finds nothing, run
last it sees what everything else left behind.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.cull.filter_23_sparse_soundings import (
    STEP_CODE, cull_sparse_soundings, usable,
)
from ymerflow_processing.core.mask import CULLED, IN_USE, provenance_key
from ymerflow_processing.core.pipeline import (
    FILTER_GROUP, Pipeline, ProcessingRunner, load_steps,
)

N = 6
GATES = 8


def delivery(**kw):
    return make_dataset(n=N, n_lines=1, n_gates=GATES, **kw)


def mask(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].inuse_key].to_numpy()


def leave(dataset, which, sounding, gates):
    """Reduce one sounding of one channel to ``gates`` usable gates."""
    key = dataset.channels[which].inuse_key
    dataset.layer_data[key].iloc[sounding, gates:] = CULLED
    return dataset


# ── What it culls ────────────────────────────────────────────────────────────

def test_a_sounding_below_the_minimum_loses_what_was_left():
    """The gates that remain are exactly the ones too few to be useful."""
    dataset = leave(delivery(), 0, sounding=2, gates=3)
    dataset, report = cull_sparse_soundings(dataset, minimum_gates=5)

    assert (mask(dataset)[2] == CULLED).all()
    assert (mask(dataset)[0] == IN_USE).all()
    assert report.distributions["soundings_culled"] == 1


def test_the_minimum_is_inclusive():
    """"At least N gates" keeps a sounding with exactly N."""
    dataset = leave(delivery(), 0, sounding=1, gates=5)
    dataset, _ = cull_sparse_soundings(dataset, minimum_gates=5)

    assert (mask(dataset)[1, :5] == IN_USE).all()


def test_it_changes_no_measurement_or_attribute():
    dataset = leave(delivery(), 0, sounding=2, gates=1)
    arrays = {k: v.copy() for k, v in dataset.layer_data.items()
              if "inuse" not in k.lower()}
    flightlines = dataset.flightlines.copy()

    dataset, _ = cull_sparse_soundings(dataset, minimum_gates=4)

    for key, array in arrays.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    pd.testing.assert_frame_equal(dataset.flightlines, flightlines)


def test_it_is_idempotent():
    dataset = leave(delivery(), 0, sounding=2, gates=3)
    dataset, _ = cull_sparse_soundings(dataset, minimum_gates=5)
    snapshot = {k: v.copy() for k, v in dataset.layer_data.items()}
    dataset, second = cull_sparse_soundings(dataset, minimum_gates=5)

    for key, array in snapshot.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    # The sounding is empty now, so it is no longer *sparse* — it is gone, and
    # this step does not claim credit for what it already took.
    assert second.distributions["soundings_culled"] == 0


# ── Per channel, not per sounding ────────────────────────────────────────────

def test_one_thin_moment_does_not_take_the_other_with_it():
    """A sounding with one usable moment is not a useless sounding, and
    disabling the good one would throw away data to punish its neighbour."""
    dataset = leave(delivery(), 0, sounding=3, gates=2)
    dataset, _ = cull_sparse_soundings(dataset, minimum_gates=5)

    assert (mask(dataset, 0)[3] == CULLED).all()
    assert (mask(dataset, 1)[3] == IN_USE).all()


def test_each_channel_is_counted_against_the_same_minimum():
    dataset = leave(leave(delivery(), 0, 1, 2), 1, 4, 2)
    dataset, report = cull_sparse_soundings(dataset, minimum_gates=5)

    assert (mask(dataset, 0)[1] == CULLED).all()
    assert (mask(dataset, 1)[4] == CULLED).all()
    assert report.distributions["soundings_culled"] == 2      # two distinct
    assert report.distributions["channel_soundings_culled"] == 2


# ── Usable means what the inversion will use ─────────────────────────────────

def test_a_gate_in_use_but_carrying_no_measurement_does_not_count():
    """Counting the mask alone would overstate what survives: a sounding of ten
    such gates is as empty as one with none."""
    dataset = delivery()
    channel = dataset.channels[0]
    dataset.layer_data[channel.data_key].iloc[2, 4:] = np.nan

    assert usable(dataset, channel)[2].sum() == 4
    dataset, _ = cull_sparse_soundings(dataset, minimum_gates=6)
    assert (mask(dataset)[2] == CULLED).all()


def test_a_gate_with_no_uncertainty_does_not_count_either():
    """The forward code needs all three together."""
    dataset = delivery()
    channel = dataset.channels[0]
    dataset.layer_data[channel.std_key].iloc[3, 2:] = np.nan

    assert usable(dataset, channel)[3].sum() == 2
    dataset, _ = cull_sparse_soundings(dataset, minimum_gates=4)
    assert (mask(dataset)[3] == CULLED).all()


def test_a_full_sounding_counts_every_gate():
    dataset = delivery()
    assert usable(dataset, dataset.channels[0])[0].sum() == GATES


# ── Already empty is not this step's doing ───────────────────────────────────

def test_a_sounding_already_emptied_is_reported_but_not_claimed():
    """Counting it would credit this filter with whatever actually emptied it."""
    dataset = leave(delivery(), 0, sounding=2, gates=0)
    dataset, report = cull_sparse_soundings(dataset, minimum_gates=5)

    outcome = report.distributions["per_channel"][str(dataset.channels[0])]
    assert outcome["soundings_already_empty"] == 1
    assert outcome["soundings_sparse"] == 0
    assert report.distributions["soundings_culled"] == 0


# ── Ordering ─────────────────────────────────────────────────────────────────

def test_run_first_it_finds_nothing():
    """Which is the whole reason 01 says it runs late."""
    _, report = cull_sparse_soundings(delivery(), minimum_gates=5)
    assert report.distributions["soundings_culled"] == 0


def test_run_after_another_filter_it_sees_what_that_one_left():
    """The same document with the steps reordered legitimately gives a different
    answer, so the report says what it counted against."""
    heights = np.full(N, 30.0); heights[2] = 250.0
    dataset = make_dataset(n=N, n_lines=1, n_gates=GATES, height=heights)
    for index in (1, 2):
        dataset.survey.gex.gex_dict[f"Channel{index}"]["RemoveInitialGates"] = 5.0

    dataset, report = ProcessingRunner(Pipeline.parse("""
        kind: filters
        steps:
          - cull_sparse_soundings: {minimum_gates: 5}
    """)).run(dataset)

    # filter_20 is a mandate, so five gates are already gone before this runs —
    # leaving three, below the minimum, for every sounding.
    culled = next(s for s in report.steps if s.step == "cull_sparse_soundings")
    assert culled.distributions["soundings_culled"] == N
    # Both channels were thinned, so the per-channel figures add to twice that.
    assert culled.distributions["channel_soundings_culled"] == 2 * N
    assert (mask(dataset) == CULLED).all()


def test_the_report_says_what_it_counted_against():
    _, report = cull_sparse_soundings(delivery(), minimum_gates=5)
    assert any("mask as it stands" in n for n in report.notes)
    assert any("belongs last" in n for n in report.notes)


# ── Provenance and plumbing ──────────────────────────────────────────────────

def test_the_culled_soundings_are_attributed_to_this_filter():
    dataset = leave(delivery(), 0, sounding=2, gates=3)
    dataset, _ = cull_sparse_soundings(dataset, minimum_gates=5)
    why = dataset.layer_data[provenance_key(dataset.channels[0].inuse_key)]

    # Only the three it actually took: the rest were already off, and belong to
    # whatever turned them off.
    assert (why.to_numpy()[2, :3] == STEP_CODE).all()
    assert (why.to_numpy()[2, 3:] == 0).all()
    assert STEP_CODE == 23


def test_the_median_usable_count_is_reported():
    """A number worth seeing before choosing a minimum."""
    dataset = leave(delivery(), 0, sounding=2, gates=3)
    _, report = cull_sparse_soundings(dataset, minimum_gates=2)

    outcome = report.distributions["per_channel"][str(dataset.channels[0])]
    assert outcome["median_usable_gates"] == GATES


@pytest.mark.parametrize("minimum", [0, -1])
def test_a_minimum_below_one_is_an_error(minimum):
    with pytest.raises(ValueError, match="at least 1"):
        cull_sparse_soundings(delivery(), minimum_gates=minimum)


def test_the_report_survives_serialization():
    dataset = leave(delivery(), 0, sounding=2, gates=3)
    _, report = cull_sparse_soundings(dataset, minimum_gates=5)
    restored = json.loads(json.dumps(report.to_dict(), default=str))

    assert restored["step"] == "cull_sparse_soundings"
    assert restored["distributions"]["soundings_culled"] == 1


def test_registered():
    assert "cull_sparse_soundings" in load_steps(FILTER_GROUP)


def test_a_sounding_thinned_in_every_channel_is_counted_once():
    """Summing the per-channel counts would report more soundings than the
    delivery has. Both figures are given, and they mean different things."""
    dataset = delivery()
    for which in (0, 1):
        dataset = leave(dataset, which, sounding=2, gates=1)

    _, report = cull_sparse_soundings(dataset, minimum_gates=5)

    assert report.distributions["soundings_culled"] == 1
    assert report.distributions["channel_soundings_culled"] == 2
    assert report.soundings_affected == 1
