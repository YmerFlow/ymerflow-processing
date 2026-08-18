"""The in-use mask and the shared culling policy.

R6's second clause: culling writes the mask. Nothing here deletes anything, and
the arrays keep their shape — a culled datum is marked, which is reversible and
auditable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.mask import (
    CULLED, IN_USE, UNTOUCHED, apply_policy, cull, ensure, in_use,
    PROVENANCE_PREFIX, enable, provenance, provenance_key, summarize,
)

HEIGHT = 6
GATES = 8


def dataset(**kw):
    return make_dataset(n=HEIGHT, n_lines=1, n_gates=GATES, **kw)


def nothing_bad():
    return np.zeros((HEIGHT, GATES), dtype=bool)


# ── Shape and materializing ──────────────────────────────────────────────────

def test_the_mask_is_per_channel_per_gate_per_sounding():
    """It reads as a single vector until you look at the shape."""
    d = dataset()
    for channel in d.channels:
        assert ensure(d, channel).shape == (HEIGHT, GATES)
    assert len({c.inuse_key for c in d.channels}) == len(d.channels)


def test_a_delivery_without_masks_gets_them_created_all_in_use():
    d = dataset()
    for channel in d.channels:
        del d.layer_data[channel.inuse_key]

    for channel in d.channels:
        assert channel.inuse_key is None      # rediscovery agrees they are gone
        mask = ensure(d, channel)
        assert (mask.to_numpy() == IN_USE).all()
        assert mask.shape == (HEIGHT, GATES)
    # And named so a file written from here rediscovers it on the way back.
    assert all(c.inuse_key is not None for c in d.channels)


def test_nothing_is_created_merely_by_looking_at_the_dataset():
    """R6 promises a QC-only run leaves the data byte-identical. Creating masks
    at load would break that for every pipeline that only measures."""
    d = dataset()
    for channel in d.channels:
        del d.layer_data[channel.inuse_key]

    before = set(d.layer_data)
    _ = d.flightlines, [str(c) for c in d.channels]
    assert set(d.layer_data) == before


def test_a_delivered_mask_is_preserved_rather_than_reset():
    """Data can arrive already culled by the contractor's own processing."""
    d = dataset()
    channel = d.channels[0]
    d.layer_data[channel.inuse_key].iloc[:, :2] = CULLED

    assert (ensure(d, channel).to_numpy()[:, :2] == CULLED).all()


# ── The policy ───────────────────────────────────────────────────────────────

def test_without_options_the_predicate_answer_is_used_as_is():
    bad = nothing_bad(); bad[2, 4] = True
    assert (apply_policy(bad) == bad).all()


def test_start_gate_discards_objections_before_it():
    """Where the predicate is not meaningful — curvature is undefined at the
    first gate, and early gates are contaminated by transmitter turn-off."""
    bad = nothing_bad(); bad[0, 0] = bad[0, 1] = bad[0, 5] = True
    out = apply_policy(bad, start_gate=3)

    assert not out[0, :3].any()
    assert out[0, 5]


def test_spread_earlier_takes_the_gates_before_a_hit():
    """The contamination is there before it is large enough to be visible — the
    gate where a transient turns negative is where the effect crosses zero, not
    where it started."""
    bad = nothing_bad(); bad[1, 5] = True
    out = apply_policy(bad, spread_earlier=2)

    assert list(out[1]) == [False, False, False, True, True, True,
                            False, False]


def test_spread_earlier_may_reach_below_the_start_gate():
    """start_gate says where the predicate is valid. The claim that an effect
    precedes its onset does not stop being true at an evaluation boundary."""
    bad = nothing_bad(); bad[0, 4] = True
    out = apply_policy(bad, start_gate=4, spread_earlier=3)

    assert out[0, 1] and out[0, 2] and out[0, 3]


def test_spread_earlier_clamps_at_the_first_gate():
    bad = nothing_bad(); bad[0, 1] = True
    out = apply_policy(bad, spread_earlier=5)
    assert out[0, 0] and out[0, 1]


def test_disable_tail_takes_everything_after_the_earliest_hit():
    """A transient does not recover once it has gone bad."""
    bad = nothing_bad(); bad[3, 5] = True
    out = apply_policy(bad, disable_tail=True)

    assert not out[3, :5].any()
    assert out[3, 5:].all()


def test_tail_and_spread_together_leave_a_usable_window():
    """The two collapse to 'usable up to N gates before the first problem',
    rather than a scatter of holes."""
    bad = nothing_bad(); bad[2, 6] = True; bad[2, 7] = True
    out = apply_policy(bad, spread_earlier=2, disable_tail=True)

    assert list(out[2]) == [False, False, False, False,
                            True, True, True, True]


def test_a_sounding_with_no_hit_is_untouched_by_tail_or_spread():
    """The bug this guards: an earliest-hit index computed for a sounding with
    no hits would cull its tail for nothing."""
    bad = nothing_bad(); bad[0, 4] = True
    out = apply_policy(bad, spread_earlier=2, disable_tail=True)

    assert out[0].any()
    for row in range(1, HEIGHT):
        assert not out[row].any(), f"sounding {row} had no objection"


def test_the_policy_is_per_sounding_not_across_the_whole_array():
    bad = nothing_bad(); bad[1, 3] = True; bad[4, 6] = True
    out = apply_policy(bad, disable_tail=True)

    assert out[1, 3:].all() and not out[1, :3].any()
    assert out[4, 6:].all() and not out[4, :6].any()


def test_a_predicate_answering_the_wrong_shape_is_an_error():
    with pytest.raises(ValueError, match="soundings, gates"):
        apply_policy(np.zeros(GATES, dtype=bool))


@pytest.mark.parametrize("kw", [{"start_gate": -1}, {"spread_earlier": -2}])
def test_negative_gate_counts_are_an_error(kw):
    with pytest.raises(ValueError, match="cannot be negative"):
        apply_policy(nothing_bad(), **kw)


# ── Culling ──────────────────────────────────────────────────────────────────

def test_culling_turns_data_off_and_changes_nothing_else():
    d = dataset()
    channel = d.channels[0]
    before = d.layer_data[channel.data_key].copy()

    bad = nothing_bad(); bad[1, 3] = True
    cull(d, channel, bad, step_code=7)

    assert d.layer_data[channel.inuse_key].to_numpy()[1, 3] == CULLED
    # The measurement itself is untouched, and the array keeps its shape.
    pd.testing.assert_frame_equal(d.layer_data[channel.data_key], before)
    assert d.layer_data[channel.inuse_key].shape == (HEIGHT, GATES)


def test_culling_only_affects_the_channel_it_was_given():
    """Each channel has its own transient; a problem in one is not a problem in
    the other."""
    d = dataset()
    bad = nothing_bad(); bad[:, :] = True
    cull(d, d.channels[0], bad, step_code=1)

    assert (d.layer_data[d.channels[1].inuse_key].to_numpy() == IN_USE).all()


def test_culling_is_idempotent():
    """R4. Subtractive semantics make this nearly free, which is most of why
    they were chosen."""
    d = dataset()
    channel = d.channels[0]
    bad = nothing_bad(); bad[2, 4] = True

    first = cull(d, channel, bad, step_code=3)
    snapshot = d.layer_data[channel.inuse_key].copy()
    second = cull(d, channel, bad, step_code=3)

    pd.testing.assert_frame_equal(d.layer_data[channel.inuse_key], snapshot)
    assert first["removed"] == 1
    assert second["removed"] == 0        # nothing left for it to take


def test_removals_are_counted_against_the_mask_as_found():
    """A delivery may arrive already culled. Counting against the total would
    claim credit for the contractor's work and double-count on a re-run."""
    d = dataset()
    channel = d.channels[0]
    d.layer_data[channel.inuse_key].iloc[:, :4] = CULLED     # arrived culled

    bad = nothing_bad(); bad[:, :6] = True                   # overlaps it
    report = cull(d, channel, bad, step_code=2)

    assert report["in_use_before"] == HEIGHT * 4             # not 48
    assert report["removed"] == HEIGHT * 2                   # only gates 4,5


def test_culling_never_turns_data_back_on():
    d = dataset()
    channel = d.channels[0]
    d.layer_data[channel.inuse_key].iloc[0, 0] = CULLED

    cull(d, channel, nothing_bad(), step_code=1)
    assert d.layer_data[channel.inuse_key].to_numpy()[0, 0] == CULLED


def test_a_mismatched_predicate_shape_is_an_error():
    d = dataset()
    with pytest.raises(ValueError, match="per sounding per"):
        cull(d, d.channels[0], np.zeros((2, 2), dtype=bool), step_code=1)


def test_a_sounding_losing_its_last_gate_is_reported():
    """What filter_23 exists to notice, surfaced as it happens."""
    d = dataset()
    bad = nothing_bad(); bad[3, :] = True
    report = cull(d, d.channels[0], bad, step_code=1)

    assert report["soundings_emptied"] == 1


# ── Provenance ───────────────────────────────────────────────────────────────

def test_the_provenance_array_cannot_be_mistaken_for_a_mask():
    """Naming patterns are anchored at the start of a key, so a suffix would
    leave the provenance array matching the in-use pattern and discovery would
    find two candidates for one channel's mask."""
    d = dataset()
    channel = d.channels[0]
    provenance(d, channel)

    key = provenance_key(channel.inuse_key)
    assert key.startswith(PROVENANCE_PREFIX)
    # Rediscovery still resolves exactly one mask per channel.
    assert [c.inuse_key for c in d.channels] == [
        "dbdt_inuse_ch1gt", "dbdt_inuse_ch2gt"]


def test_provenance_names_the_step_that_removed_each_datum():
    d = dataset()
    channel = d.channels[0]
    bad = nothing_bad(); bad[1, 2] = True
    cull(d, channel, bad, step_code=9)

    why = d.layer_data[provenance_key(channel.inuse_key)].to_numpy()
    assert why[1, 2] == 9
    assert why[0, 0] == UNTOUCHED


def test_collateral_is_recorded_with_a_negative_code():
    """So over-culling is measurable rather than a suspicion: a step whose
    collateral dwarfs its direct hits has its knobs set too aggressively."""
    d = dataset()
    channel = d.channels[0]
    direct = nothing_bad(); direct[0, 5] = True
    widened = apply_policy(direct, spread_earlier=2, disable_tail=True)

    report = cull(d, channel, widened, step_code=4, direct=direct)
    why = d.layer_data[provenance_key(channel.inuse_key)].to_numpy()

    assert why[0, 5] == 4                      # failed the predicate itself
    assert why[0, 3] == -4 and why[0, 7] == -4  # spread, and tail
    assert report["removed_direct"] == 1
    assert report["removed_collateral"] == 4


def test_the_first_step_to_cull_keeps_the_attribution():
    """Under subtractive semantics the first is the causal one. A later step
    that would also have culled it is not recorded."""
    d = dataset()
    channel = d.channels[0]
    bad = nothing_bad(); bad[2, 2] = True

    cull(d, channel, bad, step_code=1)
    cull(d, channel, bad, step_code=2)

    why = d.layer_data[provenance_key(channel.inuse_key)].to_numpy()
    assert why[2, 2] == 1


def test_provenance_fits_an_int8_both_ways():
    d = dataset()
    why = provenance(d, d.channels[0])
    assert why.to_numpy().dtype == np.int8

    for bad_code in (0, -1, 128):
        with pytest.raises(ValueError, match="1..127"):
            cull(d, d.channels[0], nothing_bad(), step_code=bad_code)


# ── Reading it back ──────────────────────────────────────────────────────────

def test_in_use_is_what_a_neighbourhood_predicate_reads():
    """Slope and curvature are computed between gates, so a culled gate is not
    a gate to compute across."""
    d = dataset()
    channel = d.channels[0]
    bad = nothing_bad(); bad[0, 3] = True
    cull(d, channel, bad, step_code=1)

    live = in_use(d, channel)
    assert live.dtype == bool
    assert not live[0, 3] and live[0, 2] and live[0, 4]


def test_the_summary_counts_what_is_left():
    d = dataset()
    bad = nothing_bad(); bad[:, :2] = True
    cull(d, d.channels[0], bad, step_code=1)

    summary = summarize(d)
    first = summary[str(d.channels[0])]
    assert first["culled"] == HEIGHT * 2
    assert first["in_use"] == HEIGHT * (GATES - 2)
    assert first["culled_fraction"] == pytest.approx(2 / GATES)
    # The untouched channel is reported too, at zero.
    assert summary[str(d.channels[1])]["culled"] == 0


def test_no_nan_ever_reaches_a_mask():
    """Downstream readers disagree about NaN — one treats it as in-use, another
    raises. Neither is wanted, so it stays confined to the filter_33 diff."""
    d = dataset()
    bad = nothing_bad(); bad[1, 1] = True
    cull(d, d.channels[0], bad, step_code=1)

    for channel in d.channels:
        values = d.layer_data[channel.inuse_key].to_numpy()
        assert np.isfinite(values).all()
        assert set(np.unique(values)) <= {IN_USE, CULLED}


# ── Enabling: the guarded exception ──────────────────────────────────────────

def test_enable_turns_data_back_on():
    d = dataset()
    channel = d.channels[0]
    bad = nothing_bad(); bad[:, :3] = True
    cull(d, channel, bad, step_code=1)

    good = np.zeros((HEIGHT, GATES), dtype=bool); good[:, :2] = True
    report = enable(d, channel, good, step_code=21)

    values = d.layer_data[channel.inuse_key].to_numpy()
    assert (values[:, :2] == IN_USE).all()
    assert (values[:, 2] == CULLED).all()          # not nominated
    assert report["restored"] == HEIGHT * 2


def test_enable_only_counts_what_was_actually_off():
    """Measured against the mask as found, like everything else."""
    d = dataset()
    channel = d.channels[0]
    bad = nothing_bad(); bad[:, 0] = True
    cull(d, channel, bad, step_code=1)

    good = np.ones((HEIGHT, GATES), dtype=bool)
    assert enable(d, channel, good, step_code=21)["restored"] == HEIGHT


def test_the_mask_disambiguates_a_provenance_code():
    """A code against a culled entry says why it is off; the same code against
    an in-use entry says it was deliberately restored."""
    d = dataset()
    channel = d.channels[0]
    bad = nothing_bad(); bad[:, :2] = True
    cull(d, channel, bad, step_code=7)
    good = np.zeros((HEIGHT, GATES), dtype=bool); good[:, 0] = True
    enable(d, channel, good, step_code=21)

    live = d.layer_data[channel.inuse_key].to_numpy()
    codes = d.layer_data[provenance_key(channel.inuse_key)].to_numpy()

    assert codes[0, 0] == 21 and live[0, 0] == IN_USE      # restored
    assert codes[0, 1] == 7 and live[0, 1] == CULLED       # why it is off
    assert codes[0, 5] == UNTOUCHED and live[0, 5] == IN_USE


def test_enable_rejects_a_bad_step_code():
    d = dataset()
    good = np.ones((HEIGHT, GATES), dtype=bool)
    for code in (0, -1, 128):
        with pytest.raises(ValueError, match="1..127"):
            enable(d, d.channels[0], good, step_code=code)


def test_enable_rejects_a_mismatched_shape():
    d = dataset()
    with pytest.raises(ValueError, match="whose\\s+mask is|mask is"):
        enable(d, d.channels[0], np.ones((2, 2), dtype=bool), step_code=21)
