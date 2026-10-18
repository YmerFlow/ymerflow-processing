"""Lining two runs up, shared by every dataset-consuming comparison.

One module rather than three copies, because all three face the same problem and
three copies would drift. What is tested here is the *identity* logic — which
sounding is which, which channel is which, which gate is which — since every
comparison built on it is only as trustworthy as the pairing underneath.
"""

import numpy as np
import pytest

from conftest import make_dataset
from ymerflow_processing.compare.align import (
    DATA, MASK, QUANTITIES, UNCERTAINTY, align_channels, align_soundings,
    axis_of, block, gate_overlap, key_for, sounding_keys,
)
from ymerflow_processing.pipeline import StepReport

N = 8
GATES = 6


def dataset(**kw):
    return make_dataset(n=N, n_lines=1, n_gates=GATES, **kw)


def strip(data, *, timestamp=True, position=True):
    """A dataset with its identity columns removed."""
    xyz = data.survey.xyz
    drop = []
    if timestamp and xyz.get_column("timestamp"):
        drop.append(xyz.get_column("timestamp"))
    if position:
        drop += [c for c in (xyz.x_column, xyz.y_column)
                 if c in data.flightlines.columns]
    xyz.flightlines = data.flightlines.drop(columns=drop)
    return data


# ── Gates, by center time ────────────────────────────────────────────────────

def test_gates_one_side_dropped_line_up_by_time_not_by_position():
    """A run that dropped its first two gates is compared over the ones it kept
    — rather than shifted against the other side by two and declared wrong
    everywhere."""
    times = np.array([1e-5, 2e-5, 4e-5, 8e-5, 1.6e-4, 3.2e-4])
    cols_a, cols_b = gate_overlap(times[2:], times)

    assert list(cols_a) == [0, 1, 2, 3]
    assert list(cols_b) == [2, 3, 4, 5]


def test_gates_missing_from_the_middle_still_line_up():
    times = np.array([1e-5, 2e-5, 4e-5, 8e-5, 1.6e-4])
    cols_a, cols_b = gate_overlap(times[[0, 1, 4]], times)
    assert list(cols_b) == [0, 1, 4]


def test_gate_times_that_share_nothing_do_not_line_up():
    assert gate_overlap(np.array([1e-5, 2e-5]), np.array([7.0, 8.0])) is None


def test_one_gate_matches_one_gate():
    """Not many-to-one: two gates close to a third must not both claim it."""
    cols_a, cols_b = gate_overlap(np.array([1e-5, 1e-5]), np.array([1e-5]))
    assert len(cols_a) == 1


def test_a_gate_time_that_is_not_a_number_is_skipped():
    times = np.array([1e-5, np.nan, 4e-5])
    cols_a, cols_b = gate_overlap(times, times)
    assert list(cols_a) == [0, 2]


def test_no_axis_means_no_overlap():
    assert gate_overlap(None, np.array([1e-5])) is None
    assert axis_of(make_dataset(n=2, n_gates=3, with_gex=False).channels[0]) \
        is None or True          # an axis may be absent; asking must not raise


# ── Soundings, by what the instrument recorded ───────────────────────────────

def test_a_timestamp_is_the_first_choice():
    """It is an identity: the instrument recorded it, and it survives
    re-export, renaming and reordering."""
    data = dataset()
    data.survey.xyz.flightlines["timestamp"] = np.arange(N) * 1e-5
    assert sounding_keys(data, 0.5)[1] == "timestamp"


def test_position_is_the_fallback():
    assert sounding_keys(dataset(), 0.5)[1] == "position"


def test_neither_available_gives_neither():
    assert sounding_keys(strip(dataset()), 0.5) == (None, None)


def test_position_matching_tolerates_rounding():
    """Good to whatever tolerance the two sides round coordinates to."""
    a, b = dataset(), dataset()
    b.flightlines[b.survey.xyz.x_column] += 0.01

    report = StepReport(step="t")
    rows_a, rows_b, how = align_soundings(a, b, 0.5, report)
    assert how == "position" and len(rows_a) == N


def test_a_rigid_offset_re_pairs_soundings_with_their_neighbors():
    """Position's one specific way of being wrong, and it does not announce
    itself: offset the two runs by five sounding spacings and each sounding
    pairs with a *neighbor* rather than failing to pair. The pairing stays
    monotone and the values stay plausible — only the overlap shrinks, by
    roughly the offset. This is why a timestamp is preferred, and why the
    unmatched fraction is reported."""
    a, b = dataset(), dataset()
    spacing = 10.0
    b.flightlines[b.survey.xyz.x_column] += 5 * spacing

    report = StepReport(step="t")
    rows_a, rows_b, how = align_soundings(a, b, 0.5, report)

    assert how == "position"
    assert len(rows_a) == N - 5              # not zero, which is the trap
    assert list(rows_a - rows_b) == [5] * (N - 5)
    assert any("rigid offset" in note for note in report.notes)


def test_a_shift_landing_between_soundings_matches_nothing():
    a, b = dataset(), dataset()
    b.flightlines[b.survey.xyz.x_column] += 5.0        # half a spacing

    with pytest.raises(ValueError, match="not the same survey"):
        align_soundings(a, b, 0.5, StepReport(step="t"))


def test_a_modest_unmatched_fraction_is_not_called_suspicious():
    """Two runs may genuinely have kept different soundings, and saying 'this
    looks like a coordinate error' every time would train people to ignore it."""
    a = make_dataset(n=40, n_lines=1, n_gates=GATES)
    b = make_dataset(n=40, n_lines=1, n_gates=GATES)
    b.survey.xyz.flightlines = b.flightlines.iloc[:-1].copy()   # 1 of 40

    report = StepReport(step="t")
    align_soundings(a, b, 0.5, report)
    assert not any("rigid offset" in note for note in report.notes)


def test_a_reordered_side_still_pairs_correctly():
    """Pairing by row would compare every sounding against a stranger."""
    a, b = dataset(), dataset()
    b.survey.xyz.flightlines = b.flightlines.iloc[::-1].copy()

    rows_a, rows_b, _ = align_soundings(a, b, 0.5, StepReport(step="t"))
    x = a.flightlines[a.survey.xyz.x_column].to_numpy()
    y = b.flightlines[b.survey.xyz.x_column].to_numpy()
    assert np.allclose(x[rows_a], y[rows_b])


def test_pairing_by_order_is_said_out_loud():
    """An assumption, not a match — and nothing in the numbers would say so."""
    report = StepReport(step="t")
    rows_a, rows_b, how = align_soundings(
        strip(dataset()), strip(dataset()), 0.5, report)

    assert how == "order"
    assert any("BY ORDER" in note for note in report.notes)


def test_unmatchable_sides_of_different_length_are_refused():
    """There is no alignment to fall back to, and guessing one would compare
    every sounding against a stranger while reporting a number."""
    a = strip(make_dataset(n=N, n_lines=1, n_gates=GATES))
    b = strip(make_dataset(n=N + 3, n_lines=1, n_gates=GATES))

    with pytest.raises(ValueError, match="no alignment to fall back to"):
        align_soundings(a, b, 0.5, StepReport(step="t"))


def test_soundings_only_one_side_has_are_reported():
    a, b = dataset(), dataset()
    keep = b.flightlines.index[:-2]
    b.survey.xyz.flightlines = b.flightlines.loc[keep].copy()

    report = StepReport(step="t")
    rows_a, rows_b, _ = align_soundings(a, b, 0.5, report)
    assert len(rows_a) == N - 2
    assert any("only in the candidate" in note for note in report.notes)


def test_the_two_sides_must_agree_on_which_identity_they_used():
    """One side matched by timestamp and the other by position would be two
    different keys compared against each other."""
    a, b = dataset(), dataset()
    a.survey.xyz.flightlines["timestamp"] = np.arange(N) * 1e-5

    report = StepReport(step="t")
    _, _, how = align_soundings(a, b, 0.5, report)
    assert how == "order"


# ── Channels, by what they physically are ────────────────────────────────────

def test_channels_pair_up():
    report = StepReport(step="t")
    pairs = align_channels(dataset(), dataset(), report)
    assert len(pairs) == 2
    assert [str(a) for a, _ in pairs] == [str(b) for _, b in pairs]


def test_a_channel_the_other_side_lacks_is_reported():
    a = dataset()
    b = make_dataset(moments=("LM",), n=N, n_lines=1, n_gates=GATES)

    report = StepReport(step="t")
    pairs = align_channels(a, b, report)
    assert len(pairs) == 1
    assert any("no channel matching" in note for note in report.notes)


def test_channels_with_no_axis_fall_back_to_the_instrument_s_numbering():
    a = make_dataset(n=N, n_lines=1, n_gates=GATES, with_gex=False)
    b = make_dataset(n=N, n_lines=1, n_gates=GATES, with_gex=False)

    pairs = align_channels(a, b, StepReport(step="t"))
    assert [(x.index, y.index) for x, y in pairs] == [(1, 1), (2, 2)]


# ── Quantities, semantically ─────────────────────────────────────────────────

def test_the_three_quantities_resolve_to_a_channel_s_own_keys():
    channel = dataset().channels[0]
    assert key_for(channel, DATA) == channel.data_key
    assert key_for(channel, UNCERTAINTY) == channel.std_key
    assert key_for(channel, MASK) == channel.inuse_key
    assert set(QUANTITIES) == {DATA, UNCERTAINTY, MASK}


def test_a_quantity_a_side_lacks_reads_as_nothing_rather_than_raising():
    data = dataset()
    channel = data.channels[0]
    del data.layer_data[channel.std_key]

    rows = np.arange(N)
    assert block(data, channel, UNCERTAINTY, rows, np.arange(GATES)) is None
    assert block(data, channel, DATA, rows, np.arange(GATES)).shape == (N, GATES)
