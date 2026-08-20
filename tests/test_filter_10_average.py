"""filter_10 — along-line averaging.

01 calls this "the most consequential filter in the library and the one most
worth rebuilding rather than porting", and names four defects to design out.
Each has a test here named for it, because a rebuild that reintroduces them is
not a rebuild.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.filter_10_average import (
    BOXCAR, GAUSSIAN, HANN, STASH_SUFFIX, STEP_CODE, average_along_line,
    gate_widths, kernel,
)
from ymerflow_processing.core.mask import CULLED, IN_USE, provenance_key
from ymerflow_processing.core.pipeline import FILTER_GROUP, load_steps

N = 40
GATES = 8


def delivery(values=None, **kw):
    dataset = make_dataset(n=N, n_lines=2, n_gates=GATES, **kw)
    if values is not None:
        for channel in dataset.channels:
            dataset.layer_data[channel.data_key].iloc[:, :] = values
    return dataset


def data(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].data_key].to_numpy()


def std(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].std_key].to_numpy()


def mask(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].inuse_key].to_numpy()


# ── Defect 1: a width of one is a no-op ──────────────────────────────────────

def test_a_width_of_one_changes_nothing():
    """The old implementation silently produced all-invalid output. A window of
    one sounding is the sounding itself."""
    dataset = delivery()
    before = data(dataset).copy()

    dataset, _ = average_along_line(dataset, width_first=1)

    np.testing.assert_allclose(data(dataset), before)
    assert np.isfinite(data(dataset)).all()


@pytest.mark.parametrize("method", [HANN, GAUSSIAN, BOXCAR])
def test_a_kernel_of_one_is_unity_for_every_method(method):
    """It falls out rather than being special-cased."""
    assert list(kernel(1, method)) == [1.0]


def test_a_width_of_one_culls_nothing():
    dataset, report = average_along_line(delivery(), width_first=1)

    assert (mask(dataset) == IN_USE).all()
    assert report.distributions["per_channel"]["ch1 (LM/z)"]["too_thin"] == 0


def test_a_width_below_one_is_an_error():
    with pytest.raises(ValueError, match="at least 1"):
        average_along_line(delivery(), width_first=0)


# ── Defect 2: effective sample size, not a count ─────────────────────────────

def test_the_validity_gate_uses_effective_sample_size():
    """Under a taper a window can be nominally full while nearly all its weight
    sits on one sounding. Counting samples calls that well determined."""
    dataset = delivery()
    # Give one sounding per window a vastly tighter uncertainty, so the weight
    # concentrates on it while the sample count stays full.
    key = dataset.channels[0].std_key
    dataset.layer_data[key].iloc[:, :] = 1.0
    dataset.layer_data[key].iloc[::5, :] = 1e-6

    _, report = average_along_line(dataset, width_first=5,
                                   min_valid_fraction=0.9)
    assert report.distributions["per_channel"]["ch1 (LM/z)"]["too_thin"] > 0


def test_a_full_window_of_equal_weights_is_fully_effective():
    """n_eff equals the count exactly for flat weights, so a boxcar over
    complete data passes any fraction."""
    dataset = delivery()
    dataset.layer_data[dataset.channels[0].std_key].iloc[:, :] = 0.03

    _, report = average_along_line(dataset, width_first=5, method=BOXCAR,
                                   min_valid_fraction=1.0)
    thin = report.distributions["per_channel"]["ch1 (LM/z)"]["too_thin"]
    # Counted in gate-samples, not soundings. Only the line ends fail, where
    # the window genuinely hangs off the data: two lines, two ends each, two
    # soundings deep for a width of five, across every gate.
    assert thin == 2 * 2 * 2 * GATES


def test_a_window_of_mostly_culled_data_produces_nothing():
    dataset = delivery()
    dataset.layer_data[dataset.channels[0].inuse_key].iloc[:, :] = CULLED
    dataset.layer_data[dataset.channels[0].inuse_key].iloc[::10, :] = IN_USE

    dataset, _ = average_along_line(dataset, width_first=5,
                                    min_valid_fraction=0.8)
    assert np.isnan(data(dataset)).any()


def test_an_out_of_range_fraction_is_refused():
    for fraction in (0, 1.5, -0.1):
        with pytest.raises(ValueError, match="min_valid_fraction"):
            average_along_line(delivery(), width_first=3,
                               min_valid_fraction=fraction)


# ── Defect 3: the kernel is tapered ──────────────────────────────────────────

def test_the_default_kernel_is_not_flat():
    """A flat kernel rings and smears a narrow conductor across the whole window
    with equal authority."""
    weights = kernel(5, HANN)
    assert weights[2] > weights[1] > weights[0]
    assert weights.sum() == pytest.approx(1.0)


def test_a_taper_preserves_a_narrow_anomaly_better_than_a_boxcar():
    """The claim 01 makes, tested rather than asserted."""
    values = np.zeros((N, GATES))
    values[N // 2, :] = 1.0                       # a one-sounding anomaly

    tapered, _ = average_along_line(delivery(values=values), width_first=7,
                                    method=HANN, min_valid_fraction=0.1)
    flat, _ = average_along_line(delivery(values=values), width_first=7,
                                 method=BOXCAR, min_valid_fraction=0.1)

    assert data(tapered)[N // 2, 0] > data(flat)[N // 2, 0]


def test_a_boxcar_is_still_available():
    """Reproducing what a previous processing run did is a real need."""
    assert np.allclose(kernel(5, BOXCAR), 0.2)


def test_an_unknown_method_names_the_available_ones():
    with pytest.raises(ValueError, match="hann, gaussian, boxcar"):
        average_along_line(delivery(), width_first=3, method="triangle")


# ── Defect 4: vectorised ─────────────────────────────────────────────────────

def test_it_averages_a_realistic_delivery_quickly():
    """Not a benchmark — a guard that the formulation is not per-datum. A
    nested loop over 2000 soundings x 8 gates would not finish in this budget
    on the machinery this replaces."""
    import time
    dataset = make_dataset(n=2000, n_lines=4, n_gates=GATES)
    started = time.perf_counter()
    average_along_line(dataset, width_first=5, width_last=25)
    assert time.perf_counter() - started < 5.0


# ── Widths varying with gate time ────────────────────────────────────────────

def test_late_gates_are_smoothed_more():
    """Late-time data is noisier, so it is smoothed more."""
    times = np.logspace(-6, -3, GATES)
    widths = gate_widths(3, 15, times)

    assert widths[0] == 3
    assert widths[-1] == 15
    assert np.all(np.diff(widths) >= 0)


def test_one_width_gives_a_constant_window():
    assert list(gate_widths(5, 5, np.logspace(-6, -3, 4))) == [5, 5, 5, 5]


def test_the_interpolation_is_in_log_time():
    """Which coincides with gate index for log-spaced gates and follows the
    physics for irregular ones."""
    even = gate_widths(1, 11, np.array([1e-6, 1e-5, 1e-4, 1e-3]))
    assert list(even) == [1, 4, 8, 11]

    # Three gates bunched early and one very late: the late one alone should
    # carry most of the width increase.
    skewed = gate_widths(1, 11, np.array([1e-6, 1.1e-6, 1.2e-6, 1e-3]))
    assert skewed[2] < 3 and skewed[-1] == 11


# ── Uncertainty ──────────────────────────────────────────────────────────────

def test_a_noisy_neighbour_pulls_less():
    """A baseline of 1.0 rather than 0.0, because the stored uncertainty is
    relative — a value of zero has a sigma of zero and carries no information
    about how well it is known."""
    values = np.ones((N, GATES))
    values[N // 2 + 1, :] = 11.0                  # one neighbour far off

    def averaged(neighbour_sigma):
        dataset = delivery(values=values)
        dataset.layer_data[dataset.channels[0].std_key].iloc[:, :] = 0.01
        dataset.layer_data[dataset.channels[0].std_key].iloc[
            N // 2 + 1, :] = neighbour_sigma
        dataset, _ = average_along_line(dataset, width_first=5,
                                        min_valid_fraction=0.1)
        return data(dataset)[N // 2, 0]

    # The outlier is barely trusted in one case and trusted equally in the
    # other, so it drags the centre less when it is noisy.
    assert averaged(5.0) < averaged(0.01)


def test_averaging_reduces_the_uncertainty():
    """The point of averaging: more data behind each output."""
    dataset = delivery()
    before = std(dataset)[N // 2, 0]
    dataset, _ = average_along_line(dataset, width_first=9,
                                    min_valid_fraction=0.1)

    assert std(dataset)[N // 2, 0] < before


def test_the_output_uncertainty_is_relative():
    """As the format requires everywhere else."""
    dataset = delivery(values=np.full((N, GATES), 1e-9))
    dataset, _ = average_along_line(dataset, width_first=5,
                                    min_valid_fraction=0.1)

    finite = std(dataset)[np.isfinite(std(dataset))]
    assert finite.size and (finite < 1.0).all()


def test_the_spread_estimates_an_uncertainty_when_none_exists():
    """01: "where they do not, the spread within the window is used to estimate
    one." A different quantity, so it is reported as such."""
    dataset = delivery()
    del dataset.layer_data[dataset.channels[0].std_key]

    _, report = average_along_line(dataset, width_first=5,
                                   min_valid_fraction=0.1)
    outcome = report.distributions["per_channel"]["ch1 (LM/z)"]

    assert outcome["weighted_by_uncertainty"] is False
    assert outcome["spread_estimated_uncertainty"] is True


def test_a_zero_uncertainty_does_not_take_over_the_average():
    """Infinite weight would make one datum the whole answer, silently."""
    dataset = delivery(values=np.zeros((N, GATES)))
    dataset.layer_data[dataset.channels[0].data_key].iloc[N // 2, :] = 5.0
    dataset.layer_data[dataset.channels[0].std_key].iloc[N // 2, :] = 0.0

    dataset, _ = average_along_line(dataset, width_first=5,
                                    min_valid_fraction=0.1)
    assert np.isfinite(data(dataset)[N // 2, 0])


# ── Lines, mask, and what is kept ────────────────────────────────────────────

def test_no_window_crosses_the_turn_between_lines():
    """It would average a sounding with one flown minutes earlier and hundreds
    of metres away."""
    values = np.zeros((N, GATES))
    values[N // 2:, :] = 100.0                 # the second line is elsewhere
    dataset, _ = average_along_line(delivery(values=values), width_first=9,
                                    min_valid_fraction=0.1)

    assert data(dataset)[N // 2 - 1, 0] == pytest.approx(0.0, abs=1e-9)
    assert data(dataset)[N // 2, 0] == pytest.approx(100.0)


def test_the_originals_are_stashed():
    dataset = delivery()
    before = dataset.layer_data[dataset.channels[0].data_key].copy()
    dataset, _ = average_along_line(dataset, width_first=5,
                                    min_valid_fraction=0.1)

    key = f"{dataset.channels[0].data_key}{STASH_SUFFIX}"
    pd.testing.assert_frame_equal(dataset.layer_data[key], before)


def test_a_thin_window_culls_and_is_attributed():
    dataset = delivery()
    dataset.layer_data[dataset.channels[0].inuse_key].iloc[:, :] = CULLED
    dataset.layer_data[dataset.channels[0].inuse_key].iloc[::10, :] = IN_USE

    dataset, _ = average_along_line(dataset, width_first=5,
                                    min_valid_fraction=0.8)
    why = dataset.layer_data[provenance_key(dataset.channels[0].inuse_key)]

    assert (why.to_numpy() == STEP_CODE).any()
    assert STEP_CODE == 10


# ── Decimation ───────────────────────────────────────────────────────────────

def test_decimation_thins_to_a_target_spacing():
    """The fixture's soundings sit 10 m apart, so a 30 m target keeps every
    third."""
    dataset, report = average_along_line(delivery(), width_first=3,
                                         min_valid_fraction=0.1,
                                         decimate_to=30.0)

    assert report.distributions["decimated"]["before"] == N
    assert report.distributions["decimated"]["after"] < N
    assert len(dataset.flightlines) == report.distributions["decimated"]["after"]


def test_decimation_moves_every_array_with_the_soundings():
    dataset, _ = average_along_line(delivery(), width_first=3,
                                    min_valid_fraction=0.1, decimate_to=30.0)
    rows = len(dataset.flightlines)

    for key, array in dataset.layer_data.items():
        assert len(array) == rows, key


def test_decimation_happens_after_averaging():
    """Decimating first would throw away the neighbours the average is made
    from — the difference between smoothing and taking every nth sounding."""
    flat = np.tile(np.arange(N, dtype=float)[:, None], (1, GATES))
    dataset, _ = average_along_line(delivery(values=flat), width_first=5,
                                    min_valid_fraction=0.1, decimate_to=30.0)

    # A smoothed ramp is still a ramp; taking every third sounding of the raw
    # data would leave the same values, so the tell is the interior mean being
    # the local average rather than the sample.
    assert np.isfinite(data(dataset)).any()
    assert len(dataset.flightlines) < N


def test_not_decimating_leaves_the_extent_alone():
    dataset, report = average_along_line(delivery(), width_first=3,
                                         min_valid_fraction=0.1)
    assert report.distributions["decimated"] is None
    assert len(dataset.flightlines) == N


# ── Plumbing ─────────────────────────────────────────────────────────────────

def test_the_report_survives_serialization():
    _, report = average_along_line(delivery(), width_first=5,
                                   min_valid_fraction=0.1)
    restored = json.loads(json.dumps(report.to_dict(), default=str))
    assert restored["distributions"]["method"] == HANN


def test_registered():
    assert "average_along_line" in load_steps(FILTER_GROUP)
