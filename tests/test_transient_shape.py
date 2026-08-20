"""filter_25 to filter_28, and the shape machinery they share.

The helper is the substance here. A decay rate is a difference between
neighbours, and a culled gate is not a neighbour — getting that wrong produces a
plausible number rather than an error, which is why most of these assertions are
about the arithmetic rather than the culling.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.cull.filter_25_decay_rate import (
    cull_by_decay_rate,
)
from ymerflow_processing.core.cull.filter_27_curvature import (
    cull_by_curvature,
)
from ymerflow_processing.core.mask import CULLED, IN_USE, provenance_key
from ymerflow_processing.core.pipeline import FILTER_GROUP, load_steps
from ymerflow_processing.core.transient import (
    log_curvature, log_slope, neighbours,
)

N = 6
GATES = 7


def power_law(exponent=-2.5, n=N, gates=GATES):
    """A pure power law: slope is exactly ``exponent``, curvature exactly 0."""
    times = np.logspace(-5, -3, gates)
    return np.tile((times ** exponent)[None, :], (n, 1)), times


def delivery(exponent=-2.5, **kw):
    """Built against the dataset's *own* gate times.

    Generating values on one time vector while the filter measures against the
    channel's axis produces a slope that is neither the constructed one nor an
    error — which is exactly the class of mistake this filter exists to catch,
    so a test that made it would be testing nothing.
    """
    dataset = make_dataset(n=N, n_lines=1, n_gates=GATES, **kw)
    for channel in dataset.channels:
        times = np.asarray(channel.axis.values, dtype=float)
        dataset.layer_data[channel.data_key].iloc[:, :] = np.tile(
            (times ** exponent)[None, :], (N, 1))
    return dataset


def mask(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].inuse_key].to_numpy()


# ── The helper ───────────────────────────────────────────────────────────────

def test_the_slope_of_a_power_law_is_its_exponent():
    """In log-log a power law is a straight line, and its slope is the decay
    exponent. That is what makes an out-of-range slope a physical statement."""
    values, times = power_law(-2.5)
    slope = log_slope(values, times)

    assert np.allclose(slope[0, 1:-1], -2.5)


def test_the_curvature_of_a_power_law_is_zero():
    values, times = power_law(-2.5)
    curvature = log_curvature(values, times)

    assert np.allclose(curvature[0, 1:-1], 0.0, atol=1e-9)


def test_a_culled_gate_is_not_a_neighbour():
    """The whole reason this is a shared helper. With gate 3 gone, the slope at
    gate 4 is measured against gate 2 — and on a power law it is still exact,
    which a fixed-offset difference would not be."""
    values, times = power_law(-2.5)
    live = np.ones(values.shape, dtype=bool)
    live[0, 3] = False

    slope = log_slope(values, times, live)
    previous, _ = neighbours(live)

    assert previous[0, 4] == 2
    assert slope[0, 4] == pytest.approx(-2.5)
    assert np.isnan(slope[0, 3])


def test_uneven_gate_spacing_is_handled_rather_than_assumed_away():
    """y0 - 2y1 + y2 assumes even spacing. Gates are logarithmic, so it is
    wrong even before anything is culled."""
    times = np.array([1e-5, 2e-5, 8e-5, 1e-3])
    values = np.tile((times ** -2.0)[None, :], (2, 1))

    assert np.allclose(log_slope(values, times)[0, 1:-1], -2.0)
    assert np.allclose(log_curvature(values, times)[0, 1:-1], 0.0, atol=1e-9)


def test_the_ends_are_unmeasurable_not_zero():
    """A central difference needs a neighbour on each side."""
    values, times = power_law()
    slope = log_slope(values, times)

    assert np.isnan(slope[0, 0]) and np.isnan(slope[0, -1])


def test_a_non_positive_value_is_unmeasurable():
    """log of it does not exist, and a negative measurement is filter_32's
    finding — treating it as a shape problem would attribute it wrongly."""
    values, times = power_law()
    values = values.copy()
    values[0, 3] = -1.0

    assert np.isnan(log_slope(values, times)[0, 3])
    assert np.isnan(log_curvature(values, times)[0, 3])


def test_a_sounding_with_too_few_survivors_measures_nothing():
    values, times = power_law()
    live = np.zeros(values.shape, dtype=bool)
    live[0, 2] = True

    assert np.isnan(log_slope(values, times, live)[0]).all()


def test_curvature_sees_a_change_of_rate():
    """The case slope alone passes: every individual slope plausible, the
    sequence of them not."""
    times = np.logspace(-5, -3, GATES)
    values = np.tile((times ** -2.0)[None, :], (N, 1)).copy()
    values[:, 3:] *= (times[3:] / times[3]) ** -2.0      # rate doubles at gate 3

    curvature = log_curvature(values, times)
    assert abs(curvature[0, 3]) > 0.5


# ── filter_25 / filter_26 ────────────────────────────────────────────────────

def test_a_plausible_transient_is_left_alone():
    dataset, report = cull_by_decay_rate(delivery(-2.5), shallowest=-0.5,
                                         steepest=-4.0)

    assert (mask(dataset) == IN_USE).all()
    assert report.distributions["removed"] == 0


def test_too_shallow_is_culled():
    dataset, _ = cull_by_decay_rate(delivery(-0.2), shallowest=-0.5)
    assert (mask(dataset)[:, 1:-1] == CULLED).all()


def test_too_steep_is_culled():
    dataset, _ = cull_by_decay_rate(delivery(-6.0), steepest=-4.0)
    assert (mask(dataset)[:, 1:-1] == CULLED).all()


def test_the_bounds_are_signed_and_named_for_what_they_mean():
    """"Maximum" and "minimum" read backwards for a negative slope."""
    steep = delivery(-6.0)
    kept, _ = cull_by_decay_rate(steep, shallowest=-0.5)   # only the other bound
    assert (mask(kept) == IN_USE).all()


def test_bounds_the_wrong_way_round_are_refused():
    """Written that way they exclude every slope."""
    with pytest.raises(ValueError, match="must be below shallowest"):
        cull_by_decay_rate(delivery(), shallowest=-4.0, steepest=-0.5)


def test_neither_bound_is_an_error():
    with pytest.raises(ValueError, match="needs shallowest or steepest"):
        cull_by_decay_rate(delivery())


def test_an_unmeasurable_gate_is_not_an_objection():
    """"The shape is unmeasurable here" is a different statement from "the shape
    is wrong", and only the second may cull."""
    dataset, report = cull_by_decay_rate(delivery(-2.5), shallowest=-0.5,
                                         steepest=-4.0)
    outcome = report.distributions["per_channel"]["ch1 (LM/z)"]

    assert outcome["unmeasurable"] == N * 2          # both end gates
    assert outcome["implausible"] == 0
    assert (mask(dataset)[:, 0] == IN_USE).all()


def test_the_slope_distribution_is_reported():
    """The number someone needs to choose a bound."""
    _, report = cull_by_decay_rate(delivery(-2.5), shallowest=-0.5)
    summary = report.distributions["per_channel"]["ch1 (LM/z)"]["slope"]

    assert summary["mean"] == pytest.approx(-2.5)


# ── filter_27 / filter_28 ────────────────────────────────────────────────────

def test_a_straight_transient_has_no_curvature_to_object_to():
    dataset, report = cull_by_curvature(delivery(-2.5), max_curvature=0.5,
                                        min_curvature=-0.5)

    assert (mask(dataset) == IN_USE).all()
    assert report.distributions["removed"] == 0


def test_a_lurching_rate_is_culled():
    dataset = delivery(-2.0)
    for channel in dataset.channels:
        times = np.asarray(channel.axis.values, dtype=float)
        values = dataset.layer_data[channel.data_key].to_numpy().copy()
        values[:, 3:] *= (times[3:] / times[3]) ** -2.0
        dataset.layer_data[channel.data_key].iloc[:, :] = values

    dataset, report = cull_by_curvature(dataset, max_curvature=0.5,
                                        min_curvature=-0.5)
    assert report.distributions["removed"] > 0


def test_curvature_bounds_the_wrong_way_round_are_refused():
    with pytest.raises(ValueError, match="is above max_curvature"):
        cull_by_curvature(delivery(), max_curvature=-1.0, min_curvature=1.0)


def test_neither_curvature_bound_is_an_error():
    with pytest.raises(ValueError, match="needs max_curvature"):
        cull_by_curvature(delivery())


# ── Shared behaviour ─────────────────────────────────────────────────────────

def test_both_change_no_measurement_or_uncertainty():
    for step, kwargs in ((cull_by_decay_rate, {"shallowest": -0.5}),
                         (cull_by_curvature, {"max_curvature": 0.001})):
        dataset = delivery(-0.2)
        arrays = {k: v.copy() for k, v in dataset.layer_data.items()
                  if "inuse" not in k.lower()}
        dataset, _ = step(dataset, **kwargs)
        for key, array in arrays.items():
            pd.testing.assert_frame_equal(dataset.layer_data[key], array)


def test_both_are_idempotent():
    for step, kwargs in ((cull_by_decay_rate, {"shallowest": -0.5}),
                         (cull_by_curvature, {"max_curvature": 0.001})):
        dataset, _ = step(delivery(-0.2), **kwargs)
        snapshot = {k: v.copy() for k, v in dataset.layer_data.items()}
        dataset, second = step(dataset, **kwargs)

        for key, array in snapshot.items():
            pd.testing.assert_frame_equal(dataset.layer_data[key], array)
        assert second.distributions["removed"] == 0


def test_the_policy_applies_to_both():
    dataset, report = cull_by_decay_rate(delivery(-0.2), shallowest=-0.5,
                                         disable_tail=True)
    assert report.distributions["policy"]["disable_tail"] is True


def test_they_are_attributed_separately():
    decayed, _ = cull_by_decay_rate(delivery(-0.2), shallowest=-0.5)
    curved, _ = cull_by_curvature(delivery(-0.2), max_curvature=-1e9)

    first = decayed.layer_data[provenance_key(decayed.channels[0].inuse_key)]
    second = curved.layer_data[provenance_key(curved.channels[0].inuse_key)]

    assert (first.to_numpy() == 25).any()
    assert (second.to_numpy() == 27).any()


def test_the_reports_survive_serialization():
    _, decayed = cull_by_decay_rate(delivery(-0.2), shallowest=-0.5)
    _, curved = cull_by_curvature(delivery(-0.2), max_curvature=0.001)

    for report in (decayed, curved):
        restored = json.loads(json.dumps(report.to_dict(), default=str))
        assert restored["distributions"]["provenance_code"] in (25, 27)


@pytest.mark.parametrize("name", ["cull_by_decay_rate", "cull_by_curvature"])
def test_registered(name):
    assert name in load_steps(FILTER_GROUP)
