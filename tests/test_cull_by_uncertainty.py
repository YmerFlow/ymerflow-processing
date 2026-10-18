"""filter_24 — points the processing already admitted it was unsure about.

The subtle part is not the comparison, it is what is being compared: the
uncertainty array is already a fraction, so this must not divide by the value
again. Doing so silently produces a number in units of one over the response and
culls almost nothing.
"""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.cull.uncertainty import (
    STEP_CODE, cull_by_uncertainty,
)
from ymerflow_processing.core.mask import CULLED, IN_USE, provenance_key
from ymerflow_processing.core.pipeline import FILTER_GROUP, load_steps

N = 5
GATES = 8


def delivery(uncertain=None, **kw):
    """A delivery at 3% throughout, with ``uncertain`` as ``{(s, g): value}``."""
    dataset = make_dataset(n=N, n_lines=1, n_gates=GATES, **kw)
    std = dataset.layer_data[dataset.channels[0].std_key]
    for (sounding, gate), value in (uncertain or {}).items():
        std.iloc[sounding, gate] = value
    return dataset


def mask(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].inuse_key].to_numpy()


# ── The comparison is direct ─────────────────────────────────────────────────

def test_a_threshold_reads_as_a_percentage():
    """0.15 means "more than fifteen percent uncertainty", which is what an
    operator means when they say it."""
    dataset, _ = cull_by_uncertainty(delivery({(1, 4): 0.20}),
                                     max_uncertainty=0.15)

    assert mask(dataset)[1, 4] == CULLED
    assert (mask(dataset)[0] == IN_USE).all()      # the rest sit at 3%


def test_the_uncertainty_is_not_divided_by_the_value_again():
    """The array is already a relative standard deviation. Dividing by a
    response of order 1e-9 would give ~1e7 and cull everything, or the inverse
    and cull nothing — either way silently."""
    dataset, report = cull_by_uncertainty(delivery(), max_uncertainty=0.15)

    assert (mask(dataset) == IN_USE).all()         # 0.03 is well under 0.15
    assert report.distributions["removed"] == 0

    strict, _ = cull_by_uncertainty(delivery(), max_uncertainty=0.01)
    assert (mask(strict) == CULLED).all()          # 0.03 is over 0.01


def test_the_threshold_is_inclusive():
    dataset, _ = cull_by_uncertainty(delivery({(1, 4): 0.15}),
                                     max_uncertainty=0.15)
    assert mask(dataset)[1, 4] == IN_USE


def test_the_uncertainty_distribution_is_reported():
    """So a delivery carrying absolute uncertainties shows magnitudes nothing
    like a percentage, rather than quietly under-culling."""
    _, report = cull_by_uncertainty(delivery(), max_uncertainty=0.15)
    summary = report.distributions["per_channel"]["ch1 (LM/z)"]["uncertainty"]

    assert summary["max"] == pytest.approx(0.03)
    assert summary["min"] == pytest.approx(0.03)


def test_a_point_with_no_uncertainty_is_not_flagged_here():
    """It is unusable rather than uncertain, which is filter_23's problem."""
    dataset, _ = cull_by_uncertainty(delivery({(2, 3): np.nan}),
                                     max_uncertainty=0.15)
    assert mask(dataset)[2, 3] == IN_USE


def test_it_changes_no_measurement_or_uncertainty():
    dataset = delivery({(1, 4): 0.9})
    arrays = {k: v.copy() for k, v in dataset.layer_data.items()
              if "inuse" not in k.lower()}

    dataset, _ = cull_by_uncertainty(dataset, max_uncertainty=0.15)

    for key, array in arrays.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


# ── Policy and plumbing ──────────────────────────────────────────────────────

def test_the_policy_applies_here_too():
    dataset, _ = cull_by_uncertainty(delivery({(2, 5): 0.9}),
                                     max_uncertainty=0.15,
                                     spread_earlier=1, disable_tail=True)

    assert (mask(dataset)[2, :4] == IN_USE).all()
    assert (mask(dataset)[2, 4:] == CULLED).all()


def test_observed_and_inferred_are_told_apart():
    dataset, report = cull_by_uncertainty(delivery({(2, 5): 0.9}),
                                          max_uncertainty=0.15, spread_earlier=1)
    codes = dataset.layer_data[
        provenance_key(dataset.channels[0].inuse_key)].to_numpy()

    assert codes[2, 5] == STEP_CODE
    assert codes[2, 4] == -STEP_CODE
    assert STEP_CODE == 24


def test_a_channel_without_uncertainties_is_noted_not_fatal():
    dataset = delivery()
    del dataset.layer_data[dataset.channels[0].std_key]

    dataset, report = cull_by_uncertainty(dataset, max_uncertainty=0.15)
    assert any("No uncertainty array" in n for n in report.notes)


def test_a_gate_already_disabled_is_not_claimed():
    dataset = delivery({(1, 5): 0.9})
    dataset.layer_data[dataset.channels[0].inuse_key].iloc[1, 5] = CULLED

    _, report = cull_by_uncertainty(dataset, max_uncertainty=0.15)
    assert report.distributions["per_channel"][
        "ch1 (LM/z)"]["over_threshold"] == 0


@pytest.mark.parametrize("threshold", [0, -0.1])
def test_a_non_positive_threshold_is_an_error(threshold):
    with pytest.raises(ValueError, match="must be positive"):
        cull_by_uncertainty(delivery(), max_uncertainty=threshold)


def test_it_is_idempotent():
    dataset, _ = cull_by_uncertainty(delivery({(1, 4): 0.9}),
                                     max_uncertainty=0.15)
    snapshot = {k: v.copy() for k, v in dataset.layer_data.items()}
    dataset, second = cull_by_uncertainty(dataset, max_uncertainty=0.15)

    for key, array in snapshot.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    assert second.distributions["removed"] == 0


def test_the_report_survives_serialization():
    _, report = cull_by_uncertainty(delivery({(1, 4): 0.9}),
                                    max_uncertainty=0.15)
    restored = json.loads(json.dumps(report.to_dict(), default=str))
    assert restored["step"] == "cull_by_uncertainty"
    assert restored["distributions"]["max_uncertainty"] == 0.15


def test_registered():
    assert "cull_by_uncertainty" in load_steps(FILTER_GROUP)
