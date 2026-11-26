"""Every per-channel parameter takes one value or a mapping by moment.

On a dual-moment survey the high moment is far cleaner than the low one at the
same gate times, so one threshold, window or starting gate cannot serve both,
and each moment is treated as independently as the steps allow. Every parameter
that acts per channel takes one value for every channel, as before, or a
mapping by channel key or moment name, resolved by
``Dataset.settings_per_channel``. The tests here run one fixture through every
such step: a scalar and a mapping saying the same thing agree exactly, each
moment may have its own value, and a mapping that names a moment the data
lacks, or leaves one out, is refused before anything is written.
"""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.coupling.reconstruction import FUZZY_FLOOR
from ymerflow_processing.filters.average import average_along_line
from ymerflow_processing.filters.cull_coupling import cull_by_coupling
from ymerflow_processing.filters.cull_curvature import cull_by_curvature
from ymerflow_processing.filters.cull_negatives import cull_negatives
from ymerflow_processing.filters.cull_slope import cull_by_slope
from ymerflow_processing.filters.cull_sparse_soundings import cull_sparse_soundings
from ymerflow_processing.filters.cull_uncertainty import cull_by_uncertainty
from ymerflow_processing.filters.noise import set_noise_floor

N, LINES, GATES = 120, 2, 12
LM, HM = "ch1 (LM/z)", "ch2 (HM/z)"


def delivery():
    """Dual-moment power-law decays, each moment carrying one of everything the
    steps look for: a smooth hump a basis cannot reproduce and whose shape is
    implausible, a sign flip, one gate far noisier than the rest, and a sounding
    with most of its gates missing."""
    dataset = make_dataset(moments=("LM", "HM"), n=N, n_lines=LINES, n_gates=GATES)
    position = np.linspace(0.0, 1.0, N)[:, None]
    for i, channel in enumerate(dataset.channels, start=1):
        times = np.asarray(channel.axis.values, dtype=float)
        amplitude = 1e-6 * i * (1.0 + 0.5 * np.sin(6.0 * position))
        exponent = 1.5 + 0.3 * position
        data = amplitude * (times / times[0]) ** -exponent
        data[20:26, 5:9] *= 1.6
        data[30, 7] = -1e-10
        data[50, 3:] = np.nan
        dataset.layer_data[channel.data_key] = pd.DataFrame(data)
        std = np.full(data.shape, 0.03)
        std[40, 6] = 0.5
        dataset.layer_data[channel.std_key] = pd.DataFrame(std)
    return dataset


def assert_same(a, b):
    """The two datasets hold the same arrays, attributes and noise model."""
    assert sorted(a.layer_data) == sorted(b.layer_data)
    for key in a.layer_data:
        pd.testing.assert_frame_equal(a.layer_data[key], b.layer_data[key],
                                      check_dtype=False)
    pd.testing.assert_frame_equal(a.flightlines, b.flightlines)
    assert a.survey.xyz.model_info == b.survey.xyz.model_info


def changed(a, b):
    try:
        assert_same(a, b)
    except AssertionError:
        return True
    return False


def arrays(dataset, which):
    """One channel's data, uncertainty and mask frames."""
    channel = dataset.channels[which]
    return [dataset.layer_data[k] for k in (channel.data_key, channel.std_key,
                                            channel.inuse_key)]


#: Each step with one scalar setting for every parameter it takes per channel,
#: and the values that leave the high moment as delivered when given to it alone.
STEPS = [
    pytest.param(cull_negatives,
                 {"start_gate": 3, "spread_earlier": 1, "disable_tail": True},
                 {"start_gate": GATES},
                 id="cull_negatives"),
    pytest.param(cull_by_uncertainty,
                 {"max_uncertainty": 0.1, "start_gate": 3, "spread_earlier": 1,
                  "disable_tail": False},
                 {"max_uncertainty": 0.9},
                 id="cull_by_uncertainty"),
    pytest.param(cull_by_slope,
                 {"shallowest": -1.2, "steepest": -4.0, "start_gate": 3,
                  "spread_earlier": 1, "disable_tail": True},
                 {"shallowest": 100.0, "steepest": -100.0},
                 id="cull_by_slope"),
    pytest.param(cull_by_curvature,
                 {"max_curvature": 1.0, "min_curvature": -5.0, "start_gate": 3,
                  "spread_earlier": 0, "disable_tail": False},
                 {"max_curvature": 1e6, "min_curvature": -1e6},
                 id="cull_by_curvature"),
    pytest.param(cull_by_coupling,
                 {"action": "cull", "n_components": 2, "min_confidence": 0.9,
                  "cull_below_confidence": None, "max_reconstruction_error": 2.5,
                  "max_uncertainty": 0.2, "window": 15.0, "start_gate": 3,
                  "spread_earlier": 1, "disable_tail": True},
                 {"max_reconstruction_error": 1e6},
                 id="cull_by_coupling"),
    pytest.param(cull_sparse_soundings, {"minimum_gates": 4}, {"minimum_gates": 1},
                 id="cull_sparse_soundings"),
    pytest.param(average_along_line,
                 {"width_first": 3, "width_last": 7, "min_valid_fraction": 0.5},
                 {"width_first": 1, "width_last": 1},
                 id="average_along_line"),
    pytest.param(average_along_line,
                 {"width_first": 3, "width_last": 7, "method": "robust_stack",
                  "min_valid_fraction": 0.5, "alpha": 0.1, "max_weight_factor": 10.0,
                  "independent_fraction": 0.5},
                 {"width_first": 1, "width_last": 1},
                 id="average_along_line-robust_stack"),
    pytest.param(set_noise_floor, {"late_gates": 4}, None, id="set_noise_floor"),
]

#: The parameters that stay one value for the step: a mode, a name, or a
#: setting about whole soundings rather than one channel's gates.
SHARED = {"action", "confidence_column", "method", "decimate_to", "line_column",
          "stash", "scope", "noise_level_1ms", "noise_exponent", "fit_exponent"}


def by_name(kw, **overrides):
    """``kw`` with every per-channel parameter spelled as a mapping by moment name."""
    spelled = {name: value if name in SHARED else {"LM": value, "HM": value}
               for name, value in kw.items()}
    spelled.update(overrides)
    return spelled


def by_key(kw):
    return {name: value if name in SHARED else {"ch1": value, "ch2": value}
            for name, value in kw.items()}


PER_CHANNEL = [pytest.param(*p.values[:2], name, id=f"{p.id}-{name}")
               for p in STEPS for name in p.values[1] if name not in SHARED]


# ── A scalar is every channel's value ────────────────────────────────────────

@pytest.mark.parametrize("step, kw, alone", STEPS)
def test_one_value_and_a_mapping_naming_every_moment_agree(step, kw, alone):
    """A document written with one value runs exactly as before, and a mapping
    saying the same thing by moment name or by channel key gives the same
    result, array for array."""
    scalar, _ = step(delivery(), **kw)
    named, _ = step(delivery(), **by_name(kw))
    keyed, _ = step(delivery(), **by_key(kw))

    assert changed(scalar, delivery())
    assert_same(named, scalar)
    assert_same(keyed, scalar)


# ── Each moment on its own ───────────────────────────────────────────────────

@pytest.mark.parametrize("step, kw, alone",
                         [p for p in STEPS if p.values[2] is not None])
def test_each_moment_may_have_its_own_value(step, kw, alone):
    """With the high moment's values set past everything in the fixture it is
    left as delivered, while the low moment is treated exactly as before."""
    both, _ = step(delivery(), **kw)
    split, report = step(delivery(), **by_name(
        kw, **{name: {"LM": kw[name], "HM": value} for name, value in alone.items()}))

    for after, same in zip(arrays(split, 0), arrays(both, 0)):
        pd.testing.assert_frame_equal(after, same, check_dtype=False)
    # Where the delivery has a value. A width of one keeps every value but has
    # nothing to say about a datum with none, and culls it as too thin.
    before = arrays(delivery(), 1)
    finite = np.isfinite(before[0].to_numpy())
    for after, delivered in zip(arrays(split, 1), before):
        np.testing.assert_allclose(after.to_numpy(dtype=float)[finite],
                                   delivered.to_numpy(dtype=float)[finite], rtol=1e-12)


def test_the_noise_estimate_may_use_a_different_tail_per_moment():
    """The moments' gate counts differ, so the last N gates differ too. The
    floor itself stays one value for the dataset."""
    _, report = set_noise_floor(delivery(), late_gates={"LM": 3, "HM": 6})
    per_channel = report.distributions["per_channel"]

    assert per_channel[LM]["late_gates"] == 3 and len(per_channel[LM]["gates_used"]) == 3
    assert per_channel[HM]["late_gates"] == 6 and len(per_channel[HM]["gates_used"]) == 6
    assert isinstance(report.distributions["model"]["level_1ms"], float)


def test_each_channel_s_outcome_records_what_it_got():
    """The top-level report keeps the parameters as given; each channel's
    outcome has the values applied to it, the floor filled in."""
    _, report = cull_by_coupling(delivery(), n_components=2,
                                 min_confidence={"LM": 0.9, "HM": 0.95},
                                 start_gate={"LM": 2, "HM": 4})
    own = report.distributions["per_channel"][HM]["settings"]

    assert own["min_confidence"] == 0.95 and own["start_gate"] == 4
    assert own["cull_below_confidence"] == pytest.approx(FUZZY_FLOOR * 0.95)
    assert report.distributions["thresholds"]["min_confidence"] == {"LM": 0.9, "HM": 0.95}
    assert report.distributions["policy"]["start_gate"] == {"LM": 2, "HM": 4}
    json.dumps(report.to_dict(), default=str)


# ── Refused before anything is written ───────────────────────────────────────

@pytest.mark.parametrize("step, kw, name", PER_CHANNEL)
def test_a_value_for_a_moment_the_data_lacks_is_refused(step, kw, name):
    with pytest.raises(KeyError, match="No channel 'XM'"):
        step(delivery(), **{**kw, name: {"LM": kw[name], "XM": kw[name]}})


@pytest.mark.parametrize("step, kw, name", PER_CHANNEL)
def test_a_mapping_must_name_every_moment_and_nothing_is_written_first(step, kw, name):
    """A moment the mapping is silent about would otherwise be treated with
    some default, or not at all, without a word."""
    dataset = delivery()
    with pytest.raises(ValueError, match=f"{name} names some channels but not"):
        step(dataset, **{**kw, name: {"HM": kw[name]}})
    assert_same(dataset, delivery())
