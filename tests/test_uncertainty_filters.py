"""filter_11 and filter_13 — adjusting uncertainty, and drawing noise from it.

filter_11 adjusts what is there; filter_12 asserts a model. filter_13 turns a
noise-free forward model into something an inversion can be fairly tested on,
and most of its substance is making a wrong ordering detectable rather than
silently plausible.
"""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.filters.inflate import inflate_uncertainty
from ymerflow_processing.dataset import stash_key
from ymerflow_processing.filters.add_noise import DEFAULT_SEED, applied, add_noise
from ymerflow_processing.filters.uncertainty import assign_uncertainty
from ymerflow_processing.filters.noise import set_noise_floor, REFERENCE_TIME
from ymerflow_processing.pipeline import (
    FILTER_GROUP, Pipeline, ProcessingRunner, load_steps,
)

N = 40
GATES = 8


def delivery(**kw):
    return make_dataset(n=N, n_lines=2, n_gates=GATES, **kw)


def realistic(seed=3):
    """A decaying transient over ambient noise, so a model can be estimated."""
    dataset = delivery()
    rng = np.random.default_rng(seed)
    times = np.asarray(dataset.channels[0].axis.values, dtype=float)
    for channel in dataset.channels:
        response = 1e-6 * (times / times[0]) ** -2.0
        ambient = rng.normal(0.0, 5e-10 * (times / REFERENCE_TIME) ** -0.5,
                             (N, GATES))
        dataset.layer_data[channel.data_key].iloc[:, :] = response[None, :] + ambient
    return dataset


def std(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].std_key].to_numpy()


# ── filter_11 ────────────────────────────────────────────────────────────────

def test_it_adds_a_fraction_arithmetically():
    """"Add two percent" moves 3% to 5%, not to 3.6%. A number someone chose is
    not an independent error, so it does not combine like one."""
    dataset, _ = inflate_uncertainty(delivery(), fraction=0.02)
    assert std(dataset)[0, 0] == pytest.approx(0.05)


def test_a_negative_fraction_reduces():
    dataset, _ = inflate_uncertainty(delivery(), fraction=-0.01)
    assert std(dataset)[0, 0] == pytest.approx(0.02)


def test_it_cannot_produce_a_negative_uncertainty():
    """One travelling downstream would be squared by something and become a
    plausible positive again, which is the worst kind of wrong."""
    dataset, report = inflate_uncertainty(delivery(), fraction=-0.5)

    assert np.allclose(std(dataset), 0.0)
    assert any("clamped there" in n for n in report.notes)
    assert report.distributions["per_channel"]["ch1 (LM/z)"][
        "clamped_at_zero"] == N * GATES


def test_a_gate_range_limits_where_it_applies():
    dataset, _ = inflate_uncertainty(delivery(), fraction=0.02, gates="0-2")

    assert np.allclose(std(dataset)[:, :3], 0.05)
    assert np.allclose(std(dataset)[:, 3:], 0.03)


def test_the_gate_idiom_is_filter_21s():
    """Learned once, works everywhere (R7)."""
    dataset, _ = inflate_uncertainty(delivery(), fraction=0.02, gates=[0, 7])

    assert std(dataset)[0, 0] == pytest.approx(0.05)
    assert std(dataset)[0, 7] == pytest.approx(0.05)
    assert std(dataset)[0, 3] == pytest.approx(0.03)


def test_it_can_apply_per_channel():
    dataset, _ = inflate_uncertainty(delivery(), fraction={"ch1": 0.02})

    assert std(dataset, 0)[0, 0] == pytest.approx(0.05)
    assert std(dataset, 1)[0, 0] == pytest.approx(0.03)


def test_it_changes_no_measurement_or_mask():
    dataset = delivery()
    keep = {k: v.copy() for k, v in dataset.layer_data.items()
            if "std" not in k.lower()}

    dataset, _ = inflate_uncertainty(dataset, fraction=0.02)

    for key, array in keep.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


def test_no_fraction_is_an_error():
    with pytest.raises(ValueError, match="needs a fraction"):
        inflate_uncertainty(delivery(), fraction=None)


def test_it_adjusts_rather_than_asserting():
    """The difference from filter_12, which replaces with a physical model.
    Running 11 twice compounds; that is what "adjust" means."""
    dataset, _ = inflate_uncertainty(delivery(), fraction=0.01)
    dataset, _ = inflate_uncertainty(dataset, fraction=0.01)
    assert std(dataset)[0, 0] == pytest.approx(0.05)


# ── filter_13 ────────────────────────────────────────────────────────────────

def test_it_perturbs_the_values():
    dataset = delivery()
    before = dataset.layer_data[dataset.channels[0].data_key].to_numpy().copy()

    dataset, _ = add_noise(dataset, seed=1)
    after = dataset.layer_data[dataset.channels[0].data_key].to_numpy()

    # atol=0: the default 1e-8 swamps a response of order 1e-9, so every
    # comparison would read as close and the test would pass on a no-op.
    assert not np.allclose(before, after, atol=0)


def test_the_spread_scales_with_the_value_not_the_relative_figure():
    """The stored uncertainty is relative, so drawing from it directly would put
    noise of order one on every datum whatever its size."""
    dataset = delivery()
    key = dataset.channels[0].data_key
    dataset.layer_data[key].iloc[:, :] = 1e-6            # a strong response
    dataset, _ = add_noise(dataset, seed=1)

    residual = dataset.layer_data[key].to_numpy() - 1e-6
    # 3% of 1e-6 is 3e-8; the scatter should be of that order, not of order 0.03.
    assert np.std(residual) == pytest.approx(3e-8, rel=0.3)


def test_the_same_seed_repeats_exactly():
    """"A fair test" is not a fair test if it cannot be repeated."""
    one, _ = add_noise(delivery(), seed=7)
    two, _ = add_noise(delivery(), seed=7)

    np.testing.assert_array_equal(
        one.layer_data[one.channels[0].data_key].to_numpy(),
        two.layer_data[two.channels[0].data_key].to_numpy())


def test_a_different_seed_gives_different_noise():
    one, _ = add_noise(delivery(), seed=7)
    two, _ = add_noise(delivery(), seed=8)

    assert not np.allclose(
        one.layer_data[one.channels[0].data_key].to_numpy(),
        two.layer_data[two.channels[0].data_key].to_numpy(), atol=0)


def test_the_noise_free_values_are_stashed():
    dataset = delivery()
    key = dataset.channels[0].data_key
    before = dataset.layer_data[key].copy()

    dataset, _ = add_noise(dataset, seed=1)

    pd.testing.assert_frame_equal(dataset.layer_data[stash_key(key)], before)


def test_it_changes_no_uncertainty_or_mask():
    dataset = delivery()
    keep = {k: v.copy() for k, v in dataset.layer_data.items()
            if "std" in k.lower() or "inuse" in k.lower()}

    dataset, _ = add_noise(dataset, seed=1)

    for key, array in keep.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


# ── Ordering, which is the hazard ────────────────────────────────────────────

def test_it_refuses_to_run_twice():
    """The classic silent error: the result is noisier by sqrt(2) and looks
    exactly like data."""
    dataset, _ = add_noise(delivery(), seed=1)

    with pytest.raises(ValueError, match="already carries synthetic noise"):
        add_noise(dataset, seed=2)


def test_a_second_pass_can_be_asked_for_explicitly():
    dataset, _ = add_noise(delivery(), seed=1)
    dataset, report = add_noise(dataset, seed=2, again=True)

    assert report.distributions["repeated"] is True


def test_the_application_is_recorded_with_its_seed():
    dataset, _ = add_noise(delivery(), seed=42)
    assert applied(dataset) == "seed=42"


def test_uniform_uncertainties_are_called_out():
    """The signature of a default from the system description rather than an
    assigned model — noise drawn from it is flat in relative terms, so late
    gates come out as clean as early ones."""
    _, report = add_noise(delivery(), seed=1)

    assert any("uniform default" in n for n in report.notes)
    assert report.distributions["per_channel"]["ch1 (LM/z)"][
        "uncertainty_uniform"] is True


def test_an_assigned_model_is_not_called_out():
    """A warning that fires on correct use is worse than none."""
    dataset, _ = set_noise_floor(realistic())
    dataset, _ = assign_uncertainty(dataset)
    _, report = add_noise(dataset, seed=1)

    assert not any("uniform default" in n for n in report.notes)


def test_the_seed_defaults_to_a_legible_marker():
    """42 is not a better number — no seed is. It is the right default because
    it reads as "nobody chose this", where an arbitrary-looking number reads as
    though someone did and had a reason."""
    _, report = add_noise(delivery())

    assert DEFAULT_SEED == 42
    assert report.distributions["seed"] == 42
    assert report.distributions["seed_source"] == "default"
    assert any("need their own seed" in n for n in report.notes)


def test_a_given_seed_is_recorded_as_given():
    """So "seed=42" in a file is distinguishable from a deliberate 42."""
    _, report = add_noise(delivery(), seed=7)

    assert report.distributions["seed_source"] == "given"
    assert not any("need their own seed" in n for n in report.notes)


def test_naming_channels_does_not_disturb_the_seed_source():
    """A name collision here would have made every channel-limited run report
    its seed as defaulted."""
    _, report = add_noise(delivery(), seed=7, channels=["ch1"])
    assert report.distributions["seed_source"] == "given"


def test_an_explicit_none_seed_is_still_an_error():
    with pytest.raises(ValueError, match="cannot be None"):
        add_noise(delivery(), seed=None)


def test_the_default_gives_every_realization_the_same_noise():
    """What a default cannot do is make two realizations independent. Detectable
    rather than silent, but the reason anything wanting independence must ask."""
    one, _ = add_noise(delivery())
    two, _ = add_noise(delivery())

    np.testing.assert_array_equal(
        one.layer_data[one.channels[0].data_key].to_numpy(),
        two.layer_data[two.channels[0].data_key].to_numpy())


# ── The chain they belong to ─────────────────────────────────────────────────

def test_the_whole_synthetic_chain_runs_in_order():
    """estimate -> assign -> perturb, which is the order 01 says matters."""
    dataset, report = ProcessingRunner(Pipeline.parse("""
        kind: filters
        steps:
          - set_noise_floor: {}
          - inflate_uncertainty: {fraction: 0.01}
          - add_noise: {seed: 11}
    """)).run(realistic())

    ran = [s.step for s in report.steps]
    assert ran[-3:] == ["set_noise_floor", "inflate_uncertainty",
                        "add_noise"]
    assert applied(dataset) == "seed=11"


def test_late_gates_get_noise_larger_than_the_datum():
    """Correct, and what makes the synthetic look like real late-time data: the
    floor has overtaken the signal, so the assigned uncertainty passed one."""
    dataset, _ = set_noise_floor(realistic())
    dataset, _ = assign_uncertainty(dataset)
    key = dataset.channels[0].data_key
    clean = dataset.layer_data[key].to_numpy().copy()

    dataset, _ = add_noise(dataset, seed=5)
    residual = np.abs(dataset.layer_data[key].to_numpy() - clean)

    assert np.median(residual[:, -1] / np.abs(clean[:, -1])) > 1.0
    assert np.median(residual[:, 0] / np.abs(clean[:, 0])) < 0.5


def test_the_reports_survive_serialization():
    dataset, inflated = inflate_uncertainty(delivery(), fraction=0.02)
    dataset, perturbed = add_noise(dataset, seed=1)

    for report in (inflated, perturbed):
        restored = json.loads(json.dumps(report.to_dict(), default=str))
        assert restored["distributions"]["per_channel"]


@pytest.mark.parametrize("name", ["inflate_uncertainty",
                                  "add_noise"])
def test_registered(name):
    assert name in load_steps(FILTER_GROUP)


# ── A raw delivery carries no uncertainties ──────────────────────────────────
# Found on the first real delivery: assign_uncertainty refused to act without an
# array to overwrite, so the whole noise half of the standard chain was dead on
# exactly the data it exists for.

def _raw(n=8, gates=5):
    """A delivery with data and no uncertainty array, as a real one arrives."""
    from conftest import make_dataset
    dataset = make_dataset(n=n, n_lines=1, n_gates=gates)
    for channel in dataset.channels:
        del dataset.layer_data[channel.std_key]
    return dataset


def _with_model(dataset):
    from ymerflow_processing.filters.noise import set_noise_floor
    dataset, _ = set_noise_floor(
        dataset, noise_level_1ms=5.0e-10, noise_exponent=-0.5)
    return dataset


def test_an_absent_uncertainty_array_is_created_rather_than_skipped():
    """Assigning uncertainty is precisely how it comes to exist."""
    from ymerflow_processing.filters.uncertainty import assign_uncertainty

    dataset = _with_model(_raw())
    assert all(c.std_key is None for c in dataset.channels)

    dataset, report = assign_uncertainty(dataset)
    assert all(c.std_key is not None for c in dataset.channels)
    assert any("created from the noise model" in n for n in report.notes)


def test_the_created_array_is_discoverable_afterwards():
    """Created under a name the next lookup finds, so nothing is kept in sync —
    Channel is frozen and channels rediscover on every access."""
    from ymerflow_processing.filters.uncertainty import assign_uncertainty

    dataset, _ = assign_uncertainty(_with_model(_raw()))
    for channel in dataset.channels:
        assert channel.std_key in dataset.layer_data
        assert dataset.layer_data[channel.std_key].shape == \
            dataset.layer_data[channel.data_key].shape


def test_creation_reports_no_before_distribution():
    """There is nothing to compare against, so it is absent rather than a
    summary of an array that never existed."""
    from ymerflow_processing.filters.uncertainty import assign_uncertainty

    _, report = assign_uncertainty(_with_model(_raw()))
    for entry in report.distributions["per_channel"].values():
        assert entry["created"] is True
        assert entry["before"] is None
        assert entry["after"]["n"] > 0


def test_an_existing_array_is_still_stashed_and_overwritten():
    """The ordinary path is unchanged."""
    from ymerflow_processing.filters.uncertainty import assign_uncertainty
    from conftest import make_dataset

    dataset = _with_model(make_dataset(n=8, n_lines=1, n_gates=5))
    key = dataset.channels[0].std_key
    dataset, report = assign_uncertainty(dataset)

    assert stash_key(key) in dataset.layer_data
    entry = report.distributions["per_channel"][str(dataset.channels[0])]
    assert entry["created"] is False and entry["before"] is not None


def test_a_channel_whose_gate_times_do_not_match_is_still_refused():
    """Gate times come from the system description, so a mismatch is a
    description that does not describe this data — not something to invent
    around."""
    from ymerflow_processing.filters.uncertainty import assign_uncertainty

    dataset = _with_model(_raw())
    channel = dataset.channels[0]
    dataset.layer_data[channel.data_key] = \
        dataset.layer_data[channel.data_key].iloc[:, :2]

    _, report = assign_uncertainty(dataset)
    assert any("does not describe this data" in n for n in report.notes)


def test_the_role_key_follows_whatever_convention_the_data_uses():
    """Both numbered conventions put the role in the same position; a named one
    has no number to assemble from, so the key is derived from the data key."""
    from ymerflow_processing.dataset import role_key

    class Fake:
        def __init__(self, data_key):
            self.data_key = data_key

    assert role_key(Fake("dbdt_ch1gt"), "std") == "dbdt_std_ch1gt"
    assert role_key(Fake("Gate_Ch01"), "std") == "STD_Ch01"
    assert role_key(Fake("Gate_Ch01"), "inuse") == "InUse_Ch01"
    assert role_key(Fake("LM_Z_G01"), "std") == "LM_Z_G01_std"
    assert role_key(Fake("LM_Z_G01"), "inuse") == "LM_Z_G01_inuse"


def test_the_role_key_is_never_the_data_key_whatever_the_case():
    """A delivery spelling its prefix `DBDT_` must still get a distinct key.
    The old case-sensitive replace found nothing and handed back the data key,
    so the uncertainties would have been written over the measurements."""
    from ymerflow_processing.dataset import role_key

    class Fake:
        def __init__(self, data_key):
            self.data_key = data_key

    assert role_key(Fake("DBDT_Ch1gt"), "std") == "dbdt_std_Ch1gt"
    assert role_key(Fake("raw_DBDT_Ch1gt"), "inuse") == "raw_dbdt_inuse_Ch1gt"
    for key in ("DBDT_Ch1gt", "raw_DBDT_Ch1gt", "GATE_CH01"):
        assert role_key(Fake(key), "std") != key
