"""cull_by_coupling: the table, the switch, the basis and the criterion.

The detector writes one table of its own every run, and ``action`` decides only
whether the table reaches the mask. Most of what is tested here is that table
behaving like the library's other derived arrays, because a new gate-shaped
family that only some detectors know about is how discovery breaks.
"""

import re

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.coupling import table
from ymerflow_processing.coupling.reconstruct import PCAReconstructor
from ymerflow_processing.coupling.reconstruction import along_line_mean
from ymerflow_processing.filters.cull_coupling import STEP_CODE, cull_by_coupling
from ymerflow_processing.filters.select_moments import select_moments, unclaimed_arrays
from ymerflow_processing.inventory import take
from ymerflow_processing.mask import provenance, provenance_key
from ymerflow_processing.methods.tem import TEM_NAMING

N, LINES, GATES = 120, 2, 12
COUPLED = slice(20, 26)   # six soundings on the first line


def survey(moments=("LM",), coupled=True, std=0.03, hump_gates=slice(5, 9),
           rows=COUPLED):
    """A 1D-looking survey: power-law decays whose amplitude and exponent drift
    smoothly along the line, which a two-component log basis reproduces
    exactly. ``coupled`` raises a smooth hump on a few soundings' mid gates,
    the elevated-but-plausible signature the shape tests cannot see."""
    dataset = make_dataset(moments=moments, n=N, n_lines=LINES, n_gates=GATES)
    times = np.asarray(dataset.channels[0].axis.values, dtype=float)
    position = np.linspace(0.0, 1.0, N)[:, None]
    for i, channel in enumerate(dataset.channels, start=1):
        amplitude = 1e-6 * i * (1.0 + 0.5 * np.sin(6.0 * position))
        exponent = 1.5 + 0.3 * position
        data = amplitude * (times / times[0]) ** -exponent
        if coupled:
            data[rows, hump_gates] *= 1.6
        dataset.layer_data[channel.data_key] = pd.DataFrame(data)
        values = np.full(data.shape, std) if np.isscalar(std) else std
        dataset.layer_data[channel.std_key] = pd.DataFrame(values)
    return dataset


def flag(dataset, **kw):
    kw.setdefault("n_components", 2)
    return cull_by_coupling(dataset, action="flag", **kw)


def coupling_table(dataset, channel=0):
    return dataset.layer_data[table.channel_key(dataset.channels[channel])].to_numpy()


# ── The table ────────────────────────────────────────────────────────────────

def test_a_suffixed_table_would_be_found_as_a_second_mask():
    """Channel naming patterns are anchored at the start of a key, so the trap
    is real for a suffix and absent for the prefix."""
    patterns = TEM_NAMING.inuse
    assert any(re.match(p, "dbdt_inuse_ch1gt_coupled", re.IGNORECASE) for p in patterns)
    assert not any(re.match(p, table.key_for("dbdt_inuse_ch1gt"), re.IGNORECASE)
                   for p in patterns)
    assert not any(re.match(p, table.key_for("LM_Z_G01_inuse"), re.IGNORECASE)
                   for p in patterns)


def test_writing_the_table_leaves_one_mask_per_channel():
    dataset = survey(moments=("LM", "HM"))
    flag(dataset)
    assert [c.inuse_key for c in dataset.channels] == ["dbdt_inuse_ch1gt", "dbdt_inuse_ch2gt"]
    assert table.key_for("dbdt_inuse_ch1gt") in dataset.layer_data


def test_the_table_is_not_reported_as_an_undeclared_array():
    """The inventory and select_moments are two detectors of the same thing;
    a new array family that only one of them knows breaks their agreement."""
    dataset = survey(moments=("LM", "HM"))
    flag(dataset)
    unresolved = set(take(dataset)["arrays"]["unresolved"])
    assert unresolved == set(unclaimed_arrays(dataset))
    assert not [key for key in unresolved if table.is_table(key)]


def test_dropping_a_channel_drops_its_table():
    dataset = survey(moments=("LM", "HM"))
    flag(dataset)
    key = table.channel_key(dataset.channels[1])
    select_moments(dataset, criterion="moment", keep=["LM"])
    assert key not in dataset.layer_data
    assert table.channel_key(dataset.channels[0]) in dataset.layer_data


def test_the_table_is_int8_in_the_provenance_language():
    dataset = survey()
    flag(dataset)
    values = coupling_table(dataset)
    assert values.dtype == np.int8
    assert set(np.unique(values)) <= {table.CLEAN, table.DIRECT, table.COLLATERAL}


def test_the_table_survives_export(tmp_path):
    from libaarhusxyz.export import msgpack as xyz_msgpack
    from ymerflow_processing.dataset import load

    dataset = survey()
    flag(dataset)
    expected = coupling_table(dataset)

    dataset.survey.dump(xyzfile=str(tmp_path / "a.xyz"), gexfile=str(tmp_path / "a.gex"))
    back = load(tmp_path / "a.xyz", tmp_path / "a.gex")
    np.testing.assert_array_equal(back.layer_data[table.channel_key(back.channels[0])], expected)

    with open(tmp_path / "a.msgpack", "wb") as f:
        dataset.survey.xyz.to_msgpack(f)
    packed = xyz_msgpack.load(str(tmp_path / "a.msgpack"))
    np.testing.assert_array_equal(packed.layer_data[table.key_for("dbdt_inuse_ch1gt")], expected)


# ── The switch ───────────────────────────────────────────────────────────────

def test_flag_writes_the_table_and_leaves_the_mask_alone():
    dataset = survey()
    channel = dataset.channels[0]
    before = dataset.layer_data[channel.inuse_key].to_numpy().copy()
    flag(dataset)
    np.testing.assert_array_equal(dataset.layer_data[channel.inuse_key], before)
    assert (coupling_table(dataset)[COUPLED] != 0).any()
    assert provenance_key(channel.inuse_key) not in dataset.layer_data


def test_flag_creates_no_mask_where_there_was_none():
    dataset = survey()
    del dataset.layer_data["dbdt_inuse_ch1gt"]
    flag(dataset)
    assert "dbdt_inuse_ch1gt" not in dataset.layer_data
    assert table.key_for("dbdt_inuse_ch1gt") in dataset.layer_data


def test_cull_gives_exactly_the_mask_that_applying_the_table_gives():
    flagged, culled = survey(), survey()
    flag(flagged)
    table.apply(flagged, flagged.channels[0], step_code=STEP_CODE)
    cull_by_coupling(culled, action="cull", n_components=2)

    key = culled.channels[0].inuse_key
    np.testing.assert_array_equal(flagged.layer_data[key], culled.layer_data[key])
    np.testing.assert_array_equal(coupling_table(flagged), coupling_table(culled))


def test_the_table_and_the_provenance_agree_on_direct_and_collateral():
    """What the step does with its decision: write it to the table, then apply
    the table. A hit, the gate before it (spread) and the tail after it."""
    dataset = survey()
    channel = dataset.channels[0]
    direct = np.zeros((N, GATES), dtype=bool)
    direct[30, 6] = True
    bad = direct.copy()
    bad[30, 5:] = True
    table.write(dataset, channel, bad, direct)
    table.apply(dataset, channel, step_code=STEP_CODE)

    marks = coupling_table(dataset)
    why = provenance(dataset, channel).to_numpy()
    np.testing.assert_array_equal(why[marks == table.DIRECT], STEP_CODE)
    np.testing.assert_array_equal(why[marks == table.COLLATERAL], -STEP_CODE)
    np.testing.assert_array_equal(why[marks == table.CLEAN], 0)
    assert (marks == table.COLLATERAL).sum() == GATES - 6


def test_the_step_records_its_hits_as_direct():
    dataset = survey()
    cull_by_coupling(dataset, action="cull", n_components=2)
    marks = coupling_table(dataset)
    why = provenance(dataset, dataset.channels[0]).to_numpy()
    assert (marks == table.DIRECT).any()
    np.testing.assert_array_equal(why[marks == table.DIRECT], STEP_CODE)
    np.testing.assert_array_equal(why[marks == table.COLLATERAL], -STEP_CODE)


def test_re_running_a_flag_pass_changes_nothing():
    dataset = survey()
    flag(dataset)
    first = coupling_table(dataset).copy()
    confidence = dataset.flightlines["qs_coupling_ch1"].to_numpy().copy()
    flag(dataset)
    np.testing.assert_array_equal(coupling_table(dataset), first)
    np.testing.assert_array_equal(dataset.flightlines["qs_coupling_ch1"], confidence)


@pytest.mark.parametrize("bad", [
    {"action": "remove"},
    {"min_confidence": 1.5},
    {"min_confidence": 0.9, "cull_below_confidence": 0.95},
    {"window": -1.0},
    {"max_reconstruction_error": 0.0},
    {"max_uncertainty": 0.0},
])
def test_a_parameter_out_of_range_is_refused(bad):
    with pytest.raises(ValueError):
        cull_by_coupling(survey(), **bad)


def test_a_channel_without_uncertainty_is_not_judged_and_says_so():
    dataset = survey()
    del dataset.layer_data[dataset.channels[0].std_key]
    _, report = flag(dataset)
    assert any("assign_uncertainty" in note for note in report.notes)
    assert table.key_for("dbdt_inuse_ch1gt") not in dataset.layer_data


# ── The criterion ────────────────────────────────────────────────────────────

def test_coupled_soundings_are_marked_and_distant_ones_are_not():
    dataset = survey()
    _, report = flag(dataset)
    marks = coupling_table(dataset) != 0
    assert marks[COUPLED].any(axis=1).all()
    far = np.r_[0:10, 40:N]
    assert not marks[far].any()
    confidence = dataset.flightlines["qs_coupling_ch1"].to_numpy()
    assert confidence[COUPLED].max() < confidence[far].min()
    assert report.distributions["per_channel"]["ch1 (LM/z)"]["marked"] == int(marks.sum())


def test_a_sounding_the_basis_reproduces_is_not_marked():
    dataset = survey(coupled=False)
    flag(dataset)
    assert not coupling_table(dataset).any()
    np.testing.assert_allclose(dataset.flightlines["qs_coupling_ch1"], 1.0)


def test_noise_dominated_gates_are_excluded_before_the_decision():
    """A hump that lies entirely past the noise cutoff is not judged at all,
    rather than judged and then excused."""
    std = np.full((N, GATES), 0.03)
    std[:, 5:] = 0.5
    dataset = survey(std=std)
    flag(dataset)
    assert not coupling_table(dataset).any()


def test_windows_are_distances_not_sounding_counts():
    """Two lines sampled at different rates average the same distance."""
    x = np.r_[np.arange(10) * 10.0, np.arange(20) * 5.0]
    y = np.r_[np.zeros(10), np.full(20, 1000.0)]
    lines = np.r_[np.zeros(10), np.ones(20)]
    spike = np.zeros(30)
    spike[[5, 20]] = 1.0
    smoothed = along_line_mean(spike, x, y, lines, length=20.0)
    assert smoothed[5] == pytest.approx(1 / 3)     # 10 m spacing: three soundings
    assert smoothed[20] == pytest.approx(1 / 5)    # 5 m spacing: five soundings


def test_a_window_never_crosses_from_one_line_to_the_next():
    x = np.r_[np.arange(5) * 10.0, np.arange(5) * 10.0]
    y = np.zeros(10)
    lines = np.r_[np.zeros(5), np.ones(5)]
    values = np.r_[np.zeros(5), np.ones(5)]
    np.testing.assert_array_equal(along_line_mean(values, x, y, lines, 1000.0), values)


# ── The basis ────────────────────────────────────────────────────────────────

def decays(n=50, gates=10, seed=1):
    rng = np.random.default_rng(seed)
    t = np.logspace(-5, -3, gates)
    return 1e-6 * rng.uniform(0.5, 2, (n, 1)) * (t / t[0]) ** -rng.uniform(1, 2, (n, 1))


def test_a_power_law_family_is_reproduced_from_two_components():
    data = decays()
    basis = PCAReconstructor(n_components=2)
    basis.fit(data)
    np.testing.assert_allclose(basis.reconstruct(data), data, rtol=1e-8)


def test_a_sounding_is_reconstructed_from_the_gates_it_has():
    data = decays()
    holed = data.copy()
    holed[::3, 7:] = np.nan
    basis = PCAReconstructor(n_components=2)
    basis.fit(holed)
    assert basis.converged_
    np.testing.assert_allclose(basis.reconstruct(holed)[::3], data[::3], rtol=1e-5)


def test_non_positive_values_are_missing_in_log_space():
    data = decays()
    data[0, 4] = -1e-9
    basis = PCAReconstructor(n_components=2)
    basis.fit(data)
    assert np.isfinite(basis.reconstruct(data)[0]).all()


def test_too_few_soundings_for_the_basis_is_refused():
    with pytest.raises(ValueError, match="needs more"):
        PCAReconstructor(n_components=4).fit(decays(n=3))
