"""filter_22 — soundings whose flight geometry broke the forward model.

The first real predicate through the mask, and the first filter whose limits
default from somewhere else. Most of what follows pins the line between a
contract limit and a physical one, because they are the same shape of number
measuring different things.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.cull.filter_22_geometry import (
    STEP_CODE, cull_by_geometry,
)
from ymerflow_processing.core.mask import CULLED, IN_USE, provenance_key
from ymerflow_processing.core.pipeline import (
    FILTER_GROUP, Pipeline, QCRunner, load_steps,
)

N = 10
GATES = 8


def flown(pitch=None, roll=None, height=None):
    return make_dataset(n=N, n_lines=1, n_gates=GATES,
                        pitch=pitch if pitch is not None else np.zeros(N) + 0.5,
                        roll=roll if roll is not None else np.zeros(N) + 0.5,
                        height=height if height is not None else np.full(N, 30.0))


def mask(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].inuse_key].to_numpy()


def judged(dataset, height="{good_max: 45, marginal_max: 65}",
           attitude="{good: 5, marginal: 10}"):
    """Run QC first, so its thresholds are on the dataset to default from."""
    dataset, _ = QCRunner(Pipeline.parse(
        f"kind: qc\nsteps:\n  - height_above_ground: {height}\n"
        f"  - system_attitude: {attitude}\n")).run(dataset)
    return dataset


# ── What it culls ────────────────────────────────────────────────────────────

def test_a_sounding_flown_too_high_is_disabled():
    heights = np.full(N, 30.0); heights[3] = 250.0
    dataset, report = cull_by_geometry(flown(height=heights), max_altitude=120)

    assert (mask(dataset)[3] == CULLED).all()
    assert (mask(dataset)[0] == IN_USE).all()
    assert report.distributions["soundings_culled"] == 1


def test_pitch_and_roll_count_in_either_direction():
    """Tilt is tilt; the loop couples differently whichever way it leans."""
    pitch = np.zeros(N); pitch[1] = 40.0; pitch[2] = -40.0
    dataset, _ = cull_by_geometry(flown(pitch=pitch), max_pitch=25)

    assert (mask(dataset)[1] == CULLED).all()
    assert (mask(dataset)[2] == CULLED).all()
    assert (mask(dataset)[0] == IN_USE).all()


def test_a_reading_too_low_to_be_real_can_be_culled_too():
    heights = np.full(N, 30.0); heights[5] = -2.0
    dataset, _ = cull_by_geometry(flown(height=heights), min_altitude=1.0)

    assert (mask(dataset)[5] == CULLED).all()


def test_it_disables_whole_soundings_across_every_channel():
    """Attitude and altitude belong to the aircraft, not to a transient. A badly
    tilted sounding is badly tilted for every moment simultaneously."""
    roll = np.zeros(N); roll[4] = 60.0
    dataset, _ = cull_by_geometry(flown(roll=roll), max_roll=25)

    for which in (0, 1):
        assert (mask(dataset, which)[4] == CULLED).all()
        assert (mask(dataset, which)[4] == CULLED).sum() == GATES


def test_it_changes_no_measurement_or_attribute():
    heights = np.full(N, 30.0); heights[2] = 300.0
    dataset = flown(height=heights)
    arrays = {k: v.copy() for k, v in dataset.layer_data.items()
              if "inuse" not in k.lower()}
    flightlines = dataset.flightlines.copy()

    dataset, _ = cull_by_geometry(dataset, max_altitude=100)

    for key, array in arrays.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    pd.testing.assert_frame_equal(dataset.flightlines, flightlines)


def test_an_unmeasured_sounding_is_not_culled_here():
    """It has no geometry on record, which is a different problem and not this
    filter's to decide. NaN fails every comparison, which is the wanted answer."""
    heights = np.full(N, 30.0); heights[6] = np.nan
    dataset, report = cull_by_geometry(flown(height=heights), max_altitude=100)

    assert (mask(dataset)[6] == IN_USE).all()
    assert report.distributions["soundings_culled"] == 0


def test_it_is_idempotent():
    heights = np.full(N, 30.0); heights[3] = 250.0
    dataset, first = cull_by_geometry(flown(height=heights), max_altitude=120)
    snapshot = {k: v.copy() for k, v in dataset.layer_data.items()}
    dataset, second = cull_by_geometry(dataset, max_altitude=120)

    for key, array in snapshot.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    assert second.distributions["soundings_culled"] == 1     # the same sounding
    assert second.distributions["per_channel"][
        str(dataset.channels[0])]["removed"] == 0            # nothing left to take


# ── Restated limits, defaulted from QC ───────────────────────────────────────

def test_a_limit_defaults_to_the_qc_out_of_spec_boundary():
    """Not the Good one. Marginal is where QC stopped accepting the sounding;
    Good is where it stopped being unremarkable."""
    heights = np.full(N, 30.0); heights[1] = 70.0; heights[2] = 50.0
    dataset, report = cull_by_geometry(judged(flown(height=heights)))

    assert report.distributions["limits"]["max_altitude"]["limit"] == 65.0
    assert (mask(dataset)[1] == CULLED).all()      # 70 is beyond marginal_max
    assert (mask(dataset)[2] == IN_USE).all()      # 50 is Marginal, not culled


def test_an_explicit_limit_wins_over_the_default():
    """A contract limit and a physical one are different numbers. A sounding at
    70 m against a 65 m contract is out of spec and still invertible."""
    heights = np.full(N, 30.0); heights[1] = 70.0
    dataset, report = cull_by_geometry(judged(flown(height=heights)),
                                       max_altitude=150)

    assert report.distributions["limits"]["max_altitude"]["limit"] == 150.0
    assert (mask(dataset)[1] == IN_USE).all()


def test_the_report_says_where_each_limit_came_from():
    """So a culled delivery can be explained without re-deriving it."""
    dataset, report = cull_by_geometry(judged(flown()), max_altitude=150)
    limits = report.distributions["limits"]

    assert limits["max_altitude"]["source"] == "given"
    assert limits["max_pitch"]["source"] == "system_attitude.marginal"
    assert any("using system_attitude's out-of-spec boundary" in n
               for n in report.notes)


def test_attitude_defaults_apply_to_both_axes():
    pitch = np.zeros(N); pitch[1] = 15.0
    roll = np.zeros(N); roll[2] = 15.0
    dataset, report = cull_by_geometry(judged(flown(pitch=pitch, roll=roll)))

    assert report.distributions["limits"]["max_pitch"]["limit"] == 10.0
    assert report.distributions["limits"]["max_roll"]["limit"] == 10.0
    assert (mask(dataset)[1] == CULLED).all()
    assert (mask(dataset)[2] == CULLED).all()


def test_min_altitude_has_no_qc_threshold_to_borrow():
    """qc_01 is one-sided, so there is nothing to default from and this stays
    unapplied rather than being invented."""
    _, report = cull_by_geometry(judged(flown()), max_altitude=150)
    assert "min_altitude" not in report.distributions["limits"]


def test_no_limits_and_no_qc_is_an_error():
    """No house numbers: what counts as too tilted is an operational judgement
    about an instrument, and a default here would silently cull a client's."""
    with pytest.raises(ValueError, match="needs at least one limit"):
        cull_by_geometry(flown())


def test_a_missing_geometry_column_is_noted_not_fatal():
    dataset = flown()
    dataset.survey.xyz.flightlines = dataset.flightlines.drop(columns=["tilt_x"])

    dataset, report = cull_by_geometry(dataset, max_pitch=10, max_roll=10)
    assert any("tilt_x" in n and "not applied" in n for n in report.notes)
    assert "max_roll" in report.distributions["limits"]


# ── Provenance and plumbing ──────────────────────────────────────────────────

def test_the_culled_soundings_are_attributed_to_this_filter():
    heights = np.full(N, 30.0); heights[3] = 250.0
    dataset, _ = cull_by_geometry(flown(height=heights), max_altitude=120)
    why = dataset.layer_data[provenance_key(dataset.channels[0].inuse_key)]

    assert (why.to_numpy()[3] == STEP_CODE).all()
    assert (why.to_numpy()[0] == 0).all()
    assert STEP_CODE == 22


def test_nothing_is_collateral():
    """No gate policy applies: the problem has no position along a transient, so
    every disabled gate is a direct consequence of the sounding's geometry."""
    heights = np.full(N, 30.0); heights[3] = 250.0
    _, report = cull_by_geometry(flown(height=heights), max_altitude=120)
    first = report.distributions["per_channel"]

    outcome = next(iter(first.values()))
    assert outcome["removed_collateral"] == 0
    assert outcome["removed_direct"] == GATES


def test_a_whole_sounding_going_is_reported_as_emptied():
    heights = np.full(N, 30.0); heights[3] = 250.0
    _, report = cull_by_geometry(flown(height=heights), max_altitude=120)

    outcome = next(iter(report.distributions["per_channel"].values()))
    assert outcome["soundings_emptied"] == 1


def test_the_report_survives_serialization():
    heights = np.full(N, 30.0); heights[3] = 250.0
    _, report = cull_by_geometry(flown(height=heights), max_altitude=120)
    restored = json.loads(json.dumps(report.to_dict(), default=str))

    assert restored["step"] == "cull_by_geometry"
    assert restored["distributions"]["soundings_culled"] == 1


def test_registered():
    assert "cull_by_geometry" in load_steps(FILTER_GROUP)
