"""What changed between two processing versions.

02's first use for the comparison shape, "currently demonstrated by eye". What
eyes cannot do is attribute a difference: spotting that a version culled more is
easy, and connecting it to the step whose threshold moved is the whole point.
"""

import json

import numpy as np
import pytest

from conftest import make_dataset
from ymerflow_processing.compare.align import DATA, MASK, UNCERTAINTY
from ymerflow_processing.compare.version_diff import (
    CONSUMES, STEP, version_diff,
)
from ymerflow_processing.mask import (
    CULLED, apply_policy, cull, mark_step,
)
from ymerflow_processing.pipeline import (
    COMPARE_GROUP, ComparisonRunner, Pipeline, QCRunner, load_steps,
)
from ymerflow_processing.qc.thresholds import GOOD, OUT_OF_SPEC, TIER_POINTS

N = 10
GATES = 6


def pair(**kw):
    return make_dataset(n=N, n_lines=1, n_gates=GATES, **kw), \
        make_dataset(n=N, n_lines=1, n_gates=GATES, **kw)


def diff(new, old, **kw):
    return version_diff([new], reference=old, **kw)


def channel_entry(report, quantity):
    channels = report.distributions["channels"]
    return next(iter(channels.values()))[quantity]


# ── The pipeline half ────────────────────────────────────────────────────────

def test_a_moved_parameter_is_named_with_both_values():
    new, old = pair()
    new.history = [{"step": "cull_by_uncertainty", "params": {"max_std": 0.3}}]
    old.history = [{"step": "cull_by_uncertainty", "params": {"max_std": 0.5}}]

    moved = diff(new, old).distributions["pipeline"]["reparameterized"]
    assert moved[0]["step"] == "cull_by_uncertainty"
    assert moved[0]["parameters"]["max_std"] == {"old": 0.5, "new": 0.3}


def test_an_added_step_is_named():
    new, old = pair()
    new.history = [{"step": "cull_negatives", "params": {}}]
    old.history = []

    pipeline = diff(new, old).distributions["pipeline"]
    assert pipeline["added"][0]["step"] == "cull_negatives"
    assert not pipeline["removed"]


def test_a_removed_step_is_named():
    new, old = pair()
    new.history = []
    old.history = [{"step": "cull_negatives", "params": {}}]

    assert diff(new, old).distributions["pipeline"]["removed"][0]["step"] == \
        "cull_negatives"


def test_the_same_step_run_twice_is_not_collapsed():
    """Keying on the name alone would fold two runs of one step together and
    report the second's parameters as the first's."""
    new, old = pair()
    new.history = [{"step": "cull_by_uncertainty", "params": {"max_std": 0.3}},
                   {"step": "cull_by_uncertainty", "params": {"max_std": 0.9}}]
    old.history = [{"step": "cull_by_uncertainty", "params": {"max_std": 0.3}},
                   {"step": "cull_by_uncertainty", "params": {"max_std": 0.5}}]

    moved = diff(new, old).distributions["pipeline"]["reparameterized"]
    assert len(moved) == 1
    assert moved[0]["at"] == 1


def test_identical_pipelines_with_differing_data_are_flagged():
    """Something outside the pipeline changed, and that is worth saying."""
    new, old = pair()
    new.history = old.history = [{"step": "cull_negatives", "params": {}}]

    assert any("outside the pipeline" in note for note in diff(new, old).notes)


# ── The mask half, attributed ────────────────────────────────────────────────

def test_extra_culling_is_counted():
    new, old = pair()
    new.layer_data[new.channels[0].inuse_key].iloc[:, 4:] = CULLED

    mask = channel_entry(diff(new, old), MASK)
    assert mask["culled_by_the_new_version"] == N * 2
    assert mask["restored_by_the_new_version"] == 0


def test_loosened_culling_is_counted_too():
    """A version that culls LESS has loosened something, and that is the change
    nobody looks for."""
    new, old = pair()
    old.layer_data[old.channels[0].inuse_key].iloc[:, 4:] = CULLED

    mask = channel_entry(diff(new, old), MASK)
    assert mask["restored_by_the_new_version"] == N * 2
    assert mask["culled_by_the_new_version"] == 0


def test_the_extra_culling_is_attributed_to_the_step_that_did_it():
    """The thing eyes cannot do. Spotting that a version culled more is easy;
    attributing it to the step whose threshold moved is what makes it
    actionable, and the dataset already knows."""
    new, old = pair()
    channel = new.channels[0]
    mark_step(new, 24, "cull_by_uncertainty")
    bad = np.tile(np.arange(GATES) >= 4, (N, 1))
    cull(new, channel, bad, step_code=24)

    mask = channel_entry(diff(new, old), MASK)
    assert mask["culled_by_step"] == {"cull_by_uncertainty": N * 2}


def test_collateral_culling_is_attributed_to_the_same_step():
    """The provenance sign says how a datum died — the predicate itself, or
    spread from it. Attribution is by step, so the sign is not a second step."""
    new, old = pair()
    channel = new.channels[0]
    mark_step(new, 25, "cull_by_decay_rate")
    trigger = np.zeros((N, GATES), dtype=bool)
    trigger[:, 4] = True
    cull(new, channel, apply_policy(trigger, spread_earlier=1),
         step_code=25, direct=trigger)

    mask = channel_entry(diff(new, old), MASK)
    assert sum(mask["culled_by_step"].values()) == N * 2
    assert list(mask["culled_by_step"]) == ["cull_by_decay_rate"]


# ── The value half ───────────────────────────────────────────────────────────

def test_values_that_moved_are_counted_and_measured():
    new, old = pair()
    new.layer_data[new.channels[0].data_key].iloc[:, 2] *= 1.5

    values = channel_entry(diff(new, old), DATA)
    assert values["n_moved"] == N
    assert values["max_relative_change"] == pytest.approx(0.5, rel=1e-6)


def test_what_is_under_a_mask_is_not_a_change():
    """Underneath a mask a run may leave anything at all."""
    new, old = pair()
    new.layer_data[new.channels[0].data_key].iloc[:, 4] = 1e9
    for dataset in (new, old):
        dataset.layer_data[dataset.channels[0].inuse_key].iloc[:, 4] = CULLED

    assert channel_entry(diff(new, old), DATA)["n_moved"] == 0


def test_a_value_that_appeared_is_not_a_change_of_magnitude():
    """It has no magnitude to average in with ones that do."""
    new, old = pair()
    old.layer_data[old.channels[0].data_key].iloc[:, 3] = np.nan

    values = channel_entry(diff(new, old), DATA)
    assert values["n_appeared"] == N
    assert values["n_moved"] == 0


def test_a_value_that_vanished_is_counted_separately():
    new, old = pair()
    new.layer_data[new.channels[0].data_key].iloc[:, 3] = np.nan

    assert channel_entry(diff(new, old), DATA)["n_vanished"] == N


def test_uncertainty_is_diffed_alongside_the_data():
    new, old = pair()
    new.layer_data[new.channels[0].std_key] *= 2.0

    assert channel_entry(diff(new, old), UNCERTAINTY)["n_moved"] == N * GATES
    assert channel_entry(diff(new, old), DATA)["n_moved"] == 0


def test_float_noise_is_not_a_change():
    new, old = pair()
    new.layer_data[new.channels[0].data_key] *= (1 + 1e-12)
    assert channel_entry(diff(new, old), DATA)["n_moved"] == 0


# ── The grade half ───────────────────────────────────────────────────────────

def test_verdicts_that_moved_are_reported_as_a_migration():
    """A reviewer asking 'did this get better' is asking this and nothing
    else."""
    document = """
        kind: qc
        steps:
          - height_above_ground: {good_max: 45, marginal_max: 65}
    """
    old = make_dataset(n=4, n_lines=1, height=[30, 30, 80, 80])
    new = make_dataset(n=4, n_lines=1, height=[30, 30, 30, 80])
    old, _ = QCRunner(Pipeline.parse(document)).run(old)
    new, _ = QCRunner(Pipeline.parse(document)).run(new)

    grades = diff(new, old).distributions["grades"]
    assert grades["qs_height"][f"{OUT_OF_SPEC} -> {GOOD}"] == 1


def test_verdicts_moving_both_ways_at_once_are_both_shown():
    """A version can improve the median while pushing a handful of soundings
    out of spec, and a single total would hide it."""
    document = """
        kind: qc
        steps:
          - height_above_ground: {good_max: 45, marginal_max: 65}
    """
    old = make_dataset(n=4, n_lines=1, height=[30, 30, 80, 80])
    new = make_dataset(n=4, n_lines=1, height=[80, 30, 30, 30])
    old, _ = QCRunner(Pipeline.parse(document)).run(old)
    new, _ = QCRunner(Pipeline.parse(document)).run(new)

    report = diff(new, old)
    migration = report.distributions["grades"]["qs_height"]
    assert migration[f"{GOOD} -> {OUT_OF_SPEC}"] == 1
    assert migration[f"{OUT_OF_SPEC} -> {GOOD}"] == 2
    assert any("both directions" in note for note in report.notes)


def test_no_verdicts_means_no_grade_section():
    new, old = pair()
    assert diff(new, old).distributions["grades"] is None


# ── It is a diff, not a judgement ────────────────────────────────────────────

def test_nothing_passes_or_fails():
    """The two versions are meant to differ, so a difference is the answer
    rather than the finding."""
    new, old = pair()
    new.layer_data[new.channels[0].data_key] *= 2.0

    report = diff(new, old)
    assert "verdict" not in report.distributions
    assert any("nothing here passes or fails" in note for note in report.notes)


# ── Contract ─────────────────────────────────────────────────────────────────

def test_it_needs_to_be_told_which_version_is_older():
    """Getting it backwards reverses the sign of every finding while reporting
    the same magnitudes."""
    new, _ = pair()
    with pytest.raises(ValueError, match="cannot be inferred"):
        version_diff([new])


def test_it_compares_two_versions_and_points_at_the_sweep_for_more():
    new, old = pair()
    with pytest.raises(ValueError, match="parameter_sweep"):
        version_diff([new, new], reference=old)


def test_it_declares_that_it_needs_datasets():
    assert CONSUMES == "datasets"
    assert ComparisonRunner(Pipeline.parse(f"""
        kind: compare
        steps:
          - {STEP}: {{}}
    """)).consumes(STEP) == "datasets"


def test_a_report_given_where_a_dataset_was_needed_says_so():
    new, _ = pair()
    with pytest.raises(TypeError, match="consumes: datasets"):
        version_diff([new], reference={"steps": []})


def test_an_empty_version_reports_only_the_pipeline():
    new, old = pair()
    new.history = [{"step": "cull_negatives", "params": {}}]
    new.survey.xyz.flightlines = new.flightlines.iloc[:0]

    report = diff(new, old)
    assert report.distributions["pipeline"]["added"]
    assert report.distributions["channels"] == {}


def test_it_runs_from_a_pipeline_document():
    new, old = pair()
    new.layer_data[new.channels[0].data_key] *= 1.5

    report = ComparisonRunner(Pipeline.parse(f"""
        kind: compare
        steps:
          - {STEP}: {{}}
    """)).run([new], reference=old)
    assert report.steps[0].soundings_affected > 0


def test_the_report_survives_serialization():
    new, old = pair()
    new.history = [{"step": "cull_by_uncertainty", "params": {"max_std": 0.3}}]
    old.history = [{"step": "cull_by_uncertainty", "params": {"max_std": 0.5}}]

    restored = json.loads(json.dumps(diff(new, old).to_dict(), default=str))
    assert restored["distributions"]["pipeline"]["reparameterized"]


def test_registered():
    assert STEP in load_steps(COMPARE_GROUP)
