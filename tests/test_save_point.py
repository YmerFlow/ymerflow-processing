"""A save point — the state a manual edit is drawn against, written where it is reached.

Most of this guards the order. The snapshot handed to the sink is the state
before the point's own edit, so the edit stays recoverable; the edit replays
whether or not there was anywhere to write the snapshot; and a bad name fails
before any data is read rather than part-way through a run.
"""

import inspect
import json
import typing

import numpy as np
import pandas as pd
import pydantic
import pytest
from click.testing import CliRunner

from conftest import make_dataset, step
from test_manual_edits import delivery, edit_set, mask
from ymerflow_processing.cli import EXIT_GOOD, EXIT_NOT_STARTED, cli
from ymerflow_processing.dataset import load
from ymerflow_processing.filters.cull_manual_edits import apply_manual_edits
from ymerflow_processing.filters.save_point import (
    STEP, save_intermediate_and_apply_diff,
)
from ymerflow_processing.mask import CULLED, IN_USE
from ymerflow_processing.pipeline import (
    FILTER_GROUP, Pipeline, ProcessingRunner, load_steps,
)


def recording_sink():
    """A sink that keeps each save point's name and its mask as emitted."""
    points = []

    def sink(name, dataset):
        points.append((name, mask(dataset).copy()))
    return sink, points


# ── The step ─────────────────────────────────────────────────────────────────

def test_the_snapshot_is_the_state_before_the_edit():
    """The editor draws this point's edit over the snapshot as an override, so
    the snapshot must not already contain it."""
    sink, points = recording_sink()
    dataset, report = save_intermediate_and_apply_diff(
        delivery(), "full-rate", diff=edit_set([2], {(0, 4): 0}), sink=sink)

    [(name, snapshot)] = points
    assert name == "full-rate"
    assert snapshot[2, 4] == IN_USE
    assert mask(dataset)[2, 4] == CULLED
    assert report.distributions["snapshot"] == "emitted"


def test_the_diff_replays_exactly_as_apply_manual_edits_does():
    diff = edit_set([1, 3], {(0, 0): 0, (1, 5): 0})
    expected, reference = apply_manual_edits(delivery(), diff, checkpoint="full-rate")
    dataset, report = save_intermediate_and_apply_diff(delivery(), "full-rate", diff=diff)

    np.testing.assert_array_equal(mask(dataset), mask(expected))
    assert report.step == STEP
    assert report.distributions["checkpoint"] == "full-rate"
    for key in ("per_channel", "soundings_edited", "removed", "restored"):
        assert report.distributions[key] == reference.distributions[key], key


def test_an_edit_may_turn_data_back_on():
    """A person overruling an automated cull, at a save point as anywhere."""
    from ymerflow_processing.filters.inuse_gates import set_inuse_gates
    dataset, _ = set_inuse_gates(delivery(), disable_gates={"ch1": 0})
    dataset, report = save_intermediate_and_apply_diff(
        dataset, "averaged", diff=edit_set([1], {(0, 0): 1}))

    assert mask(dataset)[1, 0] == IN_USE
    assert mask(dataset)[0, 0] == CULLED
    assert report.distributions["restored"] == 1


def test_with_no_diff_it_is_a_pure_save_point():
    sink, points = recording_sink()
    before = delivery()
    untouched = {k: v.copy() for k, v in before.layer_data.items()}

    dataset, report = save_intermediate_and_apply_diff(before, "averaged", sink=sink)

    assert [name for name, _ in points] == ["averaged"]
    for key, array in untouched.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    assert report.soundings_affected == 0
    assert report.distributions == {"save_point": "averaged", "snapshot": "emitted"}
    assert report.notes == []


def test_with_no_sink_the_save_point_is_skipped_and_says_so():
    """A library caller with no runner loses only the file; the edit replays."""
    dataset, report = save_intermediate_and_apply_diff(
        delivery(), "full-rate", diff=edit_set([2], {(0, 4): 0}))

    assert mask(dataset)[2, 4] == CULLED
    assert report.distributions["snapshot"] == "skipped"
    assert any("No sink was supplied" in note for note in report.notes)


def test_it_reads_the_msgpack_the_plot_editor_saves(tmp_path):
    path = tmp_path / "inuse_diff_delivery.msgpack"
    with open(path, "wb") as f:
        edit_set([2], {(0, 4): 0}).to_msgpack(f)

    dataset, report = save_intermediate_and_apply_diff(delivery(), "full-rate", diff=str(path))

    assert mask(dataset)[2, 4] == CULLED
    assert report.distributions["removed"] == 1


def test_the_report_survives_serialization():
    _, report = save_intermediate_and_apply_diff(
        delivery(), "full-rate", diff=edit_set([1], {(0, 2): 0}))
    restored = json.loads(json.dumps(report.to_dict(), default=str))
    assert restored["step"] == STEP
    assert restored["distributions"]["save_point"] == "full-rate"
    assert restored["distributions"]["snapshot"] == "skipped"


def test_the_diff_field_carries_the_media_type_the_editor_matches():
    """The plot editor offers a saved edit for a field carrying exactly this
    annotation, so it is fixed rather than ours to vary."""
    # The signature, not typing.get_type_hints, which fails on 3.10 here; see
    # the comment above the step.
    annotated = inspect.signature(save_intermediate_and_apply_diff).parameters["diff"].annotation

    assert typing.get_args(annotated)[0] is pydantic.AnyUrl
    assert ({"json_schema": {"x-url-media-type": "application/x-geophysics-xyz-model"}}
            in annotated.__metadata__)


def test_registered():
    assert STEP in load_steps(FILTER_GROUP)


# ── The runner's sink ────────────────────────────────────────────────────────

TWO_POINTS = """
    kind: filters
    steps:
      - save_intermediate_and_apply_diff: {name: full-rate}
      - set_inuse_gates: {disable_gates: {ch1: 0}}
      - save_intermediate_and_apply_diff: {name: final}
"""


def test_the_runner_hands_its_sink_to_each_save_point_as_it_is_reached():
    sink, points = recording_sink()
    dataset, report = ProcessingRunner(Pipeline.parse(TWO_POINTS), sink=sink).run(delivery())

    assert [name for name, _ in points] == ["full-rate", "final"]
    assert (points[0][1][:, 0] == IN_USE).all()     # before the cull below it
    assert (points[1][1][:, 0] == CULLED).all()     # after it
    assert (mask(dataset)[:, 0] == CULLED).all()


def test_the_sink_is_not_recorded_as_a_parameter():
    """It is the runner's, not the document's: the report and the history
    say what the document said."""
    sink, _ = recording_sink()
    dataset, report = ProcessingRunner(Pipeline.parse(TWO_POINTS), sink=sink).run(delivery())

    assert step(report, STEP).params == {"name": "full-rate"}
    recorded = [entry["params"] for entry in dataset.history if entry["step"] == STEP]
    assert recorded == [{"name": "full-rate"}, {"name": "final"}]


def test_a_runner_with_no_sink_still_replays_every_edit(tmp_path):
    path = tmp_path / "inuse_diff_delivery.msgpack"
    with open(path, "wb") as f:
        edit_set([2], {(0, 4): 0}).to_msgpack(f)
    document = Pipeline({"kind": "filters", "steps": [
        {STEP: {"name": "full-rate", "diff": str(path)}}]})
    dataset, report = ProcessingRunner(document).run(delivery())

    assert mask(dataset)[2, 4] == CULLED
    assert step(report, STEP).distributions["snapshot"] == "skipped"


# ── Names, checked before any data is read ───────────────────────────────────

def document(*names, extra=None):
    return Pipeline({"kind": "filters", "steps": [
        {STEP: dict({"name": name}, **(extra or {}))} for name in names]})


def test_a_duplicate_name_fails_when_the_runner_is_built():
    """Before run() is called, so before anything has been loaded."""
    with pytest.raises(ValueError, match="'full-rate' is used more than once"):
        ProcessingRunner(document("full-rate", "averaged", "full-rate"))


@pytest.mark.parametrize("reserved", ["output", "processed_data"])
def test_a_reserved_name_fails_when_the_runner_is_built(reserved):
    with pytest.raises(ValueError, match="reserved"):
        ProcessingRunner(document(reserved))


def test_a_save_point_with_no_name_fails_when_the_runner_is_built():
    with pytest.raises(ValueError, match="needs a 'name'"):
        ProcessingRunner(Pipeline({"kind": "filters", "steps": [STEP]}))


def test_a_document_cannot_set_the_sink():
    with pytest.raises(ValueError, match="runner supplies the sink"):
        ProcessingRunner(document("full-rate", extra={"sink": "somewhere"}))


def test_distinct_names_are_accepted():
    ProcessingRunner(document("full-rate", "averaged", "final"))


# ── The command line ─────────────────────────────────────────────────────────

@pytest.fixture
def on_disk(tmp_path):
    """A delivery, a saved edit, and a pipeline with a save point either side
    of an automated cull."""
    dataset = make_dataset(n=6, n_lines=1, n_gates=8)
    xyz, gex = tmp_path / "d.xyz", tmp_path / "sys.gex"
    dataset.survey.dump(xyzfile=str(xyz), gexfile=str(gex))
    diff = tmp_path / "inuse_diff_d.msgpack"
    with open(diff, "wb") as f:
        edit_set([2], {(0, 4): 0}).to_msgpack(f)

    def write(text):
        path = tmp_path / "pipeline.yaml"
        path.write_text(text.format(diff=diff))
        return path
    return xyz, gex, write


SAVE_POINTS = """
kind: filters
steps:
  - save_intermediate_and_apply_diff: {{name: full-rate, diff: "{diff}"}}
  - set_inuse_gates: {{disable_gates: {{ch1: 0}}}}
  - save_intermediate_and_apply_diff: {{name: final}}
"""

NO_SAVE_POINTS = """
kind: filters
steps:
  - apply_manual_edits: {{diff: "{diff}"}}
  - set_inuse_gates: {{disable_gates: {{ch1: 0}}}}
"""


def run(pipeline, xyz, gex, out):
    return CliRunner().invoke(
        cli, ["run", str(pipeline), "--xyz", str(xyz), "--gex", str(gex),
              "--out", str(out)], standalone_mode=False)


def test_the_cli_writes_one_snapshot_per_save_point(on_disk, tmp_path):
    xyz, gex, write = on_disk
    out = tmp_path / "out"

    result = run(write(SAVE_POINTS), xyz, gex, out)

    assert result.return_value == EXIT_GOOD, result.output
    for name in ("full-rate", "final"):
        for suffix in ("xyz", "gex", "msgpack"):
            assert (out / f"d.{name}.{suffix}").is_file(), (name, suffix)
    assert "wrote save point d.full-rate.xyz" in result.output


def test_each_snapshot_parses_back_as_the_state_at_its_point(on_disk, tmp_path):
    xyz, gex, write = on_disk
    out = tmp_path / "out"
    run(write(SAVE_POINTS), xyz, gex, out)

    full_rate = load(out / "d.full-rate.xyz", gex)
    final = load(out / "d.final.xyz", gex)

    assert mask(full_rate)[2, 4] == IN_USE           # before its own edit
    assert (mask(full_rate)[:, 0] == IN_USE).all()   # and before the cull
    assert mask(final)[2, 4] == CULLED
    assert (mask(final)[:, 0] == CULLED).all()


def test_writing_snapshots_does_not_change_the_output(on_disk, tmp_path):
    """The sink is handed the live dataset, so writing it must leave the data
    the rest of the run uses exactly as it was."""
    xyz, gex, write = on_disk
    run(write(SAVE_POINTS), xyz, gex, tmp_path / "with")
    run(write(NO_SAVE_POINTS), xyz, gex, tmp_path / "without")

    with_points = load(tmp_path / "with" / "d.xyz", gex)
    without = load(tmp_path / "without" / "d.xyz", gex)
    pd.testing.assert_frame_equal(with_points.flightlines, without.flightlines)
    assert set(with_points.layer_data) == set(without.layer_data)
    for key, array in without.layer_data.items():
        pd.testing.assert_frame_equal(with_points.layer_data[key], array, obj=key)


@pytest.mark.parametrize("text, said", [
    ("kind: filters\nsteps:\n  - save_intermediate_and_apply_diff: {{name: output}}\n",
     "'output' is reserved"),
    ("kind: filters\nsteps:\n"
     "  - save_intermediate_and_apply_diff: {{name: a}}\n"
     "  - save_intermediate_and_apply_diff: {{name: a}}\n",
     "'a' is used more than once"),
])
def test_the_cli_refuses_a_bad_name_before_reading_anything(on_disk, tmp_path, text, said):
    """The survey path does not exist, so reaching the loader would fail
    differently: the name is refused first."""
    _, gex, write = on_disk
    out = tmp_path / "out"

    result = run(write(text), tmp_path / "not-there.xyz", gex, out)

    assert result.return_value == EXIT_NOT_STARTED
    assert said in result.output
    assert not out.exists()
