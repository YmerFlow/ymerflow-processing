"""The command line — what makes this runnable without the platform.

Everything here writes to tmp_path and reads it back. No platform, no database,
no network: if these pass, a cron job on a bare machine works.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from click.testing import CliRunner

from conftest import make_dataset
from ymerflow_processing.core import cli as cli_module
from ymerflow_processing.core.dataset import load
from ymerflow_processing.core.cli import (
    EXIT_GOOD, EXIT_MARGINAL, EXIT_NOT_STARTED, EXIT_OUT_OF_SPEC,
    EXIT_STEP_RAISED, cli,
)
from ymerflow_processing.core.qc.rollup import NO_REASON
from ymerflow_processing.core.qc.thresholds import GOOD, OUT_OF_SPEC

QC_PIPELINE = """
kind: qc
steps:
  - height_above_ground:
      good_max: 45
      marginal_max: 65
"""


@pytest.fixture
def delivery(tmp_path):
    """A delivery on disk: three flights, the middle one flown high."""
    def build(heights, n_lines=6, n_flights=3, name="20260815"):
        dataset = make_dataset(n=len(heights), n_lines=n_lines,
                               n_flights=n_flights, height=list(heights))
        xyz = tmp_path / f"{name}.xyz"
        gex = tmp_path / "sys.gex"
        dataset.survey.dump(xyzfile=str(xyz), gexfile=str(gex))
        return xyz, gex
    return build


@pytest.fixture
def pipeline_file(tmp_path):
    def write(text, name="pipeline.yaml"):
        path = tmp_path / name
        path.write_text(text)
        return path
    return write


def invoke(*args):
    return CliRunner().invoke(cli, list(map(str, args)), standalone_mode=False)


def run_qc(pipeline, xyz, gex, out, *extra):
    return invoke("run", pipeline, "--xyz", xyz, "--gex", gex, "--out", out, *extra)


# ── Exit codes ───────────────────────────────────────────────────────────────
# The exit code answers two questions at once: did the run happen, and was the
# flight any good. 0-2 mean it ran; 3-4 mean it did not.

@pytest.mark.parametrize("height,expected", [
    (30.0, EXIT_GOOD),          # comfortably in spec
    (50.0, EXIT_MARGINAL),      # between good_max and marginal_max
    (80.0, EXIT_OUT_OF_SPEC),   # breach
])
def test_exit_code_reflects_the_worst_verdict(
        delivery, pipeline_file, tmp_path, height, expected):
    xyz, gex = delivery(np.full(40, height), n_lines=2, n_flights=1)
    result = run_qc(pipeline_file(QC_PIPELINE), xyz, gex, tmp_path / "out")
    assert result.return_value == expected


def test_a_finding_is_not_a_failure(delivery, pipeline_file, tmp_path):
    """Out of Spec still writes its report. The finding is the deliverable."""
    xyz, gex = delivery(np.full(40, 80.0), n_lines=2, n_flights=1)
    out = tmp_path / "out"
    result = run_qc(pipeline_file(QC_PIPELINE), xyz, gex, out)

    assert result.return_value == EXIT_OUT_OF_SPEC
    assert (out / "20260815.report.json").exists()
    assert (out / "20260815.xyz").exists()


def test_the_worst_verdict_wins_across_the_delivery(
        delivery, pipeline_file, tmp_path):
    """One bad flight in three is still a breach."""
    heights = np.concatenate([np.full(40, 30.0), np.full(40, 80.0),
                              np.full(40, 30.0)])
    xyz, gex = delivery(heights)
    result = run_qc(pipeline_file(QC_PIPELINE), xyz, gex, tmp_path / "out")
    assert result.return_value == EXIT_OUT_OF_SPEC


def test_a_step_raising_is_distinguished_from_a_bad_invocation(
        delivery, pipeline_file, tmp_path, monkeypatch):
    """3 is your mistake, 4 is our bug. They need different responses."""
    class Exploding:
        def __init__(self, pipeline, steps):
            pass

        def run(self, dataset):
            raise RuntimeError("boom")

    monkeypatch.setitem(cli_module.RUNNERS, "qc", Exploding)
    xyz, gex = delivery(np.full(40, 30.0), n_lines=2, n_flights=1)
    result = run_qc(pipeline_file(QC_PIPELINE), xyz, gex, tmp_path / "out")

    assert result.return_value == EXIT_STEP_RAISED


# ── Never started ────────────────────────────────────────────────────────────

def test_a_missing_survey_file_never_starts(delivery, pipeline_file, tmp_path):
    _, gex = delivery(np.full(40, 30.0), n_lines=2, n_flights=1)
    result = run_qc(pipeline_file(QC_PIPELINE), tmp_path / "absent.xyz", gex,
                    tmp_path / "out")
    assert result.return_value == EXIT_NOT_STARTED
    assert "No survey at" in result.output


def test_an_unknown_step_never_starts_and_suggests_the_real_one(
        delivery, pipeline_file, tmp_path):
    """The old failure was 'Unknown step ... Available: ' with an empty list."""
    xyz, gex = delivery(np.full(40, 30.0), n_lines=2, n_flights=1)
    typo = pipeline_file("steps: [{height_abov_ground: {good_max: 45}}]")

    result = run_qc(typo, xyz, gex, tmp_path / "out")

    assert result.return_value == EXIT_NOT_STARTED
    assert "height_above_ground" in result.output      # names the real step
    assert not (tmp_path / "out").exists()             # nothing was written


def test_nothing_is_written_when_the_pipeline_is_rejected(
        delivery, pipeline_file, tmp_path):
    """Validation happens before any data is touched."""
    xyz, gex = delivery(np.full(40, 30.0), n_lines=2, n_flights=1)
    out = tmp_path / "out"
    run_qc(pipeline_file("steps: [{nope: {}}]"), xyz, gex, out)
    assert not out.exists()


# ── Choosing the registry ────────────────────────────────────────────────────

def test_kind_in_the_document_selects_the_runner(
        delivery, pipeline_file, tmp_path):
    xyz, gex = delivery(np.full(40, 30.0), n_lines=2, n_flights=1)
    result = invoke("validate", pipeline_file(QC_PIPELINE))
    assert result.return_value == EXIT_GOOD
    assert "[qc]" in result.output


def test_the_runner_is_derived_when_no_kind_is_declared(
        delivery, pipeline_file, tmp_path):
    bare = pipeline_file(
        "steps: [{height_above_ground: {good_max: 45, marginal_max: 65}}]")
    xyz, gex = delivery(np.full(40, 30.0), n_lines=2, n_flights=1)

    result = run_qc(bare, xyz, gex, tmp_path / "out")
    assert result.return_value == EXIT_GOOD


def test_a_kind_that_disagrees_with_the_steps_is_an_error(
        delivery, pipeline_file, tmp_path):
    """Declared, but validated rather than trusted."""
    lying = pipeline_file(
        "kind: filters\nsteps: [{height_above_ground: "
        "{good_max: 45, marginal_max: 65}}]")
    xyz, gex = delivery(np.full(40, 30.0), n_lines=2, n_flights=1)

    result = run_qc(lying, xyz, gex, tmp_path / "out")

    assert result.return_value == EXIT_NOT_STARTED
    assert "declares kind: 'filters'" in result.output
    assert "registered under qc" in result.output      # says where they are


def test_group_overrides_the_document(delivery, pipeline_file, tmp_path):
    lying = pipeline_file(
        "kind: filters\nsteps: [{height_above_ground: "
        "{good_max: 45, marginal_max: 65}}]")
    xyz, gex = delivery(np.full(40, 30.0), n_lines=2, n_flights=1)

    result = run_qc(lying, xyz, gex, tmp_path / "out", "--group", "qc")
    assert result.return_value == EXIT_GOOD


# ── What gets written ────────────────────────────────────────────────────────

def _report(out, name="20260815"):
    return json.loads((out / f"{name}.report.json").read_text())


def test_the_report_carries_identity_and_data_derived_facts(
        delivery, pipeline_file, tmp_path):
    heights = np.concatenate([np.full(40, 30.0), np.full(40, 80.0),
                              np.full(40, 30.0)])
    xyz, gex = delivery(heights)
    out = tmp_path / "out"
    run_qc(pipeline_file(QC_PIPELINE), xyz, gex, out)

    report = _report(out)
    assert report["identity"] == "20260815"
    assert report["identity_source"] == "filename"
    # Read from the data, not from the filename — this is what drift orders by.
    assert report["flights"] == [1, 2, 3]


def test_a_label_overrides_the_filename_and_is_recorded_as_such(
        delivery, pipeline_file, tmp_path):
    xyz, gex = delivery(np.full(40, 30.0), n_lines=2, n_flights=1)
    out = tmp_path / "out"
    run_qc(pipeline_file(QC_PIPELINE), xyz, gex, out, "--label", "delivery_07")

    report = _report(out, "delivery_07")
    assert report["identity"] == "delivery_07"
    assert report["identity_source"] == "label"
    assert (out / "delivery_07.xyz").exists()


def test_the_report_records_the_pipeline_that_produced_it(
        delivery, pipeline_file, tmp_path):
    xyz, gex = delivery(np.full(40, 30.0), n_lines=2, n_flights=1)
    out = tmp_path / "out"
    run_qc(pipeline_file(QC_PIPELINE), xyz, gex, out)

    report = _report(out)
    assert report["pipeline"]["kind"] == "qc"
    assert report["steps"][0]["params"] == {"good_max": 45, "marginal_max": 65}


def test_verdict_columns_survive_into_the_exported_dataset(
        delivery, pipeline_file, tmp_path):
    """The exported file is what reaches a client, so the verdict must be in it."""
    xyz, gex = delivery(np.full(40, 80.0), n_lines=2, n_flights=1)
    out = tmp_path / "out"
    run_qc(pipeline_file(QC_PIPELINE), xyz, gex, out)

    written = load(out / "20260815.xyz", gex).flightlines
    assert "qs_height" in written         # the check's own verdict
    assert "qs_verdict" in written        # the rollup across checks
    assert "ql_oos" in written            # and the line's measures
    assert "ql_good" in written
    assert "ql_run" in written


def test_the_written_file_parses_back(delivery, pipeline_file, tmp_path):
    """The regression that says the output is a file and not just bytes.

    An enriched xyz is whitespace-delimited with no quoting, so two things break
    it and both did: a tier token containing a space becomes three fields, and
    an empty string occupies no field at all. Either one and the header's column
    count stops matching the rows, so the file the client receives will not open.

    Reading it back is the only assertion that catches that. Asserting on the
    header alone — which is what this file used to do — passes on a file no
    parser can read.
    """
    xyz, gex = delivery(np.full(40, 80.0), n_lines=2, n_flights=1)
    out = tmp_path / "out"
    result = run_qc(pipeline_file(QC_PIPELINE), xyz, gex, out)
    assert result.return_value == EXIT_OUT_OF_SPEC  # so verdicts are present

    written = out / "20260815.xyz"
    lines = [ln for ln in written.read_text().splitlines() if ln.strip()]
    # The header is a comment line, so it carries a leading marker the data rows
    # do not; every row after it must agree with every other.
    widths = {len(ln.split()) for ln in lines[1:]}
    assert len(widths) == 1, f"ragged file: rows have {sorted(widths)} fields"
    assert widths == {len(lines[0].split()) - 1}, "rows do not match the header"

    frame = load(written, gex).flightlines
    assert (frame["qs_height"] == OUT_OF_SPEC).all()
    assert (frame["qs_verdict"] == OUT_OF_SPEC).all()
    # Every sounding carries a reason, including the ones with nothing to say.
    assert frame["qs_reasons"].notna().all()


def test_a_clean_delivery_round_trips_too(delivery, pipeline_file, tmp_path):
    """The other half of the same bug: nothing wrong is still something to write."""
    xyz, gex = delivery(np.full(40, 30.0), n_lines=2, n_flights=1)
    out = tmp_path / "out"
    result = run_qc(pipeline_file(QC_PIPELINE), xyz, gex, out)
    assert result.return_value == EXIT_GOOD

    frame = load(out / "20260815.xyz", gex).flightlines
    assert (frame["qs_verdict"] == GOOD).all()
    assert (frame["qs_reasons"] == NO_REASON).all()


def test_re_running_replaces_that_delivery_and_leaves_others_alone(
        delivery, pipeline_file, tmp_path):
    """One record per delivery; drift accumulates across deliveries, not re-runs."""
    first, gex = delivery(np.full(40, 80.0), n_lines=2, n_flights=1,
                          name="20260815")
    second, _ = delivery(np.full(40, 30.0), n_lines=2, n_flights=1,
                         name="20260816")
    out = tmp_path / "out"
    pipeline = pipeline_file(QC_PIPELINE)

    run_qc(pipeline, first, gex, out)
    run_qc(pipeline, second, gex, out)
    # Re-run the first with a threshold that forgives it.
    relaxed = pipeline_file(
        "kind: qc\nsteps: [{height_above_ground: "
        "{good_max: 95, marginal_max: 99}}]", "relaxed.yaml")
    result = run_qc(relaxed, first, gex, out)

    assert result.return_value == EXIT_GOOD
    assert _report(out, "20260815")["steps"][0]["params"]["good_max"] == 95
    # The other delivery is untouched, so the drift series survives.
    assert _report(out, "20260816")["identity"] == "20260816"


def test_output_directory_is_created(delivery, pipeline_file, tmp_path):
    xyz, gex = delivery(np.full(40, 30.0), n_lines=2, n_flights=1)
    out = tmp_path / "deep" / "nested" / "out"
    assert run_qc(pipeline_file(QC_PIPELINE), xyz, gex, out).return_value == EXIT_GOOD
    assert out.is_dir()


# ── Introspection ────────────────────────────────────────────────────────────

def test_list_shows_registered_steps():
    result = invoke("list", "--group", "qc")
    assert "height_above_ground" in result.output


def test_list_shows_method_adapters():
    assert "tem" in invoke("list", "--group", "methods").output


def test_validate_does_not_need_a_dataset(pipeline_file):
    """The cheap pre-flight check a scheduler runs before booking a slot."""
    result = invoke("validate", pipeline_file(QC_PIPELINE))
    assert result.return_value == EXIT_GOOD
    assert "1 step(s)" in result.output
