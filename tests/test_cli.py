"""The command line — what makes this runnable without the platform.

Everything here writes to tmp_path and reads it back. No platform, no database,
no network: if these pass, a cron job on a bare machine works.
"""

import json

import numpy as np
import pytest
from click.testing import CliRunner

from conftest import make_dataset
from ymerflow_processing import cli as cli_module
from conftest import named, step, tier_names
from ymerflow_processing.dataset import load
from ymerflow_processing.cli import (
    EXIT_GOOD, EXIT_MARGINAL, EXIT_NOT_STARTED, EXIT_OUT_OF_SPEC,
    EXIT_STEP_RAISED, cli,
)
from ymerflow_processing.qc.rollup import NO_REASON
from ymerflow_processing.qc.thresholds import GOOD, OUT_OF_SPEC

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


def test_an_alias_file_and_a_crs_reach_the_loader(pipeline_file, tmp_path):
    """A contractor-named delivery runs from the command line with --alc, and
    --crs is what lets a step measure a distance or write geometry."""
    from test_dataset import _contractor_delivery
    xyz, alc, gex = _contractor_delivery(tmp_path)
    out = tmp_path / "out"

    result = run_qc(pipeline_file("kind: filters\nsteps: [{cull_negatives: {}}]"),
                    xyz, gex, out, "--alc", alc, "--crs", "EPSG:32613")

    assert result.exit_code == EXIT_GOOD, result.output
    assert "4 soundings" in result.output
    report = json.loads((out / "c.report.json").read_text())
    assert [s["step"] for s in report["steps"]] == ["inventory", "cull_negatives"]
    assert report["steps"][0]["distributions"]["channels"][0]["key"] == "ch1"


def test_a_scalefactor_on_the_command_line_is_recorded_where_the_importer_puts_it(
        pipeline_file, tmp_path):
    """A file that never went through the platform carries no scalefactor, and
    a physical noise floor compared against pV arrays as if they were V/m²
    is wrong by twelve orders of magnitude with nothing to say so."""
    from test_dataset import _contractor_delivery
    xyz, alc, gex = _contractor_delivery(tmp_path)
    out = tmp_path / "out"
    floor = ("kind: filters\nsteps:\n"
             "  - set_noise_floor: {noise_level_1ms: 2.0e-9}\n"
             "  - assign_uncertainty: {relative_fraction: 0.03}\n")

    without = run_qc(pipeline_file(floor, "a.yaml"), xyz, gex, out / "a",
                     "--alc", alc)
    assert "carries no scalefactor" in without.output

    with_it = run_qc(pipeline_file(floor, "b.yaml"), xyz, gex, out / "b",
                     "--alc", alc, "--scalefactor", "1e-12")
    assert "carries no scalefactor" not in with_it.output
    processed = load(out / "b" / "c.xyz", gex)
    assert float(processed.survey.xyz.model_info["scalefactor"]) == 1e-12


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
    assert step(report, "height_above_ground")["params"] == {
        "good_max": 45, "marginal_max": 65}
    # Every run opens by accounting for the delivery.
    assert named(report)[0] == "inventory"


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
    # Comment lines carry a leading marker and are not rows: the column header
    # is one, and so is any metadata the dataset records about itself. Every
    # actual row must agree with every other and with the header.
    comments = [ln for ln in lines if ln.lstrip().startswith("/")]
    rows = [ln for ln in lines if not ln.lstrip().startswith("/")]
    widths = {len(ln.split()) for ln in rows}
    assert len(widths) == 1, f"ragged file: rows have {sorted(widths)} fields"
    assert widths == {len(comments[-1].split()) - 1}, "rows do not match the header"

    frame = load(written, gex).flightlines
    assert set(tier_names(frame["qs_height"])) == {OUT_OF_SPEC}
    assert set(tier_names(frame["qs_verdict"])) == {OUT_OF_SPEC}
    # Every sounding carries a reason, including the ones with nothing to say.
    assert frame["qs_reasons"].notna().all()


def test_a_clean_delivery_round_trips_too(delivery, pipeline_file, tmp_path):
    """The other half of the same bug: nothing wrong is still something to write."""
    xyz, gex = delivery(np.full(40, 30.0), n_lines=2, n_flights=1)
    out = tmp_path / "out"
    result = run_qc(pipeline_file(QC_PIPELINE), xyz, gex, out)
    assert result.return_value == EXIT_GOOD

    frame = load(out / "20260815.xyz", gex).flightlines
    assert set(tier_names(frame["qs_verdict"])) == {GOOD}
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
    assert step(_report(out, "20260815"),
                "height_above_ground")["params"]["good_max"] == 95
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


# ── compare, and the two input kinds ─────────────────────────────────────────
# 04 deferred --dataset until "the rebuild's equivalence testing needs it".
# It does now: three of the four registered checks consume datasets, and
# without this they cannot be run from a command line at all.

DRIFT = """
kind: compare
steps:
  - noise_floor_drift: {}
"""

DIFF = """
kind: compare
steps:
  - version_diff: {}
"""


@pytest.fixture
def stored_report(tmp_path, pipeline_file, delivery):
    """A real report on disk, produced by a real run."""
    def build(name):
        xyz, gex = delivery(np.full(20, 30.0), n_lines=2, n_flights=1,
                            name=name)
        run_qc(pipeline_file(QC_PIPELINE), xyz, gex, tmp_path / "runs")
        return tmp_path / "runs" / f"{name}.report.json"
    return build


def test_reports_are_still_the_cheap_path(stored_report, pipeline_file,
                                          tmp_path):
    """No system description, no arrays — a pure function over small records."""
    out = tmp_path / "drift.json"
    result = invoke("compare", pipeline_file(DRIFT, "d.yaml"),
                    "--report", stored_report("20260815"),
                    "--report", stored_report("20260816"), "--out", out)

    assert result.return_value == EXIT_GOOD
    assert json.loads(out.read_text())["steps"][0]["step"] == "noise_floor_drift"


def test_a_dataset_consuming_check_runs_from_the_command_line(
        delivery, pipeline_file, tmp_path):
    """The gap this closes. Before it, version_diff was registered, tested and
    unreachable — --report fed its path to json.loads."""
    new, gex = delivery(np.full(20, 30.0), n_lines=2, n_flights=1, name="v2")
    old, _ = delivery(np.full(20, 30.0), n_lines=2, n_flights=1, name="v1")
    out = tmp_path / "diff.json"

    result = invoke("compare", pipeline_file(DIFF, "v.yaml"),
                    "--dataset", new, "--reference", old, "--gex", gex,
                    "--out", out)

    assert result.return_value == EXIT_GOOD
    written = json.loads(out.read_text())["steps"][0]
    assert written["step"] == "version_diff"
    assert written["distributions"]["alignment"]["soundings_compared"] == 20


def test_the_pipeline_is_asked_before_anything_is_read(
        delivery, pipeline_file, tmp_path):
    """Which flag to pass is not a preference: each check declares what it
    consumes, so passing the wrong one is answered rather than crashed on."""
    xyz, gex = delivery(np.full(20, 30.0), n_lines=2, n_flights=1)
    result = invoke("compare", pipeline_file(DIFF, "v.yaml"),
                    "--report", xyz, "--out", tmp_path / "o.json")

    assert result.return_value == EXIT_NOT_STARTED
    assert "consumes datasets" in result.output


def test_a_survey_file_passed_as_a_report_says_what_is_wrong(
        delivery, pipeline_file, tmp_path):
    """A raw JSONDecodeError names the column it choked on rather than the
    thing that is actually wrong."""
    xyz, gex = delivery(np.full(20, 30.0), n_lines=2, n_flights=1)
    result = invoke("compare", pipeline_file(DRIFT, "d.yaml"),
                    "--report", xyz, "--out", tmp_path / "o.json")

    assert result.return_value == EXIT_NOT_STARTED
    assert "is not a stored report" in result.output
    assert "Traceback" not in result.output


def test_datasets_need_a_system_description(delivery, pipeline_file, tmp_path):
    """A survey file does not describe its own gates, and the channels a
    comparison aligns on come from the description."""
    xyz, gex = delivery(np.full(20, 30.0), n_lines=2, n_flights=1)
    result = invoke("compare", pipeline_file(DIFF, "v.yaml"),
                    "--dataset", xyz, "--out", tmp_path / "o.json")

    assert result.return_value == EXIT_NOT_STARTED
    assert "--gex is required" in result.output


def test_a_pipeline_mixing_the_two_kinds_is_refused(pipeline_file, tmp_path,
                                                    stored_report):
    """One run loads one set of inputs, and naming which steps want which beats
    loading one kind and letting the rest fail on it."""
    document = """
        kind: compare
        steps:
          - noise_floor_drift: {}
          - version_diff: {}
    """
    result = invoke("compare", pipeline_file(document, "m.yaml"),
                    "--report", stored_report("20260815"),
                    "--out", tmp_path / "o.json")

    assert result.return_value == EXIT_NOT_STARTED
    assert "version_diff" in result.output and "noise_floor_drift" in result.output


def test_both_flags_at_once_is_refused(delivery, pipeline_file, tmp_path,
                                       stored_report):
    xyz, gex = delivery(np.full(20, 30.0), n_lines=2, n_flights=1, name="x")
    result = invoke("compare", pipeline_file(DIFF, "v.yaml"),
                    "--dataset", xyz, "--report", stored_report("20260815"),
                    "--gex", gex, "--out", tmp_path / "o.json")

    assert result.return_value == EXIT_NOT_STARTED
    assert "One run loads one set of inputs" in result.output


def test_nothing_to_compare_says_which_flag_to_pass(pipeline_file, tmp_path):
    result = invoke("compare", pipeline_file(DIFF, "v.yaml"),
                    "--out", tmp_path / "o.json")

    assert result.return_value == EXIT_NOT_STARTED
    assert "--dataset" in result.output


def test_a_gex_given_to_a_report_comparison_is_noted_not_refused(
        stored_report, pipeline_file, delivery, tmp_path):
    """Harmless, so it runs — but silently ignoring an option someone typed is
    how they come to believe it did something."""
    _, gex = delivery(np.full(4, 30.0), n_lines=1, n_flights=1, name="g")
    result = invoke("compare", pipeline_file(DRIFT, "d.yaml"),
                    "--report", stored_report("20260815"), "--gex", gex,
                    "--out", tmp_path / "o.json")

    assert result.return_value == EXIT_GOOD
    assert "ignored" in result.output


def test_the_order_of_datasets_is_preserved(delivery, pipeline_file, tmp_path):
    """02 states the sequence is ordered and a check may rely on it. A repeated
    option keeps the order typed, which a glob would not."""
    paths = []
    for index, name in enumerate(("w3", "w5", "w7")):
        xyz, gex = delivery(np.full(12, 30.0), n_lines=1, n_flights=1,
                            name=name)
        dataset = load(xyz, gex)
        dataset.history = [{"step": "average_along_line",
                            "params": {"width_first": 3 + 2 * index}}]
        dataset.survey.dump(xyzfile=str(xyz))
        paths.append(xyz)

    document = """
        kind: compare
        steps:
          - parameter_sweep: {}
    """
    out = tmp_path / "sweep.json"
    result = invoke("compare", pipeline_file(document, "s.yaml"),
                    *[arg for p in paths for arg in ("--dataset", p)],
                    "--gex", gex, "--out", out)

    assert result.return_value == EXIT_GOOD
    series = json.loads(out.read_text())["steps"][0]["distributions"]["series"]
    assert [entry["value"] for entry in series] == [3, 5, 7]


# ── Exit 5: ran, and could not grade ─────────────────────────────────────────
# The defect this closes: _VERDICT_EXIT[None] was EXIT_GOOD, so a run that
# graded nothing at all exited 0 — "completed, everything Good". Six of nine
# checks found no columns on a real delivery and the run read as a clean survey.

UNGRADEABLE = """
kind: qc
steps:
  - height_above_ground: {good_max: 45, marginal_max: 65, column: no_such_column}
"""


def test_a_run_that_could_not_grade_does_not_exit_good(delivery, pipeline_file,
                                                       tmp_path):
    """A check that was asked for and could not act leaves the flight ungraded,
    not clean."""
    from ymerflow_processing.cli import EXIT_UNGRADED

    xyz, gex = delivery(np.full(20, 30.0), n_lines=2, n_flights=1)
    document = """
        kind: qc
        steps:
          - system_attitude: {good: 5, marginal: 7.5}
    """
    out = tmp_path / "out"
    # A delivery with no attitude at all: the check cannot act.
    dataset = load(xyz, gex)
    frame = dataset.flightlines.drop(columns=["tilt_x", "tilt_y"])
    dataset.survey.xyz.flightlines = frame
    dataset.survey.dump(xyzfile=str(xyz))

    result = run_qc(pipeline_file(document, "u.yaml"), xyz, gex, out)
    assert result.return_value == EXIT_UNGRADED
    assert "UNGRADED, not clean" in result.output


def test_a_breach_still_outranks_an_ungraded_run(delivery, pipeline_file,
                                                 tmp_path):
    """A line flown out of spec is a finding someone must act on; 'some checks
    did not run' is a finding about the run. Both are on stdout, so reporting
    the breach loses nothing."""
    heights = np.concatenate([np.full(10, 30.0), np.full(10, 90.0)])
    xyz, gex = delivery(heights, n_lines=2, n_flights=1)
    document = """
        kind: qc
        steps:
          - height_above_ground: {good_max: 45, marginal_max: 65}
          - transmitter_stability: {good_fraction: 0.01, marginal_fraction: 0.02}
    """
    result = run_qc(pipeline_file(document, "b.yaml"), xyz, gex, tmp_path / "o")
    assert result.return_value == EXIT_OUT_OF_SPEC
    assert "UNGRADED" in result.output          # said, but does not set the code


def test_an_optional_check_does_not_make_a_run_ungraded(delivery, pipeline_file,
                                                        tmp_path):
    """03 requires data_misfit and culled_fraction be absent 'with no error' on
    a delivery that was never inverted."""
    xyz, gex = delivery(np.full(20, 30.0), n_lines=2, n_flights=1)
    document = """
        kind: qc
        steps:
          - height_above_ground: {good_max: 45, marginal_max: 65}
          - data_misfit: {good_max: 1.0, marginal_max: 2.0}
          - culled_fraction: {}
    """
    result = run_qc(pipeline_file(document, "o.yaml"), xyz, gex, tmp_path / "o")
    assert result.return_value == EXIT_GOOD
    assert "UNGRADED" not in result.output


def test_a_missing_powerline_monitor_is_an_absent_instrument():
    """Not every system carries one, so its absence is not a fault."""
    from ymerflow_processing.pipeline import OPTIONAL
    from ymerflow_processing.qc.powerline import powerline_noise

    dataset = make_dataset(n=8, n_lines=1, n_gates=4)
    _, report = powerline_noise(dataset)
    assert report.ungraded == OPTIONAL


def test_the_reason_is_in_the_report(delivery, pipeline_file, tmp_path):
    """Three things look identical in a report and are not."""
    xyz, gex = delivery(np.full(20, 30.0), n_lines=2, n_flights=1)
    document = """
        kind: qc
        steps:
          - height_above_ground: {good_max: 45, marginal_max: 65}
          - data_misfit: {}
    """
    out = tmp_path / "o"
    run_qc(pipeline_file(document, "r.yaml"), xyz, gex, out)
    written = {s["step"]: s["ungraded"] for s in _report(out)["steps"]}

    assert written["height_above_ground"] is None      # graded
    assert written["data_misfit"] == "optional"
