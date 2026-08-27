"""The pipeline documents that ship with the package.

`02`'s first lever against the step-list problem:

    Most users should never see the list. The standard-workflow story — load a
    standard chain and adjust it — means someone doing ordinary work never
    browses filters at all.

Which only works if the chains ship with the package, and only stays true if
they keep running. **A shipped document that has stopped working should fail the
suite rather than fail a user**, so every one of them is executed here rather
than merely parsed. Parsing proves the YAML is well formed; it does not prove a
parameter still exists.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml
from click.testing import CliRunner

from conftest import make_dataset
from ymerflow_processing.core.cli import EXIT_NOT_STARTED, cli
from ymerflow_processing.core.pipeline import (
    GROUPS, Pipeline, ProcessingRunner, QCRunner, bundled, describe_bundled,
    load_steps, resolve_group,
)

RUNNERS = {"filters": ProcessingRunner, "qc": QCRunner}
NAMES = sorted(bundled())


@pytest.fixture
def delivery():
    """A delivery with the columns the shipped documents name."""
    dataset = make_dataset(n=60, n_lines=3, n_flights=2, n_gates=8,
                           height=np.full(60, 35.0), second_altimeter=True)
    return dataset


def document(name):
    return yaml.safe_load(bundled()[name].read_text())


# ── They exist and are reachable ─────────────────────────────────────────────

def test_the_package_ships_pipelines():
    assert NAMES, "no bundled pipelines — check package_data in setup.py"


def test_no_shipped_document_names_a_column_it_cannot_know():
    """A shipped default must degrade rather than die.

    Naming a column in a shipped document overrides discovery and raises on any
    delivery that spells it differently. Both times this rule was broken the
    symptom was the same — a real delivery, a hard stop, a column name nobody
    could have guessed right for every file:

    * ``altimeter_agreement`` naming its two altimeters, which every system
      spells differently. It ships commented out instead.
    * ``correct_tilt`` naming ``tilt_x``/``tilt_y`` while the delivery spelled
      them ``AngleX``/``AngleY``.

    So the rule is general rather than a list of the two we found: a shipped
    document names no column at all. Where a step genuinely cannot work without
    one, it ships commented out with instructions."""
    offenders = []
    for name in NAMES:
        for step, params in Pipeline.load(name).steps:
            for parameter in params:
                if parameter == "columns" or parameter.endswith("_column"):
                    offenders.append(f"{name}: {step}.{parameter}")
    assert not offenders, (
        "shipped documents must let columns be discovered, since a named one "
        "raises on a delivery that spells it differently: "
        + "; ".join(offenders))


def test_each_carries_a_name_and_a_description():
    """`yf-process pipelines` lists them by description, so a document without
    one is unfindable by anyone who does not already know it exists."""
    for name in NAMES:
        doc = document(name)
        assert doc.get("name") == name, f"{name}: name key disagrees with filename"
        assert doc.get("description"), f"{name}: no description"


def test_a_name_resolves_without_a_path():
    assert Pipeline.load("acquisition-qc").steps


def test_a_real_file_beats_a_bundled_name(tmp_path):
    """The surprising direction, and the one that would have someone editing a
    file that is not the one running."""
    local = tmp_path / "acquisition-qc"
    local.write_text("kind: qc\nsteps: [{ground_speed: {good_min: 1, good_max: 2,"
                     " marginal_min: 0, marginal_max: 3}}]\n")

    import os
    here = os.getcwd()
    try:
        os.chdir(tmp_path)
        assert [n for n, _ in Pipeline.load("acquisition-qc").steps] == ["ground_speed"]
    finally:
        os.chdir(here)


def test_an_unknown_name_lists_what_there_is():
    with pytest.raises(FileNotFoundError, match="acquisition-qc"):
        Pipeline.load("no-such-pipeline")


# ── They are valid ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", NAMES)
def test_every_step_named_is_registered(name):
    """The cheap failure: a document naming a step that was renamed."""
    pipeline = Pipeline.load(name)
    group, steps = resolve_group(pipeline, list(RUNNERS))
    pipeline.validate(steps)


@pytest.mark.parametrize("name", NAMES)
def test_every_parameter_named_exists_on_its_step(name):
    """The expensive failure, and the reason parsing is not enough: a parameter
    that was renamed leaves the document parseable and the run broken."""
    import inspect

    pipeline = Pipeline.load(name)
    _, steps = resolve_group(pipeline, list(RUNNERS))
    for step, params in pipeline.steps:
        signature = inspect.signature(steps[step])
        unknown = [p for p in params if p not in signature.parameters]
        assert not unknown, f"{name}: {step} has no parameter(s) {unknown}"


@pytest.mark.parametrize("name", NAMES)
def test_every_declared_kind_matches_its_steps(name):
    """A document cannot quietly misdescribe itself."""
    pipeline = Pipeline.load(name)
    group, _ = resolve_group(pipeline, list(RUNNERS))
    assert pipeline.kind == group


# ── They run ─────────────────────────────────────────────────────────────────

def test_acquisition_qc_runs_and_grades(delivery):
    dataset, report = QCRunner(Pipeline.load("acquisition-qc")).run(delivery)

    assert dataset.flightlines["qs_height"].notna().any()
    assert "qs_score" in dataset.flightlines.columns
    # The two optional checks are absent without complaint on a raw delivery.
    steps = {s.step: s for s in report.steps}
    assert "not an error" in " ".join(steps["data_misfit"].notes)


def test_acquisition_qc_changes_no_measurement(delivery):
    """QC writes attributes; deciding what to remove is a different job."""
    before = {k: v.copy() for k, v in delivery.layer_data.items()}
    dataset, _ = QCRunner(Pipeline.load("acquisition-qc")).run(delivery)

    import pandas as pd
    for key, array in before.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


def test_standard_processing_runs_end_to_end(delivery):
    dataset, report = ProcessingRunner(
        Pipeline.load("standard-processing")).run(delivery)

    ran = [s.step for s in report.steps]
    assert "estimate_noise_model" in ran and "average_along_line" in ran
    assert dataset.n_soundings > 0


def test_standard_processing_orders_noise_before_averaging(delivery):
    """01: estimation precedes assignment, and assignment precedes averaging so
    that averaging has uncertainties to propagate. A wrong order here does not
    fail — it produces plausible nonsense — so the document is checked."""
    ran = [name for name, _ in Pipeline.load("standard-processing").steps]

    assert ran.index("estimate_noise_model") < ran.index("assign_uncertainty")
    assert ran.index("assign_uncertainty") < ran.index("average_along_line")


def test_standard_processing_culls_before_averaging():
    """So a bad gate is not smeared into its neighbours."""
    ran = [name for name, _ in Pipeline.load("standard-processing").steps]
    culls = [i for i, n in enumerate(ran) if n.startswith("cull_")]
    assert culls and max(culls) < ran.index("average_along_line")


def test_standard_processing_does_not_list_the_mandate():
    """01: filter_20 is applied automatically, not listed in a pipeline. Listing
    it would suggest it is a judgement someone makes, and it is not."""
    ran = [name for name, _ in Pipeline.load("standard-processing").steps]
    assert "disable_initial_gates" not in ran


def test_synthetic_data_runs_and_perturbs(delivery):
    dataset, report = ProcessingRunner(Pipeline.load("synthetic-data")).run(delivery)

    # Every run opens with an implicit inventory before any step acts.
    ran = [s.step for s in report.steps if s.step != "inventory"]
    assert ran == ["estimate_noise_model", "assign_uncertainty",
                   "perturb_from_uncertainty"]


def test_synthetic_data_supplies_the_noise_model_rather_than_measuring_it():
    """A forward model has no scatter to estimate from — it is noiseless by
    construction — so the model has to be stated."""
    steps = dict(Pipeline.load("synthetic-data").steps)
    assert steps["estimate_noise_model"]["noise_level_1ms"] > 0


def test_synthetic_data_assigns_before_it_draws():
    """01 singles this out. Perturbing before assignment draws from a uniform
    system-description default, which is flat in relative terms, so late gates
    come out as clean as early ones — the result looks like data and is not."""
    ran = [name for name, _ in Pipeline.load("synthetic-data").steps]
    assert ran.index("assign_uncertainty") < ran.index("perturb_from_uncertainty")


def test_synthetic_data_pins_its_seed():
    """A fair test that cannot be repeated is not a fair test."""
    steps = dict(Pipeline.load("synthetic-data").steps)
    assert "seed" in steps["perturb_from_uncertainty"]


# ── The commands that make them findable ─────────────────────────────────────

def invoke(*args):
    return CliRunner().invoke(cli, list(map(str, args)), standalone_mode=False)


def test_pipelines_lists_them_with_descriptions():
    output = invoke("pipelines").output
    for name, description in describe_bundled().items():
        assert name in output
        assert description.split()[0] in output


def test_show_prints_the_document_verbatim():
    """The comments are most of the document and explain why the order is what
    it is, so re-emitting parsed YAML would discard the useful half."""
    output = invoke("show", "standard-processing").output
    assert "THE ORDER IS THE POINT" in output
    assert yaml.safe_load(output)["name"] == "standard-processing"


def test_show_on_an_unknown_name_points_at_the_listing():
    result = invoke("show", "nope")
    assert result.return_value == EXIT_NOT_STARTED
    assert "yf-process pipelines" in result.output


def test_run_accepts_a_bundled_name(tmp_path, delivery):
    xyz, gex = tmp_path / "d.xyz", tmp_path / "d.gex"
    delivery.survey.dump(xyzfile=str(xyz), gexfile=str(gex))

    result = invoke("run", "acquisition-qc", "--xyz", xyz, "--gex", gex,
                    "--out", tmp_path / "out")
    assert result.return_value in (0, 1, 2)
    assert (tmp_path / "out" / "d.report.json").is_file()


def test_run_on_an_unknown_name_does_not_traceback():
    result = invoke("run", "nope", "--xyz", "a", "--gex", "b", "--out", "c")
    assert result.return_value == EXIT_NOT_STARTED
    assert "no bundled pipeline" in result.output
