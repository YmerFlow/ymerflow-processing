"""One parameter varied across a series — 02's averaging-width comparison.

The question a sweep answers is not which run is best. It is how the output
responds to the parameter, because the choice is a trade: a wider window
suppresses more noise and resolves less, and the number worth having is where
one stops buying the other.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from conftest import make_dataset
from ymerflow_processing.core.compare.sweep import (
    CONSUMES, STEP, parameter_sweep,
)
from ymerflow_processing.core.pipeline import (
    COMPARE_GROUP, ComparisonRunner, Pipeline, load_steps,
)

N = 12
GATES = 6


def run(width, divergence=0.0, response=None, step="average_along_line"):
    """A dataset that records having been produced at ``width``."""
    dataset = make_dataset(n=N, n_lines=1, n_gates=GATES)
    dataset.history = [{"step": step, "params": {"width": width}}]
    if divergence:
        for channel in dataset.channels:
            dataset.layer_data[channel.data_key] *= (1 + divergence)
    if response is not None:
        dataset.flightlines["resdata"] = float(response)
    return dataset


def series(*widths, **kw):
    return [run(w, **kw) for w in widths]


# ── It discovers what varied ─────────────────────────────────────────────────

def test_the_varied_parameter_is_read_from_the_runs():
    """A caller who has just produced five datasets should not have to restate
    the five numbers that produced them — restating is how a report comes to
    describe a sweep that was never run."""
    report = parameter_sweep([run(3), run(5), run(7)])

    assert report.distributions["parameter"]["name"] == "width"
    assert report.distributions["parameter"]["step"] == "average_along_line"
    assert [e["value"] for e in report.distributions["series"]] == [3, 5, 7]


def test_two_varying_parameters_are_refused():
    """A response that cannot be attributed to a parameter is not a sweep, and
    a curve drawn over the ambiguity looks exactly like a real one."""
    a, b = run(3), run(5)
    b.history = [{"step": "average_along_line",
                  "params": {"width": 5, "taper": "boxcar"}}]
    a.history = [{"step": "average_along_line",
                  "params": {"width": 3, "taper": "hann"}}]

    with pytest.raises(ValueError, match="one parameter at a time"):
        parameter_sweep([a, b])


def test_a_series_where_nothing_varies_says_so():
    report = parameter_sweep([run(5), run(5)])
    assert report.distributions["parameter"] is None
    assert any("nothing to sweep" in note for note in report.notes)


def test_runs_with_no_recorded_parameters_say_where_to_look():
    a, b = make_dataset(n=N, n_gates=GATES), make_dataset(n=N, n_gates=GATES)
    report = parameter_sweep([a, b])
    assert any("rather than by calling the step directly" in note
               for note in report.notes)


def test_a_parameter_whose_value_is_a_list_is_still_comparable():
    a, b = run(3), run(5)
    a.history = [{"step": "select_channels", "params": {"keep": ["LM"]}}]
    b.history = [{"step": "select_channels", "params": {"keep": ["LM", "HM"]}}]

    assert parameter_sweep([a, b]).distributions["parameter"]["name"] == "keep"


# ── The response ─────────────────────────────────────────────────────────────

def test_divergence_is_measured_against_the_control():
    control = run(1)
    report = parameter_sweep([run(3, divergence=0.1), run(5, divergence=0.2)],
                             reference=control)

    values = [e["divergence_from_control"]
              for e in report.distributions["series"]]
    assert values[0] == pytest.approx(0.1, rel=1e-6)
    assert values[1] == pytest.approx(0.2, rel=1e-6)


def test_without_a_control_the_first_run_is_the_baseline():
    """The conventional sweep: a series starting from the current setting."""
    report = parameter_sweep(series(3, 5, 7))

    assert report.distributions["control_is_a_member_of_the_series"]
    assert any("first run is the baseline" in note for note in report.notes)


def test_a_named_column_is_traced_alongside_the_divergence():
    """A noise estimate or a misfit is a better answer where one exists, which
    is why any column can be named."""
    runs = [run(3, response=2.0), run(5, response=1.5), run(7, response=1.4)]
    report = parameter_sweep(runs, response_column="resdata")

    assert [e["response"] for e in report.distributions["series"]] == \
        [2.0, 1.5, 1.4]


def test_a_named_column_no_run_carries_is_said_rather_than_silently_dropped():
    report = parameter_sweep(series(3, 5), response_column="resdata")
    assert any("only the divergence" in note for note in report.notes)


def test_divergence_ignores_what_no_run_kept():
    from ymerflow_processing.core.mask import CULLED
    control = run(1)
    varied = run(3)
    varied.layer_data[varied.channels[0].data_key].iloc[:, 5] = 1e9
    for dataset in (control, varied):
        for channel in dataset.channels:
            dataset.layer_data[channel.inuse_key].iloc[:, 5] = CULLED

    report = parameter_sweep([varied, run(5)], reference=control)
    assert report.distributions["series"][0]["divergence_from_control"] == \
        pytest.approx(0.0)


# ── Where the response flattens ──────────────────────────────────────────────

def test_the_flattening_point_is_found():
    """Beyond this width, nothing changes — which is the averaging-width
    answer."""
    runs = [run(3, response=10.0), run(5, response=5.0), run(7, response=4.9),
            run(9, response=4.85)]
    report = parameter_sweep(runs, response_column="resdata")

    assert report.distributions["levels_off_at"]["at"] == 5


def test_a_response_that_never_flattens_reports_none():
    runs = [run(3, response=10.0), run(5, response=7.0), run(7, response=4.0),
            run(9, response=1.0)]
    assert parameter_sweep(runs, response_column="resdata"
                           ).distributions["levels_off_at"] is None


def test_the_flattening_point_is_an_observation_not_a_recommendation():
    """The trade a sweep explores is the user's to make, and a knee in one
    response is not a knee in the one they care about."""
    runs = [run(3, response=10.0), run(5, response=5.0), run(7, response=4.9),
            run(9, response=4.85)]
    report = parameter_sweep(runs, response_column="resdata")

    assert any("not a recommendation" in note for note in report.notes)
    assert any("nothing here passes, fails, or recommends" in note
               for note in report.notes)


def test_too_few_points_to_see_a_shape_reports_none():
    runs = [run(3, response=10.0), run(5, response=5.0)]
    assert parameter_sweep(runs, response_column="resdata"
                           ).distributions["levels_off_at"] is None


def test_a_series_given_out_of_order_is_sorted_by_the_parameter():
    """Otherwise the curve zigzags and a knee lands wherever it doubled back."""
    runs = [run(9, response=4.85), run(3, response=10.0), run(5, response=5.0),
            run(7, response=4.9)]
    report = parameter_sweep(runs, response_column="resdata")

    assert [e["value"] for e in report.distributions["series"]] == [3, 5, 7, 9]
    assert report.distributions["levels_off_at"]["at"] == 5
    assert any("given out of order" in note for note in report.notes)


def test_a_non_numeric_parameter_is_left_in_the_order_given():
    a, b, c = run(3), run(5), run(7)
    for dataset, taper in ((a, "hann"), (b, "boxcar"), (c, "triangle")):
        dataset.history = [{"step": "average_along_line",
                            "params": {"taper": taper}}]

    report = parameter_sweep([a, b, c])
    assert report.distributions["levels_off_at"] is None
    assert any("not numeric" in note for note in report.notes)


# ── Contract ─────────────────────────────────────────────────────────────────

def test_a_sweep_needs_more_than_one_run():
    with pytest.raises(ValueError, match="version_diff"):
        parameter_sweep([run(3)])


def test_it_declares_that_it_needs_datasets():
    assert CONSUMES == "datasets"
    assert ComparisonRunner(Pipeline.parse(f"""
        kind: compare
        steps:
          - {STEP}: {{}}
    """)).consumes(STEP) == "datasets"


def test_a_report_given_where_a_dataset_was_needed_says_so():
    with pytest.raises(TypeError, match="consumes: datasets"):
        parameter_sweep([run(3), {"steps": []}])


def test_alignment_notes_are_not_repeated_once_per_run():
    """They belong to the sweep as a whole; one note per run would bury the
    report under four copies of the same sentence."""
    control = run(1)
    runs = series(3, 5, 7)
    for dataset in runs + [control]:
        xyz = dataset.survey.xyz
        xyz.flightlines = dataset.flightlines.drop(
            columns=[c for c in (xyz.x_column, xyz.y_column)
                     if c in dataset.flightlines.columns])

    report = parameter_sweep(runs, reference=control)
    assert sum("BY ORDER" in note for note in report.notes) == 1


def test_a_real_sweep_over_a_real_filter():
    """The discovery premise end to end: the runner writes history, and the
    sweep reads the widths back out of it rather than being told."""
    from ymerflow_processing.core.pipeline import Pipeline, ProcessingRunner

    runs = []
    for width in (3, 5, 7, 9):
        dataset, _ = ProcessingRunner(Pipeline.parse(f"""
            kind: filters
            steps:
              - average_along_line: {{width_first: {width}}}
        """)).run(make_dataset(n=40, n_lines=1, n_gates=GATES))
        runs.append(dataset)

    report = parameter_sweep(runs)
    assert report.distributions["parameter"] == {
        "step": "average_along_line", "at": 0, "name": "width_first"}
    assert [e["value"] for e in report.distributions["series"]] == [3, 5, 7, 9]


def test_it_runs_from_a_pipeline_document():
    report = ComparisonRunner(Pipeline.parse(f"""
        kind: compare
        steps:
          - {STEP}: {{response_column: resdata}}
    """)).run([run(3, response=2.0), run(5, response=1.5)])

    assert report.steps[0].distributions["parameter"]["name"] == "width"


def test_the_report_survives_serialization():
    runs = [run(3, response=10.0), run(5, response=5.0), run(7, response=4.9)]
    restored = json.loads(json.dumps(
        parameter_sweep(runs, response_column="resdata").to_dict(),
        default=str))
    assert restored["distributions"]["series"][0]["value"] == 3


def test_registered():
    assert STEP in load_steps(COMPARE_GROUP)
