"""Line selection — classify, then select on what was classified.

The pair 01 decided filter_01-04 should collapse into. The interesting
assertions are about the two things the plan predates: a label that cannot be
written to a whitespace-delimited format, and a subset that must move every
array together or silently misalign the dataset.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.lines import (
    DEFAULT_LINE_TYPE, LINE_TYPE_COLUMN, classify_lines, select_lines,
)
from ymerflow_processing.core.pipeline import (
    FILTER_GROUP, Pipeline, ProcessingRunner, load_steps,
)

N = 12
GATES = 8


def delivery(height=None, **kw):
    """Three lines of four soundings; the middle one flown high."""
    if height is None:
        height = np.concatenate([np.full(4, 30.0), np.full(4, 600.0),
                                 np.full(4, 30.0)])
    return make_dataset(n=N, n_lines=3, n_gates=GATES, height=list(height), **kw)


# ── Classifying ──────────────────────────────────────────────────────────────

def test_a_high_line_is_labelled_and_nothing_is_removed():
    """01: "Nothing is removed — this only classifies, so a later selection step
    can act on the label.\""""
    dataset, report = classify_lines(delivery(), criterion="height_threshold",
                                     threshold=500, line_type="High-Altitude")
    frame = dataset.flightlines

    assert len(frame) == N
    assert set(frame.loc[frame["line_id"] == 101, LINE_TYPE_COLUMN]) == {
        "High-Altitude"}
    assert set(frame.loc[frame["line_id"] == 100, LINE_TYPE_COLUMN]) == {
        DEFAULT_LINE_TYPE}
    assert report.distributions["lines_classified"] == 1


def test_it_is_the_line_mean_that_decides():
    """A test line is flown high throughout; one high sounding on a production
    line is qc_01's finding, not a reclassification."""
    heights = np.full(N, 30.0)
    # Above the threshold on its own, but four soundings to a line, so the mean
    # is 172.5 and the line is not a test line.
    heights[0] = 600.0
    dataset, report = classify_lines(delivery(height=heights),
                                     criterion="height_threshold",
                                     threshold=500, line_type="High-Altitude")

    assert set(dataset.flightlines[LINE_TYPE_COLUMN]) == {DEFAULT_LINE_TYPE}
    assert report.distributions["line_means"]["100"] == pytest.approx(172.5)


def test_lines_can_be_named_explicitly():
    dataset, _ = classify_lines(delivery(), criterion="explicit",
                                lines=[100, 102], line_type="Reflight")
    frame = dataset.flightlines

    assert set(frame.loc[frame["line_id"] == 100, LINE_TYPE_COLUMN]) == {"Reflight"}
    assert set(frame.loc[frame["line_id"] == 101, LINE_TYPE_COLUMN]) == {
        DEFAULT_LINE_TYPE}


def test_naming_a_line_that_is_not_there_is_an_error():
    with pytest.raises(KeyError, match="999"):
        classify_lines(delivery(), criterion="explicit", lines=[999],
                       line_type="Reflight")


def test_classifying_twice_layers_rather_than_resetting():
    """The default is what a line is until something says otherwise, not
    something reapplied on every pass."""
    dataset, _ = classify_lines(delivery(), criterion="explicit", lines=[100],
                                line_type="Reflight")
    dataset, _ = classify_lines(dataset, criterion="explicit", lines=[102],
                                line_type="Tie")
    frame = dataset.flightlines

    assert set(frame.loc[frame["line_id"] == 100, LINE_TYPE_COLUMN]) == {"Reflight"}
    assert set(frame.loc[frame["line_id"] == 102, LINE_TYPE_COLUMN]) == {"Tie"}


def test_the_vocabulary_is_free_form():
    """A survey defines its own categories; there is no fixed list."""
    dataset, _ = classify_lines(delivery(), criterion="explicit", lines=[100],
                                line_type="Calibration-Pass-2")
    assert "Calibration-Pass-2" in set(dataset.flightlines[LINE_TYPE_COLUMN])


def test_a_label_with_whitespace_is_refused_where_it_is_set():
    """The same defect that made every enriched export unreadable when a verdict
    tier contained a space — refused at writing rather than found at reading."""
    with pytest.raises(ValueError, match="whitespace-delimited"):
        classify_lines(delivery(), criterion="explicit", lines=[100],
                       line_type="High Altitude")


def test_the_refusal_suggests_a_usable_label():
    with pytest.raises(ValueError, match="High-Altitude"):
        classify_lines(delivery(), criterion="explicit", lines=[100],
                       line_type="High Altitude")


def test_a_criterion_missing_its_parameter_is_an_error():
    with pytest.raises(ValueError, match="needs a threshold"):
        classify_lines(delivery(), criterion="height_threshold",
                       line_type="High-Altitude")


def test_an_unknown_criterion_names_the_available_ones():
    with pytest.raises(ValueError, match="height_threshold, explicit"):
        classify_lines(delivery(), criterion="auto", line_type="X")


# ── Selecting ────────────────────────────────────────────────────────────────

def test_selecting_by_line_type_drops_the_rest():
    dataset, _ = classify_lines(delivery(), criterion="height_threshold",
                                threshold=500, line_type="High-Altitude")
    dataset, report = select_lines(dataset, criterion="line_type",
                                   line_types=[DEFAULT_LINE_TYPE])

    assert len(dataset.flightlines) == 8
    assert set(dataset.flightlines["line_id"]) == {100, 102}
    assert report.distributions["soundings"]["dropped"] == 4


def test_selecting_explicit_lines_needs_no_classification():
    dataset, _ = select_lines(delivery(), criterion="explicit_lines",
                              lines=[101])

    assert set(dataset.flightlines["line_id"]) == {101}
    assert len(dataset.flightlines) == 4


def test_every_array_moves_with_the_soundings():
    """The one that matters: an array left at full length would address a
    different sounding than its neighbour, silently."""
    dataset, _ = select_lines(delivery(), criterion="explicit_lines",
                              lines=[102])

    assert len(dataset.flightlines) == 4
    for key, array in dataset.layer_data.items():
        assert len(array) == 4, key


def test_the_surviving_rows_are_the_right_ones():
    """Not merely the right count. The third line's data must be what is left."""
    dataset = delivery()
    marker = dataset.layer_data[dataset.channels[0].data_key]
    marker.iloc[8:12, 0] = -7.0                  # line 102's first gate

    dataset, _ = select_lines(dataset, criterion="explicit_lines", lines=[102])
    kept = dataset.layer_data[dataset.channels[0].data_key]

    assert (kept.iloc[:, 0] == -7.0).all()


def test_the_index_is_contiguous_afterwards():
    dataset, _ = select_lines(delivery(), criterion="explicit_lines", lines=[102])

    assert list(dataset.flightlines.index) == [0, 1, 2, 3]
    for array in dataset.layer_data.values():
        assert list(array.index) == [0, 1, 2, 3]


def test_selecting_nothing_is_refused():
    """A selection matching no line is almost always a typo, and an empty
    dataset fails much later and much less clearly."""
    with pytest.raises(ValueError, match="No sounding matches"):
        select_lines(delivery(), criterion="explicit_lines", lines=[999])


def test_selecting_by_type_before_anything_classified_says_so():
    with pytest.raises(KeyError, match="classify_lines first"):
        select_lines(delivery(), criterion="line_type",
                     line_types=["Production"])


def test_an_array_not_indexed_by_sounding_is_left_and_noted():
    dataset = delivery()
    dataset.layer_data["model"] = pd.DataFrame(np.ones((25, 3)))

    dataset, report = select_lines(dataset, criterion="explicit_lines",
                                   lines=[102])

    assert len(dataset.layer_data["model"]) == 25
    assert any("not indexed by sounding" in n for n in report.notes)


# ── Together ─────────────────────────────────────────────────────────────────

def test_the_pair_composes_from_one_pipeline():
    """01: filter_04 "complements filter_02, which assigns the labels this one
    selects on"."""
    dataset, report = ProcessingRunner(Pipeline.parse("""
        kind: filters
        steps:
          - classify_lines:
              criterion: height_threshold
              threshold: 500
              line_type: High-Altitude
          - select_lines:
              criterion: line_type
              line_types: [Production]
    """)).run(delivery())

    assert set(dataset.flightlines["line_id"]) == {100, 102}
    assert set(dataset.flightlines[LINE_TYPE_COLUMN]) == {DEFAULT_LINE_TYPE}


def test_the_column_fits_a_shapefile_field():
    assert len(LINE_TYPE_COLUMN) <= 10


@pytest.mark.parametrize("name", ["classify_lines", "select_lines"])
def test_registered(name):
    assert name in load_steps(FILTER_GROUP)
