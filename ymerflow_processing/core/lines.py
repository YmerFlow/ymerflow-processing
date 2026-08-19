"""Line selection — classifying flightlines, and reducing to the ones wanted.

``01`` lists four filters here and then decides they are two:

    **``Classify lines``** (was filter_01 + filter_02) and **``Select lines``**
    (was filter_03 + filter_04). Each takes a **``criterion``** naming how the
    lines are chosen, and only the parameters belonging to that criterion.

They share a module because they are a designed pair — ``01`` says filter_04
"complements filter_02, which assigns the labels this one selects on" — and
because they share the column one writes and the other reads. Splitting them
would put that column's name in two files and its meaning in neither.

``criterion``, not a boolean
---------------------------
R7, and ``01`` argues it at length. With exactly two options a bool is
equivalent, but ``auto_high_altitude: true`` bakes one specific rule into a
*parameter name*, so a third criterion — by date flown, by flight number — forces
either a second boolean that can contradict the first or a rename that breaks
existing pipelines. A criterion field takes a new value and nothing else changes.

It also avoids the boolean that silently switches which *other* parameters
matter, where half the signature is dead depending on a flag and nothing says
which half.

``line_type``, not ``tag``
--------------------------
``tag`` is taken: ``ProcessTag`` is a first-class model on the platform with its
own table, router and selector widget, meaning labels on process *versions*.
Reusing the word for flightline classification would collide inside one UI.
``line_type`` is also more accurate than ``flight_type``, since a flight contains
many lines and the classification is per line.

Values are free-form, with ``Production`` as the default until something sets
otherwise — a survey defines its own categories.

Two constraints the plan predates
---------------------------------
**A label may not contain whitespace.** The survey xyz format is
whitespace-delimited with no quoting, so ``"High Altitude"`` becomes two fields
and the file it is written to will not parse back. This is the same defect that
made every enriched export unreadable when a verdict tier contained a space, so
it is refused at the point of writing rather than discovered at the point of
reading. ``High-Altitude`` is fine.

**A label is a string, so the plot layer cannot draw it.** YmerFlow's plot layer
enumerates columns that are typed arrays and a label column is not one. Verdicts
solved this by being numeric with a fixed legend; ``line_type`` cannot, because
the vocabulary is deliberately open. It exports to GeoPackage and shapefile,
where it is exactly what someone wants to symbolise on, and it is readable in the
xyz — it simply is not available as a plot channel. Worth knowing rather than
discovering.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .pipeline import StepReport

log = logging.getLogger(__name__)

#: Where a flightline's classification lives. Nine characters, so it survives a
#: shapefile field name intact.
LINE_TYPE_COLUMN = "line_type"

#: What a line is until something classifies it otherwise.
DEFAULT_LINE_TYPE = "Production"

#: How ``classify_lines`` chooses which lines to label.
HEIGHT_THRESHOLD = "height_threshold"
EXPLICIT = "explicit"
CLASSIFY_CRITERIA = (HEIGHT_THRESHOLD, EXPLICIT)

#: How ``select_lines`` chooses which lines to keep.
LINE_TYPE = "line_type"
EXPLICIT_LINES = "explicit_lines"
SELECT_CRITERIA = (LINE_TYPE, EXPLICIT_LINES)


def _lines(dataset, line_column):
    frame = dataset.flightlines
    if frame is None or len(frame) == 0:
        return frame, None
    column = line_column or dataset.survey.xyz.line_id_column
    if column is None or column not in frame.columns:
        raise KeyError(
            f"No line identifier column {column!r} in flightlines. "
            f"Present: {', '.join(map(str, frame.columns))}"
        )
    return frame, column


def _check_label(label: str) -> str:
    """A label that cannot be written is refused where it is set."""
    label = str(label)
    if not label:
        raise ValueError("A line_type cannot be empty.")
    if any(character.isspace() for character in label):
        raise ValueError(
            f"line_type {label!r} contains whitespace. The survey xyz format is "
            f"whitespace-delimited with no quoting, so this would become two "
            f"fields and the file would not parse back. Use "
            f"{label.replace(' ', '-')!r} or similar."
        )
    return label


def classify_lines(dataset, criterion: str, line_type: str,
                   threshold: float | None = None, lines=None,
                   altitude_column: str | None = None,
                   line_column: str | None = None,
                   default_line_type: str = DEFAULT_LINE_TYPE):
    """Label flightlines, without removing anything.

    :param criterion: ``height_threshold`` — label lines whose *mean* height
        exceeds ``threshold``, which is how high-altitude test lines are found;
        or ``explicit`` — label the lines named in ``lines``.
    :param line_type: The label to apply. Free-form, but see the module
        docstring: no whitespace.
    :param threshold: Height above ground, for ``height_threshold``.
    :param lines: Line identifiers, for ``explicit``.
    :param altitude_column: Which column holds height. Discovered if unset.
    :param line_column: Which column identifies the line. Discovered if unset.
    :param default_line_type: What unclassified lines are called.

    Nothing is removed — this only classifies, so a later ``select_lines`` can
    act on the label.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step="classify_lines")
    if criterion not in CLASSIFY_CRITERIA:
        raise ValueError(
            f"Unknown criterion {criterion!r}. Available: "
            f"{', '.join(CLASSIFY_CRITERIA)}"
        )
    line_type = _check_label(line_type)
    default_line_type = _check_label(default_line_type)

    frame, column = _lines(dataset, line_column)
    report.soundings_total = 0 if frame is None else len(frame)
    if frame is None or len(frame) == 0:
        report.notes.append("No soundings; nothing to classify.")
        return dataset, report

    if LINE_TYPE_COLUMN not in frame.columns:
        frame[LINE_TYPE_COLUMN] = default_line_type

    identifiers = frame[column]
    if criterion == HEIGHT_THRESHOLD:
        if threshold is None:
            raise ValueError(
                "A 'height_threshold' criterion needs a threshold, in the "
                "altitude column's units."
            )
        altitude_column = altitude_column or dataset.survey.xyz.alt_column
        if altitude_column is None or altitude_column not in frame.columns:
            raise KeyError(
                f"No altitude column {altitude_column!r} in flightlines. "
                f"Present: {', '.join(map(str, frame.columns))}"
            )
        # The *mean* over the line, not any single sounding: a test line is
        # flown high throughout, and one high sounding on a production line is
        # qc_01's finding rather than a reclassification.
        means = frame.groupby(column, observed=True)[altitude_column].mean()
        chosen = sorted(means[means > float(threshold)].index)
        report.distributions = {"criterion": criterion,
                                "threshold": float(threshold),
                                "column": altitude_column,
                                "line_means": {str(k): float(v)
                                               for k, v in means.items()}}
    else:
        if lines is None:
            raise ValueError("An 'explicit' criterion needs a list of lines.")
        wanted = {str(v) for v in (lines if isinstance(lines, (list, tuple, set))
                                   else [lines])}
        present = {str(v) for v in identifiers.unique()}
        missing = sorted(wanted - present)
        if missing:
            raise KeyError(
                f"Line(s) {', '.join(missing)} are not in this dataset. "
                f"Present: {', '.join(sorted(present))}"
            )
        chosen = sorted(v for v in identifiers.unique() if str(v) in wanted)
        report.distributions = {"criterion": criterion,
                                "lines": [str(v) for v in chosen]}

    selected = identifiers.isin(chosen).to_numpy()
    frame.loc[selected, LINE_TYPE_COLUMN] = line_type

    report.soundings_affected = int(selected.sum())
    report.distributions.update({
        "line_type": line_type,
        "lines_classified": len(chosen),
        "line_types": {str(k): int(v) for k, v
                       in frame[LINE_TYPE_COLUMN].value_counts().items()},
    })
    if not chosen:
        report.notes.append(
            f"No line matched; every line keeps its existing classification."
        )
    log.info("classify_lines: labelled %d line(s) as %s", len(chosen), line_type)
    return dataset, report


def select_lines(dataset, criterion: str, line_types=None, lines=None,
                 line_column: str | None = None):
    """Reduce the dataset to the flightlines wanted. **Drops soundings.**

    :param criterion: ``line_type`` — keep lines carrying one of
        ``line_types``; or ``explicit_lines`` — keep the lines named in
        ``lines``.
    :param line_types: Classifications to keep, for ``line_type``.
    :param lines: Line identifiers to keep, for ``explicit_lines``.
    :param line_column: Which column identifies the line. Discovered if unset.

    This changes the dataset's extent: soundings are removed from
    ``flightlines`` and from every ``layer_data`` array together, so the two
    cannot drift apart.

    **Refuses to remove everything**, the same as ``select_channels``: a
    selection matching no line is almost always a typo, and an empty dataset
    fails much later and much less clearly.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step="select_lines")
    if criterion not in SELECT_CRITERIA:
        raise ValueError(
            f"Unknown criterion {criterion!r}. Available: "
            f"{', '.join(SELECT_CRITERIA)}"
        )

    frame, column = _lines(dataset, line_column)
    report.soundings_total = 0 if frame is None else len(frame)
    if frame is None or len(frame) == 0:
        report.notes.append("No soundings; nothing to select.")
        return dataset, report

    if criterion == LINE_TYPE:
        if line_types is None:
            raise ValueError("A 'line_type' criterion needs line_types.")
        if LINE_TYPE_COLUMN not in frame.columns:
            raise KeyError(
                f"No {LINE_TYPE_COLUMN!r} column — nothing has classified these "
                f"lines yet. Run classify_lines first, or select by "
                f"'explicit_lines'."
            )
        wanted = {str(v) for v in (line_types
                                   if isinstance(line_types, (list, tuple, set))
                                   else [line_types])}
        keep = frame[LINE_TYPE_COLUMN].astype(str).isin(wanted).to_numpy()
        present = sorted({str(v) for v in frame[LINE_TYPE_COLUMN].unique()})
        detail = f"line_type in {sorted(wanted)}. Present: {', '.join(present)}"
    else:
        if lines is None:
            raise ValueError("An 'explicit_lines' criterion needs lines.")
        wanted = {str(v) for v in (lines if isinstance(lines, (list, tuple, set))
                                   else [lines])}
        keep = frame[column].astype(str).isin(wanted).to_numpy()
        present = sorted({str(v) for v in frame[column].unique()})
        detail = f"line in {sorted(wanted)}. Present: {', '.join(present)}"

    if not keep.any():
        raise ValueError(f"No sounding matches {detail}")

    before_lines = frame[column].nunique()
    dropped = int((~keep).sum())

    # flightlines and every array together, positionally, then reindexed as one
    # — so nothing can end up addressing a different sounding than its neighbour.
    dataset.survey.xyz.flightlines = frame[keep].reset_index(drop=True)
    for key, array in list(dataset.layer_data.items()):
        if len(array) == len(keep):
            dataset.layer_data[key] = array[keep].reset_index(drop=True)
        else:
            report.notes.append(
                f"Array {key!r} has {len(array)} rows against "
                f"{len(keep)} soundings and was left alone; it is not indexed "
                f"by sounding."
            )

    kept_frame = dataset.flightlines
    report.soundings_affected = dropped
    report.distributions = {
        "criterion": criterion,
        "soundings": {"before": len(keep), "after": len(kept_frame),
                      "dropped": dropped},
        "lines": {"before": int(before_lines),
                  "after": int(kept_frame[column].nunique())},
    }
    log.info("select_lines: kept %d of %d sounding(s)",
             len(kept_frame), len(keep))
    return dataset, report
