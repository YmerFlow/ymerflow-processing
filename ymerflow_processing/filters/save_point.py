"""A save point: write the state reached here, then replay its manual edit.

**What it does to the dataset:** nothing, unless it carries a diff; then exactly
what ``apply_manual_edits`` does — sets the in-use mask wherever the edit says
so, in either direction. No measurement, uncertainty or array shape changes.

Why one step does both
----------------------
A manual edit is a difference from a state, and the state it is a difference
from is this save point. Splitting them would mean inventing a reference to
stand in for the one the pairing already carries; ``apply_manual_edits`` keeps
its ``checkpoint`` name beside its ``diff`` for the same reason. This step makes
that checkpoint something a person can open in the plot editor rather than only
a name recorded on the run.

A chain usually wants edits at more than one stage — at full rate after the
automatic culls, and again on the averaged result — so a pipeline carries one
save point per stage, each under its own name.

The order is the point
----------------------
The snapshot goes out **first**, before the edit touches the mask. It is the
state the steps above produced, so the editor can draw this point's edit over
it as an override that can be toggled back off. And because the edit is stored
as a diff rather than baked into the snapshot, re-running the chain does not
silently undo it.

The library writes no files
---------------------------
Materializing intermediate state is the runner's job: the platform writes a
dataset per save point beside the pipeline's usual output, and the command line
writes ``<label>.<name>.xyz`` beside its final output. The runner hands the step
a ``sink(name, dataset)`` and the step calls it. With no sink — a library caller
with no runner — the snapshot is skipped, the report says so, and the edit
still replays.

The sink writes as soon as it is called. A full-rate snapshot of a large survey
is gigabytes, so nothing is held until the end of the run.

The save point's name is checked before any data is read (unique within the
pipeline, and not ``output`` or ``processed_data``, which name the final
output); see :meth:`ymerflow_processing.pipeline.Pipeline.validate`.
"""

import logging
import typing

import pydantic

from ..pipeline import StepReport
from .cull_manual_edits import apply_manual_edits

log = logging.getLogger(__name__)

CATEGORY = "cull gates"

STEP = "save_intermediate_and_apply_diff"

#: A URL to an in-use edit saved by the plot editor. The editor offers a saved
#: edit for any field carrying this media type, and matches it exactly, so the
#: annotation is fixed rather than ours to vary.
ManualEditUrl = typing.Annotated[
    pydantic.AnyUrl,
    {"json_schema": {"x-url-media-type": "application/x-geophysics-xyz-model"}},
]


# ``diff`` is optional through its default alone, not ``| None``: on Python 3.10
# a union hashes its members, and an Annotated carrying a dict cannot be hashed.
# swaggerspect drops None from a union anyway, so the field's schema is the same.
# For the same reason typing.get_type_hints fails on this function under 3.10;
# inspect.signature, which swaggerspect and the reference read, does not.
def save_intermediate_and_apply_diff(
    dataset,
    name: typing.Annotated[str, {"json_schema": {"title": "Save point name"}}],
    diff: typing.Annotated[ManualEditUrl,
                           {"json_schema": {"title": "Manual edit"}}] = None,
    sink=None,
):
    """Save the state reached here under a name, then replay its manual edit.

    Parameters
    ----------
    dataset : Dataset
        The data as the steps above left it. Handed to ``sink`` unchanged; its
        in-use masks are then edited if ``diff`` is given.
    name : str
        A name for the intermediate state saved here. It appears in the plot
        editor under Dataset, and is the state this point's manual edits are a
        difference from.
    diff : ManualEditUrl, optional
        An in-use edit saved from a plot of this save point. Filled in by the
        plot editor; replayed after the save point is written, so re-running
        keeps it. Leave empty for a pure save point.
    sink : callable or None
        Supplied by the runner, never by the pipeline document: called as
        ``sink(name, dataset)`` to write the save point. ``None`` skips it.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``. The report's ``distributions`` carry
        ``save_point`` (the name) and ``snapshot`` (``"emitted"`` or
        ``"skipped"``), and, with a diff, the replay's sounding and per-channel
        gate counts from ``apply_manual_edits``.
    """
    emitted = sink is not None
    if emitted:
        sink(name, dataset)

    if diff is None:
        report = StepReport(step=STEP)
        frame = dataset.flightlines
        report.soundings_total = 0 if frame is None else len(frame)
        report.soundings_affected = 0
    else:
        # apply_manual_edits opens a path or an already-loaded XYZ; a URL the
        # form validated arrives as an AnyUrl, which is a path once it is a str.
        if isinstance(diff, pydantic.AnyUrl):
            diff = str(diff)
        dataset, report = apply_manual_edits(dataset, diff, checkpoint=name)
        report.step = STEP

    report.distributions["save_point"] = name
    report.distributions["snapshot"] = "emitted" if emitted else "skipped"
    if not emitted:
        report.notes.append(
            f"No sink was supplied, so save point {name!r} was not written. "
            f"Only a runner writes save points; the edit, if any, still replays.")
    log.info("%s: save point %r %s", STEP, name,
             "emitted" if emitted else "skipped (no sink)")
    return dataset, report
