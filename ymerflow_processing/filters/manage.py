"""filter_14–19 — moving and reshaping the dataset's own structure.

``01`` groups these as *"Data management"* and is explicit about what they are
for: *"These exist to work around import problems and format mismatches, not to
do geophysics."* Worth repeating here so they are not mistaken for core
capability — **if import improves, demand for them falls.**

Two operations over two kinds of thing
--------------------------------------
The six are a two-by-three grid, and that is why they share a module rather than
sitting in six files of ten lines each. R2 asks for readability; splitting a
symmetry across six files hides it.

===========  ======================  ====================
operation    per-sounding attribute  per-gate array
===========  ======================  ====================
copy         filter_14               filter_15
rename       filter_16               filter_17
remove       filter_18               filter_19
===========  ======================  ====================

**Attributes** are one value per sounding — position, altitude, attitude,
current. **Arrays** are a value per sounding per gate: the measurements, their
uncertainties, their in-use flags.

They stay six entry points, because ``01`` argues the grouping is a display
concern rather than a filter-identity one, and because one filter with a
six-valued mode is the thing R7 exists to prevent.

Overwriting is reported, not warned
-----------------------------------
All six replace an existing destination if there is one. ``01`` notes they
currently *warn* when they do, and that under R3 that belongs in the structured
record instead — so it is a note on the step report rather than a printed line
that scrolls past.

Renaming an array a channel is discovered by
--------------------------------------------
These operate on raw ``layer_data`` keys, and channel discovery matches on those
keys. Renaming ``dbdt_ch1gt`` to something that matches no pattern makes the
channel vanish from the model — not an error, and not something this can prevent
without knowing what was intended. What it can do is say so, and it does.
"""

import logging

from ..pipeline import StepReport

log = logging.getLogger(__name__)

CATEGORY = "housekeeping"


def _report(step, dataset, kind, **fields):
    report = StepReport(step=step)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)
    report.distributions = {"kind": kind, **fields}
    return report


def _attributes(dataset):
    frame = dataset.flightlines
    if frame is None:
        raise ValueError("This dataset has no per-sounding attributes.")
    return frame


def _require(container, name, kind):
    if name not in container:
        present = ", ".join(map(str, list(container)[:20]))
        raise KeyError(
            f"No {kind} named {name!r}. Present: {present}"
            f"{' ...' if len(list(container)) > 20 else ''}"
        )


def _note_discovery_loss(dataset, report, before):
    """Say so if a rename or removal took a channel out of the model."""
    after = {c.key for c in dataset.channels}
    lost = sorted(before - after)
    if lost:
        report.notes.append(
            f"Channel(s) {', '.join(lost)} are no longer discovered — an array "
            f"they were identified by has been renamed or removed. That may be "
            f"intended; nothing else will act on them now."
        )


# ── Attributes ───────────────────────────────────────────────────────────────

def copy_attribute(dataset, source: str, destination: str):
    """Copy a per-sounding attribute, leaving the original (filter_14).

    Parameters
    ----------
    dataset : Dataset
        The dataset whose ``flightlines`` gain a column.
    source : str
        The column to copy.
    destination : str
        The name to copy it under.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    frame = _attributes(dataset)
    _require(frame.columns, source, "attribute")
    report = _report("copy_attribute", dataset, "attribute",
                     source=source, destination=destination)
    if destination in frame.columns:
        report.notes.append(f"Overwrote existing attribute {destination!r}.")
    frame[destination] = frame[source]
    log.info("copy_attribute: %s -> %s", source, destination)
    return dataset, report


def rename_attribute(dataset, source: str, destination: str):
    """Rename a per-sounding attribute (filter_16).

    Parameters
    ----------
    dataset : Dataset
        The dataset whose ``flightlines`` column is renamed.
    source : str
        The column to rename.
    destination : str
        Its new name.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    frame = _attributes(dataset)
    _require(frame.columns, source, "attribute")
    report = _report("rename_attribute", dataset, "attribute",
                     source=source, destination=destination)
    if destination in frame.columns:
        report.notes.append(f"Overwrote existing attribute {destination!r}.")
    dataset.survey.xyz.flightlines = frame.rename(columns={source: destination})
    log.info("rename_attribute: %s -> %s", source, destination)
    return dataset, report


def remove_attribute(dataset, name: str):
    """Remove a per-sounding attribute (filter_18).

    Parameters
    ----------
    dataset : Dataset
        The dataset whose ``flightlines`` lose a column.
    name : str
        The column to remove.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    frame = _attributes(dataset)
    _require(frame.columns, name, "attribute")
    report = _report("remove_attribute", dataset, "attribute", removed=name)
    dataset.survey.xyz.flightlines = frame.drop(columns=[name])
    log.info("remove_attribute: %s", name)
    return dataset, report


# ── Arrays ───────────────────────────────────────────────────────────────────

def copy_array(dataset, source: str, destination: str):
    """Copy a per-gate array, leaving the original (filter_15).

    Parameters
    ----------
    dataset : Dataset
        The dataset whose ``layer_data`` gains an array.
    source : str
        The array to copy.
    destination : str
        The key to copy it under.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    _require(dataset.layer_data, source, "array")
    report = _report("copy_array", dataset, "array",
                     source=source, destination=destination)
    if destination in dataset.layer_data:
        report.notes.append(f"Overwrote existing array {destination!r}.")
    dataset.layer_data[destination] = dataset.layer_data[source].copy()
    report.distributions["shape"] = tuple(dataset.layer_data[source].shape)
    log.info("copy_array: %s -> %s", source, destination)
    return dataset, report


def rename_array(dataset, source: str, destination: str):
    """Rename a per-gate array (filter_17).

    Parameters
    ----------
    dataset : Dataset
        The dataset whose ``layer_data`` key is renamed.
    source : str
        The array to rename.
    destination : str
        Its new key.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    _require(dataset.layer_data, source, "array")
    before = {c.key for c in dataset.channels}
    report = _report("rename_array", dataset, "array",
                     source=source, destination=destination)
    if destination in dataset.layer_data:
        report.notes.append(f"Overwrote existing array {destination!r}.")
    dataset.layer_data[destination] = dataset.layer_data.pop(source)
    report.distributions["shape"] = tuple(dataset.layer_data[destination].shape)
    _note_discovery_loss(dataset, report, before)
    log.info("rename_array: %s -> %s", source, destination)
    return dataset, report


def remove_array(dataset, name: str):
    """Remove a per-gate array (filter_19).

    Parameters
    ----------
    dataset : Dataset
        The dataset whose ``layer_data`` loses an array.
    name : str
        The array to remove.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    _require(dataset.layer_data, name, "array")
    before = {c.key for c in dataset.channels}
    report = _report("remove_array", dataset, "array", removed=name)
    report.distributions["shape"] = tuple(dataset.layer_data[name].shape)
    del dataset.layer_data[name]
    _note_discovery_loss(dataset, report, before)
    log.info("remove_array: %s", name)
    return dataset, report
