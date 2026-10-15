"""filter_33 — replay a hand-authored edit as part of an automated pipeline.

**What it does to the dataset:** sets the in-use mask wherever the edit says so,
either direction. It changes no measurement, no uncertainty, and no array shape
(R6).

``01``: *"This is what makes the whole chain reproducible in the presence of
human judgment. Without it, any manual editing forces the pipeline to be
abandoned in favor of a hand-edited file."*

That is the whole argument. Someone looks at a transient in a viewer, disables
six gates a predicate could not have known were bad, and the question is whether
re-running the pipeline destroys that. Expressed as a **difference against a
named state** rather than as a replacement of it, it does not.

The format is not ours to design
--------------------------------
``01``'s 2026-08-15 addition found that the diff already exists and is already
wired end to end on the platform. It is a ``libaarhusxyz`` msgpack XYZ carrying:

``flightlines.apply_idx``
    The global sounding indices the edit touches — so the diff carries its own
    addressing rather than depending on row order surviving whatever ran before
    it. This is the part that makes replay safe.
``layer_data["InUse_{channel}"]``
    One array per channel, with **NaN meaning "no override"**. Only non-NaN
    entries say anything.
``model_info.diff_dummy``
    A marker identifying the file as a diff.

So this filter's job is narrow: read that sparse XYZ, and apply its non-NaN
values at the soundings ``apply_idx`` names.

**It carries only the mask.** ``01`` notes this agrees exactly with R6, and that
it is coherent rather than a gap — a hand-edited *value* would be unattributable,
whereas a hand-disabled *datum* is a judgement with a reason behind it.

The second exception to subtractive culling
-------------------------------------------
An edit may turn data back on, and a person overruling an automated cull is the
clearest case there is for that. This and filter_21 are the only two steps
permitted to, and both say so.

Ordering matters, and it is the whole point. R6: *"Automated culling runs first,
the manual edit replays last. Because the manual edit is stored as a diff rather
than baked into state, re-running the pipeline does not silently undo it —
which is the standard failure mode when automated and manual edits mix."*

The checkpoint
--------------
``01`` decided the checkpoint and the edit stay one step, because *"a diff is
meaningless without knowing which state it is a difference from"* and splitting
them would require inventing a reference to replace one that is implicit in the
pairing.

Here the checkpoint is a **name recorded on the run**, not a file this writes.
Materialising and versioning intermediate state is the platform's job — it
already creates a process version per edit — and a library that wrote snapshot
files as a side effect of filtering would be doing the platform's work badly.
What this owes is the name, so the pairing survives into the record.

NaN is consumed here and goes no further
----------------------------------------
The sentinel is the diff's, not the mask's. A ``NaN`` written into a main in-use
array would be read inconsistently downstream — one path in simpeg tests
``inuse == 0``, where NaN is not zero and the datum silently survives; another
does ``& inuse`` and raises. Neither happens, because nothing here writes a NaN:
the overrides are read out and the sentinel is discarded.
"""

from __future__ import annotations

import logging
import re

import numpy as np

from ..mask import CULLED, IN_USE, cull, enable, mark_step
from ..pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "apply_manual_edits"

#: This filter's number in ``building_plans/01``, used as its provenance code.
STEP_CODE = 33

#: The column carrying the diff's own addressing.
APPLY_INDEX = "apply_idx"

#: How a diff names the channel an override belongs to. Matched on a captured
#: index rather than by substring, the same discipline channel discovery uses.
DIFF_INUSE = (r"^inuse_ch(\d+)", r"^dbdt_inuse_ch(\d+)")


def _channel_index(key: str) -> int | None:
    for pattern in DIFF_INUSE:
        match = re.match(pattern, str(key), flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def _open(diff):
    """The diff as an XYZ, whether given as one or as a path.

    R8 asks for external inputs typed rather than as paths, and an already-loaded
    XYZ is the typed form. A path is accepted because a standalone run has
    nothing to materialise it — that is the same layering ``load`` follows.
    """
    if hasattr(diff, "layer_data"):
        return diff
    from libaarhusxyz.xyz import XYZ
    return XYZ(str(diff))


def apply_manual_edits(dataset, diff: str | XYZ, checkpoint: str | None = None):
    """Apply a hand-authored in-use diff at the soundings it names.

    :param diff: The edit set — a ``libaarhusxyz`` XYZ, or a path to one.
    :param checkpoint: A name for the state this diff is a difference from.
        Recorded on the run so the pairing survives; nothing is written here.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    edit = _open(diff)
    if not getattr(edit, "model_info", {}).get("diff_dummy"):
        report.notes.append(
            "This file carries no 'diff_dummy' marker, so it may not be an edit "
            "set. Applying it anyway; only its non-NaN in-use entries are read."
        )

    edited = getattr(edit, "flightlines", None)
    if edited is None or APPLY_INDEX not in getattr(edited, "columns", []):
        raise KeyError(
            f"The edit set has no {APPLY_INDEX!r} column. That column carries "
            f"the sounding indices the edit applies to, so without it the diff "
            f"has no addressing and cannot be replayed safely."
        )

    where = np.asarray(edited[APPLY_INDEX], dtype=int)
    outside = where[(where < 0) | (where >= len(frame))]
    if outside.size:
        raise IndexError(
            f"The edit set names sounding(s) "
            f"{', '.join(map(str, sorted(set(outside.tolist()))))}, but this "
            f"dataset has {len(frame)}. The diff was authored against a "
            f"different state — applying it would edit the wrong soundings."
        )

    by_index = {c.index: c for c in dataset.channels}
    per_channel, unmatched, touched = {}, [], False

    for key, array in edit.layer_data.items():
        index = _channel_index(key)
        channel = by_index.get(index) if index is not None else None
        if channel is None:
            continue

        overrides = np.asarray(array, dtype=float)
        if overrides.shape[0] != where.size:
            raise ValueError(
                f"Edit array {key!r} has {overrides.shape[0]} rows against "
                f"{where.size} entries in {APPLY_INDEX}; they must agree."
            )
        gates = dataset.layer_data[channel.data_key].shape[1]
        if overrides.shape[1] != gates:
            unmatched.append(
                f"{key} ({overrides.shape[1]} gates against {gates})")
            continue

        # NaN means "no override" and is the diff's sentinel, not the mask's.
        # It is read out here and goes no further.
        stated = np.isfinite(overrides)
        on = np.zeros((len(frame), gates), dtype=bool)
        off = np.zeros((len(frame), gates), dtype=bool)
        on[where] = stated & (overrides != CULLED)
        off[where] = stated & (overrides == CULLED)

        outcome = {"overrides": int(stated.sum())}
        if off.any():
            outcome["disabled"] = cull(dataset, channel, off,
                                       step_code=STEP_CODE)
        if on.any():
            outcome["enabled"] = enable(dataset, channel, on,
                                        step_code=STEP_CODE)
        if off.any() or on.any():
            touched = True
        per_channel[str(channel)] = outcome

    if unmatched:
        report.notes.append(
            f"Edit array(s) {', '.join(unmatched)} do not match this dataset's "
            f"gate count and were not applied."
        )
    if not per_channel:
        report.notes.append(
            "No in-use array in the edit set matched a channel here; nothing "
            "was applied."
        )
        return dataset, report

    if touched:
        mark_step(dataset, STEP_CODE, STEP)

    restored = sum(c.get("enabled", {}).get("restored", 0)
                   for c in per_channel.values())
    removed = sum(c.get("disabled", {}).get("removed", 0)
                  for c in per_channel.values())

    report.soundings_affected = int(np.unique(where).size)
    report.distributions = {
        "checkpoint": checkpoint,
        "soundings_edited": int(np.unique(where).size),
        "per_channel": per_channel,
        "restored": restored,
        "removed": removed,
        "provenance_code": STEP_CODE,
    }
    if checkpoint is None:
        report.notes.append(
            "No checkpoint name was given. A diff is a difference from some "
            "state, and naming that state is what lets a re-run be checked "
            "against the one the edit was authored against."
        )
    log.info("%s: %d sounding(s) edited, %d restored, %d removed",
             STEP, int(np.unique(where).size), restored, removed)
    return dataset, report
