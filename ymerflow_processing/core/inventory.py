"""What arrived, and what of it this library can actually reason about.

Every run's first step, taken implicitly. It reads and changes nothing.

Why it exists
-------------
Steps operate through the channel model — ``Channel.array_keys``, discovered by
matching array names and reading the system description. That model is the
library's whole view of the data, and **anything outside it is invisible rather
than absent**. The two are easy to confuse and behave completely differently.

The case that made this concrete: SkyTEM's data files carry a horizontal
component, but **both** the GEX and the ALC are built for the Z component only.
A dual-moment delivery then
carries three gate arrays against two declared channels. Nothing declares the
third, so discovery never registers it, and ``select_channels(keep=["z"])``
reported *"nothing removed"* while a non-Z array sat untouched in ``layer_data``.
Every statement in that sentence was true of channels and false of the file.

So the run opens by saying what came in, what was resolved, and what was not —
before any step gets to act as though the model were the whole dataset.

Resolved, and preserved
-----------------------
**Resolved** is what discovery could account for: arrays claimed by a channel,
with a component and a moment where the system description declares them.

**Unresolved** is everything else. It is not dropped and not disabled — it simply
travels through the pipeline untouched, because every channel-based step steps
over it. That is the important consequence and the reason this is reported rather
than logged: an operator asking for a Z-only dataset should learn at the top of
the run that the file also contains something the library cannot classify, not
discover it downstream when an inversion behaves oddly.

Undeclared is not the same as unusable. At least one such array is real coherent
signal, and the recovery path is to describe it — amend the system description so
discovery can claim it — rather than to delete it.

Nothing here judges
-------------------
No thresholds, no verdicts, no mask. It is an accounting of the delivery as
handed over, which makes it safe as an implicit step in any pipeline, including
a QC-only one where R6 promises the data comes out byte-identical.
"""

from __future__ import annotations

import logging

import numpy as np

from .mask import is_provenance
from .pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "inventory"


def _shape(frame) -> tuple:
    return tuple(getattr(frame, "shape", ()))


def take(dataset) -> dict:
    """Account for the delivery: what arrived, and what was resolved from it."""
    frame = dataset.flightlines
    channels = dataset.channels
    layer_data = dataset.layer_data

    claimed: dict[str, str] = {}
    for channel in channels:
        for key in channel.array_keys:
            if key in layer_data:
                claimed[key] = str(channel)

    # Anything the same shape as a claimed array is per-sounding-per-gate data
    # the model failed to account for. Detected by shape because a name we
    # recognized would already have been claimed.
    gate_shapes = {_shape(layer_data[k]) for k in claimed}
    unresolved, other = {}, {}
    for key, array in layer_data.items():
        # Provenance is gate-shaped and claimed by no channel, so the shape test
        # picks it up — and calling a column this library wrote "something the
        # model failed to account for" is the opposite of the truth.
        if key in claimed or is_provenance(key):
            continue
        (unresolved if _shape(array) in gate_shapes else other)[key] = _shape(array)

    resolved = []
    for channel in channels:
        axis = channel.axis
        resolved.append({
            "channel": str(channel),
            "key": channel.key,
            "index": channel.index,
            "moment": channel.name,
            "component": channel.component,
            "arrays": {
                "data": channel.data_key,
                "uncertainty": channel.std_key,
                "in_use": channel.inuse_key,
            },
            "gates": int(len(axis.values)) if axis.values is not None else None,
            "axis": axis.name,
            "dipole_moment": getattr(channel, "dipole_moment", None),
        })

    system = {}
    blocks = getattr(dataset.survey.gex, "gex_dict", None) or {}
    for name, block in blocks.items():
        if not str(name).startswith("Channel") or not isinstance(block, dict):
            continue
        system[name] = {
            "gates": block.get("NoGates"),
            "moment": block.get("TransmitterMoment"),
            "component": block.get("ReceiverPolarizationXYZ"),
            "remove_initial_gates": block.get("RemoveInitialGates"),
        }

    return {
        "method": dataset.method,
        "soundings": 0 if frame is None else int(len(frame)),
        "attributes": [] if frame is None else [str(c) for c in frame.columns],
        "arrays": {
            "total": len(layer_data),
            "resolved": {k: v for k, v in sorted(claimed.items())},
            "unresolved": dict(sorted(unresolved.items())),
            "other": dict(sorted(other.items())),
        },
        "channels": resolved,
        "system_description": system or None,
    }


def inventory(dataset):
    """List what the delivery contains and what of it was resolved.

    Reads only. Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    account = take(dataset)

    report.soundings_total = account["soundings"]
    report.distributions = account

    channels = account["channels"]
    report.notes.append(
        f"{account['soundings']} sounding(s), {len(account['attributes'])} "
        f"attribute(s), {account['arrays']['total']} array(s); resolved "
        f"{len(channels)} channel(s) as method {account['method']!r}."
    )
    for channel in channels:
        report.notes.append(
            f"  {channel['channel']}: {channel['gates']} gate(s), "
            f"data {channel['arrays']['data']}, "
            f"in-use {channel['arrays']['in_use'] or 'none yet'}."
        )

    unresolved = account["arrays"]["unresolved"]
    if unresolved:
        listing = ", ".join(f"{k} {v}" for k, v in unresolved.items())
        report.notes.append(
            f"{len(unresolved)} array(s) match the shape of gate data but are "
            f"claimed by no channel: {listing}. Nothing will act on them — "
            f"every step works through the channel model, so they travel "
            f"through the pipeline untouched. A vendor may ship a component it "
            f"omits from the system description; describing it in the system "
            f"description is what makes it usable."
        )

    if not channels:
        report.notes.append(
            "No channels resolved. A scalar instrument correctly has none; "
            "otherwise the array names or the system description do not match "
            "what this method expects."
        )

    log.info("%s: %d sounding(s), %d channel(s), %d unresolved array(s)",
             STEP, account["soundings"], len(channels), len(unresolved))
    return dataset, report
