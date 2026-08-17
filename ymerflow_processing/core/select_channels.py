"""Reduce a dataset to the channels an inversion will accept.

**What it does to the dataset:** removes every array belonging to the channels
not kept — values, uncertainties and in-use mask alike — and, unless told
otherwise, the matching blocks from the system description. Channels that are
kept are untouched, byte for byte.

This is not culling. Culling writes the in-use mask, which is reversible and
auditable (R6); this *removes arrays*, changing the dataset's structure. The
distinction matters because the reason for doing it is structural: a forward
modelling code that accepts only vertical-component data will not merely ignore
X and Y, it will fail on them. Marking them not-in-use would not help, because
they would still be there.

Two axes, one step
------------------
R1 is explicit that component and moment are *different axes*. A
three-component dual-moment instrument declares six channels — one per
(moment, component) pair — so both selections are the same operation over a
different attribute, and R7 says that is exactly when to use a ``criterion``:

.. code-block:: yaml

    - Select channels: {criterion: component, keep: [z]}    # Z only
    - Select channels: {criterion: moment,    keep: [HM]}    # high moment only

Chaining them narrows on both, and each criterion carries only its own
parameters, so nothing is dead depending on a flag.

Selection is by *keep* rather than by *drop* throughout, matching ``Select
lines`` in ``building_plans/01``. A pipeline then states what it wants, which
stays correct when a survey turns up with a channel nobody anticipated — a
drop-list would silently let it through.

Channel numbering is left alone
-------------------------------
Keeping the vertical component of a three-component instrument leaves channels
3 and 6, not 1 and 2, and that is deliberate: **survivors keep their original
numbers and their original array names.**

Renumbering them was implemented and removed. It made the exported file stop
corresponding to the delivered one — the same array name meaning different data
either side of processing, with the mapping recorded only in a run report rather
than in the file anyone opens. Renaming a client's data to satisfy a positional
convention is too high a price for the tidiness.

It was only ever needed because a consumer addressed channels by literal name.
That is being fixed at the source instead: SimPEG's ``DualMomentTEMXYZSystem``
resolves its two moments by receiver orientation and dipole moment, so channels
3 and 6 are as readable as 1 and 2 (``YmerFlow/simpeg#3``).
"""

from __future__ import annotations

import logging

from .pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "select_channels"

#: What each criterion reads off a channel. Adding one is a line here plus a
#: line of documentation, not a new step.
CRITERIA = {
    "component": lambda channel: channel.component,
    "moment": lambda channel: channel.name,
    "key": lambda channel: channel.key,
}


def _matches(value, keep: set[str]) -> bool:
    return value is not None and str(value).strip().lower() in keep


def select_channels(dataset, criterion: str, keep, drop_system_blocks: bool = True):
    """Keep only the channels whose ``criterion`` value is in ``keep``.

    :param criterion: How channels are chosen — ``component`` (``x``/``y``/``z``,
        from the receiver polarization), ``moment`` (the transmitter moment name,
        such as ``LM``), or ``key`` (``ch1``, always available).
    :param keep: Which values to keep. Matched case-insensitively, so ``Z`` and
        ``z`` are the same request.
    :param drop_system_blocks: Also remove the discarded channels' blocks from
        the system description, so it does not go on declaring channels whose
        data has gone. Set false to leave the system description alone.

    Returns ``(dataset, StepReport)``.

    **Refuses to remove everything.** A criterion that matches no channel is
    almost always a typo or a wrong assumption about the data, and a dataset with
    no data in it fails much later and much less clearly. It raises here instead,
    naming what was actually present.
    """
    report = StepReport(step=STEP)

    if criterion not in CRITERIA:
        raise ValueError(
            f"Unknown criterion {criterion!r}. Available: {', '.join(sorted(CRITERIA))}"
        )
    if isinstance(keep, str):
        keep = [keep]
    wanted = {str(k).strip().lower() for k in keep}
    read = CRITERIA[criterion]

    channels = dataset.channels
    if not channels:
        report.notes.append("No channels found; nothing to select.")
        return dataset, report

    kept = [c for c in channels if _matches(read(c), wanted)]
    discarded = [c for c in channels if c not in kept]

    present = sorted({str(read(c)) for c in channels if read(c) is not None})
    if not kept:
        found = ", ".join(present) if present else (
            "(none — the system description may not declare it)")
        raise ValueError(
            f"No channel has {criterion} in {sorted(wanted)}. Present: {found}"
        )

    report.distributions = {
        "channels": {
            "before": [str(c) for c in channels],
            "after": [str(c) for c in kept],
            "criterion": criterion,
            f"{criterion}s_present": present,
        }
    }

    if not discarded:
        report.notes.append(
            f"Every channel already has {criterion} in {sorted(wanted)}; "
            "nothing removed."
        )
        return dataset, report

    removed = []
    for channel in discarded:
        for key in channel.array_keys:
            if key in dataset.layer_data:
                del dataset.layer_data[key]
                removed.append(key)
        if drop_system_blocks:
            block = f"Channel{channel.index}"
            if block in getattr(dataset.system, "gex_dict", {}):
                del dataset.system.gex_dict[block]

    report.points_total = len(channels)
    report.points_affected = len(discarded)
    report.distributions["arrays_removed"] = removed
    report.notes.append(
        f"Removed {len(discarded)} channel(s): "
        f"{', '.join(str(c) for c in discarded)}"
    )
    log.info("%s: kept %d of %d channel(s) by %s",
             STEP, len(kept), len(channels), criterion)

    return dataset, report
