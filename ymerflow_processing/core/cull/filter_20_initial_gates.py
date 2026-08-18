"""filter_20 — disable the early gates the system description declares unusable.

**What it does to the dataset:** turns off the first *N* gates of each channel in
the in-use mask, where *N* is what the system description says. It changes no
measurement, no uncertainty, and no array shape (R6).

The instrument's own statement about itself
-------------------------------------------
``01`` describes this as "the early gates that the system description declares
unusable — the instrument's own statement about which gates are contaminated by
the transmitter turn-off."

The physics: SkyTEM's early gates sit in the transmitter's turn-off, and cannot
be inverted without modeling the system primary field. The sign reversal is
visible in the data at those times. ``RemoveInitialGates`` is the vendor
declaring where that ends.

A record of a window, not an instruction to re-slice one
--------------------------------------------------------
``NoGates`` and ``RemoveInitialGates`` are read from the **same origin**: index 0
is both the first delivered gate column and the first row of the gate-time table.
Delivered data width, ``NoGates``, and the time axis are 1:1 and head-aligned.

So ``RemoveInitialGates`` masks the first *N* gates *within* that window. It is
**not** an index that shifts the time axis, and it never narrows an array. Two
ways to get this wrong, both silent:

- Slicing the gate-time table by it. The table is already trimmed to ``NoGates``;
  slicing again would misalign every gate against its time, and the lengths would
  still look plausible.
- Trimming the data columns. Everything here is captured in the mask, and
  nothing in the pipeline may narrow a gate axis.

Why a filter and not the importer
---------------------------------
It is arguably a correction rather than a quality judgement, which is an argument
for applying it at import so data always arrives "as the vendor intended". As a
filter it is better: visible in the flow, versioned, re-runnable, reversible, and
a standalone run gets it without the platform. ``01`` had it as a filter and that
is where it stays.

It is idempotent with an importer that also applies it, because culling is
subtractive — running both leaves exactly the same mask as running either.

Reversible, and deliberately so
-------------------------------
filter_21 can enable these gates again. Someone doing full primary-field modeling
legitimately wants them back, which is why this writes a mask rather than
removing anything.

Sources
-------
``RemoveInitialGates`` and ``NoGates`` are per-channel entries in the SkyTEM GEX
system description; the 1:1 alignment between them and the delivered data was
established by counting rows across a corpus of real system descriptions rather
than assumed.
"""

from __future__ import annotations

import logging

import numpy as np

from ..mask import cull, ensure, mark_step
from ..pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "disable_initial_gates"

#: This filter's number in ``building_plans/01``, used as its provenance code.
STEP_CODE = 20


def declared_initial_gates(dataset, channel) -> int | None:
    """How many initial gates the system description declares unusable.

    ``None`` when it says nothing — which is different from saying zero, and is
    reported differently.
    """
    gex = dataset.survey.gex
    if gex is None:
        return None
    try:
        declared = gex.remove_initial_gates(channel.index)
    except (KeyError, AttributeError, IndexError):
        return None
    if declared is None or not np.isfinite(float(declared)):
        return None
    return int(declared)


def disable_initial_gates(dataset, gates: dict | int | None = None):
    """Disable each channel's declared initial gates in the in-use mask.

    :param gates: Override the system description, per channel key or moment
        name (``{"ch1": 6}``), or one count for every channel. Left unset — the
        normal case — each channel's own declaration is used. An override is for
        a delivery whose description is known to be wrong, and it is recorded in
        the report alongside what was declared, so the two never get confused.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    channels = dataset.channels
    if not channels:
        report.notes.append("No channels; nothing to disable.")
        return dataset, report

    overrides = dataset.resolve_per_channel(gates) if gates is not None else {}

    per_channel, undeclared = {}, []
    for channel in channels:
        declared = declared_initial_gates(dataset, channel)
        count = overrides.get(channel.key, declared)
        if declared is None and channel.key not in overrides:
            undeclared.append(str(channel))
            continue

        width = ensure(dataset, channel).shape[1]
        count = int(count or 0)
        if count < 0:
            raise ValueError(
                f"{channel} declares {count} initial gates to remove; a count "
                f"of gates cannot be negative."
            )
        if count > width:
            raise ValueError(
                f"{channel} declares {count} initial gates to remove but has "
                f"only {width} gates. The declaration and the delivery "
                f"disagree, which is a fact about the data worth knowing "
                f"rather than something to clamp silently."
            )

        # Position-based, and the whole column for those gates. The .loc
        # equivalent is inclusive on integer labels and would take one gate too
        # many — the only sharp edge in this filter.
        bad = np.zeros(ensure(dataset, channel).shape, dtype=bool)
        bad[:, :count] = True

        outcome = cull(dataset, channel, bad, step_code=STEP_CODE)
        outcome["declared"] = declared
        outcome["applied"] = count
        if channel.key in overrides:
            outcome["overridden"] = True
        per_channel[str(channel)] = outcome

    if undeclared:
        report.notes.append(
            f"The system description declares no initial gates to remove for "
            f"{', '.join(undeclared)}; those channels are untouched. Name them "
            f"under 'gates:' if the delivery needs it anyway."
        )
    if not per_channel:
        return dataset, report

    mark_step(dataset, STEP_CODE, STEP)
    removed = sum(c["removed"] for c in per_channel.values())
    report.soundings_affected = 0        # gates, not soundings; see distributions
    report.distributions = {
        "per_channel": per_channel,
        "removed": removed,
        "provenance_code": STEP_CODE,
    }
    log.info("%s: disabled %d gate-samples across %d channel(s)",
             STEP, removed, len(per_channel))
    return dataset, report
