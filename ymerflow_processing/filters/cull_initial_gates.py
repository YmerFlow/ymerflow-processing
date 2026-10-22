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

A mandate, not a step someone remembers
---------------------------------------
This is **applied automatically** — by :func:`ymerflow_processing.dataset.load`
when a survey and a system description are joined, and again at the top of every
pipeline run for a dataset built some other way. It is not something a pipeline
opts into.

The reason is that it is not a judgement. Every other culling filter decides
something: a threshold was chosen, a predicate objected, an operator could
reasonably disagree. This one reads the instrument's own statement about which of
its samples are usable. Data outside that window is not data anyone decided to
drop — it is samples the vendor already said cannot be inverted. Applying it is
part of reading the file correctly, in the same way the gate times are.

Left optional it is the single easiest thing to forget, and forgetting it is
silent: the inversion runs, converges, and is confidently wrong about the
shallow section. So "what is in the system description is what you get."

It is **not registered as a step** (unregistered 2026-10-21): a step that is not
a decision has no place in a form or a pipeline document, and listing it beside
the real culls invited people to think it was optional. The function stays
callable from Python for the one case the automatic pass cannot serve — a
delivery whose description is known to be wrong, where ``gates:`` overrides the
declared count. Running it again changes nothing, because culling is subtractive.
For a wrong declaration in a pipeline, ``set_inuse_gates`` is the step.

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

import logging

import numpy as np

from ..mask import cull, ensure, mark_step
from ..pipeline import StepReport

log = logging.getLogger(__name__)

CATEGORY = "cull gates"

STEP = "disable_initial_gates"

#: This filter's number in ``building_plans/01``, used as its provenance code.
STEP_CODE = 20


def declared_initial_gates(dataset, channel) -> int | None:
    """How many initial gates the system description declares unusable.

    Parameters
    ----------
    dataset : Dataset
        Whose system description is read.
    channel : Channel
        The channel asked about.

    Returns
    -------
    int | None
        The declared count, or ``None`` when the description says nothing —
        which is different from saying zero, and is reported differently.
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


def honor_system_description(dataset):
    """Apply the declaration, quietly, wherever a dataset comes into being.

    The mandate form: no report, no parameters, nothing to configure. Safe to
    call repeatedly — culling is subtractive, so this is a no-op on a dataset
    that already has it.

    Parameters
    ----------
    dataset : Dataset
        The dataset to apply its own system description to; its in-use masks
        are written.

    Returns
    -------
    int
        The number of gate-samples disabled, so a caller can say whether
        anything happened without this needing to know how to say it.
    """
    try:
        _, report = disable_initial_gates(dataset)
    except ValueError:
        # A description that disagrees with the delivery is worth raising from
        # an explicit step, where someone asked and is watching. Raising it from
        # a load would make an unreadable file out of a readable one, so the
        # explicit step is left to report it properly.
        log.warning("%s: the system description disagrees with the delivery; "
                    "the declared initial gates were not applied. Run the step "
                    "explicitly to see the detail.", STEP)
        return 0
    return (report.distributions or {}).get("removed", 0)


def disable_initial_gates(dataset, gates: dict | int | None = None):
    """Disable each channel's declared initial gates in the in-use mask.

    Parameters
    ----------
    dataset : Dataset
        The dataset to cull; its in-use masks are written.
    gates : dict | int | None
        Override the system description, per channel key or moment name
        (``{"ch1": 6}``), or one count for every channel. Left unset — the
        normal case — each channel's own declaration is used. An override is
        for a delivery whose description is known to be wrong, and it is
        recorded in the report alongside what was declared, so the two never
        get confused.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
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

        # From the data array, not the mask: asking the mask would materialize
        # one, and a channel declaring zero should leave the delivery untouched.
        width = dataset.layer_data[channel.data_key].shape[1]
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

        if count == 0:
            # Nothing to disable, and materializing a mask to say so would
            # write arrays into a delivery that had none — which is exactly the
            # promise lazy materialization exists to keep. Declaring zero is a
            # real answer, so it is still reported.
            per_channel[str(channel)] = {"channel": str(channel), "removed": 0,
                                         "declared": declared, "applied": 0}
            continue

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

    removed = sum(c["removed"] for c in per_channel.values())
    if removed:
        mark_step(dataset, STEP_CODE, STEP)
    report.soundings_affected = 0        # gates, not soundings; see distributions
    report.distributions = {
        "per_channel": per_channel,
        "removed": removed,
        "provenance_code": STEP_CODE,
    }
    log.info("%s: disabled %d gate-samples across %d channel(s)",
             STEP, removed, len(per_channel))
    return dataset, report
