"""filter_32 — disable negative measurements after a nominated gate.

**What it does to the dataset:** turns off gates whose measured value is
negative, and whatever the culling policy widens that to. It changes no
measurement, no uncertainty, and no array shape (R6).

``01``: "For a standard transient the response should be positive throughout; a
sign change indicates coupling, an IP effect, or noise dominating."

That is a **physical plausibility test**, and the cleanest one in the library: it
needs no threshold, because zero is not a number anyone chose. A decaying
transient over a resistive earth is positive at every gate, so a negative sample
is not a small measurement — it is a measurement of something the forward model
does not describe.

The policy is the point here
----------------------------
This is the filter the shared culling policy was written for, and both knobs earn
their place on physical grounds rather than as tuning:

**Later gates go too** (``disable_tail``). Once coupling or noise dominates, it
dominates everything after — later gates are smaller, so the same interference is
a larger fraction of them. A transient does not recover once it has gone bad.

**Earlier gates go too** (``spread_earlier``). The gate where the sign flips is
where the contamination becomes *visible*, not where it started. It was already
there in the gate before, large enough to matter and not yet large enough to
overwhelm the response. Culling only the visibly negative gate keeps the last
contaminated positive one.

Both are off by default. They widen a cull beyond what was actually observed, and
a step should not do that unasked — the provenance array records the difference
precisely so an operator can see how much of a cull was inference rather than
observation. For negatives the usual setting is ``spread_earlier: 1`` with
``disable_tail: true``.

Why a starting gate
-------------------
Early gates are contaminated by the transmitter turn-off and legitimately change
sign there; that is what filter_20 disables them for. Evaluating this filter over
them would flag the vendor's own known-bad window as a discovery. ``start_gate``
puts the evaluation where the response is supposed to be clean.

Zero is not negative
--------------------
Only values strictly below zero are flagged. An exact zero is not a sign change,
and in practice means a dead channel or a fill value rather than a reversal.
"""

import logging

import numpy as np

from ..mask import apply_policy, cull, in_use, mark_step
from ..pipeline import StepReport

log = logging.getLogger(__name__)

CATEGORY = "cull gates"

STEP = "cull_negatives"

#: This filter's number in ``building_plans/01``, used as its provenance code.
STEP_CODE = 32


def cull_negatives(dataset, start_gate: int = 0, spread_earlier: int = 0,
                   disable_tail: bool = False):
    """Disable gates whose measured value is negative.

    Parameters
    ----------
    dataset : Dataset
        The dataset to cull; its in-use masks are written.
    start_gate : int
        Where evaluation begins, 0-based. Early gates sit in the transmitter
        turn-off and legitimately change sign there.
    spread_earlier : int
        Also disable this many gates before each negative — the contamination
        precedes its visible onset.
    disable_tail : bool
        Also disable everything after the earliest negative in that sounding,
        since a transient does not recover.

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
        report.notes.append("No channels; nothing to inspect.")
        return dataset, report

    per_channel, culled_any = {}, False
    for channel in channels:
        values = dataset.layer_data[channel.data_key].to_numpy(dtype=float)

        # Strictly below zero: an exact zero is not a sign change, and usually
        # means a dead channel or a fill value rather than a reversal. NaN is
        # not negative either, and fails this comparison as it should.
        negative = values < 0

        # Only where the datum is still live — a gate something else already
        # removed is not this filter's finding, and counting it would inflate
        # what this step appears to have discovered.
        negative &= in_use(dataset, channel)

        widened = apply_policy(negative, start_gate=start_gate,
                               spread_earlier=spread_earlier,
                               disable_tail=disable_tail)
        # The policy discards objections before start_gate, so the direct hits
        # are what survived it rather than the raw comparison.
        direct = negative.copy()
        direct[:, :start_gate] = False

        outcome = {
            "negative_gates": int(direct.sum()),
            "soundings_with_a_negative": int(direct.any(axis=1).sum()),
            "first_negative_gate": (int(np.argmax(direct.any(axis=0)))
                                    if direct.any() else None),
        }
        if widened.any():
            culled_any = True
            outcome.update(cull(dataset, channel, widened,
                                step_code=STEP_CODE, direct=direct))
        else:
            outcome["removed"] = 0
        per_channel[str(channel)] = outcome

    if culled_any:
        mark_step(dataset, STEP_CODE, STEP)

    removed = sum(c["removed"] for c in per_channel.values())
    collateral = sum(c.get("removed_collateral", 0) for c in per_channel.values())
    report.soundings_affected = max(
        (c["soundings_with_a_negative"] for c in per_channel.values()),
        default=0)
    report.distributions = {
        "per_channel": per_channel,
        "removed": removed,
        "policy": {"start_gate": start_gate, "spread_earlier": spread_earlier,
                   "disable_tail": disable_tail},
        "provenance_code": STEP_CODE,
    }

    if collateral:
        # Stated, not judged. A ratio threshold was tried here and removed: with
        # disable_tail on a long transient, collateral dominating direct hits is
        # the *intended* behaviour, so any ratio that fires on misuse fires on
        # correct use too. What is wanted is the number in front of an operator,
        # which is a different thing from a rule about it — and inventing that
        # rule would be the house number this library refuses everywhere else.
        report.notes.append(
            f"{collateral} of {removed} disabled gate(s) were collateral — "
            f"inferred from the policy rather than observed negative. That is "
            f"what spread_earlier and disable_tail are for; the split is here "
            f"so the size of the inference is visible."
        )
    log.info("%s: disabled %d gate-sample(s), %d of them collateral",
             STEP, removed, collateral)
    return dataset, report
