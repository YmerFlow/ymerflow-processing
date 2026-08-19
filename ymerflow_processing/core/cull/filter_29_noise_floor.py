"""filter_29 — disable data that has fallen below the ambient noise floor.

**What it does to the dataset:** turns off gates whose measured value is at or
below the noise floor, and whatever the culling policy widens that to. It changes
no measurement, no uncertainty and no array shape (R6).

``01``: *"Below the floor, the instrument is measuring the environment rather
than the ground."*

The same statement as filter_12, used the other way
---------------------------------------------------
``01`` says this *"shares its noise model with filter_12, and the two should
share an implementation. It is the same physical statement used once to assign
uncertainty and once to cull."*

They now do, and neither carries a floor of its own — R9 puts the model on the
dataset and both read it. That is the defect being designed out: the two used to
default differently, one to the system description and the other to a number
roughly ten times higher than the same instrument's documented empirical floor,
and a run could assign uncertainty with one and cull with the other with nothing
detecting it.

Which means the honest thing to say about this filter is that **it may be
redundant with filter_12** on a dataset where uncertainties were assigned from
the model. A gate below the floor already carries a relative uncertainty past
one, and an inversion that weights by uncertainty has already discounted it. This
removes it outright instead, which matters for a consumer that thresholds on the
mask rather than weighting, and for anyone reading how much of a sounding was
real.

Not the same statement as filter_24
-----------------------------------
filter_24 culls what the *processing* claimed to know about a point. This culls
where the *measurement* is below what the environment contributes. Both can be
true of one datum, and neither implies the other: a carefully processed sounding
with a tight uncertainty can still be entirely below the floor at late time.

``margin`` rather than a second floor
-------------------------------------
The floor is the dataset's, not this filter's. What is local to culling is how
far above it a datum must sit to be believed, and that is a multiplier on the
shared floor rather than a number that can drift away from it.
"""

from __future__ import annotations

import logging

import numpy as np

from ..mask import apply_policy, cull, in_use, mark_step
from ..noise import recorded
from ..pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "cull_below_noise_floor"

#: This filter's number in ``building_plans/01``, used as its provenance code.
STEP_CODE = 29


def cull_below_noise_floor(dataset, margin: float = 1.0, start_gate: int = 0,
                           spread_earlier: int = 0, disable_tail: bool = False):
    """Disable gates at or below the dataset's noise floor.

    :param margin: How many times the floor a datum must exceed to be kept.
        ``1.0`` culls at the floor itself; ``2.0`` demands twice it. This is a
        multiplier on the shared floor rather than a floor of its own, so it
        cannot drift away from what filter_12 assigned uncertainty with.
    :param start_gate: Where evaluation begins, 0-based.
    :param spread_earlier: Also disable this many gates before each hit.
    :param disable_tail: Also disable everything after the earliest hit in that
        sounding. Well justified here — the floor is crossed once and the signal
        keeps decaying, so later gates are further below it.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if margin <= 0:
        raise ValueError(
            f"margin multiplies the noise floor and must be positive; got "
            f"{margin:g}.")

    model = recorded(dataset)
    if model is None:
        raise ValueError(
            "This dataset carries no noise model. Run estimate_noise_model "
            "first — R9 keeps one definition on the dataset so this cannot cull "
            "against a different floor than filter_12 assigned uncertainty with."
        )

    channels = dataset.channels
    if not channels:
        report.notes.append("No channels; nothing to inspect.")
        return dataset, report

    per_channel, culled_any = {}, False
    for channel in channels:
        values = dataset.layer_data[channel.data_key].to_numpy(dtype=float)
        times = np.asarray(channel.axis.values, dtype=float)
        if times.size != values.shape[1]:
            continue

        # Converted into the data's own normalization. Skipping this is wrong by
        # the dipole moment and nothing would signal it.
        floor = model.floor(times,
                            dipole_moment=getattr(channel, "dipole_moment", None),
                            normalization=dataset.normalization)

        # Magnitude against the floor: a negative datum is below it in the sense
        # that matters, but that is filter_32's finding to make and not this
        # one's, so the comparison is on size.
        below = np.abs(values) <= (np.abs(floor)[None, :] * margin)
        below &= np.isfinite(values) & in_use(dataset, channel)

        widened = apply_policy(below, start_gate=start_gate,
                               spread_earlier=spread_earlier,
                               disable_tail=disable_tail)
        direct = below.copy()
        direct[:, :start_gate] = False

        outcome = {
            "below_floor": int(direct.sum()),
            "first_gate_below": (int(np.argmax(direct.any(axis=0)))
                                 if direct.any() else None),
            "floor_at_last_gate": float(floor[-1]) if floor.size else None,
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
    report.soundings_affected = removed
    report.distributions = {
        "model": model.to_dict(),
        "margin": float(margin),
        "normalization": dataset.normalization,
        "per_channel": per_channel,
        "removed": removed,
        "policy": {"start_gate": start_gate, "spread_earlier": spread_earlier,
                   "disable_tail": disable_tail},
        "provenance_code": STEP_CODE,
    }
    log.info("%s: disabled %d gate-sample(s) at or below %.3gx the floor",
             STEP, removed, margin)
    return dataset, report
