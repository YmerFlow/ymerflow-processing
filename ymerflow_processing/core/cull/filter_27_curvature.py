"""filter_27 and filter_28 — a transient changing its decay rate too fast.

**What it does to the dataset:** disables data whose local curvature falls
outside a plausible range, plus whatever the culling policy widens that to. It
changes no measurement, no uncertainty and no array shape (R6).

``01``: *"Where slope tests catch data decaying at an impossible rate, curvature
tests catch data changing its decay rate impossibly fast."*

That sentence is the whole reason both exist. A transient can pass a slope test
at every gate while lurching between rates from one gate to the next — each
individual slope plausible, the sequence of them not. Coupling to a nearby
conductor does exactly that.

The second derivative in log-log space, over surviving gates
-------------------------------------------------------------
Computed by the shared helper in ``core/transient.py``, using the non-uniform
three-point divided difference rather than ``y₀ − 2y₁ + y₂``. The familiar form
assumes even spacing; gates are logarithmic and culling makes the gaps variable
on top of that, so the familiar form is wrong twice over and silently.

Undefined at the ends, which ``01`` states
-------------------------------------------
*"Curvature is undefined at the first and last gate."* It is also undefined at
the first and last *surviving* gate, at a non-positive value, and in a sounding
with fewer than three survivors. All of those are ``NaN`` and none of them is an
objection — "the shape is unmeasurable here" is a different statement from "the
shape is wrong", and only the second may cull.

**There are no defaults**, for the same reason as the decay rate: where the
plausible band lies depends on the instrument and the ground.
"""

from __future__ import annotations

import logging

import numpy as np

from ..distribution import summarize
from ..mask import apply_policy, cull, in_use, mark_step
from ..pipeline import StepReport
from ..transient import log_curvature

log = logging.getLogger(__name__)

STEP = "cull_by_curvature"

#: The first of the pair's numbers in ``building_plans/01``. filter_28 is the
#: other bound on the same measurement and shares it.
STEP_CODE = 27


def cull_by_curvature(dataset, max_curvature: float | None = None,
                      min_curvature: float | None = None, start_gate: int = 0,
                      spread_earlier: int = 0, disable_tail: bool = False):
    """Disable data whose local curvature is implausible.

    :param max_curvature: Curvature above this is culled (filter_27).
    :param min_curvature: Curvature below this is culled (filter_28).
    :param start_gate: Where evaluation begins, 0-based.
    :param spread_earlier: Also disable this many gates before each hit.
    :param disable_tail: Also disable everything after the earliest hit.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if max_curvature is None and min_curvature is None:
        raise ValueError(
            "cull_by_curvature needs max_curvature or min_curvature. With "
            "neither it would do nothing, which is better said by leaving the "
            "step out.")
    if (max_curvature is not None and min_curvature is not None
            and min_curvature > max_curvature):
        raise ValueError(
            f"min_curvature ({min_curvature:g}) is above max_curvature "
            f"({max_curvature:g}); written that way round they exclude every "
            f"value.")

    per_channel, culled_any = {}, False
    for channel in dataset.channels:
        values = dataset.layer_data[channel.data_key].to_numpy(dtype=float)
        times = np.asarray(channel.axis.values, dtype=float)
        if times.size != values.shape[1]:
            continue

        live = in_use(dataset, channel)
        curvature = log_curvature(values, times, live)

        bad = np.zeros(values.shape, dtype=bool)
        if max_curvature is not None:
            bad |= curvature > max_curvature
        if min_curvature is not None:
            bad |= curvature < min_curvature
        bad &= np.isfinite(curvature) & live

        widened = apply_policy(bad, start_gate=start_gate,
                               spread_earlier=spread_earlier,
                               disable_tail=disable_tail)
        direct = bad.copy()
        direct[:, :start_gate] = False

        outcome = {
            "implausible": int(direct.sum()),
            "unmeasurable": int((~np.isfinite(curvature) & live).sum()),
            "curvature": summarize(curvature[np.isfinite(curvature)]),
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
        "per_channel": per_channel,
        "removed": removed,
        "bounds": {"max_curvature": max_curvature,
                   "min_curvature": min_curvature},
        "policy": {"start_gate": start_gate, "spread_earlier": spread_earlier,
                   "disable_tail": disable_tail},
        "provenance_code": STEP_CODE,
    }
    log.info("%s: disabled %d gate-sample(s)", STEP, removed)
    return dataset, report
