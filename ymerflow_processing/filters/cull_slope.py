"""filter_25 and filter_26 — a transient decaying at an impossible rate.

**What it does to the dataset:** disables data whose local decay rate falls
outside a plausible range, plus whatever the culling policy widens that to. It
changes no measurement, no uncertainty and no array shape (R6).

``01``: *"disable points where the local decay rate of the transient — the slope
in log-time, log-amplitude space — is shallower than a maximum (filter_25) or
steeper than a minimum (filter_26)."*

One step, two bounds, because ``01`` writes them as one heading and they are two
bounds on one measurement rather than two predicates. That is the same shape as
every threshold pair in the QC group and is not the thirteen-valued mode ``01``
warns against.

A physical statement, not a threshold on a number
--------------------------------------------------
``01``: *"A transient decaying far too steeply or far too shallowly is not
describing an earth; it is describing noise, coupling, or an instrument problem.
For reference, a homogeneous half-space and ambient noise have characteristically
different slopes, and real data lies between them."*

That last sentence is what makes this checkable at all. The bounds are not
arbitrary: they bracket the range between two known behaviours, and a slope
outside it says the sample is not a measurement of ground.

Signed, and named for what it means
------------------------------------
The slope of a decaying transient is **negative** — amplitude falls as time
rises — so "maximum" and "minimum" read backwards for anyone thinking in decay
rates. The parameters are named ``shallowest`` and ``steepest`` instead, and take
signed values:

    shallowest: -0.5      # closer to flat than this is not a decay
    steepest:   -4.0      # faster than this is not an earth

A slope of ``-0.2`` is shallower than ``-0.5`` and is culled by the first;
``-6.0`` is steeper than ``-4.0`` and is culled by the second.

**There are no defaults.** Where the plausible band lies depends on the
instrument and the ground, and a house number would silently cull a client's
data.
"""

import logging

import numpy as np

from ..distribution import summarize
from ..mask import apply_policy, cull, in_use, mark_step
from ..pipeline import StepReport
from ..transient import log_slope

log = logging.getLogger(__name__)

CATEGORY = "cull gates"

STEP = "cull_by_slope"

#: The first of the pair's numbers in ``building_plans/01``. filter_26 is the
#: other bound on the same measurement and shares it.
STEP_CODE = 25


def cull_by_slope(dataset, shallowest: float | None = None,
                       steepest: float | None = None, start_gate: int = 0,
                       spread_earlier: int = 0, disable_tail: bool = False):
    """Disable data whose local decay rate is implausible.

    Parameters
    ----------
    dataset : Dataset
        The dataset to cull; its in-use masks are written.
    shallowest : float | None
        Slopes above this — closer to flat — are culled (filter_25). Signed and
        negative for a decay.
    steepest : float | None
        Slopes below this — falling faster — are culled (filter_26).
    start_gate : int
        Where evaluation begins, 0-based.
    spread_earlier : int
        Also disable this many gates before each hit.
    disable_tail : bool
        Also disable everything after the earliest hit.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if shallowest is None and steepest is None:
        raise ValueError(
            "cull_by_slope needs shallowest or steepest. With neither it "
            "would do nothing, which is better said by leaving the step out.")
    if (shallowest is not None and steepest is not None
            and steepest > shallowest):
        raise ValueError(
            f"steepest ({steepest:g}) must be below shallowest "
            f"({shallowest:g}); these are signed slopes, and a decay is "
            f"negative. Written the other way round they exclude every slope.")

    per_channel, culled_any = {}, False
    for channel in dataset.channels:
        values = dataset.layer_data[channel.data_key].to_numpy(dtype=float)
        times = np.asarray(channel.axis.values, dtype=float)
        if times.size != values.shape[1]:
            continue

        live = in_use(dataset, channel)
        slope = log_slope(values, times, live)

        # NaN is "unmeasurable here" — an end gate, a non-positive value, a
        # sounding with too few survivors — and must not become an objection.
        bad = np.zeros(values.shape, dtype=bool)
        if shallowest is not None:
            bad |= slope > shallowest
        if steepest is not None:
            bad |= slope < steepest
        bad &= np.isfinite(slope) & live

        widened = apply_policy(bad, start_gate=start_gate,
                               spread_earlier=spread_earlier,
                               disable_tail=disable_tail)
        direct = bad.copy()
        direct[:, :start_gate] = False

        outcome = {
            "implausible": int(direct.sum()),
            "unmeasurable": int((~np.isfinite(slope) & live).sum()),
            "slope": summarize(slope[np.isfinite(slope)]),
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
    report.gates_affected = removed       # gates, not soundings
    report.distributions = {
        "per_channel": per_channel,
        "removed": removed,
        "bounds": {"shallowest": shallowest, "steepest": steepest},
        "policy": {"start_gate": start_gate, "spread_earlier": spread_earlier,
                   "disable_tail": disable_tail},
        "provenance_code": STEP_CODE,
    }
    log.info("%s: disabled %d gate-sample(s)", STEP, removed)
    return dataset, report
