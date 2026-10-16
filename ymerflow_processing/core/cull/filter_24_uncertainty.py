"""filter_24 — disable data points whose assigned uncertainty is too large.

**What it does to the dataset:** turns off gates whose uncertainty exceeds a
threshold, and whatever the culling policy widens that to. It changes no
measurement, no uncertainty, and no array shape (R6).

``01``: "disables individual data points whose assigned uncertainty exceeds a
threshold, expressed as a fraction of the value."

The uncertainty is already a fraction
-------------------------------------
That phrasing invites dividing the uncertainty by the value. **Do not** — in this
format the uncertainty array already *is* the fraction. The convention is
AarhusInv's relative standard deviation, and the forward code confirms it by
inverting the operation when it writes uncertainties back out:

    std = np.abs(1 / self.inv.invProb.dmisfit.W.diagonal()
                 / self.inv.invProb.dmisfit.data.dobs)

Dividing by ``dobs`` to recover the stored value means the stored value was
already relative. The system description agrees: ``UniformDataSTD`` is a plain
``0.03``, three percent, not an amplitude.

So the test is a direct comparison, and a threshold of ``0.15`` means "disable
anything with more than fifteen percent uncertainty" — which is what an operator
means when they say it. Dividing again would have produced a number with units of
one over the response, silently, and culled almost nothing.

The distribution of the values compared is reported, so a delivery whose
uncertainties are *absolute* would show magnitudes nothing like a percentage and
be obvious rather than quietly under-culled.

Uncertainty is not noise
------------------------
This culls what the *processing* claims to know about a point, not what the
instrument measured. A large assigned uncertainty means an earlier stage — stacking,
averaging, the vendor's own processing — already concluded the point was poorly
determined. That is a different statement from filter_29's, which says the
measurement is below the ambient noise floor and so is measuring the environment.
Both can be true, and neither implies the other.

**There is no default threshold.** What uncertainty is tolerable depends on the
inversion and the survey, and a house number would silently cull a client's data.
"""

from __future__ import annotations

import logging

import numpy as np

from ..distribution import summarize
from ..mask import apply_policy, cull, in_use, mark_step
from ..pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "cull_by_uncertainty"

#: This filter's number in ``building_plans/01``, used as its provenance code.
STEP_CODE = 24


def cull_by_uncertainty(dataset, max_uncertainty: float, start_gate: int = 0,
                        spread_earlier: int = 0, disable_tail: bool = False):
    """Disable gates whose relative uncertainty exceeds ``max_uncertainty``.

    Parameters
    ----------
    max_uncertainty : float
        Largest relative uncertainty to keep, as a fraction — ``0.15`` is
        fifteen percent. Inclusive: a point exactly at the limit is kept. **No
        default:** what is tolerable depends on the inversion.
    start_gate : int
        Where evaluation begins, 0-based.
    spread_earlier : int
        Also disable this many gates before each hit.
    disable_tail : bool
        Also disable everything after the earliest hit in that sounding.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    max_uncertainty = float(max_uncertainty)
    if max_uncertainty <= 0:
        raise ValueError(
            f"max_uncertainty is a fraction and must be positive; got "
            f"{max_uncertainty:g}. Zero would disable everything with any "
            f"uncertainty at all, which is every point."
        )

    channels = dataset.channels
    if not channels:
        report.notes.append("No channels; nothing to inspect.")
        return dataset, report

    per_channel, culled_any, without = {}, False, []
    for channel in channels:
        if not channel.std_key or channel.std_key not in dataset.layer_data:
            without.append(str(channel))
            continue

        std = dataset.layer_data[channel.std_key].to_numpy(dtype=float)

        # A direct comparison: the array is already a relative standard
        # deviation. NaN is not greater than anything and is not flagged here —
        # a point with no uncertainty assigned is filter_23's problem, since it
        # is unusable rather than uncertain.
        excessive = (std > max_uncertainty) & in_use(dataset, channel)

        widened = apply_policy(excessive, start_gate=start_gate,
                               spread_earlier=spread_earlier,
                               disable_tail=disable_tail)
        direct = excessive.copy()
        direct[:, :start_gate] = False

        outcome = {
            "column": channel.std_key,
            "over_threshold": int(direct.sum()),
            # So a delivery carrying *absolute* uncertainties shows magnitudes
            # nothing like a percentage, rather than quietly under-culling.
            "uncertainty": summarize(std[np.isfinite(std)]),
        }
        if widened.any():
            culled_any = True
            outcome.update(cull(dataset, channel, widened,
                                step_code=STEP_CODE, direct=direct))
        else:
            outcome["removed"] = 0
        per_channel[str(channel)] = outcome

    if without:
        report.notes.append(
            f"No uncertainty array for {', '.join(without)}; those channels "
            f"were not inspected."
        )
    if not per_channel:
        return dataset, report

    if culled_any:
        mark_step(dataset, STEP_CODE, STEP)

    removed = sum(c["removed"] for c in per_channel.values())
    report.distributions = {
        "per_channel": per_channel,
        "removed": removed,
        "max_uncertainty": max_uncertainty,
        "policy": {"start_gate": start_gate, "spread_earlier": spread_earlier,
                   "disable_tail": disable_tail},
        "provenance_code": STEP_CODE,
    }
    log.info("%s: disabled %d gate-sample(s) above %.3g relative uncertainty",
             STEP, removed, max_uncertainty)
    return dataset, report
