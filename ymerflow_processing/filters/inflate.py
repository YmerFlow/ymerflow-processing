"""filter_11 — adjust uncertainty over a range of gates.

**What it does to the dataset:** adds a fixed fraction to the uncertainty of
nominated gates. It changes no measurement, no mask and no array shape.

``01``: *"adds a fixed fractional uncertainty to a specified range of gates — for
example, inflating uncertainty by 2% across all gates, or over a chosen
subrange. Negative values reduce it."*

Adjusting, not asserting
------------------------
This is the counterpart to filter_12 and the difference is the whole reason both
exist. filter_12 **replaces** uncertainty with a physical model — a floor and a
relative fraction, derived from the data and the system description. This
**adjusts** whatever is already there, by a number someone chose.

So it is an operator's knob, not a measurement, and it is honest about that:
there is no model behind the fraction and none is implied. The usual reason to
reach for it is knowing something the model does not — a gate range affected by a
known interference, a calibration one does not fully trust.

Arithmetic, not in quadrature
-----------------------------
Combining two *independent* uncertainties is a quadrature sum. This is not that:
``01`` says "adds" and "negative values reduce it", and a negative cannot be
subtracted in quadrature without going imaginary at the crossing point.

Reading it as arithmetic is what makes the parameter mean what an operator says
out loud — "add two percent" moves 3% to 5%, not to 3.6%. The trade is that this
does not compose the way independent errors do, which is correct, because a
number someone chose is not an independent error.

Uncertainty cannot go below zero. A reduction that would take it there clamps at
zero and the count is reported, rather than a negative uncertainty travelling
downstream to be squared by something.

Gate ranges use filter_21's idiom
---------------------------------
A single index, a list, or a ``"first-last"`` range, inclusive — the same forms
``nominate_gates`` takes, so the idiom is learned once and works everywhere (R7).
"""

import logging

import numpy as np
import pandas as pd

from .cull_nominate_gates import parse_gates
from ..distribution import summarize
from ..pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "inflate_uncertainty"


def inflate_uncertainty(dataset, fraction: float | dict[str, float],
                        gates: int | str | list[int | str] | dict | None = None):
    """Add ``fraction`` to the uncertainty of the nominated gates.

    Parameters
    ----------
    dataset : Dataset
        The dataset whose uncertainty arrays are increased.
    fraction : float | dict
        How much to add, as a fraction — ``0.02`` is two percentage points.
        Negative reduces. Per channel key or moment name (``{"ch1": 0.02}``),
        or one value for every channel.
    gates : int | str | list | dict | None
        Which gates, per channel or one nomination for all. A single index, a
        list, or an inclusive ``"first-last"`` range. Every gate if unset.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if fraction is None:
        raise ValueError(
            "inflate_uncertainty needs a fraction. With none it would do "
            "nothing, which is better said by leaving the step out.")

    channels = dataset.channels
    if not channels:
        report.notes.append("No channels; nothing to adjust.")
        return dataset, report

    amounts = dataset.resolve_per_channel(fraction)
    ranges = dataset.resolve_per_channel(gates) if gates is not None else {}

    per_channel, missing = {}, []
    for channel in channels:
        if channel.key not in amounts:
            continue
        if not channel.std_key or channel.std_key not in dataset.layer_data:
            missing.append(str(channel))
            continue

        array = dataset.layer_data[channel.std_key]
        values = array.to_numpy(dtype=float)
        width = values.shape[1]
        chosen = (parse_gates(ranges[channel.key], width)
                  if channel.key in ranges else list(range(width)))
        if not chosen:
            continue

        amount = float(amounts[channel.key])
        adjusted = values.copy()
        adjusted[:, chosen] = values[:, chosen] + amount

        # A negative uncertainty travelling downstream would be squared by
        # something and become a plausible positive again, which is the worst
        # kind of wrong. Clamped, and the count reported.
        floored = adjusted < 0
        adjusted = np.where(floored, 0.0, adjusted)

        dataset.layer_data[channel.std_key] = pd.DataFrame(
            adjusted, index=array.index, columns=array.columns)
        per_channel[str(channel)] = {
            "column": channel.std_key,
            "fraction": amount,
            "gates": chosen,
            "clamped_at_zero": int(floored.sum()),
            "before": summarize(values[:, chosen].ravel()),
            "after": summarize(adjusted[:, chosen].ravel()),
        }
        if floored.any():
            report.notes.append(
                f"{int(floored.sum())} value(s) in {channel.std_key!r} would "
                f"have gone below zero and were clamped there.")

    if missing:
        report.notes.append(
            f"No uncertainty array for {', '.join(missing)}; those channels "
            f"were not adjusted.")
    if not per_channel:
        report.notes.append("No nominated channel matched; nothing was adjusted.")
        return dataset, report

    report.distributions = {"per_channel": per_channel}
    log.info("%s: adjusted %d channel(s)", STEP, len(per_channel))
    return dataset, report
