"""filter_12 — replace uncertainties with the dataset's noise model.

**What it does to the dataset:** overwrites the uncertainty arrays. It changes no
measurement, no mask and no array shape — but it is the first step in this
library that writes something an inversion weights by, so the originals are
stashed.

``01``: *"Takes an input dataset and replaces its uncertainty estimates with a
noise model, rather than adjusting whatever was already there."* The distinction
from filter_11 is exactly that — this asserts, filter_11 adjusts.

It carries no model of its own
------------------------------
R9: *"The noise model is defined once, as a property of the dataset, by an
explicit estimation step. Every consumer reads it. No filter carries its own copy
of the floor or the slope."*

So this takes no floor and no slope. It reads what ``estimate_noise_model``
recorded, and refuses to run if nothing did. That refusal is the point: the
defect R9 exists to prevent is assigning uncertainty with one floor and culling
with another, which used to be possible and which nothing detected.

What it writes
--------------
A relative standard deviation per datum — the floor's contribution and the
relative fraction in quadrature. As the signal decays toward the floor the
assigned uncertainty grows past one, which is how a gate dies honestly rather
than looking usable forever.

Normalization is applied here, not assumed: the floor is stated before division
by transmitter moment, and comparing it against moment-normalized data without
converting is wrong by three to four orders of magnitude for a high moment, with
nothing to signal it.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .distribution import summarize
from .noise import recorded
from .pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "assign_uncertainty"

#: Suffix for the stashed originals. This is the first step that overwrites
#: something an inversion weights by, so what was delivered is kept.
STASH_SUFFIX = "_raw"


def assign_uncertainty(dataset, stash: bool = True):
    """Replace uncertainties with the dataset's recorded noise model.

    :param stash: Keep the delivered uncertainties alongside, under the same key
        with ``_raw`` appended.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    model = recorded(dataset)
    if model is None:
        raise ValueError(
            "This dataset carries no noise model. Run estimate_noise_model "
            "first — R9 keeps one definition on the dataset so uncertainty "
            "cannot be assigned with one floor and data culled with another."
        )

    channels = dataset.channels
    if not channels:
        report.notes.append("No channels; nothing to assign.")
        return dataset, report

    per_channel, missing = {}, []
    for channel in channels:
        if not channel.std_key or channel.std_key not in dataset.layer_data:
            missing.append(str(channel))
            continue

        values = dataset.layer_data[channel.data_key]
        times = np.asarray(channel.axis.values, dtype=float)
        if times.size != values.shape[1]:
            missing.append(f"{channel} (no gate times)")
            continue

        before = dataset.layer_data[channel.std_key]
        if stash:
            dataset.layer_data[f"{channel.std_key}{STASH_SUFFIX}"] = before.copy()

        relative = model.relative(
            values.to_numpy(dtype=float), times,
            dipole_moment=getattr(channel, "dipole_moment", None),
            normalization=dataset.normalization)
        dataset.layer_data[channel.std_key] = pd.DataFrame(
            relative, index=values.index, columns=values.columns)

        finite = relative[np.isfinite(relative)]
        per_channel[str(channel)] = {
            "column": channel.std_key,
            "before": summarize(before.to_numpy(dtype=float).ravel()),
            "after": summarize(finite),
            # Where the floor has overtaken the signal entirely. Not an error —
            # it is the model saying those gates are noise, which is what the
            # absolute component exists to say.
            "beyond_unity": int((finite > 1.0).sum()),
        }

    if missing:
        report.notes.append(
            f"No uncertainty array or gate times for {', '.join(missing)}; "
            f"those channels were not assigned.")
    if not per_channel:
        return dataset, report

    report.distributions = {
        "model": model.to_dict(),
        "normalization": dataset.normalization,
        "per_channel": per_channel,
        "stashed": STASH_SUFFIX if stash else None,
    }
    log.info("%s: assigned from a floor of %.3g at 1 ms across %d channel(s)",
             STEP, model.level_1ms, len(per_channel))
    return dataset, report
