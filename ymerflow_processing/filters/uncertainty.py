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

So this takes no floor and no slope. It reads what ``set_noise_floor``
recorded, and refuses to run if nothing did. That refusal is the point: the
defect R9 exists to prevent is assigning uncertainty with one floor and culling
with another, which used to be possible and which nothing detected.

What it does take is the **proportional error** — calibration, geometry,
primary-field removal, the "3 %" of an inversion's error model. That is not a
property of the noise floor, so it is not set with it (Ben, 2026-10-21): the
floor says how quiet the instrument is, the fraction says how far to trust the
data, and one step sets one thing. Absent, it comes from the system
description's ``UniformDataSTD``; it is never estimated.

What it writes
--------------
A relative standard deviation per datum — the floor's contribution and the
relative fraction in quadrature. As the signal decays toward the floor the
assigned uncertainty grows past one, which is how a gate dies honestly rather
than looking usable forever.

A supplied floor is physical (V/Am², before moment normalization), so assigning
divides it by the dipole moment and compares it against the physical form
``dbdt × scalefactor × GateFactor`` — the same floor SimPEG weights by at
inversion. Skipping the moment, or ``scalefactor × GateFactor``, is wrong by
orders of magnitude with nothing to signal it; the noise model carries which the
floor is and both factors default to 1 when the dataset has no GEX.
"""

import logging

import numpy as np
import pandas as pd

from ..dataset import role_key
from ..distribution import summarize
from .noise import gate_factor, recorded, scalefactor
from ..pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "assign_uncertainty"
CATEGORY = "noise"

#: Suffix for the stashed originals. This is the first step that overwrites
#: something an inversion weights by, so what was delivered is kept.
STASH_SUFFIX = "_raw"


def declared_fraction(dataset, channel) -> float | None:
    """The channel's ``UniformDataSTD`` from the system description, if declared.

    Parameters
    ----------
    dataset : Dataset
        Whose system description is read.
    channel : Channel
        The channel asked about.

    Returns
    -------
    float | None
        The declared proportional error, or ``None`` when the description says
        nothing.
    """
    gex = getattr(dataset.survey, "gex", None)
    block = getattr(gex, "gex_dict", {}).get(f"Channel{channel.index}", {}) if gex else {}
    value = block.get("UniformDataSTD") if isinstance(block, dict) else None
    return None if value is None else float(value)


def assign_uncertainty(dataset, relative_fraction: float | None = None,
                       stash: bool = True):
    """Replace uncertainties with the noise floor and a proportional error.

    Parameters
    ----------
    dataset : Dataset
        The dataset whose uncertainty arrays are replaced. Must carry a noise
        floor — see ``set_noise_floor``.
    relative_fraction : float | None
        The proportional error as a fraction of signal — the "3 %" of an
        inversion's error model, combined with the floor in quadrature. Absent
        takes each channel's ``UniformDataSTD`` from the system description,
        and a description that declares none gives a floor-only uncertainty,
        which is reported. **Never estimated** — it has no separable signature
        in the data, and estimating it would invent a number with a
        measurement's authority.
    stash : bool
        Keep the delivered uncertainties alongside, under the same key with
        ``_raw`` appended.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    model = recorded(dataset)
    if model is None:
        raise ValueError(
            "This dataset carries no noise floor. Run set_noise_floor "
            "first — R9 keeps one definition on the dataset so uncertainty "
            "cannot be assigned with one floor and data culled with another."
        )
    if relative_fraction is not None and relative_fraction < 0:
        raise ValueError(
            f"relative_fraction is a fraction of signal and cannot be negative; "
            f"got {relative_fraction:g}.")

    channels = dataset.channels
    if not channels:
        report.notes.append("No channels; nothing to assign.")
        return dataset, report

    per_channel, missing, created, floor_only = {}, [], [], []
    for channel in channels:
        if relative_fraction is not None:
            fraction, source = float(relative_fraction), "supplied"
        else:
            declared = declared_fraction(dataset, channel)
            fraction = 0.0 if declared is None else declared
            source = "floor only" if declared is None else "system description"
            if declared is None:
                floor_only.append(str(channel))
        values = dataset.layer_data[channel.data_key]
        times = np.asarray(channel.axis.values, dtype=float)
        if times.size != values.shape[1]:
            missing.append(f"{channel} (no gate times)")
            continue

        # A raw delivery legitimately carries no uncertainties — assigning them
        # is precisely how they come to exist, and refusing to act without an
        # array to overwrite left the whole noise half of the standard chain
        # dead on exactly the data it is for. The mask already worked this way:
        # created under a discoverable name so the next lookup finds it.
        std_key = channel.std_key
        if not std_key or std_key not in dataset.layer_data:
            std_key = role_key(channel, "std")
            created.append(str(channel))
            before = None
        else:
            before = dataset.layer_data[std_key]
            if stash:
                dataset.layer_data[f"{std_key}{STASH_SUFFIX}"] = before.copy()

        relative = model.relative(
            values.to_numpy(dtype=float), times,
            dipole_moment=getattr(channel, "dipole_moment", None),
            scalefactor=scalefactor(dataset),
            gate_factor=gate_factor(dataset, channel),
            relative_fraction=fraction)
        dataset.layer_data[std_key] = pd.DataFrame(
            relative, index=values.index, columns=values.columns)

        finite = relative[np.isfinite(relative)]
        per_channel[str(channel)] = {
            "column": std_key,
            "relative_fraction": fraction,
            "relative_fraction_source": source,
            "created": before is None,
            "before": (None if before is None
                       else summarize(before.to_numpy(dtype=float).ravel())),
            "after": summarize(finite),
            # Where the floor has overtaken the signal entirely. Not an error —
            # it is the model saying those gates are noise, which is what the
            # absolute component exists to say.
            "beyond_unity": int((finite > 1.0).sum()),
        }

    if missing:
        report.notes.append(
            f"No gate times for {', '.join(missing)}; those channels were not "
            f"assigned. Gate times come from the system description, so this "
            f"is a description that does not describe this data.")
    if created:
        report.notes.append(
            f"The delivery carried no uncertainty array for "
            f"{', '.join(created)}; one was created from the noise model. That "
            f"is the ordinary case for raw data — there is nothing to compare "
            f"against, so the 'before' distribution is absent rather than "
            f"empty.")
    if floor_only:
        report.notes.append(
            f"No relative_fraction was given and the system description "
            f"declares no UniformDataSTD for {', '.join(floor_only)}, so those "
            f"channels carry the floor alone — no proportional error.")
    if not per_channel:
        return dataset, report

    report.distributions = {
        "model": model.to_dict(),
        "relative_fraction": relative_fraction,
        "per_channel": per_channel,
        "stashed": STASH_SUFFIX if stash else None,
    }
    log.info("%s: assigned from a floor of %.3g at 1 ms across %d channel(s)",
             STEP, model.level_1ms, len(per_channel))
    return dataset, report
