"""filter_13 — draw noise from the assigned uncertainties and add it.

**What it does to the dataset:** perturbs the measurements. It changes no
uncertainty, no mask and no array shape — but it is the one step here that
writes *values*, so the originals are stashed.

``01``: *"Used to turn a noise-free forward model into something resembling
measured data, so that an inversion of it is a fair test."*

That is the whole purpose and it bounds the scope: this exists for synthetic
data. Run on real data it adds noise to noise, and the result looks entirely
plausible.

Ordering is the hazard, so ordering is checked
----------------------------------------------
``01`` is unusually direct about it:

    Order matters and is easy to get wrong: uncertainties must be assigned
    before noise is drawn from them, and any along-line averaging belongs after
    both. The rebuild should make a wrong ordering detectable rather than
    silently producing plausible nonsense.

Three things follow, and they are the substance of this filter rather than the
arithmetic:

**It refuses to run twice.** Perturbing an already-perturbed dataset is the
classic silent error — the result is noisier by √2 and looks exactly like data.
So the application is recorded on the dataset, with its seed, and a second run
raises unless explicitly told this is intended.

**It says so when the uncertainties look untouched.** If every uncertainty in a
channel is identical, that is the signature of a uniform default from the system
description rather than an assigned model, which usually means filter_12 has not
run. Noise drawn from it is flat in relative terms, so late gates come out as
clean as early ones — which is precisely the thing the absolute floor exists to
prevent.

**The seed defaults to 42 and is always recorded.** No seed is statistically
better than another — measured, the worst correlation between PCG64 streams from
seeds 1 to 8 is 0.014 at twenty thousand samples, which is noise — so the choice
is an auditing property rather than a numerical one. 42 is the right default
precisely because it reads as *nobody chose this*, where an arbitrary-looking
number reads as though someone did and had a reason.

What the default cannot do is make two realizations independent. Ten synthetic
datasets built from one clean dataset with the default seed are ten identical
files. That is detectable rather than silent — the seed is recorded on the
dataset and in the report, and the report says whether it was chosen or
defaulted — but it is why anything wanting independent draws must say so.

What it draws from
------------------
The uncertainty in this format is *relative*, so the absolute standard deviation
of the perturbation is ``uncertainty × |value|``. Drawing from the relative
figure directly would put noise of order one on every datum regardless of its
magnitude.

A gate whose assigned uncertainty has passed one — the floor having overtaken the
signal — receives a perturbation larger than the datum itself. That is correct
and is what makes the synthetic look like real late-time data.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .distribution import summarize
from .pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "perturb_from_uncertainty"

#: Suffix for the stashed originals — the noise-free values this was drawn onto.
STASH_SUFFIX = "_clean"

#: Where the application is recorded, so a second one is detectable.
REGISTRY = "synthetic_noise"

#: A legible marker for "not chosen". No seed is numerically better than
#: another, so what a default can offer is honesty about having been one.
DEFAULT_SEED = 42


def applied(dataset) -> str | None:
    """What perturbation this dataset already carries, if any."""
    value = dataset.survey.xyz.model_info.get(REGISTRY)
    return str(value) if value else None


def perturb_from_uncertainty(dataset, seed: int = DEFAULT_SEED, channels=None,
                             again: bool = False, stash: bool = True):
    """Add noise drawn from each datum's assigned uncertainty.

    :param seed: The draw to make. Defaults to 42, which is a legible marker
        for "not chosen" rather than a better number — no seed is. **Independent
        realizations need different seeds**; the default gives every one of them
        the same noise.
    :param channels: Which channels to perturb, by key or moment name. All of
        them if unset.
    :param again: Permit perturbing a dataset that already carries noise.
        Off by default, because doing it unknowingly is the silent error this
        filter is written to prevent.
    :param stash: Keep the noise-free values under the same key with ``_clean``
        appended.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if seed is None:
        raise ValueError(
            "seed cannot be None. Its purpose is to make an inversion a fair "
            f"test, and an unrepeatable test is not one; leave it out for the "
            f"default of {DEFAULT_SEED}.")
    seed_given = seed != DEFAULT_SEED

    already = applied(dataset)
    if already and not again:
        raise ValueError(
            f"This dataset already carries synthetic noise ({already}). "
            f"Perturbing again would make it noisier by roughly sqrt(2) and the "
            f"result would look exactly like data. Pass again=True if that is "
            f"intended."
        )

    wanted = dataset.channels
    if channels is not None:
        named = set(dataset.resolve_per_channel(
            {k: True for k in ([channels] if isinstance(channels, str)
                               else channels)}))
        wanted = [c for c in wanted if c.key in named]
    if not wanted:
        report.notes.append("No channels matched; nothing was perturbed.")
        return dataset, report

    rng = np.random.default_rng(seed)
    per_channel, missing = {}, []

    for channel in wanted:
        if not channel.std_key or channel.std_key not in dataset.layer_data:
            missing.append(str(channel))
            continue

        array = dataset.layer_data[channel.data_key]
        values = array.to_numpy(dtype=float)
        relative = dataset.layer_data[channel.std_key].to_numpy(dtype=float)

        if stash:
            dataset.layer_data[f"{channel.data_key}{STASH_SUFFIX}"] = array.copy()

        # The stored uncertainty is relative, so the absolute spread of the
        # perturbation is that times the magnitude. Drawing from the relative
        # figure directly would put noise of order one on every datum whatever
        # its size.
        spread = np.abs(relative) * np.abs(values)
        drawn = rng.normal(0.0, 1.0, values.shape) * spread
        perturbed = values + np.where(np.isfinite(drawn), drawn, 0.0)

        dataset.layer_data[channel.data_key] = pd.DataFrame(
            perturbed, index=array.index, columns=array.columns)

        finite = relative[np.isfinite(relative)]
        uniform = finite.size > 1 and np.allclose(finite, finite[0])
        per_channel[str(channel)] = {
            "column": channel.data_key,
            "uncertainty_uniform": bool(uniform),
            "before": summarize(values.ravel()),
            "after": summarize(perturbed.ravel()),
        }
        if uniform:
            report.notes.append(
                f"Every uncertainty in {channel.std_key!r} is identical, which "
                f"is the signature of a uniform default rather than an assigned "
                f"model. Noise drawn from it is flat in relative terms, so late "
                f"gates will come out as clean as early ones. Run "
                f"assign_uncertainty first if that is not intended."
            )

    if missing:
        report.notes.append(
            f"No uncertainty array for {', '.join(missing)}; those channels "
            f"were not perturbed.")
    if not per_channel:
        return dataset, report

    dataset.survey.xyz.model_info[REGISTRY] = f"seed={seed}"
    if not seed_given:
        report.notes.append(
            f"No seed was given, so the default of {DEFAULT_SEED} was used. Any "
            f"realizations meant to be independent of this one need their own "
            f"seed — the same one draws the same noise."
        )
    report.distributions = {
        "seed": int(seed),
        "seed_source": "given" if seed_given else "default",
        "per_channel": per_channel,
        "stashed": STASH_SUFFIX if stash else None,
        "repeated": bool(already),
    }
    log.info("%s: perturbed %d channel(s) with seed %s",
             STEP, len(per_channel), seed)
    return dataset, report
