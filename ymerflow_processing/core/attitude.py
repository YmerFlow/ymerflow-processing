"""filter_06 and filter_07 — the transmitter was not level, and what to do.

A matched pair. filter_06 corrects for tilt; filter_07 declares there is none to
correct for. They share a module because 07 exists precisely to make 06 a no-op,
and because both write the same attitude columns and stash them the same way.

Why tilt has to be corrected at all
-----------------------------------
``01``: *"the 1D forward model assumes a horizontal loop; without it, tilt is
indistinguishable from geology."*

That is the whole argument, and it is worth stating in the direction that bites:
a systematically tilted stretch of line does not look like bad data. It looks
like a resistivity change, and it inverts into one.

The correction
--------------
The transmitter's dipole moment points along the loop normal. Tilt the loop by
``θ`` from horizontal and the vertical component of that moment falls to
``m cos θ`` — the horizontal component excites a mode a 1D earth model does not
represent. The receiver coil is rigidly mounted on the same frame, so it is
tilted by the same ``θ``, and the vertical field it senses projects onto its
normal by another ``cos θ``.

Both together:

    measured ≈ horizontal_response × cos²θ

so the correction divides by ``cos²θ``, with

    cos θ = cos(pitch) · cos(roll)

exactly, as qc_02 uses. This is first order: it accounts for the projection of a
vertical-dipole response and not for the horizontal mode the tilt also excites.
That is the standard airborne TEM treatment and the reason tilt is *also* a QC
threshold — beyond a few degrees the neglected term stops being negligible, and
the answer is to cull the sounding (filter_22) rather than to correct it harder.

Idempotency, which ``01`` calls out explicitly
----------------------------------------------
*"Must be idempotent — applying it twice would apply the correction twice."*

Applying it twice is not detectable from the values: a doubly corrected sounding
is a plausible sounding of a different earth. So the application is recorded on
the dataset and a second pass raises unless explicitly asked for — the same
guard filter_13 uses, for the same reason.

The recorded attitude is not cleared afterwards. It remains true that the
aircraft was tilted; what changes is that the values no longer carry the
consequence. Clearing it would make a corrected dataset indistinguishable from
one flown level, and qc_02 would then report perfect attitude on data that was
salvaged rather than well flown.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .distribution import summarize
from .pipeline import StepReport

log = logging.getLogger(__name__)

#: Where the delivered attitude is kept. Ten characters each.
STASHED_PITCH = "tilt_x_raw"
STASHED_ROLL = "tilt_y_raw"

#: Where a tilt correction records itself, so a second one is detectable.
REGISTRY = "tilt_corrected"


def corrected(dataset) -> str | None:
    """What tilt correction this dataset already carries, if any."""
    value = dataset.survey.xyz.model_info.get(REGISTRY)
    return str(value) if value else None


def _attitude(frame, pitch_column, roll_column):
    for role, name in (("pitch", pitch_column), ("roll", roll_column)):
        if name not in frame.columns:
            raise KeyError(
                f"No {role} column {name!r} in flightlines. "
                f"Present: {', '.join(map(str, frame.columns))}"
            )
    return (frame[pitch_column].to_numpy(dtype=float),
            frame[roll_column].to_numpy(dtype=float))


def correct_tilt(dataset, pitch_column: str = "tilt_x",
                 roll_column: str = "tilt_y", max_correction: float = 4.0,
                 again: bool = False, stash: bool = True):
    """Rescale amplitudes to what a horizontal transmitter would have measured.

    :param pitch_column: Which column holds pitch, along the flight direction.
    :param roll_column: Which holds roll, across it.
    :param max_correction: Largest factor to apply. Beyond this the first-order
        treatment has stopped being defensible, so the sounding is left
        uncorrected and reported rather than scaled by a number the physics does
        not support. ``4.0`` corresponds to about 45° of total tilt.
    :param again: Permit correcting a dataset that already carries a correction.
    :param stash: Keep the delivered values alongside.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step="correct_tilt")
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    already = corrected(dataset)
    if already and not again:
        raise ValueError(
            f"This dataset already carries a tilt correction ({already}). "
            f"Applying it twice would apply the correction twice, and a doubly "
            f"corrected sounding is a plausible sounding of a different earth. "
            f"Pass again=True if that is intended."
        )
    if frame is None or len(frame) == 0 or not dataset.channels:
        report.notes.append("No soundings or no channels; nothing to correct.")
        return dataset, report

    pitch, roll = _attitude(frame, pitch_column, roll_column)
    cosine = np.cos(np.radians(pitch)) * np.cos(np.radians(roll))
    tilt = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    with np.errstate(divide="ignore", invalid="ignore"):
        factor = 1.0 / cosine ** 2
    # Beyond the limit the neglected horizontal mode is no longer small, so
    # scaling harder is not a better answer — culling is (filter_22).
    too_tilted = ~np.isfinite(factor) | (factor > max_correction)
    factor = np.where(too_tilted, 1.0, factor)

    per_channel = {}
    for channel in dataset.channels:
        array = dataset.layer_data[channel.data_key]
        if stash:
            dataset.layer_data[f"{channel.data_key}_untilted"] = array.copy()
        values = array.to_numpy(dtype=float)
        dataset.layer_data[channel.data_key] = pd.DataFrame(
            values * factor[:, None], index=array.index, columns=array.columns)
        per_channel[str(channel)] = {"column": channel.data_key}

    dataset.survey.xyz.model_info[REGISTRY] = (
        f"cos2 pitch={pitch_column} roll={roll_column}")

    report.soundings_affected = int((factor != 1.0).sum())
    report.distributions = {
        "factor": summarize(factor),
        "total_tilt_degrees": summarize(tilt),
        "left_uncorrected": int(too_tilted.sum()),
        "max_correction": float(max_correction),
        "per_channel": per_channel,
        "repeated": bool(already),
    }
    if too_tilted.any():
        report.notes.append(
            f"{int(too_tilted.sum())} sounding(s) tilt beyond a correction of "
            f"{max_correction:g} and were left as measured. Past that the "
            f"first-order treatment is not defensible; cull them with "
            f"cull_by_geometry rather than scaling them harder."
        )
    report.notes.append(
        "The recorded attitude is unchanged. The aircraft was still tilted; "
        "what changed is that the values no longer carry the consequence."
    )
    log.info("correct_tilt: median factor %.4g across %d channel(s)",
             float(np.median(factor)), len(per_channel))
    return dataset, report


def zero_attitude(dataset, pitch_column: str = "tilt_x",
                  roll_column: str = "tilt_y", stash: bool = True):
    """Force the recorded attitude to zero, keeping the originals.

    ``01``: *"Used when measured attitude is unreliable or was never recorded,
    so that a downstream tilt correction becomes a no-op rather than applying
    garbage. If the attitude fields are absent entirely, they are created as
    zero."*

    Both halves matter. Creating the columns means a delivery with no attitude at
    all can still run a pipeline that mentions tilt, and the correction it
    applies is exactly one.

    :param stash: Keep the delivered attitude alongside. Where the columns did
        not exist there is nothing to keep, and that is reported.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step="zero_attitude")
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if frame is None or len(frame) == 0:
        report.notes.append("No soundings; nothing to zero.")
        return dataset, report

    created, zeroed = [], {}
    for role, name, stash_name in (("pitch", pitch_column, STASHED_PITCH),
                                   ("roll", roll_column, STASHED_ROLL)):
        if name not in frame.columns:
            created.append(name)
            frame[name] = 0.0
            continue
        values = frame[name].to_numpy(dtype=float)
        if stash:
            frame[stash_name] = values
        zeroed[role] = summarize(values)
        frame[name] = 0.0

    if created:
        report.notes.append(
            f"Attitude column(s) {', '.join(created)} did not exist and were "
            f"created as zero. There was nothing to stash for them."
        )
    report.soundings_affected = len(frame)
    report.distributions = {
        "columns": {"pitch": pitch_column, "roll": roll_column},
        "created": created,
        "before": zeroed,
        "stashed": ({"pitch": STASHED_PITCH, "roll": STASHED_ROLL}
                    if stash and zeroed else None),
    }
    report.notes.append(
        "A tilt correction applied after this is exactly one, which is the "
        "point: a no-op rather than a correction derived from attitude nobody "
        "trusts."
    )
    log.info("zero_attitude: zeroed %s", ", ".join(zeroed) or "nothing")
    return dataset, report
