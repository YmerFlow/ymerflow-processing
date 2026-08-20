"""filter_08 and filter_09 — the conventions and constants the data is stated in.

filter_08 converts amplitudes between normalization conventions. filter_09 fills
the per-sounding transmitter current from the system description. They share a
module because both take a number from the system description and impose it on
the data, and because the second is one of the inputs the first depends on.

filter_08 — where unit confusion concentrates
---------------------------------------------
``01`` says so directly, and prescribes the fix:

    This filter is where unit confusion concentrates. The three conventions
    differ by orders of magnitude and are easy to mistake for one another. The
    rebuild should make the current convention an explicit property of the
    dataset that travels with it, rather than something the user asserts at this
    step and hopes is right.

So this takes **no parameter saying what the data currently is**. It reads
``Dataset.normalization``, converts, and updates it. The old signature — "which
convention the input is currently in" — is the defect: a wrong assertion there is
undetectable, and being wrong by the dipole moment is being wrong by three or
four orders of magnitude with nothing in the values to give it away.

The three conventions
---------------------
``raw``
    Instrument units, as delivered.
``rx``
    Divided by receiver area and turns. Volts per square metre.
``rx_tx``
    Also divided by the transmitter dipole moment. What the forward code expects,
    because misfit is meaningless unless observed and predicted share a
    convention.

Each step is a division by a positive constant, so every conversion is invertible
and the whole thing composes: asking for a convention the data is already in is a
no-op rather than an error.

The convention travels with the dataset
---------------------------------------
It is recorded in ``model_info``, so a dataset written out and read back declares
what it is rather than defaulting to whatever the loader was told. That is the
half of ``01``'s prescription that a constructor argument alone cannot provide —
an assertion made once at load time does not survive the file.

filter_09 — the nominal current
-------------------------------
``01``: *"populates the per-sounding transmitter current from the nominal value
in the system description, overwriting whatever was measured. Used when measured
current is missing, noisy, or known to be wrong."*

Deliberately destructive, and stashed for that reason. qc_04 exists to tell you
whether you *should*: it measures stability against the moment's own mean and
agreement with this same nominal, and reports the offset without penalising it.
Running this filter is the decision qc_04 informs, and the two read the same
declared value.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .dataset import NORMALIZATIONS, RAW, RX_NORMALIZED, RX_TX_NORMALIZED
from .distribution import summarize
from .pipeline import StepReport

log = logging.getLogger(__name__)

#: Where the convention is recorded, so it survives a round trip through a file.
REGISTRY = "normalization"

#: Order matters: each convention is the one before it divided by a constant.
ORDER = (RAW, RX_NORMALIZED, RX_TX_NORMALIZED)


def recorded_normalization(dataset) -> str | None:
    """The convention a dataset declares in its own metadata, if any."""
    value = dataset.survey.xyz.model_info.get(REGISTRY)
    return str(value) if value in NORMALIZATIONS else None


def _receiver_factor(dataset, channel) -> float | None:
    """Receiver area times turns, from the system description."""
    block = getattr(dataset.survey.gex, "gex_dict", {}).get(
        f"Channel{channel.index}", {})
    general = getattr(dataset.survey.gex, "gex_dict", {}).get("General", {})
    if not isinstance(block, dict):
        return None
    area = block.get("RxCoilArea") or general.get("RxCoilArea")
    turns = block.get("RxCoilNumber") or general.get("RxCoilNumber") or 1.0
    if area in (None, 0):
        return None
    return float(area) * float(turns)


def _steps(source: str, target: str):
    """The conversions between two conventions, as (name, direction) pairs."""
    low, high = ORDER.index(source), ORDER.index(target)
    names = ("receiver", "moment")
    if high > low:
        return [(names[i], "divide") for i in range(low, high)]
    return [(names[i], "multiply") for i in reversed(range(high, low))]


def convert_normalization(dataset, to: str = RX_TX_NORMALIZED,
                          stash: bool = True):
    """Convert amplitudes into ``to``, from whatever the dataset declares.

    :param to: The convention to produce. Defaults to ``rx_tx``, which is what
        the forward code expects.
    :param stash: Keep the values as they were alongside.

    **It takes no parameter saying what the data currently is.** That assertion
    is the defect ``01`` names — a wrong one is undetectable and wrong by orders
    of magnitude. The dataset declares its own convention and this updates it.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step="convert_normalization")
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if to not in NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalization {to!r}. Available: {', '.join(NORMALIZATIONS)}")

    source = recorded_normalization(dataset) or dataset.normalization
    report.distributions = {"from": source, "to": to, "converted": []}

    if source == to:
        dataset.survey.xyz.model_info[REGISTRY] = to
        report.notes.append(
            f"Already in {to!r}; nothing was converted. Each conversion is a "
            f"division by a positive constant, so asking for the convention the "
            f"data is already in is a no-op rather than an error."
        )
        return dataset, report

    if not dataset.channels:
        report.notes.append("No channels; nothing to convert.")
        return dataset, report

    conversions = _steps(source, to)
    per_channel, missing = {}, []

    for channel in dataset.channels:
        factor = 1.0
        unknown = []
        for name, direction in conversions:
            if name == "receiver":
                value = _receiver_factor(dataset, channel)
            else:
                value = getattr(channel, "dipole_moment", None)
            if value in (None, 0) or not np.isfinite(value):
                unknown.append(name)
                continue
            factor = factor / value if direction == "divide" else factor * value
        if unknown:
            missing.append(f"{channel} ({', '.join(unknown)})")
            continue

        array = dataset.layer_data[channel.data_key]
        if stash:
            dataset.layer_data[f"{channel.data_key}_{source}"] = array.copy()
        values = array.to_numpy(dtype=float)
        dataset.layer_data[channel.data_key] = pd.DataFrame(
            values * factor, index=array.index, columns=array.columns)
        per_channel[str(channel)] = {
            "factor": float(factor),
            "before": summarize(values.ravel()),
            "after": summarize((values * factor).ravel()),
        }

    if missing:
        raise ValueError(
            f"Cannot convert {', '.join(missing)}: the system description does "
            f"not give the constant(s) needed. Converting the rest would leave "
            f"one dataset in two conventions, which is the failure this filter "
            f"exists to prevent."
        )

    dataset.normalization = to
    dataset.survey.xyz.model_info[REGISTRY] = to
    report.soundings_affected = report.soundings_total
    report.distributions.update({
        "converted": [name for name, _ in conversions],
        "per_channel": per_channel,
        "stashed": f"_{source}" if stash else None,
    })
    log.info("convert_normalization: %s -> %s across %d channel(s)",
             source, to, len(per_channel))
    return dataset, report


def apply_nominal_current(dataset, columns=None, stash: bool = True):
    """Overwrite the per-sounding transmitter current with the declared nominal.

    :param columns: Which ``flightlines`` column holds each channel's current,
        per channel key or moment name. Discovered if unset.
    :param stash: Keep the measured values alongside.

    Deliberately destructive; qc_04 is what tells you whether you should. It
    measures stability against the moment's own mean and agreement with this
    same declared nominal, and reports the offset without penalising it.

    Returns ``(dataset, StepReport)``.
    """
    from .qc.qc_04_current import _nominal_current, find_current_columns

    report = StepReport(step="apply_nominal_current")
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    channels = dataset.channels
    if frame is None or len(frame) == 0 or not channels:
        report.notes.append("No soundings or no channels; nothing to apply.")
        return dataset, report

    named = dataset.resolve_per_channel(columns) if columns else {}
    found = find_current_columns(frame, channels)

    per_channel, undeclared, uncolumned = {}, [], []
    for channel in channels:
        nominal = _nominal_current(dataset, channel)
        if nominal in (None, 0) or not np.isfinite(nominal):
            undeclared.append(str(channel))
            continue

        column = named.get(channel.key) or found.get(channel.key)
        if column is None:
            # No column to overwrite, so one is created — a delivery that never
            # recorded current is exactly the case 01 names.
            column = f"{(channel.name or channel.key).lower()}current"
            uncolumned.append(f"{channel} -> {column}")
            before = None
        else:
            before = summarize(frame[column].to_numpy(dtype=float))
            if stash:
                # Not truncated to ten characters. "LMcurrent_raw"[:10] is
                # "LMcurrent_", which is neither readable nor safely unique —
                # two current columns would collide. The vector export already
                # detects truncation collisions and refuses before writing,
                # which is the right place for that check to live rather than
                # in every step guessing at it.
                frame[f"{column}_raw"] = frame[column]

        frame[column] = float(nominal)
        per_channel[str(channel)] = {"column": column, "nominal": float(nominal),
                                     "before": before}

    if undeclared:
        report.notes.append(
            f"The system description declares no nominal current for "
            f"{', '.join(undeclared)}; those channels were left alone.")
    if uncolumned:
        report.notes.append(
            f"No current column existed for {', '.join(uncolumned)}, so one was "
            f"created. That is the case this filter is named for.")
    if not per_channel:
        return dataset, report

    report.soundings_affected = report.soundings_total
    report.distributions = {"per_channel": per_channel,
                            "stashed": "_raw" if stash else None}
    report.notes.append(
        "The measured current is overwritten by declaration. qc_04 is what says "
        "whether it should be: it measures stability against the moment's own "
        "mean and reports the offset from this same nominal without penalising "
        "it."
    )
    log.info("apply_nominal_current: set %d channel(s) from the description",
             len(per_channel))
    return dataset, report
