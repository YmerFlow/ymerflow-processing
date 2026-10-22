"""filter_09 — the nominal current.

The amplitude convention is no longer a step. Data on disk is in exactly the
SimPEG static_instrument format — raw ``dbdt_chNgt`` per unit transmitter moment,
with ``scalefactor`` (model_info) and ``GateFactor`` (GEX) kept as metadata and
applied at read, never baked into the arrays — fixed upstream by the importer.
The old ``declare_normalization`` / ``convert_normalization`` steps and the
three-valued ``Dataset.normalization`` they drove are retired: the convert step
was commented out of the standard pipeline, the declare step only re-asserted
what import wrote, and the one engine that actually inverts the data
(``simpeg.static_instrument``) ignores such a field and reads the fixed
convention directly. What remains of this module is the one thing that is about a
number from the system description and not about that convention: filter_09.

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

import logging

import numpy as np

from ..dataset import STASH_PREFIX, stash_key
from ..distribution import summarize
from ..pipeline import StepReport

log = logging.getLogger(__name__)

CATEGORY = "corrections"


def apply_nominal_current(dataset, columns: str | dict[str, str] | None = None, stash: bool = True):
    """Overwrite the per-sounding transmitter current with the declared nominal.

    Parameters
    ----------
    dataset : Dataset
        The dataset whose current columns are overwritten.
    columns : dict | None
        Which ``flightlines`` column holds each channel's current, per channel
        key or moment name. Discovered if unset.
    stash : bool
        Keep the measured values alongside, under the column's name with
        ``orig_`` in front.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.

    Notes
    -----
    Deliberately destructive; qc_04 is what tells you whether you should. It
    measures stability against the moment's own mean and agreement with this
    same declared nominal, and reports the offset without penalising it.
    """
    from ..qc.current import _nominal_current, find_current_columns

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
                frame[stash_key(column)] = frame[column]

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
                            "stashed": STASH_PREFIX if stash else None}
    report.notes.append(
        "The measured current is overwritten by declaration. qc_04 is what says "
        "whether it should be: it measures stability against the moment's own "
        "mean and reports the offset from this same nominal without penalising "
        "it."
    )
    log.info("apply_nominal_current: set %d channel(s) from the description",
             len(per_channel))
    return dataset, report
