"""Misfit between measured data and a forward synthetic, as a column.

``01``'s resolution of filter_31's oddity:

    the comparison runner computes misfit between the inversion and the measured
    data and writes it as a column. filter_31 then stops being special — it
    becomes ``cull where near_infrastructure AND misfit > threshold``, both
    columns on the dataset, thresholded like any other culling filter. That makes
    the loop explicit as steps rather than one filter reaching backwards, and
    misfit becomes generally available — plottable, exportable in the vector
    output, usable by QC.

Every word of that is followed except *"the comparison runner"*. Comparison is
N things to **one report**; this is a dataset and an external input to a dataset
with new columns, which is the filter contract exactly. The inversion result
arrives the way filter_05's terrain raster and filter_33's edit diff arrive —
R8's typed external input — and the library needs no inversion machinery to read
one, because an inversion result *is* an XYZ.

The quantity, and why it is divided
------------------------------------
**Measured minus forward synthetic**, divided by the uncertainty the inversion
weighted by. The division is not a detail. A transient spans three or four orders
of magnitude from first gate to last, so a raw difference of fixed size is
enormous at late time and negligible at early time — no single threshold on it is
wrong in neither direction.

Divided, the misfit is dimensionless and comparable across the whole transient:
**how many standard deviations the model missed by**, which is a statement about
the fit rather than about the amplitude.

Two ways in, because two things produce results
------------------------------------------------
An inversion writes the normalized residual itself as ``dbdt_err_ch{n}gt``, being
``(dobs - dpred) * W.diagonal()``, and that is used where present. A plain forward
model writes only predictions, so the difference is taken here and normalized by
the dataset's own uncertainty.

What it writes, and why those names
------------------------------------
**Per gate**, under the same key an inversion would have used. A dataset whose
misfit came from an inversion and one whose misfit was computed here are then
indistinguishable, and anything reading misfit finds it the same way in both.

**Per sounding**, as ``resdata`` — the root-mean-square of the gatewise misfit
over the gates that are in use. That is the column qc_11 grades, it is seven
characters so it survives a shapefile field name, and it is what the inversion
itself writes under that name.

Only over gates in use, because a culled gate is not evidence about the fit. A
sounding whose bad gates were removed before inverting should not carry their
residual afterwards.
"""

import logging
import re

import numpy as np
import pandas as pd

from .distribution import summarize
from .mask import in_use
from .pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "compute_misfit"

#: How a result names its gatewise residual. Matched on a captured channel index
#: rather than by substring, the discipline channel discovery keeps.
MISFIT_PATTERNS = (r"^dbdt_err_ch(\d+)", r"^err_ch(\d+)")

#: Where the per-sounding root-mean-square misfit goes. The name the inversion
#: itself uses, and seven characters, so it survives a shapefile field.
RMS_COLUMN = "resdata"


def misfit_key(channel) -> str:
    """Where a channel's gatewise misfit lives, by the inversion's own naming."""
    data = channel.data_key
    # Inserted after the prefix as the delivery spells it. A case-sensitive
    # replace would find nothing in `DBDT_Ch01` and return the data key, and
    # `compute_misfit` would then write the misfit over the measurements.
    at = data.lower().find("dbdt_")
    if at >= 0:
        after = at + len("dbdt_")
        return data[:after] + "err_" + data[after:]
    return f"err_{data}"


def find_misfit(source, channel, shape=None):
    """A stated gatewise residual for this channel, if the source carries one.

    Looked up first under the key this library would write for the channel,
    which is the only route for a convention that names its channels rather
    than numbering them — ``err_LM_Z_G01`` captures no index for the numbered
    patterns to match on.
    """
    layer_data = getattr(source, "layer_data", None) or {}
    wanted = misfit_key(channel).lower()
    for key, array in layer_data.items():
        if str(key).lower() == wanted:
            values = np.asarray(array, dtype=float)
            if shape is None or values.shape == shape:
                return values
    for key, array in layer_data.items():
        for pattern in MISFIT_PATTERNS:
            found = re.match(pattern, str(key), flags=re.IGNORECASE)
            if found and int(found.group(1)) == channel.index:
                values = np.asarray(array, dtype=float)
                if shape is None or values.shape == shape:
                    return values
    return None


def _as_xyz(result):
    """The result as an XYZ, whether given as one, a Dataset, or a path."""
    if hasattr(result, "layer_data"):
        return result
    if hasattr(result, "survey"):
        return result.survey.xyz
    from libaarhusxyz.xyz import XYZ
    return XYZ(str(result))


def gatewise_misfit(dataset, result, channel) -> np.ndarray | None:
    """Measured minus forward synthetic for one channel, in sigmas.

    ``None`` when the result carries neither a residual nor a prediction for
    this channel. Raises when it carries predictions but the dataset has no
    uncertainty to normalize the difference by, since a raw difference cannot be
    thresholded across a transient spanning orders of magnitude.
    """
    xyz = _as_xyz(result)
    shape = dataset.layer_data[channel.data_key].shape

    stated = find_misfit(xyz, channel, shape)
    if stated is not None:
        return stated

    predicted = xyz.layer_data.get(channel.data_key)
    if predicted is None:
        return None
    synthetic = np.asarray(predicted, dtype=float)
    if synthetic.shape != shape:
        return None

    measured = dataset.layer_data[channel.data_key].to_numpy(dtype=float)
    if not channel.std_key or channel.std_key not in dataset.layer_data:
        raise ValueError(
            f"The result for {channel} carries predictions but no residual, and "
            f"this dataset has no uncertainty to normalize the difference by. A "
            f"raw difference cannot be thresholded across a transient spanning "
            f"orders of magnitude."
        )
    relative = dataset.layer_data[channel.std_key].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = np.abs(relative) * np.abs(measured)
        return np.where(sigma > 0, (measured - synthetic) / sigma, np.nan)


def ensure_misfit(dataset, channel, result=None):
    """This channel's gatewise misfit, computing it if it is not there yet.

    The column is the canonical form, so anything wanting misfit looks for it
    first. But a caller who has the result to hand should not have to run a
    separate step to make one appear — so where the column is absent and a
    result was given, it is computed here and **written**, which means the next
    consumer finds it rather than computing it again.

    ``None`` when there is no column and no result, which is the caller's cue to
    say what it needs rather than to guess.
    """
    shape = dataset.layer_data[channel.data_key].shape
    existing = find_misfit(dataset.survey.xyz, channel, shape)
    if existing is not None:
        return existing
    if result is None:
        return None

    computed = gatewise_misfit(dataset, result, channel)
    if computed is None:
        return None
    reference = dataset.layer_data[channel.data_key]
    dataset.layer_data[misfit_key(channel)] = pd.DataFrame(
        computed, index=reference.index, columns=reference.columns)
    return computed


def compute_misfit(dataset, result, rms_column: str = RMS_COLUMN):
    """Write misfit between the data and a forward synthetic as columns.

    Parameters
    ----------
    result
        A previous inversion's or forward model's output — an XYZ, a Dataset,
        or a path to one.
    rms_column : str
        Where the per-sounding root-mean-square misfit goes.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    channels = dataset.channels
    if frame is None or len(frame) == 0 or not channels:
        report.notes.append("No soundings or no channels; nothing to compute.")
        return dataset, report

    per_channel, without, squares, counts = {}, [], None, None
    for channel in channels:
        misfit = gatewise_misfit(dataset, result, channel)
        if misfit is None:
            without.append(str(channel))
            continue

        reference = dataset.layer_data[channel.data_key]
        key = misfit_key(channel)
        dataset.layer_data[key] = pd.DataFrame(
            misfit, index=reference.index, columns=reference.columns)

        # A culled gate is not evidence about the fit: a sounding whose bad
        # gates were removed before inverting should not carry their residual.
        live = in_use(dataset, channel) & np.isfinite(misfit)
        usable = np.where(live, misfit, np.nan)
        squares = (np.nansum(usable ** 2, axis=1) if squares is None
                   else squares + np.nansum(usable ** 2, axis=1))
        counts = (live.sum(axis=1) if counts is None
                  else counts + live.sum(axis=1))

        per_channel[str(channel)] = {
            "column": key,
            "misfit": summarize(misfit[np.isfinite(misfit)]),
            "gates_used": int(live.sum()),
        }

    if without:
        report.notes.append(
            f"The result carries neither a residual nor a prediction for "
            f"{', '.join(without)}; those channels have no misfit column.")
    if not per_channel:
        return dataset, report

    with np.errstate(divide="ignore", invalid="ignore"):
        rms = np.where(counts > 0, np.sqrt(squares / counts), np.nan)
    frame[rms_column] = rms

    report.soundings_affected = int(np.isfinite(rms).sum())
    report.distributions = {
        "per_channel": per_channel,
        "rms_column": rms_column,
        "rms": summarize(rms),
    }
    report.notes.append(
        f"Misfit is in standard deviations — how many the model missed by — so "
        f"a threshold on it means the same thing at every gate. The "
        f"per-sounding figure in {rms_column!r} is over gates in use only."
    )
    log.info("%s: wrote misfit for %d channel(s), median RMS %.3g",
             STEP, len(per_channel), float(np.nanmedian(rms)))
    return dataset, report
