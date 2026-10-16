"""filter_22 — disable soundings whose flight geometry broke the forward model.

**What it does to the dataset:** turns off entire soundings in the in-use mask —
every gate of every channel — where recorded attitude or altitude falls outside
acceptable bounds. It changes no measurement, no uncertainty, and no array shape
(R6).

``01`` describes these as "soundings where the geometry assumption underlying the
forward model has broken down." That is the whole justification, and it is worth
being precise about: the forward model places a transmitter loop of known area at
a known height and orientation above a layered earth. Tilt the loop far enough
and its coupling to the ground is no longer what the model assumes; lift it far
enough and the signal is dominated by something other than the section beneath
it. Inverting such a sounding does not fail — it produces a model, confidently,
of an earth that was never measured.

Restating the limits, and where the defaults come from
------------------------------------------------------
``01`` suggests sharing thresholds with qc_01 and qc_02 rather than restating
them. This restates them, because **culling and contract compliance are
different questions**. A sounding flown at 70 m against a 65 m contract is out of
spec and still perfectly invertible; one at 200 m is neither. A contract limit is
what the operator agreed to fly, and a culling limit is where the physics stops
holding — the same shape of number, measuring different things, and forcing them
to be equal would make one of them wrong.

But they are related, and "where did QC stop calling this acceptable" is a
sensible starting point for "where has the geometry broken down". So each limit
**defaults to the corresponding QC threshold** where one was recorded — the
out-of-spec boundary, not the good one, since that is the point QC itself stopped
accepting the sounding. Set explicitly, the given value wins and the default is
never consulted.

That needs the number to outlive the run that chose it, which is why the QC
checks record what they judged against on the dataset. QC and culling are
separate pipeline documents and often separate invocations.

Whole soundings, every channel
------------------------------
Attitude and altitude are properties of the *aircraft*, not of a transient. A
badly tilted sounding is badly tilted for every moment and every component
simultaneously, so this writes across every channel — a row of the mask rather
than a column.

That is also why it takes no gate policy. ``spread_earlier`` and ``disable_tail``
describe how a problem propagates *along a transient*, and this problem has no
position along one.

Sources
-------
The geometry the forward model assumes — loop area, height, orientation — enters
through the dipole approximation used to compute the primary field; see
``em/tem/method.py`` for the moment. Departure from level flight changes the
component of the field coupling into the receiver, which is the same physical
statement filter_06 uses to *correct* small tilts rather than cull them. This
filter is for the tilts too large to correct.

**There are no default limits.** Unset and with no QC thresholds recorded, this
raises rather than guessing: what counts as too tilted or too high is an
operational judgement about an instrument, and a house number here would silently
cull a client's data.
"""

from __future__ import annotations

import logging

import numpy as np

from ..mask import cull, mark_step
from ..pipeline import StepReport
from ..qc.thresholds import recorded_thresholds

log = logging.getLogger(__name__)

STEP = "cull_by_geometry"

#: This filter's number in ``building_plans/01``, used as its provenance code.
STEP_CODE = 22

#: Which QC check each limit falls back to, and which of its bounds. Always the
#: out-of-spec boundary — the point QC itself stopped accepting the sounding —
#: rather than the Good one, which is where it stopped being unremarkable.
DEFAULT_FROM = {
    "max_pitch": ("system_attitude", "marginal"),
    "max_roll": ("system_attitude", "marginal"),
    "max_altitude": ("height_above_ground", "marginal_max"),
}


def _limit(name, given, thresholds, report):
    """One limit: what was given, or the QC threshold it defaults to."""
    if given is not None:
        return float(given), "given"
    step, bound = DEFAULT_FROM.get(name, (None, None))
    spec = thresholds.get(step) or {}
    if bound in spec:
        report.notes.append(
            f"{name} was not given; using {step}'s out-of-spec boundary "
            f"({bound}={spec[bound]:g}) as the starting point."
        )
        return float(spec[bound]), f"{step}.{bound}"
    return None, None


def cull_by_geometry(
    dataset,
    max_pitch: float | None = None,
    max_roll: float | None = None,
    max_altitude: float | None = None,
    min_altitude: float | None = None,
    pitch_column: str = "tilt_x",
    roll_column: str = "tilt_y",
    altitude_column: str | None = None,
):
    """Disable soundings flown outside the geometry the forward model assumes.

    Parameters
    ----------
    max_pitch : float | None
        Beyond ±this, in the attitude column's units, the sounding is disabled.
        Defaults to ``system_attitude``'s out-of-spec boundary if that check
        has run.
    max_roll : float | None
        The same, across the flight direction.
    max_altitude : float | None
        Above this the sounding is disabled. Defaults to
        ``height_above_ground``'s out-of-spec boundary if recorded.
    min_altitude : float | None
        Below this the sounding is disabled — a reading too low to be real
        usually means the altimeter, not the aircraft. **No default:** qc_01 is
        one-sided and has no minimum to borrow.
    pitch_column : str
        Which ``flightlines`` column holds pitch.
    roll_column : str
        Which holds roll.
    altitude_column : str | None
        Which holds height above ground. Discovered if unset.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if frame is None or len(frame) == 0:
        report.notes.append("No soundings; nothing to cull.")
        return dataset, report

    thresholds = recorded_thresholds(dataset)
    limits, sources = {}, {}
    for name, given in (("max_pitch", max_pitch), ("max_roll", max_roll),
                        ("max_altitude", max_altitude)):
        limits[name], sources[name] = _limit(name, given, thresholds, report)
    limits["min_altitude"] = None if min_altitude is None else float(min_altitude)
    sources["min_altitude"] = "given" if min_altitude is not None else None

    if all(v is None for v in limits.values()):
        raise ValueError(
            "cull_by_geometry needs at least one limit. None was given, and no "
            "QC thresholds are recorded on this dataset to default from — run "
            "the QC checks first, or state the limits here. There are no "
            "built-in values: what counts as too tilted or too high is an "
            "operational judgement about an instrument."
        )

    altitude_column = altitude_column or dataset.survey.xyz.alt_column
    bad = np.zeros(len(frame), dtype=bool)
    measured = {}

    for name, column, test in (
        ("max_pitch", pitch_column, lambda v, l: np.abs(v) > l),
        ("max_roll", roll_column, lambda v, l: np.abs(v) > l),
        ("max_altitude", altitude_column, lambda v, l: v > l),
        ("min_altitude", altitude_column, lambda v, l: v < l),
    ):
        limit = limits[name]
        if limit is None:
            continue
        if column is None or column not in frame.columns:
            report.notes.append(
                f"No column {column!r} for {name}; that limit was not applied."
            )
            continue
        values = frame[column].to_numpy(dtype=float)
        # NaN fails every comparison, so an unmeasured sounding is not culled
        # here. It has no geometry on record, which is a different problem and
        # not this filter's to decide.
        failed = test(values, limit) & np.isfinite(values)
        bad |= failed
        measured[name] = {"column": column, "limit": limit,
                          "source": sources[name], "soundings": int(failed.sum())}

    if not measured:
        report.notes.append("No geometry columns found; nothing was culled.")
        return dataset, report

    per_channel = {}
    for channel in dataset.channels:
        gates = dataset.layer_data[channel.data_key].shape[1]
        # A row of the mask, not a column: attitude and altitude belong to the
        # aircraft, so a bad sounding is bad for every gate of every channel.
        across = np.repeat(bad[:, None], gates, axis=1)
        per_channel[str(channel)] = cull(dataset, channel, across,
                                         step_code=STEP_CODE)

    if bad.any():
        mark_step(dataset, STEP_CODE, STEP)

    report.soundings_affected = int(bad.sum())
    report.distributions = {
        "limits": measured,
        "soundings_culled": int(bad.sum()),
        "per_channel": per_channel,
        "provenance_code": STEP_CODE,
    }
    log.info("%s: disabled %d of %d sounding(s) across %d channel(s)",
             STEP, int(bad.sum()), len(frame), len(per_channel))
    return dataset, report
