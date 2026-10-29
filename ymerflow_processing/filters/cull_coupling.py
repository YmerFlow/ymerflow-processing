"""Coupled data: transients the survey's own basis cannot reproduce.

**What it does to the dataset:** writes a per-gate coupling table for every
channel it judges and a per-sounding confidence column, every run. With
``action: cull`` it then turns off what the table marks. It changes no
measurement, no uncertainty and no array shape (R6).

The gap it fills
----------------
``cull_negatives``, ``cull_by_curvature`` and ``cull_by_slope`` judge one
sounding's shape at a time. They catch the oscillatory kind of coupling, which
breaks monotonicity, and **cannot catch the smoothly elevated kind**: nothing
about its shape is wrong. What is wrong is that it does not look like the
transients around it. This step fits a basis to the survey and asks, per gate,
whether each transient is something the basis can reproduce, by the method of
Asif et al. (2022), Algorithm 1 (``coupling/reconstruction.py``).

Detect and cull, never repair: coupled data cannot be corrected.

One table, and a switch that applies it
---------------------------------------
``action: flag`` writes the table and the confidence and leaves the mask alone;
``action: cull`` writes the same table and then applies it. One code path, so
nothing can drift between a flagging run and a culling run, and the table
survives both (``coupling/table.py``).

Where it sits
-------------
**After** ``assign_uncertainty``: the error is in sigmas of the assigned
uncertainty, so there is nothing to divide by before it. **Before**
``average_along_line``: averaging smears a coupling signature into its
neighbors, and before averaging is the data the method was built for.

Where it does not suit
----------------------
The criterion tests **one-dimensionality, not cleanliness**. Genuine 2D or 3D
geology, a steeply dipping conductor above all, reconstructs badly too and is
marked as coupled. Where conductive targets are expected, flag rather than
cull, and review. A coupling that mimics a 1D response reconstructs well and is
kept. Oscillatory coupling is ``cull_negatives``' business, not this step's.

The defaults are Asif et al.'s, from a survey with measured stack scatter and a
trained auto-encoder; a modeled uncertainty and a fitted basis move them, so
they are a starting point for calibration, not a calibration.
"""

import logging

import numpy as np

from ..coupling import table
from ..coupling.reconstruct import PCAReconstructor
from ..coupling.reconstruction import FUZZY_FLOOR, detect, positions
from ..distribution import summarize
from ..mask import apply_policy, mark_step
from ..pipeline import StepReport

log = logging.getLogger(__name__)

CATEGORY = "cull gates"

STEP = "cull_by_coupling"

#: Provenance code. Unassigned in ``building_plans/01``; 26, 28 and 34 are the
#: candidates, and 28 sorts between curvature (27) and the noise floor (29).
STEP_CODE = 28

ACTIONS = ("flag", "cull")


def cull_by_coupling(dataset, action: str = "flag", max_reconstruction_error: float = 2.5,
                     min_confidence: float = 0.86,
                     cull_below_confidence: float | None = None,
                     window: float = 15.0, max_uncertainty: float = 0.2,
                     n_components: int = 4, confidence_column: str = "qs_coupling",
                     start_gate: int = 0, spread_earlier: int = 1,
                     disable_tail: bool = True):
    """Mark, and optionally cull, gates the survey's basis cannot reproduce.

    Parameters
    ----------
    dataset : Dataset
        The dataset to judge; its coupling tables and confidence columns are
        written, and with ``action: cull`` its in-use masks.
    action : str
        ``flag`` writes the coupling table and confidence only; ``cull`` also
        applies the table to the mask.
    max_reconstruction_error : float
        A gate reconstructs when its error is below this many standard
        deviations of its assigned uncertainty. Asif et al.'s ``α``, 2.5.
        Lower is stricter.
    min_confidence : float
        A sounding whose smoothed confidence, the fraction of its gates that
        reconstruct, exceeds this keeps every gate. Asif et al.'s ``β_max``,
        0.86 for their airborne survey (0.99 on the ground). Higher is
        stricter. A fraction, 0..1.
    cull_below_confidence : float | None
        Below this smoothed confidence every gate of the sounding is culled;
        between it and ``min_confidence`` only gates within one standard
        deviation survive. ``0.80 × min_confidence`` if unset, as in Asif et al.
        A fraction, at most ``min_confidence``.
    window : float
        The along-line length the data are averaged over before they are
        judged, in the survey's horizontal units (meters for a projected CRS).
        The confidence is averaged over twice this. Asif et al. average 7 raw
        stacks; 15 m is that at a 10 Hz SkyTEM spacing of about 2.2 m. Set by
        the sounding spacing, not a strictness dial: a larger window smears
        the coupling.
    max_uncertainty : float
        From the first gate whose relative uncertainty, after averaging,
        exceeds this, the transient is not judged: that is noise, not coupling.
        Asif et al.'s ``Δ_max``, 20%. A fraction.
    n_components : int
        Components kept in the reconstruction basis. Too many and the basis
        learns the coupling as well as the decay.
    confidence_column : str
        Prefix of the per-sounding confidence column written for each channel,
        suffixed with the channel key (``qs_coupling_ch1``). The smoothed
        fraction of gates that reconstruct, 0..1; ``NaN`` where nothing could
        be judged.
    start_gate : int
        Where the culling policy begins, 0-based.
    spread_earlier : int
        Also cull this many gates before each hit.
    disable_tail : bool
        Also cull everything after the earliest hit in that sounding.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.

    Raises
    ------
    ValueError
        If a parameter is out of range, or a channel has too few soundings to
        fit a basis of ``n_components``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if action not in ACTIONS:
        raise ValueError(f"action must be one of {', '.join(ACTIONS)}; got {action!r}")
    max_reconstruction_error = float(max_reconstruction_error)
    min_confidence = float(min_confidence)
    floor = (FUZZY_FLOOR * min_confidence if cull_below_confidence is None
             else float(cull_below_confidence))
    window, max_uncertainty = float(window), float(max_uncertainty)
    if max_reconstruction_error <= 0:
        raise ValueError(f"max_reconstruction_error is in standard deviations and must "
                         f"be positive; got {max_reconstruction_error:g}")
    if not 0 < min_confidence <= 1:
        raise ValueError(f"min_confidence is a fraction in (0, 1]; got {min_confidence:g}")
    if not 0 <= floor <= min_confidence:
        raise ValueError(f"cull_below_confidence must lie in [0, min_confidence = "
                         f"{min_confidence:g}]; got {floor:g}")
    if window < 0:
        raise ValueError(f"window is a distance and cannot be negative; got {window:g}")
    if max_uncertainty <= 0:
        raise ValueError(f"max_uncertainty is a fraction and must be positive; got "
                         f"{max_uncertainty:g}")

    channels = dataset.channels
    if not channels:
        report.notes.append("No channels; nothing to judge.")
        return dataset, report
    if positions(dataset) is None:
        report.notes.append("No sounding positions; the along-line windows cannot be "
                            "formed, so nothing was judged.")
        return dataset, report

    per_channel, without, culled_any = {}, [], False
    for channel in channels:
        if not channel.std_key or channel.std_key not in dataset.layer_data:
            without.append(str(channel))
            continue

        basis = PCAReconstructor(n_components=n_components)
        direct, confidence, error = detect(
            dataset, channel, basis,
            max_reconstruction_error=max_reconstruction_error,
            min_confidence=min_confidence, cull_below_confidence=floor,
            window=window, max_uncertainty=max_uncertainty)
        widened = apply_policy(direct, start_gate=start_gate,
                               spread_earlier=spread_earlier, disable_tail=disable_tail)
        direct[:, :start_gate] = False
        table.write(dataset, channel, widened, direct)
        frame[f"{confidence_column}_{channel.key}"] = confidence

        outcome = {
            "table": table.channel_key(channel),
            "confidence_column": f"{confidence_column}_{channel.key}",
            "marked": int(widened.sum()),
            "marked_direct": int(direct.sum()),
            "soundings_judged": int(np.isfinite(confidence).sum()),
            "confidence": summarize(confidence[np.isfinite(confidence)]),
            "error": summarize(error[np.isfinite(error)]),
            "basis": {"iterations": basis.iterations_, "converged": basis.converged_},
        }
        if basis.converged_ is False:
            report.notes.append(
                f"The basis for {channel} did not converge in {basis.iterations_} "
                f"rounds of imputation; its reconstruction errors are less certain.")
        if action == "cull" and widened.any():
            culled_any = True
            outcome.update(table.apply(dataset, channel, step_code=STEP_CODE))
        else:
            outcome["removed"] = 0
        per_channel[str(channel)] = outcome

    if without:
        report.notes.append(f"No uncertainty array for {', '.join(without)}; those "
                            f"channels were not judged. Run assign_uncertainty first.")
    if not per_channel:
        return dataset, report
    if culled_any:
        mark_step(dataset, STEP_CODE, STEP)

    report.distributions = {
        "per_channel": per_channel,
        "action": action,
        "marked": sum(c["marked"] for c in per_channel.values()),
        "removed": sum(c["removed"] for c in per_channel.values()),
        "thresholds": {"max_reconstruction_error": max_reconstruction_error,
                       "min_confidence": min_confidence,
                       "cull_below_confidence": floor, "window": window,
                       "max_uncertainty": max_uncertainty,
                       "n_components": n_components},
        "policy": {"start_gate": start_gate, "spread_earlier": spread_earlier,
                   "disable_tail": disable_tail},
        "provenance_code": STEP_CODE,
    }
    log.info("%s (%s): marked %d gate-sample(s), removed %d", STEP, action,
             report.distributions["marked"], report.distributions["removed"])
    return dataset, report
