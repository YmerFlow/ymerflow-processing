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
uncertainty, so there is nothing to divide by before it. Where that
uncertainty is stack scatter alone, tighter than any basis reproduces a
transient, ``relative_floor`` loosens the test without touching what the
inversion weights by. **Before**
``average_along_line``: averaging smears a coupling signature into its
neighbors, and before averaging is the data the method was built for.

Where it does not suit
----------------------
The criterion tests **one-dimensionality, not cleanliness**. Genuine 2D or 3D
geology, a steeply dipping conductor above all, reconstructs badly too and is
marked as coupled. Where conductive targets are expected, flag rather than
cull, and review. A coupling that mimics a 1D response reconstructs well and is
kept. Oscillatory coupling is ``cull_negatives``' business, not this step's.

The defaults were calibrated on one VTEM survey edited by hand on every line,
against a modeled uncertainty and the fitted PCA basis (the plan's "What the
calibration found"): eight components and ``β_max`` 0.99 catch about 40% of
the processor's culls at 50% precision on lines held out, where Asif et al.'s
airborne values (four components, 0.86) caught 30% at 32%. Other vendors and
systems are uncalibrated, and the step stays a reviewer's aid run as ``flag``
until a survey's own numbers say otherwise.
"""

import logging

import numpy as np

from ..coupling import table
from ..coupling.reconstruct import PCAReconstructor
from ..coupling.reconstruction import FUZZY_FLOOR, detect, positions
from ..distribution import summarize
from ..mask import apply_policy, check_policy, mark_step
from ..pipeline import StepReport

log = logging.getLogger(__name__)

CATEGORY = "cull gates"

STEP = "cull_by_coupling"

#: Provenance code. Unassigned in ``building_plans/01``; 26, 28 and 34 are the
#: candidates, and 28 sorts between curvature (27) and the noise floor (29).
STEP_CODE = 28

ACTIONS = ("flag", "cull")


def _applied(own: dict) -> dict:
    """One channel's settings as applied: floats, the floor filled in, refused
    when out of range, before any table or mask is written."""
    alpha = float(own["max_reconstruction_error"])
    beta = float(own["min_confidence"])
    floor = (FUZZY_FLOOR * beta if own["cull_below_confidence"] is None
             else float(own["cull_below_confidence"]))
    window, delta = float(own["window"]), float(own["max_uncertainty"])
    relative_floor = float(own["relative_floor"])
    if alpha <= 0:
        raise ValueError(f"max_reconstruction_error is in standard deviations and must "
                         f"be positive; got {alpha:g}")
    if not 0 < beta <= 1:
        raise ValueError(f"min_confidence is a fraction in (0, 1]; got {beta:g}")
    if not 0 <= floor <= beta:
        raise ValueError(f"cull_below_confidence must lie in [0, min_confidence = "
                         f"{beta:g}]; got {floor:g}")
    if window < 0:
        raise ValueError(f"window is a distance and cannot be negative; got {window:g}")
    if delta <= 0:
        raise ValueError(f"max_uncertainty is a fraction and must be positive; got "
                         f"{delta:g}")
    if relative_floor < 0:
        raise ValueError(f"relative_floor is a fraction of the value and cannot be "
                         f"negative; got {relative_floor:g}")
    check_policy(own)
    return {**own, "max_reconstruction_error": alpha, "relative_floor": relative_floor,
            "min_confidence": beta,
            "cull_below_confidence": floor, "window": window, "max_uncertainty": delta,
            "n_components": int(own["n_components"])}


def cull_by_coupling(dataset, action: str = "flag",
                     max_reconstruction_error: float | dict[str, float] = 2.5,
                     relative_floor: float | dict[str, float] = 0.0,
                     min_confidence: float | dict[str, float] = 0.99,
                     cull_below_confidence: float | dict[str, float] | None = None,
                     window: float | dict[str, float] = 15.0,
                     max_uncertainty: float | dict[str, float] = 0.2,
                     n_components: int | dict[str, int] = 8,
                     confidence_column: str = "qs_coupling",
                     start_gate: int | dict[str, int] = 0,
                     spread_earlier: int | dict[str, int] = 1,
                     disable_tail: bool | dict[str, bool] = True):
    """Mark, and optionally cull, gates the survey's basis cannot reproduce.

    The basis is fitted per channel and each channel is judged against its
    own, so every threshold and policy parameter is per channel too: one value
    for every channel, or a mapping by channel key or moment name
    (``{"LM": 0.86, "HM": 0.99}``) naming every channel. ``action`` and
    ``confidence_column`` describe the step and stay one value.

    Parameters
    ----------
    dataset : Dataset
        The dataset to judge; its coupling tables and confidence columns are
        written, and with ``action: cull`` its in-use masks.
    action : str
        ``flag`` writes the coupling table and confidence only; ``cull`` also
        applies the table to the mask.
    max_reconstruction_error : float | dict
        A gate reconstructs when its error is below this many standard
        deviations of its assigned uncertainty. Asif et al.'s ``α``, 2.5.
        Lower is stricter. One value, or a mapping by channel key or moment
        name.
    relative_floor : float | dict
        A proportional error the test allows the basis, as a fraction of the
        value: the standard deviation ``max_reconstruction_error`` counts in
        is the assigned uncertainty and this in quadrature,
        ``sqrt(assigned² + (relative_floor × value)²)``. For this test only:
        the dataset's uncertainties are not changed, and ``max_uncertainty``
        still reads the assigned one. Needed where the assigned uncertainty is
        stack scatter alone, a fraction of a percent on early gates, which no
        basis reproduces a transient to; the defaults were calibrated against
        a modeled uncertainty, a noise floor and a proportional error of a
        few percent. 0, the default, counts in the assigned uncertainty as it
        is. One value, or a mapping by channel key or moment name
        (``{"LM": 0.0, "HM": 0.02}``).
    min_confidence : float | dict
        A sounding whose smoothed confidence, the fraction of its gates that
        reconstruct, exceeds this keeps every gate. Asif et al.'s ``β_max``;
        0.99 calibrated on a hand-edited VTEM survey (their airborne value was
        0.86, their ground value 0.99). Higher is stricter. A fraction, 0..1.
        One value, or a mapping by channel key or moment name.
    cull_below_confidence : float | dict | None
        Below this smoothed confidence every gate of the sounding is culled;
        between it and ``min_confidence`` only gates within one standard
        deviation survive. ``0.80 × min_confidence`` if unset, as in Asif et al.
        A fraction, at most ``min_confidence``. One value, or a mapping by
        channel key or moment name.
    window : float | dict
        The along-line length the data are averaged over before they are
        judged, in the survey's horizontal units (meters for a projected CRS).
        The confidence is averaged over twice this. Asif et al. average 7 raw
        stacks; 15 m is that at a 10 Hz SkyTEM spacing of about 2.2 m. Set by
        the sounding spacing, not a strictness dial: a larger window smears
        the coupling. One value, or a mapping by channel key or moment name.
    max_uncertainty : float | dict
        From the first gate whose relative uncertainty, after averaging,
        exceeds this, the transient is not judged: that is noise, not coupling.
        Asif et al.'s ``Δ_max``, 20%. A fraction. One value, or a mapping by
        channel key or moment name.
    n_components : int | dict
        Components kept in the reconstruction basis. A bigger basis trades
        recall for precision; eight was the best of 3-8 on a hand-edited VTEM
        survey. Too many and the basis learns the coupling as well as the
        decay. One value, or a mapping by channel key or moment name.
    confidence_column : str
        Prefix of the per-sounding confidence column written for each channel,
        suffixed with the channel key (``qs_coupling_ch1``). The smoothed
        fraction of gates that reconstruct, 0..1; ``NaN`` where nothing could
        be judged.
    start_gate : int | dict
        Where the culling policy begins, 0-based, counted in each channel's
        own gates. One value, or a mapping by channel key or moment name.
    spread_earlier : int | dict
        Also cull this many gates before each hit. One value, or a mapping by
        channel key or moment name.
    disable_tail : bool | dict
        Also cull everything after the earliest hit in that sounding. One
        value, or a mapping by channel key or moment name.

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

    channels = dataset.channels
    if not channels:
        report.notes.append("No channels; nothing to judge.")
        return dataset, report
    if positions(dataset) is None:
        report.notes.append("No sounding positions; the along-line windows cannot be "
                            "formed, so nothing was judged.")
        return dataset, report

    settings = {key: _applied(own) for key, own in dataset.settings_per_channel(
        max_reconstruction_error=max_reconstruction_error,
        relative_floor=relative_floor, min_confidence=min_confidence,
        cull_below_confidence=cull_below_confidence, window=window,
        max_uncertainty=max_uncertainty, n_components=n_components,
        start_gate=start_gate, spread_earlier=spread_earlier,
        disable_tail=disable_tail).items()}

    per_channel, without, culled_any = {}, [], False
    for channel in channels:
        if not channel.std_key or channel.std_key not in dataset.layer_data:
            without.append(str(channel))
            continue

        own = settings[channel.key]
        basis = PCAReconstructor(n_components=own["n_components"])
        direct, confidence, error = detect(
            dataset, channel, basis,
            max_reconstruction_error=own["max_reconstruction_error"],
            relative_floor=own["relative_floor"],
            min_confidence=own["min_confidence"],
            cull_below_confidence=own["cull_below_confidence"],
            window=own["window"], max_uncertainty=own["max_uncertainty"])
        widened = apply_policy(direct, start_gate=own["start_gate"],
                               spread_earlier=own["spread_earlier"],
                               disable_tail=own["disable_tail"])
        direct[:, :own["start_gate"]] = False
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
            "settings": own,
        }
        if basis.gates_ is None:
            # No gate to evaluate, so detect() fitted nothing: an empty table,
            # no confidence, and the reason said rather than a failure.
            outcome["basis"] = None
            report.notes.append(
                f"{channel} had no gate to judge: every sounding is already culled "
                f"or empty, or its uncertainty is past max_uncertainty from the "
                f"first gate. Nothing was marked.")
        else:
            left_out = int((~basis.gates_).sum())
            outcome["basis"] = {"iterations": basis.iterations_,
                                "converged": basis.converged_,
                                "gates_left_out": left_out}
            if basis.converged_ is False:
                report.notes.append(
                    f"The basis for {channel} did not converge in {basis.iterations_} "
                    f"rounds of imputation; its reconstruction errors are less certain.")
            if left_out:
                report.notes.append(
                    f"{left_out} gate(s) of {channel} were available in too few "
                    f"soundings to fit into the basis and were not judged.")
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
        # As given, scalar or mapping; each channel's outcome has what it got.
        "thresholds": {"max_reconstruction_error": max_reconstruction_error,
                       "relative_floor": relative_floor,
                       "min_confidence": min_confidence,
                       "cull_below_confidence": cull_below_confidence, "window": window,
                       "max_uncertainty": max_uncertainty,
                       "n_components": n_components},
        "policy": {"start_gate": start_gate, "spread_earlier": spread_earlier,
                   "disable_tail": disable_tail},
        "provenance_code": STEP_CODE,
    }
    log.info("%s (%s): marked %d gate-sample(s), removed %d", STEP, action,
             report.distributions["marked"], report.distributions["removed"])
    return dataset, report
