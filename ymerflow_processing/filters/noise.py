"""The noise model — defined once, held by the dataset, read by everyone.

R9:

    The noise model is defined **once**, as a property of the dataset, by an
    explicit estimation step. Every consumer reads it. No filter carries its own
    copy of the floor or the slope.

The reason is not tidiness. Two filters used to encode the same physical
statement independently and their defaults disagreed — one defaulting the floor
to the system description, the other to ``1e-8``, against documented empirical
values around ``5e-10`` to ``1.5e-9`` for the same instrument. So uncertainty
could be assigned with one floor and data culled with another, and the run would
complete with nothing detecting it. One definition makes that unrepresentable.

Two components, combining in quadrature
---------------------------------------
**An absolute floor.** Ambient environmental noise, independent of signal
strength, decaying with time as a power law. This is what makes late gates fail
*honestly*.

**A relative fraction.** Calibration, geometry, primary-field removal — error
proportional to signal.

``01`` is emphatic about why both are needed: *"A relative-only model never lets
data die — the fractional uncertainty stays flat forever and every gate looks
usable, including gates that are pure noise."*

Since this format stores uncertainty as a **relative** standard deviation, the
floor enters as its own relative contribution — ``floor(t) / |value|`` — which
grows without bound as the signal decays. That is the whole point.

Numerically identical to SimPEG's floor
---------------------------------------
The data on disk is the SimPEG static_instrument format: raw ``dbdt`` per unit
transmitter moment, with ``scalefactor`` (model_info) and ``GateFactor`` (GEX)
applied at read, never baked into the arrays. So the floor here is SimPEG's
``uncert_array`` floor exactly — ``N(t) = (noise_level_1ms / dipole_moment) ·
(t·1000)^exponent`` — with ``noise_level_1ms`` in physical units (V/Am², before
moment normalization), the ``1e3`` ms reference, the ``−0.5`` default exponent,
and the per-channel nominal ``ApproxDipoleMoment`` divided out. This is the one
amplitude conversion that remains, and it is irreducible.

A **supplied** floor is physical, so it is divided by the moment and compared
against the physical data the step materializes transiently —
``dbdt × scalefactor × GateFactor`` — then discarded; the step persists only a
relative fraction / culling mask in the raw-format arrays. An **estimated** floor
is measured from the raw arrays by ``_scatter``, so it is already in their units:
``scalefactor`` and ``GateFactor`` cancel in the ratio ``floor / |value|`` and no
moment division is needed. The ``physical`` flag on the model records which of
the two it is, and both ``scalefactor`` and ``GateFactor`` default to ``1.0`` when
the dataset carries no GEX (a synthetic system).

Getting this wrong is silent and catastrophic: a supplied floor (physical V/Am²)
divided by the raw ``dbdt`` without ``scalefactor × GateFactor`` is off by exactly
that product — ~10¹² with the usual ``scalefactor = 1e-12``. The cross-check test
against SimPEG's ``uncert_array`` is what pins it.

Estimated from the scatter, not from the level
----------------------------------------------
At late gates the measured value is signal *plus* noise, and taking its magnitude
as the noise would count whatever response remains. What separates them is that
**geology varies smoothly along a line and noise does not**: the
sounding-to-sounding difference at a late gate is noise-dominated even where
signal is still present.

So the floor is estimated from the robust scatter of consecutive differences,
divided by ``sqrt(2)`` because differencing two independent samples adds their
variances. Median absolute deviation rather than a standard deviation, for the
usual reason — a few spikes would inflate σ and raise the floor that is supposed
to detect them.

The relative fraction is never estimated
----------------------------------------
R9 is explicit. It is systematic — calibration, geometry, primary-field removal —
and has no separable signature in the data the way the ambient floor does at late
gates. Estimating it would be inventing a number with a measurement's authority.
It comes from the system description's ``UniformDataSTD``, or is supplied.

An override changes what is applied, never what is measured
-----------------------------------------------------------
R9 again, and it shapes the interface: a supplied value is used, **and the
estimate is made anyway and reported beside it**. *"You set 1e-9; the data
suggests 3e-9"* is itself a finding — either the number came from a different
survey, or something is noisier than expected. A fitted slope far from −0.5
usually means the late gates are not ambient noise at all.

That is also why these are optional parameters rather than a mode switch: absent
means estimate, supplied means use, and the common middle case — estimate the
floor, hold the slope — cannot be expressed by an either/or.
"""

import logging
from dataclasses import dataclass, field

import numpy as np

from ..pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "estimate_noise_model"

#: The floor is quoted at this time, so a number is comparable between surveys
#: and instruments. One millisecond is late enough to be ambient for every
#: airborne TEM system in use and round enough to say aloud.
REFERENCE_TIME = 1e-3

#: Ambient noise falls off roughly as the square root of time. A fitted value far
#: from this usually means the gates it was fitted over are not ambient noise.
DEFAULT_EXPONENT = -0.5

#: Differencing two independent samples adds their variances.
DIFFERENCE_TO_SIGMA = np.sqrt(2.0)

#: Scales a median absolute deviation to a standard deviation for normal data.
MAD_TO_SIGMA = 1.4826

PER_LINE, PER_FLIGHT, PER_DATASET = "per_line", "per_flight", "per_dataset"
SCOPES = (PER_LINE, PER_FLIGHT, PER_DATASET)

#: Where the model is recorded, so every consumer reads the same one.
REGISTRY = "noise_model"


def scalefactor(dataset) -> float:
    """The model_info ``scalefactor`` (raw → V/m²), or ``1.0`` when absent.

    Not baked into the arrays; read here only to materialize the physical form a
    supplied floor is compared against. A synthetic dataset carries none, and
    ``1.0`` is then the identity SimPEG assumes.
    """
    value = dataset.survey.xyz.model_info.get("scalefactor", 1.0)
    return float(value) if value else 1.0


def gate_factor(dataset, channel) -> float:
    """The channel's GEX ``GateFactor`` (receiver geometry), or ``1.0`` when absent.

    A ``SyntheticTEMXYZSystem``-style dataset carries no GEX and behaves as
    ``GateFactor = 1`` throughout, so a missing GEX or field is tolerated rather
    than an error — exactly as SimPEG's ``dual.py`` falls back with ``or 1.0``.
    """
    gex = getattr(dataset.survey, "gex", None)
    block = getattr(gex, "gex_dict", {}).get(f"Channel{channel.index}", {}) if gex else {}
    value = block.get("GateFactor", 1.0) if isinstance(block, dict) else 1.0
    return float(value) if value else 1.0


@dataclass
class NoiseModel:
    """The floor, its slope, and the relative fraction — with their provenance.

    Parameters
    ----------
    level_1ms : float | None
        The absolute noise floor at the reference time of 1 ms. ``None`` when
        it has not been estimated or supplied.
    exponent : float
        The power-law slope of the floor in time; ``-0.5`` is typical.
    relative_fraction : float | None
        Systematic error as a fraction of signal.
    physical : bool
        Whether ``level_1ms`` is in physical units (V/Am², before moment
        normalization) — ``True`` for a supplied floor, which is then divided by
        the dipole moment and compared against physical data. ``False`` for an
        estimated floor, measured from the raw arrays and already in their units.
        Not decoration: a physical floor compared against raw data without the
        moment and ``scalefactor × GateFactor`` is wrong by orders of magnitude.
    scope : str
        The grain the floor was estimated over — ``per_line``, ``per_flight``
        or ``per_dataset``.
    supplied : tuple
        Which fields were supplied rather than estimated.
    measured : dict
        What estimation found, even where a supplied value is applied. An
        override changes what is applied, never what is measured (R9).
    """

    level_1ms: float | None = None
    exponent: float = DEFAULT_EXPONENT
    relative_fraction: float | None = None
    physical: bool = False
    scope: str = PER_DATASET
    #: Which values were supplied rather than estimated, and what the estimate
    #: said anyway. R9: an override changes what is applied, never what is
    #: measured.
    supplied: tuple = ()
    measured: dict = field(default_factory=dict)

    def floor(self, times, dipole_moment=None):
        """The absolute floor at each time, SimPEG's ``noise`` term exactly.

        For a **physical** (supplied) floor this is ``(level_1ms / dipole_moment)
        · (t·1e3)^exponent`` — identical to SimPEG's ``uncert_array`` noise, in
        physical V/m². The moment division is the one amplitude conversion that
        remains, and it is silently catastrophic if skipped. For an **estimated**
        floor this is ``level_1ms · (t·1e3)^exponent``, already in the raw array
        units it was measured from, and no moment is needed.

        Parameters
        ----------
        times : array_like
            Gate centre times, in seconds.
        dipole_moment : float | None
            The channel's transmitter moment, needed only for a physical floor.

        Returns
        -------
        numpy.ndarray
            One floor per time.

        Raises
        ------
        ValueError
            If the model has no floor, or the floor is physical and no dipole
            moment is known.
        """
        if self.level_1ms is None:
            raise ValueError(
                "This noise model has no floor. Run estimate_noise_model, or "
                "supply noise_level_1ms."
            )
        times = np.asarray(times, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            level = self.level_1ms * (times / REFERENCE_TIME) ** self.exponent

        if not self.physical:
            return level
        if dipole_moment in (None, 0) or not np.isfinite(dipole_moment):
            raise ValueError(
                "The floor is stated in physical units (V/Am², before moment "
                "normalization) and must be divided by the channel's dipole "
                "moment — but none is known for this channel."
            )
        return level / dipole_moment

    def floor_in_array_units(self, times, dipole_moment=None,
                             scalefactor=1.0, gate_factor=1.0):
        """The floor in the raw on-disk array units ``dbdt`` is stored in.

        A physical floor is divided back by ``scalefactor × gate_factor`` so it
        can be compared against, or divided into, the raw arrays — the transient
        physical form ``floor()`` returns lives only long enough for this. An
        estimated floor is already in those units and passes through unchanged,
        the factors cancelling as the module docstring describes.
        """
        floor = self.floor(times, dipole_moment)
        if self.physical:
            return floor / (scalefactor * gate_factor)
        return floor

    def relative(self, values, times, dipole_moment=None,
                 scalefactor=1.0, gate_factor=1.0):
        """Relative uncertainty per datum: the two components in quadrature.

        The floor's contribution is ``floor / |value|`` with both in the raw
        array units — so as the signal decays toward the floor, the assigned
        uncertainty grows toward one and past it, which is how a gate dies
        honestly. For a physical floor the comparison is equivalently
        ``floor_physical / |data × scalefactor × gate_factor|``, which is what
        makes it numerically identical to SimPEG's.

        Parameters
        ----------
        values : array_like
            The raw data arrays, soundings by gates.
        times : array_like
            Gate centre times, one per gate.
        dipole_moment : float | None
            The channel's transmitter moment, for a physical floor.
        scalefactor, gate_factor : float
            The model_info ``scalefactor`` and the channel's GEX ``GateFactor``,
            both ``1.0`` when absent. Used only for a physical floor.

        Returns
        -------
        numpy.ndarray
            Relative uncertainty, the same shape as ``values``.
        """
        values = np.asarray(values, dtype=float)
        floor = self.floor_in_array_units(times, dipole_moment,
                                          scalefactor, gate_factor)
        relative = 0.0 if self.relative_fraction is None else self.relative_fraction

        with np.errstate(divide="ignore", invalid="ignore"):
            from_floor = np.abs(floor)[None, :] / np.abs(values)
        from_floor = np.where(np.isfinite(from_floor), from_floor, np.inf)
        return np.sqrt(relative ** 2 + from_floor ** 2)

    def to_dict(self) -> dict:
        """The model as plain data, for a report or the dataset's metadata.

        Returns
        -------
        dict
            Every field, with ``supplied`` as a list.
        """
        return {"level_1ms": self.level_1ms, "exponent": self.exponent,
                "relative_fraction": self.relative_fraction,
                "physical": self.physical, "scope": self.scope,
                "supplied": list(self.supplied), "measured": self.measured}


def _encode(model: NoiseModel) -> str:
    parts = [f"level_1ms={model.level_1ms:g}" if model.level_1ms is not None else "",
             f"exponent={model.exponent:g}",
             (f"relative_fraction={model.relative_fraction:g}"
              if model.relative_fraction is not None else ""),
             f"physical={'true' if model.physical else 'false'}",
             f"scope={model.scope}"]
    return " ".join(p for p in parts if p)


def store(dataset, model: NoiseModel) -> None:
    """Record the model on the dataset, so every consumer reads the same one.

    Parameters
    ----------
    dataset : Dataset
        Where the model is recorded, in ``model_info``.
    model : NoiseModel
        The model to record.
    """
    dataset.survey.xyz.model_info[REGISTRY] = _encode(model)


def recorded(dataset) -> NoiseModel | None:
    """The model this dataset carries, or ``None`` if none was estimated.

    Parameters
    ----------
    dataset : Dataset
        Whose ``model_info`` is read.

    Returns
    -------
    NoiseModel | None
        The recorded model, or ``None``.
    """
    raw = dataset.survey.xyz.model_info.get(REGISTRY)
    if not raw:
        return None
    if not isinstance(raw, str):
        raw = " ".join(map(str, raw))

    fields: dict = {}
    for entry in raw.split():
        name, _, value = entry.partition("=")
        if not value:
            continue
        if name == "scope":
            fields[name] = value
        elif name == "physical":
            fields[name] = value == "true"
        else:
            try:
                fields[name] = float(value)
            except ValueError:
                continue
    return NoiseModel(**fields) if fields else None


def _scatter(values, groups) -> np.ndarray:
    """Robust per-gate noise, from consecutive differences within each group.

    Geology varies smoothly along a line and noise does not, so the
    sounding-to-sounding difference is noise-dominated even where signal remains.
    """
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    per_gate = np.full(values.shape[1], np.nan)

    for gate in range(values.shape[1]):
        column = values[:, gate]
        pieces = []
        for key in np.unique(groups):
            rows = groups == key
            series = column[rows]
            if series.size < 3:
                continue
            step = np.diff(series)
            pieces.append(step[np.isfinite(step)])
        if not pieces:
            continue
        differences = np.concatenate(pieces)
        if differences.size < 3:
            continue
        spread = np.median(np.abs(differences - np.median(differences)))
        per_gate[gate] = spread * MAD_TO_SIGMA / DIFFERENCE_TO_SIGMA
    return per_gate


def _fit(times, noise, exponent=None):
    """Fit ``level_1ms`` and, unless supplied, the exponent, in log-log space."""
    times = np.asarray(times, dtype=float)
    noise = np.asarray(noise, dtype=float)
    usable = np.isfinite(times) & np.isfinite(noise) & (times > 0) & (noise > 0)
    if usable.sum() < 2:
        return None, exponent if exponent is not None else DEFAULT_EXPONENT

    x = np.log(times[usable] / REFERENCE_TIME)
    y = np.log(noise[usable])
    if exponent is None:
        slope, intercept = np.polyfit(x, y, 1)
    else:
        slope = float(exponent)
        intercept = float(np.mean(y - slope * x))
    return float(np.exp(intercept)), float(slope)


def estimate_noise_model(dataset, scope: str = PER_DATASET,
                         noise_level_1ms: float | None = None,
                         noise_exponent: float | None = None,
                         relative_fraction: float | None = None,
                         late_gates: int | None = None):
    """Estimate the dataset's noise model, and record it for every consumer.

    Parameters
    ----------
    dataset : Dataset
        The dataset to estimate from; the model is recorded on it and no
        measurement is changed.
    scope : str
        ``per_line``, ``per_flight`` or ``per_dataset`` — the grain the
        estimate is made over. Not "per survey": this library has no concept of
        one, ``Survey`` is already the data object's class, and a campaign may
        map to one dataset, part of one, or several.
    noise_level_1ms : float | None
        The absolute floor at 1 ms in physical units (V/Am², before moment
        normalization) — the instrument's own dB/dt noise, the same number
        SimPEG's ``uncertainties__noise_level_1ms`` takes. Applied to the data it
        is divided by each channel's dipole moment and compared against the
        physical form ``dbdt × scalefactor × GateFactor``. Absent means estimate,
        and an estimated floor is in the raw arrays' own units instead.
    noise_exponent : float | None
        The power-law slope. Absent means fit it; ``-0.5`` is the usual value
        and a fitted number far from it usually means the gates involved are
        not ambient noise.
    relative_fraction : float | None
        Systematic error as a fraction of signal. Absent takes it from the
        system description's ``UniformDataSTD``. **Never estimated** — it has
        no separable signature in the data, and estimating it would invent a
        number with a measurement's authority.
    late_gates : int | None
        How many trailing gates to estimate from. Enough to fit a slope, few
        enough to be noise-dominated; the last third by default. The per-gate
        scatter is reported so the choice is checkable rather than trusted.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.

    Notes
    -----
    A supplied value is used *and* the estimate is made anyway and reported
    beside it.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if scope not in SCOPES:
        raise ValueError(
            f"Unknown scope {scope!r}. Available: {', '.join(SCOPES)}")

    channels = dataset.channels
    if frame is None or len(frame) == 0 or not channels:
        report.notes.append("No soundings or no channels; nothing to estimate.")
        return dataset, report

    if scope == PER_LINE:
        groups = frame[dataset.survey.xyz.line_id_column].to_numpy()
    elif scope == PER_FLIGHT:
        column = dataset.flight_column
        groups = (frame[column].to_numpy() if column
                  else np.zeros(len(frame)))
        if column is None:
            report.notes.append(
                "No flight identifier; the estimate was made over the whole "
                "delivery rather than per flight.")
    else:
        groups = np.zeros(len(frame))

    per_channel, levels, slopes = {}, [], []
    for channel in channels:
        values = dataset.layer_data[channel.data_key].to_numpy(dtype=float)
        times = np.asarray(channel.axis.values, dtype=float)
        if times.size != values.shape[1]:
            continue

        trailing = late_gates or max(3, values.shape[1] // 3)
        first = max(0, values.shape[1] - int(trailing))
        noise = _scatter(values, groups)

        level, slope = _fit(times[first:], noise[first:], noise_exponent)
        per_channel[str(channel)] = {
            "estimated_level_1ms": level,
            "estimated_exponent": slope,
            "gates_used": list(range(first, values.shape[1])),
            "per_gate_noise": [None if not np.isfinite(v) else float(v)
                               for v in noise],
        }
        if level is not None:
            levels.append(level)
            slopes.append(slope)

    measured_level = float(np.median(levels)) if levels else None
    measured_slope = float(np.median(slopes)) if slopes else DEFAULT_EXPONENT

    # The system description's own figure, which R9 says is where the relative
    # fraction comes from and which is never estimated.
    declared = None
    gex = dataset.survey.gex
    for channel in channels:
        block = getattr(gex, "gex_dict", {}).get(f"Channel{channel.index}", {})
        if isinstance(block, dict) and block.get("UniformDataSTD") is not None:
            declared = float(block["UniformDataSTD"])
            break

    supplied = tuple(
        name for name, value in (("level_1ms", noise_level_1ms),
                                 ("exponent", noise_exponent),
                                 ("relative_fraction", relative_fraction))
        if value is not None)

    # A supplied floor is physical (V/Am², pre-moment), as the parameter
    # documents, so it is divided by the dipole moment and compared against the
    # physical data. An estimated one is measured from the raw arrays, so it is
    # already in their units — marking it physical would have `floor()` divide it
    # by the moment a second time, a floor orders of magnitude off with nothing
    # to signal it.
    model = NoiseModel(
        level_1ms=noise_level_1ms if noise_level_1ms is not None else measured_level,
        exponent=(noise_exponent if noise_exponent is not None
                  else measured_slope),
        relative_fraction=(relative_fraction if relative_fraction is not None
                           else declared),
        physical=noise_level_1ms is not None,
        scope=scope,
        supplied=supplied,
        measured={"level_1ms": measured_level, "exponent": measured_slope},
    )
    if model.level_1ms is None:
        report.notes.append(
            "The floor could not be estimated and none was supplied, so no "
            "model was recorded. There may be too few soundings per group to "
            "measure a scatter from.")
        return dataset, report

    store(dataset, model)

    for name, applied, seen in (("level_1ms", model.level_1ms, measured_level),
                                ("exponent", model.exponent, measured_slope)):
        if name in supplied and seen is not None:
            report.notes.append(
                f"{name} was supplied as {applied:g}; the data suggests "
                f"{seen:g}. Either the number came from a different survey, or "
                f"something is not as expected."
            )
    if relative_fraction is None and declared is None:
        report.notes.append(
            "No relative fraction was supplied and the system description "
            "declares no UniformDataSTD, so the model carries a floor only.")

    report.distributions = {
        "model": model.to_dict(),
        "per_channel": per_channel,
        "relative_fraction_source": (
            "supplied" if relative_fraction is not None
            else "system description" if declared is not None else None),
    }
    log.info("%s: floor %s at 1 ms, exponent %.3g, relative %s (scope %s)",
             STEP, "n/a" if model.level_1ms is None else f"{model.level_1ms:.3g}",
             model.exponent, model.relative_fraction, scope)
    return dataset, report
