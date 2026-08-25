"""Time-domain EM: how its arrays are named, and what their axis means.

The whole of TEM's contribution to channel discovery. ``core`` finds *which*
channels exist and *which arrays belong to each*; this says what they are called
and that the second axis is gate time. Everything else is generic.

A moment is TEM's name for a channel — one transmitter waveform, with its own
current, its own gate series and its own gate count. A single-moment system
(HeliTEM, VTEM, Xcite) yields one; SkyTEM 304 yields two; a three-moment system
yields three. None of that is special-cased anywhere: the count comes from the
data (R1).

Gate times
----------
Taken from the system description, which lists each gate as three numbers —
center, open, close. The **center** is what the axis carries, because it is the
time a gate's measurement is attributed to when a transient is modeled or its
decay differentiated.

``RemoveInitialGates`` is deliberately *not* applied here. It is the
instrument's statement that its earliest gates are contaminated by transmitter
turn-off, which is a statement about which data to disable — filter_20's job,
writing the mask (R6) — not about what time a gate sits at. Those early gates
have genuinely negative center times relative to turn-off, and that is exactly
why the system says to drop them. Removing them from the axis instead would
misalign the axis against arrays that still carry every gate.

Source for the gate-center convention: verified directly against SkyTEM 304M and
306HP system descriptions, where each ``GateTime`` row's first value is the
arithmetic center of its second and third (open and close) to within rounding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ...core.dataset import Axis, Channel, ChannelNaming, discover_by_naming

log = logging.getLogger(__name__)


#: Matches a moment named rather than numbered — ``LM_Z_G01``, ``HM_X_G01``.
#: Capturing the moment and the component separately is the point: the pair is
#: the identity, and ``G01`` is not a channel index (it is the same on every
#: array in the file).
#: **Anchored at both ends.** Unanchored it also matches ``LM_Z_G01_inuse`` —
#: the mask this library writes for such a channel — so the mask would be
#: discovered as a second *data* array for the same channel. Exactly the trap the
#: mask's provenance prefix was moved to avoid, met from the other direction.
NAMED_MOMENT = r"^([a-z]+)_([xyz])_g\d+$"

#: The mask for a named moment, as ``mask._inuse_key_for`` writes it: the data
#: key with the role appended. Matched so the array this library created is
#: claimed by the channel it belongs to rather than reported as unclaimed.
#:
#: There is deliberately **no std pattern** for this family. Nothing writes one:
#: a delivery in this convention ships no uncertainties, and ``assign_uncertainty``
#: skips a channel that has none rather than inventing an array. Adding a pattern
#: for a name nothing produces is how the "two families" claim above this came to
#: be wrong.
NAMED_MOMENT_INUSE = r"^([a-z]+)_([xyz])_g\d+_inuse$"


def _by_moment_and_component(groups, system) -> int | None:
    """Which channel the system description says is this moment and component.

    Asked rather than assumed, and this is the whole reason the resolver exists.
    Nothing in an array called ``HM_Z_G01`` says whether the high moment is
    channel 1 or channel 2 — only the system description knows, by declaring
    ``TransmitterMoment`` and ``ReceiverPolarizationXYZ`` per channel.

    **A component the description does not declare resolves to nothing**, and
    that is correct rather than a gap. SkyTEM's data files carry a horizontal
    component, but the system description and the alias file are built for the
    Z component only. An
    array nothing claims is exactly what ``select_channels.unclaimed_arrays``
    exists to report — and it must not reach an inversion, since
    ``DualMomentTEMXYZSystem`` hardcodes ``rx_orientation = 'z'`` and never reads
    the declared polarization, so a three-component delivery presented as
    moments is read as a confident wrong model rather than refused.

    Without a system description there is no answer, and **order is not an
    answer** — sorted by name a high moment precedes a low one, so guessing
    would put the moments the wrong way round with nothing to give it away.
    """
    if system is None:
        log.warning(
            "arrays are named by moment (%s) but there is no system description "
            "to say which channel that is; they cannot be claimed. Order is not "
            "a fallback here — sorted by name a high moment precedes a low one.",
            "_".join(groups))
        return None

    moment, component = (str(g).strip().lower() for g in groups[:2])
    for index in range(1, MAX_CHANNELS + 1):
        if not getattr(system, "gex_dict", {}).get(f"Channel{index}"):
            break
        declared = _system_value(system, index, "transmitter_moment")
        polarization = _component(system, index)
        if (declared is not None
                and str(declared).strip().lower() == moment
                and polarization == component):
            return index
    return None


#: Where to stop looking for ``Channel{n}`` blocks. A guard against an unbounded
#: loop on a malformed description, not a claim about how many a system may have.
MAX_CHANNELS = 32

#: How time-domain arrays are named in ``layer_data``. **Three** families are in
#: circulation and all three appear in data we are handed:
#:
#: * the AarhusInv-style ``dbdt_ch1gt`` names,
#: * the SkyTEM-style ``Gate_Ch01`` names,
#: * and the delivered-product style ``LM_Z_G01``, which identifies a channel by
#:   moment and component rather than by number.
#:
#: The first two capture the index directly. The third captures a name, and
#: :func:`_by_moment_and_component` asks the system description what it means.
#:
TEM_NAMING = ChannelNaming(
    data=(r"^dbdt_ch(\d+)", r"^gate_ch(\d+)", NAMED_MOMENT),
    std=(r"^dbdt_std_ch(\d+)", r"^std_ch(\d+)"),
    inuse=(r"^dbdt_inuse_ch(\d+)", r"^inuse_ch(\d+)", NAMED_MOMENT_INUSE),
    resolve=_by_moment_and_component,
)


@dataclass(frozen=True)
class Moment(Channel):
    """A transmitter moment: a channel whose axis is gate time.

    Adds the one property a channel does not have and a transient needs — the
    transmitter's dipole moment, which sets the amplitude the earth was driven
    with and so the scale of the response. Derived as turns x area x nominal
    current; see :func:`_dipole_moment`.
    """

    dipole_moment: float | None = None    # A m^2, = N * I * A

    @property
    def gate_times(self):
        """Gate center times in seconds, one per gate."""
        return self.axis.values

    @property
    def n_gates(self) -> int | None:
        return self.axis.size


def _gate_axis(system, index: int) -> Axis:
    """The gate-time axis for one channel, read from the system description.

    Two gate-table layouts are in circulation, and which one a file uses depends
    on the receiver the system was flown with:

    ==================  ==========  ================================
    system              receiver    gate tables in the general block
    ==================  ==========  ================================
    304M, 304HR 75 Hz   classic     one shared ``GateTime``
    306HP, 312HP        DigiTEM     ``GateTimeLM`` *and* ``GateTimeHM``
    ==================  ==========  ================================

    The DigiTEM receiver digitises each moment independently, so its moments
    carry genuinely separate gate series rather than two windows onto one. Both
    layouts are resolved by ``libaarhusxyz.GEX.gate_times``, which prefers the
    moment-specific table and falls back to the shared one — so this reads
    through that rather than reaching into the dictionary and having to know.

    Degrades to an empty axis rather than raising. Channel discovery runs
    everywhere — including in ``__repr__`` — and data without a system
    description is a legitimate state, not an error. A system description that
    is *present* but disagrees with itself is worth knowing about, so that is
    logged rather than passed over.
    """
    if system is None:
        return Axis(name="gate time", unit="s")
    try:
        gates = np.asarray(system.gate_times(index), dtype=float)
    except (KeyError, IndexError, AssertionError, AttributeError) as exc:
        log.warning(
            "no gate times for channel %d in the system description (%s); "
            "its axis will be empty", index, exc,
        )
        return Axis(name="gate time", unit="s")

    # Three columns: center, open, close. Take the center.
    centers = gates[:, 0] if gates.ndim == 2 and gates.shape[1] >= 1 else gates.ravel()
    return Axis(values=centers, name="gate time", unit="s")


def _system_value(system, index: int, getter: str):
    """One per-channel value from the system description, or ``None``."""
    if system is None:
        return None
    try:
        return getattr(system, getter)(index)
    except (KeyError, IndexError, AssertionError, AttributeError):
        return None


def _component(system, index: int) -> str | None:
    """Which field component this channel measures, lowercased, or ``None``.

    Read from ``ReceiverPolarizationXYZ`` in the channel block. A single-component
    system declares ``Z`` on every channel; a three-component receiver declares
    one channel per component, so a three-component dual-moment instrument
    appears as six channels rather than two channels holding three arrays each.

    This is what a later step selects on when an inversion accepts only one
    component — SimPEG will not take X and Y alongside Z, so they have to be
    removed from the dataset, not merely ignored during discovery.
    """
    # Read through the system description's own accessor rather than reaching
    # into its dictionary — unlike ApproxDipoleMoment, this one has one.
    value = _system_value(system, index, "rx_orientation")
    if value is None:
        return None
    # Arrives as a bare string, occasionally wrapped by the parser.
    text = str(value[0] if isinstance(value, (list, tuple)) and value else value)
    return text.strip().lower() or None


def _dipole_moment(system, index: int) -> float | None:
    """The transmitter's magnetic dipole moment for one channel, in A m².

    Not stated in the system description — *derived* from it, as the product of
    the three quantities that define a current loop's magnetic moment:

        m = N · I · A

    for N turns of loop area A carrying current I. This is the definition of the
    magnetic dipole moment of a planar current loop (see e.g. Griffiths,
    *Introduction to Electrodynamics*, or Jackson, *Classical Electrodynamics*,
    ch. 5), and it is what makes the moments of a dual-moment system
    comparable — a SkyTEM 304M's low moment is ~3.1e3 A m² against a high
    moment of ~1.5e5, and the response scales with it.

    ``libaarhusxyz`` performs this product at parse time and stores the result
    as ``ApproxDipoleMoment``, drawing turns from ``NumberOfTurns{moment}`` in
    the general block — falling back to a plain ``NumberOfTurns`` for a
    single-moment system — so it is read from there rather than recomputed. It
    is approximate in the sense that ``TxApproximateCurrent`` is the *nominal*
    current; the current actually flown is a per-sounding attribute, and where
    that matters it is what should be used.
    """
    if system is None:
        return None
    block = getattr(system, "gex_dict", {}).get(f"Channel{index}", {})
    value = block.get("ApproxDipoleMoment")
    return None if value is None else float(value)


class TEMMethod:
    """Discovers time-domain moments. Registered as ``ymerflow.processing.methods``."""

    name = "tem"

    def channels(self, dataset) -> list[Moment]:
        """Every moment present, ordered by channel index.

        Returns ``[]`` when nothing in ``layer_data`` is named like time-domain
        data, which is how this adapter says "not my data".
        """
        layer_data = dataset.layer_data
        if not layer_data:
            return []

        system = dataset.system
        found = discover_by_naming(layer_data.keys(), TEM_NAMING, system)

        moments = []
        for index, keys in found.items():
            axis = _gate_axis(system, index)
            name = _system_value(system, index, "transmitter_moment")
            moments.append(Moment(
                key=f"ch{index}",
                index=index,
                # A blank TransmitterMoment is as good as absent; the system
                # description is allowed to leave it empty and several do.
                name=name or None,
                data_key=keys["data_key"],
                std_key=keys.get("std_key"),
                inuse_key=keys.get("inuse_key"),
                axis=axis,
                component=_component(system, index),
                dipole_moment=_dipole_moment(system, index),
            ))
            _check_gate_count(moments[-1], layer_data)
        return moments


def _check_gate_count(moment: Moment, layer_data) -> None:
    """Warn when the system description and the data disagree on gate count.

    They should match exactly. When they do not, something upstream has trimmed
    one and not the other, and every gate-indexed operation after this point is
    working on a misaligned axis. Silent misalignment is the expensive failure
    here, so it is worth a line in the log even though it is not fatal.
    """
    if moment.n_gates is None:
        return
    array = layer_data.get(moment.data_key)
    if array is None or getattr(array, "ndim", 0) != 2:
        return
    if array.shape[1] != moment.n_gates:
        log.warning(
            "moment %s: system description gives %d gates but %r has %d columns",
            moment, moment.n_gates, moment.data_key, array.shape[1],
        )
