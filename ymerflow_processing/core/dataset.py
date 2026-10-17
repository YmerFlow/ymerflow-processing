"""The data object every step operates on.

A thin layer over ``libaarhusxyz.Survey`` (MIT) that adds the things the pipeline
needs and nothing else. It deliberately knows nothing about pipelines — the runner
composes steps, the data does not run itself.

Two tables underneath, and the distinction matters throughout:

``flightlines``
    One row per sounding. Position, altitude, attitude, line identity — and QC
    verdicts, which are written here as ordinary columns (R6).

``layer_data``
    One array per quantity, shaped (soundings x gates). The measurements, their
    uncertainties, and the in-use mask. Culling writes here (R6).

Channels, not moments
---------------------
Every airborne method this library will grow into has the same shape: a table of
per-sounding attributes, plus zero or more **named families of parallel arrays —
data, uncertainty, in-use — sharing one second axis.** What differs between
methods is only what that axis *means*, and how the arrays are named:

======  =========================  =====================
method  array families             second axis
======  =========================  =====================
TEM     moments (LM, HM, ...)      gate time, seconds
FEM     frequencies                frequency, Hz
mag     sensors (often just one)   none — scalar
rad     spectra, windows           energy, keV
grav    tensor components          component
======  =========================  =====================

So this module owns the *vocabulary* (:class:`Channel`, :class:`Axis`) and the
*mechanism* (pattern-driven discovery over ``layer_data`` keys). It does not own
the patterns or the meaning — those belong to the method packages, which
register a :class:`MethodAdapter` under the ``ymerflow.processing.methods``
entry-point group. Adding gravity later is a new package and one adapter, with
nothing here to change.

This is what keeps ``core/`` genuinely method-agnostic instead of holding TEM
naming under a general-sounding label. It is also what ``building_plans/01``
already requires without naming it, in the filter_10 decision: the along-line
averaging engine is generic and belongs here, while its gate-time-varying window
widths belong in ``em/tem/``. A generic engine can only taper along the array
axis if it can read that axis without knowing it is time.
"""

import copy
import json
import importlib.metadata
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import libaarhusxyz
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Method packages register a channel-discovery adapter here. The fourth group,
#: alongside the three runner groups in ``pipeline.py``.
METHOD_GROUP = "ymerflow.processing.methods"


# ── Normalization conventions ────────────────────────────────────────────────
# Data arrives in one of these and the difference is orders of magnitude. The
# dataset carries which one it is in so a step can check rather than trust
# (see filter_08 in building_plans/01).

RAW = "raw"                       # instrument units, receiver response not removed
RX_NORMALIZED = "rx"              # V/m^2  — divided by receiver area x turns
RX_TX_NORMALIZED = "rx_tx"        # V/(A m^4) — additionally by transmitter moment

NORMALIZATIONS = (RAW, RX_NORMALIZED, RX_TX_NORMALIZED)


# ── The second axis of an array family ───────────────────────────────────────

@dataclass(frozen=True)
class Axis:
    """What the columns of a channel's arrays are indexed by.

    Carried by ``core`` and interpreted by nobody here. A method package fills
    it in — gate center times for TEM, frequencies for FEM — and a generic
    step may read ``values`` to weight or taper along it without ever learning
    what the quantity is.

    Parameters
    ----------
    values : array_like | None
        The coordinate of each column. ``None`` for a method whose channels
        have no meaningful second axis, such as a scalar magnetometer.
    name : str | None
        What the quantity is — ``"gate time"``, ``"frequency"``.
    unit : str | None
        Its unit — ``"s"``, ``"Hz"``.
    """

    values: Any = None
    name: str | None = None          # "gate time", "frequency", "energy"
    unit: str | None = None          # "s", "Hz", "keV"

    @property
    def size(self) -> int | None:
        return None if self.values is None else len(self.values)


# ── One family of parallel arrays ────────────────────────────────────────────

@dataclass(frozen=True)
class Channel:
    """One named family of parallel arrays in ``layer_data``.

    Identity is deliberately two-part, because neither half is sufficient alone:

    ``key``
        The stable addressing token — ``"ch1"``, ``"ch2"`` — derived from the
        array names themselves, so it always exists.
    ``name``
        The readable alias the instrument gives it, such as ``"LM"``. Nicer to
        write in a pipeline document, but optional: the system description may
        leave it blank, and ``libaarhusxyz`` explicitly tolerates that.

    :meth:`Dataset.resolve_per_channel` accepts either, so a pipeline can say
    ``{LM: 15, HM: 30}`` and still work on data whose moments are unnamed.

    Parameters
    ----------
    key : str
        The stable addressing token, ``"ch1"``.
    index : int
        The channel's number, 1-based, as the instrument numbers them.
    data_key : str
        The ``layer_data`` array holding the values.
    name : str | None
        The instrument's alias for it, such as ``"LM"``.
    std_key : str | None
        The uncertainties array, where present.
    inuse_key : str | None
        The in-use mask, where present.
    axis : Axis
        What the arrays' columns are indexed by.
    component : str | None
        Which field component this channel measures — ``"x"``, ``"y"`` or
        ``"z"`` — where the system description says.
    """

    key: str
    index: int                       # 1-based, as the instrument numbers them
    data_key: str                    # the array in layer_data holding values
    name: str | None = None
    std_key: str | None = None       # uncertainties, where present
    inuse_key: str | None = None     # the in-use mask, where present
    axis: Axis = field(default_factory=Axis)

    #: Which field component this channel measures — "x", "y", "z" — where the
    #: system description says. A multi-component receiver declares one channel
    #: per component, so component is an attribute *of* a channel rather than a
    #: second axis inside one. That is what keeps R1's "components are a
    #: different axis from moments" true without nesting anything: a
    #: three-component dual-moment system is simply six channels.
    component: str | None = None

    @property
    def array_keys(self) -> list[str]:
        """Every ``layer_data`` array belonging to this channel."""
        return [k for k in (self.data_key, self.std_key, self.inuse_key)
                if k is not None]

    def __str__(self) -> str:
        parts = [p for p in (self.name, self.component) if p]
        return f"{self.key} ({'/'.join(parts)})" if parts else self.key


# ── How a method names its arrays ────────────────────────────────────────────

@dataclass(frozen=True)
class ChannelNaming:
    """Regular expressions matching a method's ``layer_data`` array names.

    Each pattern must capture exactly one group: the channel index. Patterns are
    matched case-insensitively and anchored at the start of the key.

    Matching on a captured index rather than on a substring is deliberate.
    Substring matching — which is what ``libaarhusxyz``'s own helpers do — makes
    ``dbdt_ch1`` match ``dbdt_ch10``, so a ten-channel instrument would silently
    fold two channels together. Nothing has ten channels today; the failure
    would be invisible when something does.

    Parameters
    ----------
    data : tuple[str, ...]
        Patterns for the data arrays.
    std : tuple[str, ...]
        Patterns for the uncertainty arrays.
    inuse : tuple[str, ...]
        Patterns for the in-use masks.
    resolve : callable | None
        Turns what a pattern captured, when it is not a digit string, into a
        channel index: ``resolve(groups, context) -> int | None``.

    Notes
    -----
    **Not every convention numbers its channels.** Some name them. A real
    SkyTEM delivery ships ``LM_Z_G01`` and ``HM_Z_G01``, identifying a channel
    by transmitter moment and receiver component with no number anywhere —
    ``G01`` is the same on every array and is not a channel index.

    So a pattern may capture something other than a digit string, and
    :attr:`resolve` turns what it captured into an index. A pattern capturing a
    single all-digit group still needs no resolver: that is the index.

    The split stays where the module docstrings put it. ``core`` owns the
    mechanism — find which channels exist, and which arrays belong to each. The
    method adapter owns the meaning, and *which channel is called LM* is
    meaning: only the system description knows, and this package may not read
    one.
    """

    data: tuple[str, ...]
    std: tuple[str, ...] = ()
    inuse: tuple[str, ...] = ()

    #: ``(captured groups, context) -> index or None``, for patterns that
    #: capture a name rather than a number. ``None`` from the resolver means
    #: *this array belongs to no channel* — which is a real answer, not a
    #: failure: a vendor may ship a component its own system description does
    #: not declare, and an array nothing claims is reported by
    #: ``select_channels.unclaimed_arrays`` rather than guessed at.
    resolve: Callable[[tuple[str, ...], Any], int | None] | None = None


def _match_index(key: str, patterns: tuple[str, ...],
                 naming: "ChannelNaming | None" = None,
                 context=None) -> int | None:
    """The channel index ``key`` encodes, or ``None`` if it encodes none.

    A single all-digit group is the index itself. Anything else is handed to
    ``naming.resolve``, which is how a convention that names its channels rather
    than numbering them gets an index without ``core`` learning what a moment is.
    """
    for pattern in patterns:
        match = re.match(pattern, key, flags=re.IGNORECASE)
        if not match:
            continue
        groups = match.groups()
        if len(groups) == 1 and groups[0] and groups[0].isdigit():
            return int(groups[0])
        resolver = getattr(naming, "resolve", None)
        if resolver is None:
            log.warning(
                "%r matched a pattern capturing %r, which is not an index, and "
                "its naming supplies no resolver; the array is unclaimed",
                key, groups)
            return None
        return resolver(groups, context)
    return None


def discover_by_naming(layer_data_keys, naming: ChannelNaming,
                       context=None) -> dict[int, dict]:
    """Group ``layer_data`` keys by the channel index they encode.

    The mechanism half of channel discovery: it finds *which* channels exist and
    *which arrays belong to each*, and knows nothing about what they measure. A
    method adapter supplies the naming and adds the meaning.

    Parameters
    ----------
    layer_data_keys : iterable of str
        The ``layer_data`` keys to group.
    naming : ChannelNaming
        The method's patterns, and its resolver for a named convention.
    context : object | None
        Passed through to ``naming.resolve`` untouched. For TEM it is the
        system description, which is the only thing that knows which channel a
        moment name refers to.

    Returns
    -------
    dict[int, dict]
        ``{index: {"data_key": ..., "std_key": ..., "inuse_key": ...}}``. A
        channel with no data array is not a channel — uncertainties or a mask
        without values is a broken import, not a measurement — so those are
        dropped with a warning rather than returned half-formed.
    """
    found: dict[int, dict] = {}

    # Order matters only defensively: the roles have distinct anchored prefixes,
    # but checking the more specific ones first means a sloppier pattern added
    # later cannot quietly capture another role's keys.
    for role, patterns in (("inuse_key", naming.inuse),
                           ("std_key", naming.std),
                           ("data_key", naming.data)):
        for key in layer_data_keys:
            index = _match_index(key, patterns, naming, context)
            if index is None:
                continue
            slot = found.setdefault(index, {})
            if role in slot:
                log.warning(
                    "channel %d: both %r and %r look like its %s; keeping %r",
                    index, slot[role], key, role.replace("_key", ""), slot[role],
                )
                continue
            slot[role] = key

    complete = {}
    for index in sorted(found):
        if "data_key" not in found[index]:
            log.warning(
                "channel %d has %s but no data array; ignoring it",
                index, " and ".join(sorted(found[index])),
            )
            continue
        complete[index] = found[index]
    return complete


# ── What a method package provides ───────────────────────────────────────────

@runtime_checkable
class MethodAdapter(Protocol):
    """What a method package registers under ``ymerflow.processing.methods``.

    Two responsibilities, and only these two: say how this method names its
    arrays, and say what their second axis means. Everything else about channel
    discovery is handled here in ``core``.
    """

    name: str

    def channels(self, dataset: "Dataset") -> list[Channel]:
        """Every channel this method can find in ``dataset``, or ``[]``.

        Returning ``[]`` is how an adapter says "this is not my data", so
        discovery can simply try each in turn.

        Parameters
        ----------
        dataset : Dataset
            The data to look at.

        Returns
        -------
        list[Channel]
            The channels found, in a stable order.
        """
        ...


def load_method_adapters() -> list[MethodAdapter]:
    """Every registered method adapter, in a stable order.

    Sorted by entry-point name because ``importlib.metadata`` does not promise
    an order, and which adapter claims a dataset must not depend on install
    order.

    Returns
    -------
    list[MethodAdapter]
        One instance per registered entry point.
    """
    adapters = []
    for ep in sorted(importlib.metadata.entry_points(group=METHOD_GROUP),
                     key=lambda e: e.name):
        loaded = ep.load()
        # An entry point may name either a class or a ready-made instance.
        adapters.append(loaded() if isinstance(loaded, type) else loaded)
    return adapters


# ── Column conventions ───────────────────────────────────────────────────────
# libaarhusxyz resolves position, altitude and line identity by trying a list of
# known spellings. It has nothing for flight, so the same idiom is continued
# here rather than each caller inventing its own lookup.

#: Where the processing history lives, so it round-trips with the delivery. A
#: file that says how it was processed is worth having on its own, and two of
#: the comparison checks read datasets off disk and would otherwise see none.
HISTORY = "processing_history"

#: Where the coordinate reference system lives. **The platform's key, not one
#: invented here** — its import process writes an integer EPSG code there and
#: every process downstream reads it, so using the same key is what lets a
#: dataset carry its projection between the two.
PROJECTION = "projection"


def _epsg_code(value) -> int | None:
    """``value`` as a whole-number EPSG code, or ``None`` if it is not one.

    Numeric rather than textual because a round trip writes the integer back as
    ``"32610.0"``: ``libaarhusxyz`` stores ``model_info`` values as floats, so a
    string test would recognise a projection before the file was saved and not
    after.
    """
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return int(number) if number > 0 and number == int(number) else None

#: Spellings a flight identifier turns up under.
FLIGHT_COLUMNS = ("flight", "flight_no", "flightno", "flight_id", "flightid")

#: Spellings pitch and roll turn up under. ``libaarhusxyz`` knows these too, but
#: reaches them by generating capitalisation *variants* — lower, upper, title —
#: which cannot produce an interior capital. A real delivery spells them
#: ``AngleX`` and ``AngleY``, and no variant of ``anglex`` is ``AngleX``, so its
#: lookup returns nothing and attitude silently disappears from the run. Matching
#: case-insensitively instead has no such gap (YmerFlow/libaarhusxyz#6).
PITCH_COLUMNS = ("tilt_x", "txpitch", "anglex", "pitch", "tx_pitch")
ROLL_COLUMNS = ("tilt_y", "txroll", "angley", "roll", "tx_roll")

#: Spellings **height above ground** turns up under — not elevation. The two are
#: different quantities and one file here carries both: ``Height`` at tens of
#: metres and ``Alt`` at hundreds, the latter being height above the geoid. A
#: name alone cannot always separate them, which is why
#: :func:`plausible_height` exists and why the column used is reported.
HEIGHT_COLUMNS = ("tx_altitude", "txaltitude", "tx_alt", "height",
                  "altitude", "alt")

#: Where a normalized timestamp lands, and what it is counted in. libaarhusxyz
#: builds this from the date and time columns as *days since 1900-01-01*.
TIMESTAMP_EPOCH = "1900-01-01"


#: How each convention spells a channel's roles, keyed by the prefix its data
#: array carries. ``None`` means the role has no established spelling in that
#: convention and the generic suffix form is used instead.
ROLE_SPELLINGS = {
    "gate_": {"std": "STD_", "inuse": "InUse_"},
    "dbdt_": {"std": "dbdt_std_", "inuse": "dbdt_inuse_"},
}


def role_key(channel, role: str) -> str:
    """A key for one of a channel's arrays, derived from its **data key**.

    Both numbered conventions put the role in the same position, so the key is
    the data key with the role substituted rather than a name assembled from a
    channel number. That is what makes this work for a convention nobody has
    enumerated — including one that identifies its channels by name, where
    there is no number to assemble from.

    Used wherever the library has to *create* an array a delivery did not carry:
    a raw delivery legitimately ships neither a mask nor uncertainties, and
    creating them under a discoverable name is what lets the next lookup find
    them without anything being kept in sync (``Channel`` is frozen and
    ``Dataset.channels`` rediscovers on every access).

    Parameters
    ----------
    channel : Channel
        The channel whose array is wanted; only its ``data_key`` is read.
    role : str
        ``"std"`` or ``"inuse"``.

    Returns
    -------
    str
        The ``layer_data`` key for that role.
    """
    data = channel.data_key
    lowered = data.lower()
    for prefix, roles in ROLE_SPELLINGS.items():
        # Located on the lowered copy and spliced into the original, so a
        # delivery spelling the prefix `DBDT_` gets the role substituted too.
        # `str.replace` on the original is case-sensitive and, finding nothing,
        # would hand back the data key itself as the key for the role.
        at = lowered.find(prefix)
        if at >= 0:
            return data[:at] + roles[role] + data[at + len(prefix):]
    return f"{data}_{role}"


def find_column(frame, spellings) -> str | None:
    """The first of ``spellings`` present in ``frame``, matched case-insensitively.

    Case-insensitively because a round trip through the xyz format changes case,
    and because generating capitalisation variants — which is the other way to
    do this — cannot produce an interior capital like ``AngleX``. That gap is
    not hypothetical: it is why attitude vanished from the first real delivery
    this library was pointed at.

    Order is priority: the most specific spelling first, so a file carrying both
    a specific and a generic name gets the specific one.

    Parameters
    ----------
    frame : pandas.DataFrame | None
        Where to look; ``None`` finds nothing.
    spellings : sequence of str
        Lower-case candidates, most specific first.

    Returns
    -------
    str | None
        The column as it is actually spelled in ``frame``, or ``None``.
    """
    if frame is None:
        return None
    lookup = {str(c).strip().lower(): c for c in frame.columns}
    for name in spellings:
        found = lookup.get(name.strip().lower())
        if found is not None:
            return found
    return None


def find_flight_column(frame) -> str | None:
    """The column identifying which flight each sounding belongs to, or ``None``.

    Parameters
    ----------
    frame : pandas.DataFrame | None
        The ``flightlines`` to search.

    Returns
    -------
    str | None
        The first of ``FLIGHT_COLUMNS`` present, as spelled in the frame.
    """
    return find_column(frame, FLIGHT_COLUMNS)


# ── Copying a survey ─────────────────────────────────────────────────────────

def _copy_survey(survey):
    """Deep-copy a ``Survey`` component by component.

    A compatibility shim, kept deliberately.

    ``GEX.__getattr__`` used to proxy straight into its underlying dict, so
    probing it for ``__deepcopy__`` — which ``deepcopy`` does for every object it
    visits — raised ``KeyError`` rather than the ``AttributeError`` the protocol
    expects, and the copy died. The same trap caught ``copy``, ``pickle`` and
    anything else that feature-detects by attribute.

    **That is fixed** in the fork this package pins
    (YmerFlow/libaarhusxyz#3), so ``copy.deepcopy(survey)`` now works. It is
    still broken in the version published to PyPI, which is what an installation
    that ignores our pin would get. Copying component-wise works against both, so
    this stays until the fork is the only path anyone can install — then it
    reduces to ``copy.deepcopy(survey)`` and this function goes away.

    ``XYZ`` copies cleanly and the system description is a plain dict, so the two
    halves are copied separately and reassembled. Worth doing here rather than at
    every call site: "steps return a new dataset rather than mutate" is the
    library's convention, which makes this the most-travelled path in it.
    """
    xyz = copy.deepcopy(survey.xyz)
    gex = type(survey.gex)(copy.deepcopy(survey.gex.gex_dict))
    orig = copy.deepcopy(survey.orig_xyz) if survey.orig_xyz is not None else None
    return type(survey)(xyz, gex, orig_xyz=orig)


# ── The dataset ──────────────────────────────────────────────────────────────

class Dataset:
    """A survey, its system description, and what has been done to it.

    Parameters
    ----------
    survey : libaarhusxyz.Survey
        The data file and its system description, already read.
    normalization : str
        The convention the amplitudes are in; one of ``NORMALIZATIONS``.
    crs : str | None
        The coordinate reference system, where known. Left unset, whatever the
        delivery carries is kept.
    """

    def __init__(
        self,
        survey: libaarhusxyz.Survey,
        normalization: str = RX_NORMALIZED,
        crs: str | None = None,
    ):
        if normalization not in NORMALIZATIONS:
            raise ValueError(
                f"normalization must be one of {NORMALIZATIONS}, got {normalization!r}"
            )
        self.survey = survey
        self.normalization = normalization
        # Only when stated. Assigning None would clobber a projection the
        # delivery already carries, and a dataset imported through the platform
        # always carries one.
        if crs is not None:
            self.crs = crs
        #: Injectable for testing and for callers who know what they have.
        #: ``None`` means "ask the entry-point group".
        self._adapters: list[MethodAdapter] | None = None

    # ── The two tables ───────────────────────────────────────────────────────

    @property
    def flightlines(self):
        """Per-sounding attributes. QC verdicts live here as columns."""
        return self.survey.xyz.flightlines

    @property
    def layer_data(self):
        """Per-gate arrays. The in-use mask lives here."""
        return self.survey.xyz.layer_data

    @property
    def system(self):
        """The system description."""
        return self.survey.gex

    @property
    def n_soundings(self) -> int:
        return 0 if self.flightlines is None else len(self.flightlines)

    @property
    def flight_column(self) -> str | None:
        """Which column says what flight a sounding belongs to, if any.

        A delivered file commonly holds three or four flights, so this is what
        lets one run report per-flight as well as per-line.
        """
        frame = self.flightlines
        return None if frame is None else find_flight_column(frame)

    def describe(self) -> dict:
        """The facts about *what this data is*, for a run report.

        Distinct from the report's identity, which is a human label. These are
        read from the data, so they stay true when a file is renamed or
        re-exported — which is what makes them safe to order a cross-delivery
        drift series by.

        Returns
        -------
        dict
            Sounding and channel counts, the flights present, the time range,
            and the CRS — plain values, ``None`` where the data says nothing.
        """
        frame = self.flightlines
        described: dict = {"flights": None, "time_range": None}
        if frame is None or len(frame) == 0:
            return described

        flight_column = self.flight_column
        if flight_column is not None:
            flights = pd.unique(frame[flight_column].dropna())
            described["flights"] = sorted(_plain(f) for f in flights)

        described["time_range"] = self._time_range()
        return described

    def _time_range(self) -> list[str] | None:
        """First and last acquisition time, ISO-formatted, or ``None``.

        Two shapes are handled because both occur. A survey that has been
        normalized carries a single ``timestamp`` column counted in *days since
        1900-01-01*, which is libaarhusxyz's own convention. One that has not
        still has whatever date and time columns it was delivered with.
        """
        frame = self.flightlines
        xyz = self.survey.xyz

        timestamp = xyz.get_column("timestamp")
        if timestamp is not None and timestamp in frame.columns:
            days = pd.to_numeric(frame[timestamp], errors="coerce").dropna()
            if not days.empty:
                moments = (pd.Timestamp(TIMESTAMP_EPOCH)
                           + pd.to_timedelta(days, unit="D"))
                return [moments.min().isoformat(), moments.max().isoformat()]

        date = xyz.get_column("date")
        if date is not None and date in frame.columns:
            values = frame[date].dropna().astype(str)
            # A '*' sentinel marks an edited sounding in some deliveries.
            values = values[values.str.strip().ne("") & values.ne("*")]
            if not values.empty:
                return [str(values.min()), str(values.max())]
        return None

    # ── Channels, discovered rather than assumed ─────────────────────────────

    @property
    def adapters(self) -> list[MethodAdapter]:
        return load_method_adapters() if self._adapters is None else self._adapters

    @property
    def channels(self) -> list[Channel]:
        """Every channel present, in a stable order.

        Asked of the data rather than assumed anywhere. A step operating on
        "the data" operates on all of these unless told otherwise (R1) — so a
        single-moment system and a three-moment system need no special casing,
        and a scalar magnetometer correctly reports none at all.

        Deliberately not cached: steps add and drop arrays, and a stale channel
        list would be a subtle, silent wrong answer. Discovery is a regex over a
        few dozen keys.
        """
        for adapter in self.adapters:
            found = adapter.channels(self)
            if found:
                return found
        return []

    @property
    def adapter(self) -> MethodAdapter | None:
        """The registered method that claimed this data, if any."""
        for adapter in self.adapters:
            if adapter.channels(self):
                return adapter
        return None

    @property
    def method(self) -> str | None:
        """Which registered method claimed this data, if any."""
        adapter = self.adapter
        return None if adapter is None else adapter.name

    @property
    def n_channels(self) -> int:
        return len(self.channels)

    def channel(self, wanted: str) -> Channel:
        """The channel addressed by ``wanted``, matched on key or on name.

        Parameters
        ----------
        wanted : str
            A channel key (``"ch1"``) or name (``"LM"``).

        Returns
        -------
        Channel
            The matching channel.

        Raises
        ------
        KeyError
            If nothing matches, naming what is present.
        """
        channels = self.channels
        for ch in channels:
            if wanted == ch.key or (ch.name is not None and wanted == ch.name):
                return ch
        raise KeyError(
            f"No channel {wanted!r}. Present: "
            f"{', '.join(str(c) for c in channels) or '(none)'}"
        )

    def resolve_per_channel(self, value) -> dict[str, Any]:
        """Expand a per-channel setting to every channel present.

        A scalar means "apply to all"; a mapping is used as given and may be
        addressed by either key or name. This is how R1's rule that no default
        may name a specific moment stays workable — a caller passes one number
        and it reaches every channel, however many there are.

        Parameters
        ----------
        value : object | dict
            A scalar for every channel, or a mapping by channel key or name.

        Returns
        -------
        dict[str, Any]
            Keyed by channel ``key``, so a consumer never has to care which
            spelling the caller used.
        """
        channels = self.channels
        if not isinstance(value, dict):
            return {ch.key: value for ch in channels}

        resolved: dict[str, Any] = {}
        for wanted, setting in value.items():
            # A YAML mapping with bare 1: 2: keys arrives as ints, not strings.
            resolved[self.channel(str(wanted)).key] = setting
        return resolved

    # ── Moments: the time-domain name for a channel ──────────────────────────

    @property
    def moments(self) -> list[Channel]:
        """Every transmitter moment — TEM's name for a channel.

        Kept because ``building_plans`` refers to moments throughout, and it is
        what a geophysicist says. It resolves to :attr:`channels` on time-domain
        data and refuses on anything else, rather than quietly returning
        frequencies or sensors under the wrong noun.
        """
        method = self.method
        if method not in (None, "tem"):
            raise TypeError(
                f"This is {method} data; it has channels, not moments. "
                f"Use .channels — see R1 in building_plans/01_filter_inventory.md."
            )
        return self.channels

    @property
    def n_moments(self) -> int:
        return len(self.moments)

    def resolve_per_moment(self, value):
        """Per-moment settings; see :meth:`resolve_per_channel`.

        Parameters
        ----------
        value : object | dict
            A scalar for every moment, or a mapping by key or name.

        Returns
        -------
        dict[str, Any]
            Keyed by channel ``key``.
        """
        return self.resolve_per_channel(value)

    # ── Where on earth it is ─────────────────────────────────────────────────

    @property
    def crs(self) -> str | None:
        """The coordinate reference system the positions are in, or ``None``.

        Held in ``model_info["projection"]``, which is **the platform's own
        convention rather than one invented here**: the import process takes the
        projection as a required parameter, writes it there once as an integer
        EPSG code, and every process downstream reads it from there. A dataset
        that came through the platform therefore already knows where it is, and
        a step that asked the caller to restate it would be asking for something
        the file is carrying.

        Read back as ``"EPSG:32610"`` because that is what ``pyproj`` and
        ``geopandas`` take, and stored as the bare integer because that is what
        the platform writes and reads. Anything that is not an EPSG code — a
        WKT or proj string — is stored and returned unchanged.

        A round trip through the file turns the integer into ``"32610.0"``,
        because ``libaarhusxyz`` writes ``model_info`` values as floats. So the
        code is recognised numerically rather than by ``isdigit``, which would
        have quietly stopped recognising it the moment the dataset was saved.
        """
        value = self.survey.xyz.model_info.get(PROJECTION)
        if value is None or value == "":
            return None
        code = _epsg_code(value)
        return str(value).strip() if code is None else f"EPSG:{code}"

    @crs.setter
    def crs(self, value) -> None:
        if value is None:
            self.survey.xyz.model_info.pop(PROJECTION, None)
            return
        text = str(value).strip()
        if text.upper().startswith("EPSG:"):
            text = text.split(":", 1)[1].strip()
        code = _epsg_code(text)
        self.survey.xyz.model_info[PROJECTION] = (
            str(value) if code is None else code)

    # ── What was done to it ──────────────────────────────────────────────────

    @property
    def history(self) -> list[dict]:
        """One entry per step applied, in order. The runner records; steps do not.

        Held in ``model_info`` rather than on the object, so it **survives a
        round trip through a file**. That is not tidiness: ``version_diff``
        reads history to say which parameter moved, and ``parameter_sweep``
        discovers the swept parameter from it — and both of those read datasets
        off disk. An in-memory-only history would leave the diff reporting "no
        pipeline difference" between two files that were processed differently,
        which is a confident wrong answer rather than a missing one.

        The same argument ``mask`` makes for its provenance legend: *the file
        has to carry its own legend because whoever opens it will not have the
        run report.* A delivery that says how it was processed is worth having
        on its own.
        """
        raw = self.survey.xyz.model_info.get(HISTORY)
        if not raw:
            return []
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                log.warning("%s is not readable as processing history; "
                            "treating it as empty", HISTORY)
                return []
        return list(raw)

    @history.setter
    def history(self, entries) -> None:
        self.survey.xyz.model_info[HISTORY] = json.dumps(
            list(entries), default=str)

    def record_step(self, step: str, params: dict | None = None) -> None:
        """Append one applied step.

        Used instead of ``history.append``, which would append to a list
        reconstructed from the file and thrown away.

        Parameters
        ----------
        step : str
            The step's registered name.
        params : dict | None
            The parameters it ran with, as given in the pipeline document.
        """
        self.history = self.history + [{"step": step, "params": params or {}}]

    # ── Copying ──────────────────────────────────────────────────────────────

    def copy(self) -> "Dataset":
        """A deep copy, so a step can return a new dataset rather than mutate.

        Returns
        -------
        Dataset
            Independent of this one in every array and attribute.
        """
        # type(self) rather than Dataset: a subclass must not be silently
        # downgraded to the base class by copying.
        new = type(self)(_copy_survey(self.survey), self.normalization, self.crs)
        # history is not copied: it lives in model_info, which the survey copy
        # already brought along. Copying it here would work but would give the
        # field two homes, and the one that round-trips should be the only one.
        new._adapters = self._adapters
        return new

    def __repr__(self) -> str:
        return (
            f"<Dataset {self.n_soundings} soundings, "
            f"method={self.method!r}, {self.n_channels} channel(s), "
            f"normalization={self.normalization!r}, "
            f"{len(self.history)} steps applied>"
        )


def _plain(value):
    """A numpy scalar as a plain Python value, so a report can be serialized."""
    return value.item() if isinstance(value, np.generic) else value


# ── Loading ──────────────────────────────────────────────────────────────────

def load(xyz, gex, normalization: str | None = None,
         crs: str | None = None) -> Dataset:
    """A dataset read from a survey file and a system description.

    Both are paths. That is deliberate and is what keeps the library usable on
    its own: R8's layering is *the platform materialises, the library reads*, so
    nothing here resolves an asset identifier or reaches for a storage backend.
    Hand it two files and it works, with no platform and no credentials.

    Parameters
    ----------
    xyz : str | Path
        The survey file.
    gex : str | Path
        The system description.
    normalization : str | None
        The convention the data is in. Unset, the one the file declares is
        used, and ``rx`` if it declares none. Given, it wins over the file —
        a caller who names one is correcting the record.
    crs : str | None
        The coordinate reference system the positions are in, where known.

    Returns
    -------
    Dataset
        The two files as one object, with the system description's declared
        initial gates already disabled.

    Raises
    ------
    FileNotFoundError
        If either path does not exist.

    Notes
    -----
    Missing files are reported here rather than surfacing from inside
    ``libaarhusxyz`` as something less obvious.

    **The system description's declaration is applied here.** A GEX that
    declares initial gates unusable has those gates disabled in the in-use mask
    before the dataset is returned — this is the moment the two files become one
    thing, and it is what makes "what is in the system description is what you
    get" a true statement rather than an instruction someone has to follow.

    It is not a culling decision and does not read as one: nobody chose a
    threshold, the instrument stated which of its own samples cannot be
    inverted. Left to a pipeline step it is the easiest thing in the library to
    forget, and forgetting it is silent — the inversion runs, converges, and is
    confidently wrong about the shallow section.

    ``filter_21`` puts them back for anyone doing full primary-field modeling,
    which is why this writes a mask rather than removing anything.
    """
    for role, path in (("survey", xyz), ("system description", gex)):
        if not Path(path).exists():
            raise FileNotFoundError(f"No {role} at {path}")
    survey = libaarhusxyz.Survey(str(xyz), str(gex))

    # A convention the file declares beats the caller's default. filter_08 says
    # 01's fix is to make the convention "an explicit property of the dataset
    # that travels with it" — an assertion made once at load time does not
    # survive the file, so a written dataset records what it is and this reads
    # it back. An explicit argument still wins, since a caller who names one is
    # correcting the record rather than guessing — which is why the default is
    # None and not RX_NORMALIZED: with the latter as a sentinel, a caller
    # explicitly asking for rx on a file declaring rx_tx was overruled.
    if normalization is None:
        declared = survey.xyz.model_info.get("normalization")
        normalization = (declared if declared in NORMALIZATIONS
                         else RX_NORMALIZED)

    dataset = Dataset(survey, normalization, crs)

    # Imported here rather than at module scope: the culling package imports
    # Dataset, so a top-level import back would be a cycle.
    from .cull.filter_20_initial_gates import honor_system_description
    honor_system_description(dataset)
    return dataset
