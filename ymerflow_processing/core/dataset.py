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

from __future__ import annotations

import copy
import importlib.metadata
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

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


@dataclass
class NoiseModel:
    """The noise model, held once and read by every consumer (R9).

    Defined in one place so that assignment and culling cannot disagree about
    it. ``floor_1ms`` and ``exponent`` may be estimated from the data;
    ``relative_fraction`` never is — it is systematic and has no separable
    signature to measure.
    """

    floor_1ms: float | None = None       # V/m^2, receiver-area normalized
    exponent: float = -0.5
    relative_fraction: float | None = None

    #: How each value was arrived at — "estimated", "supplied", or "system".
    provenance: dict[str, str] = field(default_factory=dict)

    #: What estimation found, even where a supplied value is being applied.
    #: An override changes what is applied, never what is measured (R9).
    estimated: dict[str, float] = field(default_factory=dict)

    def is_complete(self) -> bool:
        return self.floor_1ms is not None and self.relative_fraction is not None


# ── The second axis of an array family ───────────────────────────────────────

@dataclass(frozen=True)
class Axis:
    """What the columns of a channel's arrays are indexed by.

    Carried by ``core`` and interpreted by nobody here. A method package fills
    it in — gate center times for TEM, frequencies for FEM — and a generic
    step may read ``values`` to weight or taper along it without ever learning
    what the quantity is.

    ``values`` is ``None`` for a method whose channels have no meaningful second
    axis, such as a scalar magnetometer.
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
    """

    data: tuple[str, ...]
    std: tuple[str, ...] = ()
    inuse: tuple[str, ...] = ()


def _match_index(key: str, patterns: tuple[str, ...]) -> int | None:
    """The channel index ``key`` encodes, or ``None`` if it is not one of these."""
    for pattern in patterns:
        match = re.match(pattern, key, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return None


def discover_by_naming(layer_data_keys, naming: ChannelNaming) -> dict[int, dict]:
    """Group ``layer_data`` keys by the channel index they encode.

    The mechanism half of channel discovery: it finds *which* channels exist and
    *which arrays belong to each*, and knows nothing about what they measure. A
    method adapter supplies the naming and adds the meaning.

    Returns ``{index: {"data_key": ..., "std_key": ..., "inuse_key": ...}}``.
    A channel with no data array is not a channel — uncertainties or a mask
    without values is a broken import, not a measurement — so those are dropped
    with a warning rather than returned half-formed.
    """
    found: dict[int, dict] = {}

    # Order matters only defensively: the roles have distinct anchored prefixes,
    # but checking the more specific ones first means a sloppier pattern added
    # later cannot quietly capture another role's keys.
    for role, patterns in (("inuse_key", naming.inuse),
                           ("std_key", naming.std),
                           ("data_key", naming.data)):
        for key in layer_data_keys:
            index = _match_index(key, patterns)
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
        """
        ...


def load_method_adapters() -> list[MethodAdapter]:
    """Every registered method adapter, in a stable order.

    Sorted by entry-point name because ``importlib.metadata`` does not promise
    an order, and which adapter claims a dataset must not depend on install
    order.
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

#: Spellings a flight identifier turns up under.
FLIGHT_COLUMNS = ("flight", "Flight", "flight_no", "FlightNo", "flight_id", "FlightID")

#: Where a normalized timestamp lands, and what it is counted in. libaarhusxyz
#: builds this from the date and time columns as *days since 1900-01-01*.
TIMESTAMP_EPOCH = "1900-01-01"


def find_flight_column(frame) -> str | None:
    """The column identifying which flight each sounding belongs to, or ``None``."""
    for name in FLIGHT_COLUMNS:
        if name in frame.columns:
            return name
    return None


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
    """A survey, its system description, and what has been done to it."""

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
        self.crs = crs
        self.noise_model = NoiseModel()
        #: One entry per step applied, in order. The runner appends; steps do not.
        self.history: list[dict] = []
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
        """The channel addressed by ``wanted``, matched on key or on name."""
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

        Returns a mapping keyed by channel ``key``, so a consumer never has to
        care which spelling the caller used.
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
        """Per-moment settings; see :meth:`resolve_per_channel`."""
        return self.resolve_per_channel(value)

    # ── Copying ──────────────────────────────────────────────────────────────

    def copy(self) -> "Dataset":
        """A deep copy, so a step can return a new dataset rather than mutate."""
        # type(self) rather than Dataset: a subclass must not be silently
        # downgraded to the base class by copying.
        new = type(self)(_copy_survey(self.survey), self.normalization, self.crs)
        new.noise_model = copy.deepcopy(self.noise_model)
        new.history = list(self.history)
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

def load(xyz, gex, normalization: str = RX_NORMALIZED, crs: str | None = None) -> Dataset:
    """A dataset read from a survey file and a system description.

    Both are paths. That is deliberate and is what keeps the library usable on
    its own: R8's layering is *the platform materialises, the library reads*, so
    nothing here resolves an asset identifier or reaches for a storage backend.
    Hand it two files and it works, with no platform and no credentials.

    Missing files are reported here rather than surfacing from inside
    ``libaarhusxyz`` as something less obvious.
    """
    for role, path in (("survey", xyz), ("system description", gex)):
        if not Path(path).exists():
            raise FileNotFoundError(f"No {role} at {path}")
    return Dataset(libaarhusxyz.Survey(str(xyz), str(gex)), normalization, crs)
