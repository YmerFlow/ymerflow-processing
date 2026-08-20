"""Synthetic fixtures.

Everything here is built in memory. No survey files, no system-description
files, no network, no platform — the library must be runnable and testable
standalone, and a test suite that needs a real survey on disk is the first thing
to break that.

Synthetic data also buys coverage real data cannot: we have dual-moment SkyTEM
to hand, but R1 is a claim about *one to N* moments, and the only way to test a
three-moment system and a single-moment system is to construct them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from libaarhusxyz import Survey
from libaarhusxyz.gex import GEX
from libaarhusxyz.xyz import XYZ

from ymerflow_processing.core.dataset import Dataset
from ymerflow_processing.core.qc.thresholds import labels
from ymerflow_processing.em.tem.method import TEMMethod


def make_infrastructure(path=None, lines=((0, 500), (1000, 500)),
                        crs="EPSG:32612"):
    """Mapped features as vector geometry — a GeoDataFrame, or written to disk.

    Written rather than mocked where a path is wanted, because the interesting
    failures are geopandas' and shapely's: CRS reprojection, empty geometries,
    and the nearest-neighbour query.
    """
    import geopandas as gpd
    from shapely.geometry import LineString

    if lines and not isinstance(lines[0][0], (list, tuple)):
        lines = [lines]
    frame = gpd.GeoDataFrame(
        {"kind": ["powerline"] * len(lines)},
        geometry=[LineString(coords) for coords in lines], crs=crs)
    if path is None:
        return frame
    frame.to_file(path)
    return path


def make_terrain(path, elevation=200.0, bounds=(-100, -100, 1000, 1000),
                 cell=10.0, nodata=-9999.0, crs="EPSG:32612", holes=None):
    """A terrain model on disk: a constant surface, optionally with nodata.

    Written rather than mocked, because the interesting failures are rasterio's
    — CRS mismatch, nodata, and sampling outside the footprint — and a mock
    would assert against my idea of them rather than against the library.
    """
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin

    left, bottom, right, top = bounds
    width = int(np.ceil((right - left) / cell))
    height = int(np.ceil((top - bottom) / cell))
    grid = np.full((height, width), float(elevation), dtype="float32")
    for row, column in (holes or []):
        grid[row, column] = nodata

    with rasterio.open(
        path, "w", driver="GTiff", height=height, width=width, count=1,
        dtype="float32", crs=crs, nodata=nodata,
        transform=from_origin(left, top, cell, cell),
    ) as handle:
        handle.write(grid, 1)
    return path


def step(report, name):
    """One step's report, by name.

    By name rather than by position, because every run opens with an implicit
    inventory and a pipeline may grow implicit steps again — a test asserting on
    ``steps[0]`` is asserting on the ordering rather than on the step it means.
    """
    steps = report["steps"] if isinstance(report, dict) else report.steps
    for entry in steps:
        found = entry["step"] if isinstance(entry, dict) else entry.step
        if found == name:
            return entry
    raise AssertionError(f"No step {name!r} in the report. Ran: {named(report)}")


def named(report):
    """Every step that ran, in order."""
    steps = report["steps"] if isinstance(report, dict) else report.steps
    return [e["step"] if isinstance(e, dict) else e.step for e in steps]


def tier_names(codes):
    """Verdict codes back to tier names, for readable assertions.

    Verdicts are stored as float32 codes so the frontend can plot them — a
    string column fails its ``ArrayBuffer.isView`` filter and never appears at
    all. Tests still assert in the vocabulary the domain uses, because
    ``== [GOOD, GOOD, OOS]`` says what a test means and ``== [2.0, 2.0, 0.0]``
    does not.
    """
    return labels(codes)


def make_gex(moments=("LM", "HM"), n_gates=8, turns=(1.0, 4.0), currents=(9.0, 110.0),
             components=None):
    """A system description with one ``ChannelN`` block per moment.

    Shaped like a real SkyTEM GEX after parsing: gate times as an (n, 3) array
    of center / open / close, and the per-channel entries the library reads.

    ``components`` gives each channel its receiver polarization. A
    single-component instrument declares ``Z`` throughout; a three-component one
    declares a channel per component, so ``moments`` and ``components`` are
    given per channel and a 3C dual-moment system is six of them.
    """
    # Log-spaced gates, as a real transient is sampled. Center is the midpoint
    # of open and close, which is the convention verified against SkyTEM files.
    opens = np.logspace(-6, -3, n_gates)
    closes = opens * 1.3
    centers = (opens + closes) / 2
    gate_times = np.column_stack([centers, opens, closes])

    components = components or ["Z"] * len(moments)
    gex: dict = {"General": {"GateTime": gate_times, "TxLoopArea": 342.0}}
    for i, moment in enumerate(moments, start=1):
        gex["General"][f"NumberOfTurns{moment}"] = turns[i - 1 if i <= len(turns) else -1]
        gex[f"Channel{i}"] = {
            "NoGates": float(n_gates),
            "TransmitterMoment": moment,
            "TxApproximateCurrent": currents[i - 1 if i <= len(currents) else -1],
            "UniformDataSTD": 0.03,
            "ReceiverPolarizationXYZ": components[i - 1],
            "RemoveInitialGates": 0.0,
            "GateTimeShift": 0.0,
            "MeaTimeDelay": 0.0,
            # Derived by libaarhusxyz at parse time; supplied here because we
            # hand it a dict and so bypass the parser.
            "ApproxDipoleMoment": (
                turns[i - 1 if i <= len(turns) else -1]
                * 342.0
                * currents[i - 1 if i <= len(currents) else -1]
            ),
        }
    return GEX(gex)


def make_flightlines(n=40, n_lines=2, height=None, n_flights=None,
                     pitch=None, roll=None, speed=None, seconds=None,
                     second_altimeter=None):
    """A per-sounding attribute table with the columns QC actually reads.

    ``n_flights`` groups the lines into flights, the way a delivery holding
    three or four flights does. ``None`` omits the flight column entirely, which
    is a case QC has to survive.
    """
    per_line = n // n_lines
    lines = np.repeat(np.arange(100, 100 + n_lines), per_line)[:n]
    if height is None:
        height = np.full(n, 40.0)

    frame = pd.DataFrame({
        "line_id": lines,
        # 10 m spacing along a straight east-west line, so contiguous-run
        # distances are trivially checkable by hand.
        "x": np.tile(np.arange(per_line) * 10.0, n_lines)[:n],
        "y": np.repeat(np.arange(n_lines) * 1000.0, per_line)[:n],
        "tx_alt": np.asarray(height, dtype=float),
        "topo": np.full(n, 250.0),
        # tilt_x is pitch (along flight), tilt_y is roll (across it). Default to
        # a small non-zero wobble: real attitude is never identically zero, and
        # qc_02 treats an all-zero series as a zeroed sensor rather than a
        # perfectly level flight.
        "tilt_x": np.linspace(-2.0, 2.0, n) if pitch is None
                  else np.asarray(pitch, dtype=float),
        "tilt_y": np.linspace(1.0, -1.0, n) if roll is None
                  else np.asarray(roll, dtype=float),
    })
    if second_altimeter is not None:
        # A second, independent altimeter, for the checks that compare them.
        # Absent by default: not every aircraft carries two.
        frame["alt_two"] = np.asarray(second_altimeter, dtype=float)
    if speed is not None:
        frame["GdSpeed"] = np.asarray(speed, dtype=float)
    if seconds is not None:
        # libaarhusxyz counts timestamp in days since 1900-01-01.
        frame["timestamp"] = np.asarray(seconds, dtype=float) / 86400.0
    if n_flights is not None:
        # Whole lines belong to a flight; a line is never split across two.
        lines_per_flight = max(1, n_lines // n_flights)
        frame["Flight"] = np.minimum(
            (lines - 100) // lines_per_flight, n_flights - 1) + 1
    return frame


def make_dataset(moments=("LM", "HM"), n=40, n_lines=2, n_gates=8,
                 height=None, naming="dbdt", with_gex=True, n_flights=None,
                 components=None, pitch=None, roll=None,
                 speed=None, seconds=None, second_altimeter=None):
    """A :class:`Dataset` with one array family per moment.

    ``naming`` picks which of the two conventions in circulation the arrays are
    named by — ``dbdt`` for the AarhusInv style, ``gate`` for the SkyTEM style.
    Both must be discovered identically.
    """
    flightlines = make_flightlines(n, n_lines, height, n_flights, pitch, roll,
                                   speed, seconds, second_altimeter)

    layer_data = {}
    for i in range(1, len(moments) + 1):
        if naming == "dbdt":
            data_key, std_key, inuse_key = (
                f"dbdt_ch{i}gt", f"dbdt_std_ch{i}gt", f"dbdt_inuse_ch{i}gt")
        else:
            data_key, std_key, inuse_key = (
                f"Gate_Ch{i:02d}", f"STD_Ch{i:02d}", f"InUse_Ch{i:02d}")
        layer_data[data_key] = pd.DataFrame(
            np.full((n, n_gates), 1e-9 * i), columns=range(n_gates))
        layer_data[std_key] = pd.DataFrame(
            np.full((n, n_gates), 0.03), columns=range(n_gates))
        layer_data[inuse_key] = pd.DataFrame(
            np.ones((n, n_gates), dtype=int), columns=range(n_gates))

    # model_info is present on any XYZ parsed from a file, and dumping requires
    # it, so the fixture carries it too — a fixture that cannot round-trip is
    # not modeling the real object.
    xyz = XYZ({"flightlines": flightlines,
               "layer_data": layer_data,
               "model_info": {},
               "file_meta": {}})
    gex = make_gex(moments, n_gates, components=components) if with_gex else GEX({})
    dataset = Dataset(Survey(xyz, gex))
    # Injected rather than resolved through entry points, so the tests do not
    # depend on the package being installed in the running interpreter.
    dataset._adapters = [TEMMethod()]
    return dataset


@pytest.fixture
def dual_moment():
    return make_dataset(moments=("LM", "HM"))


@pytest.fixture
def single_moment():
    """A HeliTEM/VTEM-shaped system: one moment, and nothing special-cased."""
    return make_dataset(moments=("LM",))


@pytest.fixture
def three_moment():
    """No such instrument is to hand; R1 says it must work anyway."""
    return make_dataset(moments=("LM", "MM", "HM"))
