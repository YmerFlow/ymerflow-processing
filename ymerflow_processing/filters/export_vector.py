"""Write a dataset's attributes out as vector geometry.

**The artifact that leaves the building.** ``03`` calls the vector export the way
QC results reach people who do not use the platform — the client, the
acquisition contractor, the regulator — and for a funded program it is
plausibly a contract deliverable rather than an internal convenience.

It is deliberately *not* a QC feature. ``03`` makes that argument itself: once a
classification is just a column, exporting classified geometry is a general
dataset capability, so this serves anything wanting geometry out and the QC
deliverable is not a bespoke code path.

**What it does to the dataset:** nothing. It reads and writes files.

Two geometries, because they answer different questions
-------------------------------------------------------
**Points** say *where along a line* the problem was. One feature per sounding,
carrying every attribute.

**Lines** say *which lines failed*, which is what a reflight conversation is
about. One feature per flightline, carrying the per-line, per-flight and
per-delivery measures — which are constant along a line by construction. Someone
opens this in their GIS and sees the failures without joining anything.

GeoPackage, and no shapefile
----------------------------
GeoPackage carries a real CRS, an R-tree index, unlimited field names, NULLs
and several geometry types in one file. Shapefile was supported for the contract
that might name it, and dropped on 2026-10-22 (Ben): its ten-character field
names had become a constraint on every column name in the library, and once
``orig_tx_alt`` pushed past it the choice was to keep bending the library around
a 1998 format or to stop writing it. GeoJSON stays as a wire format for the web,
with the CRS caveat noted at write time. A ``.shp`` path is refused with a
message saying so, rather than quietly written as something else.

**Unclassified is written as an explicit label**, not a NULL. ``03`` gives a
metric with no threshold configured *no classification at all* rather than a
guessed "Good", and the one artifact a client reads has to keep that distinction
legible. A GIS legend reading ``Unclassified`` does; a blank cell reads as
"nothing to see". The report says what the label was.

The output CRS is explicit, always
----------------------------------
``03``: the output CRS is a project decision and cannot be derived from the
input. Contractor delivery CRS and client output CRS are routinely different, so
this takes the output CRS as a required parameter and **fails rather than
guessing**. Reprojection needs the source CRS too; that comes from the dataset,
and its absence is an error rather than an assumption that the two match.

Sources
-------
GeoPackage is OGC 12-128r19. GeoJSON is RFC 7946.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ..dataset import FLIGHT_COLUMNS
from ..pipeline import StepReport
from ..qc.thresholds import labels, verdict_columns

log = logging.getLogger(__name__)

CATEGORY = "housekeeping"

STEP = "export_vector"

#: Output extension to OGR driver. GeoPackage first because it is the default.
DRIVERS = {".gpkg": "GPKG", ".geojson": "GeoJSON", ".json": "GeoJSON"}

#: What "no classification" is written as. See the module docstring.
UNCLASSIFIED = "Unclassified"

#: Format names, for a caller that chooses one rather than implying it through
#: a filename. R8 says format is a loader concern on the way in; on the way out
#: it is a delivery concern, and neither is a property of the processing.
FORMATS = {"gpkg": "GPKG", "geojson": "GeoJSON", "json": "GeoJSON"}

#: Said in one place, because both the format and the extension paths say it.
NO_SHAPEFILE = ("Shapefile is not written (dropped 2026-10-22): its "
                "ten-character field names were constraining every column name "
                "in the library. Write GeoPackage (.gpkg); a GIS that needs a "
                "shapefile can export one from it.")


def _driver_for(path: Path, format: str | None = None) -> str:
    """The OGR driver, from a stated format or failing that the extension."""
    if format is not None:
        name = str(format).lower().lstrip(".")
        if name == "shp":
            raise ValueError(NO_SHAPEFILE)
        if name not in FORMATS:
            raise ValueError(
                f"Unknown format {format!r}. Available: "
                f"{', '.join(sorted(FORMATS))} — gpkg unless there is a reason "
                f"not to."
            )
        return FORMATS[name]

    suffix = path.suffix.lower()
    if suffix == ".shp":
        raise ValueError(NO_SHAPEFILE)
    if suffix not in DRIVERS:
        raise ValueError(
            f"Cannot tell a format from {path.name!r}, and none was given. Use "
            f"one of {', '.join(sorted(DRIVERS))}, or pass format= — .gpkg "
            f"unless there is a reason not to."
        )
    return DRIVERS[suffix]


def _labelled(frame: pd.DataFrame, verdicts: list[str],
              unclassified: str) -> pd.DataFrame:
    """Verdict codes written back as tier names, absences spelled out.

    The one place the numeric encoding is undone. Verdicts are stored as codes
    because a string column is invisible to the plot layer, but this file is
    opened in a GIS by a person, and ``0`` in a legend is worse than useless
    where ``OOS`` is self-explanatory.

    Unclassified becomes an explicit label rather than a null, so that "not
    measured" and "measured and fine" cannot read as the same thing in the one
    artifact a client actually opens. Both drivers write the same label so two
    exports of one dataset cannot disagree.

    ``verdicts`` names the columns to translate; it cannot be inferred, since
    encoding verdicts as numbers made them indistinguishable by dtype from the
    measurements sitting beside them.
    """
    out = frame.copy()
    for name in verdicts:
        if name not in out.columns:
            continue
        names = pd.Series(labels(out[name]), index=out.index, dtype=object)
        out[name] = names.where(names.notna(), unclassified)
    return out


def _line_features(frame, line_column, geometry_of):
    """One row per flightline: its identity, and the columns constant along it.

    Per-line, per-flight and per-delivery measures are constant within a line by
    construction, so taking the first value of each is exact rather than a
    summary. Per-sounding columns vary along the line and are deliberately left
    to the point layer.
    """
    constant = [c for c in frame.columns
                if str(c).startswith(("ql_", "qf_", "qd_"))]
    keep = [line_column] + [c for c in frame.columns
                            if str(c).lower() in FLIGHT_COLUMNS]
    rows, geometries = [], []
    for line, block in frame.groupby(line_column, sort=True, observed=True):
        geometry = geometry_of(block)
        if geometry is None:
            continue
        record = {c: block[c].iloc[0] for c in dict.fromkeys(keep + constant)}
        record["n_sound"] = len(block)
        rows.append(record)
        geometries.append(geometry)
    return pd.DataFrame(rows), geometries


def build_layers(dataset, crs, source_crs=None, columns=None,
                 lines: bool = True, unclassified: str = UNCLASSIFIED):
    """The dataset as named vector layers, in the output CRS.

    **This is the typed output**, and it is separate from writing on purpose.
    R8 says of the input side that "format is a loader concern, not a filter
    concern"; the same holds in reverse. What this step produces is *vector
    geometry* — what someone later chooses to serialise it as is a question
    about delivery, not about processing.

    Splitting it means the platform can build once and serve GeoPackage or
    GeoJSON on demand without re-running anything, and the standalone CLI can
    still write a file. Both read the same layers, so the two routes cannot
    drift.

    Parameters
    ----------
    dataset : Dataset
        Whose ``flightlines`` become features. Not changed.
    crs : str
        The output CRS, e.g. ``"EPSG:32613"``.
    source_crs : str | None
        What the dataset's coordinates are in. Taken from the dataset if it
        knows; an error if neither says.
    columns : list[str] | None
        Which attributes to carry. All of ``flightlines`` by default.
    lines : bool
        Also build one feature per flightline.
    unclassified : str
        What a verdict with no classification is written as.

    Returns
    -------
    tuple
        ``({name: GeoDataFrame}, notes)`` — the layers by name, and the notes
        the build produced for the step report.
    """
    import geopandas as gpd
    from shapely.geometry import LineString, Point

    notes: list[str] = []
    frame = dataset.flightlines
    if frame is None or len(frame) == 0:
        return {}, ["No soundings; nothing to build."]

    if not crs:
        raise ValueError(
            "An output CRS is required. It is a project decision and cannot be "
            "derived from the input — a contractor's delivery CRS and a "
            "client's output CRS are routinely different."
        )
    source = source_crs or dataset.crs
    if not source:
        raise ValueError(
            "The dataset does not say what CRS its coordinates are in, so they "
            "cannot be reprojected to the output CRS. Supply 'source_crs:'."
        )

    xyz = dataset.survey.xyz
    if xyz.x_column not in frame.columns or xyz.y_column not in frame.columns:
        return {}, ["No position columns; nothing to place."]

    # Which columns hold verdict codes rather than measurements. Read from the
    # dataset's own record rather than inferred: the two are both float32 and
    # nothing in the values distinguishes ql_verdict from ql_score.
    verdicts = verdict_columns(dataset)

    wanted = list(columns) if columns else [
        c for c in frame.columns if c not in (xyz.x_column, xyz.y_column)]
    missing = [c for c in wanted if c not in frame.columns]
    if missing:
        raise KeyError(
            f"No column(s) {', '.join(map(repr, missing))} in flightlines.")

    attributes = _labelled(frame[wanted], verdicts, unclassified)
    points = gpd.GeoDataFrame(
        attributes,
        geometry=[Point(x, y) for x, y in zip(frame[xyz.x_column],
                                              frame[xyz.y_column])],
        crs=source,
    )
    if str(source) != str(crs):
        points = points.to_crs(crs)
    built = {"soundings": points}

    line_column = xyz.line_id_column
    if lines and line_column and line_column in frame.columns:
        def geometry_of(block):
            # Two points is the minimum for a line; a single sounding on a line
            # has a position but no path.
            coordinates = list(zip(block[xyz.x_column], block[xyz.y_column]))
            return LineString(coordinates) if len(coordinates) > 1 else None

        records, geometries = _line_features(
            _labelled(frame, verdicts, unclassified), line_column, geometry_of)
        if geometries:
            flightlines = gpd.GeoDataFrame(records, geometry=geometries,
                                           crs=source)
            if str(source) != str(crs):
                flightlines = flightlines.to_crs(crs)
            built["flightlines"] = flightlines
    elif lines:
        notes.append("No line identifier, so no flightline geometry was built.")

    return built, notes


def export_vector(
    dataset,
    path: str,
    crs: str,
    source_crs: str | None = None,
    columns: list[str] | None = None,
    lines: bool = True,
    unclassified: str = UNCLASSIFIED,
    format: str | None = None,
):
    """Write the dataset's per-sounding attributes out as vector geometry.

    A thin writer over :func:`build_layers`, which is where the geometry is
    actually made. That split is the point: the layers are the product and the
    file is one way of delivering them.

    Parameters
    ----------
    dataset : Dataset
        Whose ``flightlines`` are written. Returned unchanged.
    path : str | Path
        Where to write.
    crs : str
        **Output** CRS, e.g. ``"EPSG:32613"``. Required and never guessed: it
        is a project decision, and a contractor's delivery CRS and a client's
        output CRS are routinely different.
    source_crs : str | None
        What the dataset's coordinates are in. Taken from the dataset if it
        knows; an error if neither says, since reprojecting from an assumed CRS
        produces plausible, wrong positions.
    columns : list[str] | None
        Which attributes to write. All of ``flightlines`` by default.
    lines : bool
        Also write one feature per flightline. GeoPackage takes both layers in
        one file; GeoJSON holds one, so this writes a second file alongside.
    unclassified : str
        What a verdict with no classification is written as.
    format : str | None
        Which format to write — ``gpkg`` or ``geojson``. Taken from the
        extension if unset, which is what a standalone run usually wants.
        Stated explicitly it wins, so a caller serving a chosen format is not
        obliged to encode that choice in a filename. Shapefile is refused.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``. The dataset is returned unchanged — this
        reads and writes files.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    path = Path(path)
    driver = _driver_for(path, format)
    # Only GeoPackage holds several layers in one file; the GeoJSON driver
    # allows neither layers nor appending.
    holds_layers = driver == "GPKG"

    built, notes = build_layers(dataset, crs, source_crs, columns, lines,
                                unclassified)
    report.notes.extend(notes)
    if not built:
        report.notes.append("There is nothing to export; no file was written.")
        return dataset, report

    points = built["soundings"]
    attributes = points.drop(columns=points.geometry.name)

    source = source_crs or dataset.crs
    if driver == "GeoJSON" and str(crs).upper() not in ("EPSG:4326", "OGC:CRS84"):
        report.notes.append(
            f"RFC 7946 mandates WGS84 lon/lat for GeoJSON and removed the crs "
            f"member, so this file records no CRS and readers will assume "
            f"EPSG:4326 while the coordinates are {crs}. GeoJSON is a wire "
            f"format; write GeoPackage to keep a declared CRS."
        )

    written: list[str] = []
    layers = {}
    for index, (name, layer) in enumerate(built.items()):
        if holds_layers:
            layer.to_file(path, driver=driver, layer=name, engine="pyogrio",
                          mode="a" if index else "w")
            written.append(f"{path} :: {name}")
        elif index == 0:
            layer.to_file(path, driver=driver, engine="pyogrio")
            written.append(str(path))
        else:
            # One layer per file, so the lines go beside the points.
            beside = path.with_name(f"{path.stem}_lines{path.suffix}")
            layer.to_file(beside, driver=driver, engine="pyogrio")
            written.append(str(beside))
            report.notes.append(
                f"{driver} holds one layer per file, so the flightlines "
                f"were written beside the soundings as {beside.name}."
            )
        layers[name] = len(layer)

    report.distributions = {
        "files": written,
        "layers": layers,
        "driver": driver,
        "crs": {"source": str(source), "output": str(crs),
                "reprojected": str(source) != str(crs)},
        "unclassified_written_as": unclassified,
        "fields": [str(c) for c in attributes.columns],
    }
    log.info("%s: %s, %s", STEP, driver,
             ", ".join(f"{n} {k}" for k, n in layers.items()))
    return dataset, report
