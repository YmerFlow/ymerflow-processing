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

GeoPackage by default
---------------------
Shapefile only when a contract names it. Everything awkward below is a shapefile
limitation that GeoPackage does not have — it carries a real CRS, an R-tree
index, unlimited field names, NULLs, several geometry types in one file, and is
several times smaller.

The one that actually costs something: **shapefile cannot distinguish NULL from
empty.** That collides with ``03``'s deliberate rule that a metric with no
threshold configured gets *no classification at all* rather than a guessed
"Good". If unclassified and Good arrive at a client as the same thing, the
distinction is lost in the one artifact they read. So unclassified is written as
an **explicit label**, and the report says what that label was.

Both formats write the same label rather than GeoPackage carrying a true NULL:
two exports of one dataset disagreeing about what "not measured" looks like
would be worse than GeoPackage under-using its own capability.

Field names are also limited to ten characters in a shapefile, and two names
sharing their first ten collide *silently* — one replacing the other. That is
checked before writing rather than discovered afterwards in a deliverable.

The output CRS is explicit, always
----------------------------------
``03``: the output CRS is a project decision and cannot be derived from the
input. Contractor delivery CRS and client output CRS are routinely different, so
this takes the output CRS as a required parameter and **fails rather than
guessing**. Reprojection needs the source CRS too; that comes from the dataset,
and its absence is an error rather than an assumption that the two match.

Sources
-------
Shapefile's limits are from the format specification itself — ESRI, *ESRI
Shapefile Technical Description*, 1998 — and the DBF field-name and type limits
it inherits. GeoPackage is OGC 12-128r19.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .pipeline import StepReport
from .qc.thresholds import labels, verdict_columns

log = logging.getLogger(__name__)

STEP = "export_vector"

#: Output extension to OGR driver. GeoPackage first because it is the default.
DRIVERS = {".gpkg": "GPKG", ".shp": "ESRI Shapefile", ".geojson": "GeoJSON",
           ".json": "GeoJSON"}

#: Shapefile field names are limited to this, and names sharing their first
#: this-many collide silently.
SHAPEFILE_NAME_LIMIT = 10

#: A shapefile component that reaches this is unreadable by much software.
SHAPEFILE_SIZE_LIMIT = 2 * 1024 ** 3

#: What "no classification" is written as, since a shapefile cannot express the
#: absence of a value. See the module docstring.
UNCLASSIFIED = "Unclassified"


#: Format names, for a caller that chooses one rather than implying it through
#: a filename. R8 says format is a loader concern on the way in; on the way out
#: it is a delivery concern, and neither is a property of the processing.
FORMATS = {"gpkg": "GPKG", "shp": "ESRI Shapefile", "geojson": "GeoJSON",
           "json": "GeoJSON"}


def _driver_for(path: Path, format: str | None = None) -> str:
    """The OGR driver, from a stated format or failing that the extension."""
    if format is not None:
        name = str(format).lower().lstrip(".")
        if name not in FORMATS:
            raise ValueError(
                f"Unknown format {format!r}. Available: "
                f"{', '.join(sorted(FORMATS))} — gpkg unless a contract names "
                f"something else."
            )
        return FORMATS[name]

    suffix = path.suffix.lower()
    if suffix not in DRIVERS:
        raise ValueError(
            f"Cannot tell a format from {path.name!r}, and none was given. Use "
            f"one of {', '.join(sorted(DRIVERS))}, or pass format= — .gpkg "
            f"unless a contract names something else."
        )
    return DRIVERS[suffix]


def truncation_collisions(columns, limit: int = SHAPEFILE_NAME_LIMIT) -> dict:
    """Names that would become the same field once a shapefile truncates them.

    Silent in the format itself: one column simply replaces the other in the
    written file, and nothing reports it. Worth catching before a client opens
    the deliverable rather than after.
    """
    seen: dict[str, list[str]] = {}
    for name in columns:
        seen.setdefault(str(name)[:limit], []).append(str(name))
    return {short: names for short, names in seen.items() if len(names) > 1}


def _labelled(frame: pd.DataFrame, verdicts: list[str],
              unclassified: str) -> pd.DataFrame:
    """Verdict codes written back as tier names, absences spelled out.

    The one place the numeric encoding is undone. Verdicts are stored as codes
    because a string column is invisible to the plot layer, but this file is
    opened in a GIS by a person, and ``0`` in a legend is worse than useless
    where ``OOS`` is self-explanatory.

    Unclassified becomes an explicit label rather than a null, because shapefile
    cannot distinguish NULL from empty — "not measured" and "measured and fine"
    would arrive as the same thing in the one artifact a client actually reads.
    Both drivers write the same label so two exports of one dataset cannot
    disagree.

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
                            if str(c).lower() in ("flight", "flight_no",
                                                  "flight_id")]
    rows, geometries = [], []
    for line, block in frame.groupby(line_column, sort=True, observed=True):
        geometry = geometry_of(block)
        if geometry is None:
            continue
        record = {c: block[c].iloc[0] for c in dict.fromkeys(keep + constant)}
        # Seven characters: the same discipline the checks keep.
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

    Splitting it means the platform can build once and serve GeoPackage,
    shapefile or GeoJSON on demand without re-running anything, and the
    standalone CLI can still write a file. Both read the same layers, so the two
    routes cannot drift.

    Returns ``({name: GeoDataFrame}, notes)``.
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

    :param path: Where to write.
    :param crs: **Output** CRS, e.g. ``"EPSG:32613"``. Required and never
        guessed: it is a project decision, and a contractor's delivery CRS and a
        client's output CRS are routinely different.
    :param source_crs: What the dataset's coordinates are in. Taken from the
        dataset if it knows; an error if neither says, since reprojecting from
        an assumed CRS produces plausible, wrong positions.
    :param columns: Which attributes to write. All of ``flightlines`` by default.
    :param lines: Also write one feature per flightline. GeoPackage takes both
        layers in one file; a shapefile holds one geometry type, so this writes
        a second file alongside.
    :param unclassified: What a verdict with no classification is written as.
    :param format: Which format to write — ``gpkg``, ``shp`` or ``geojson``.
        Taken from the extension if unset, which is what a standalone run
        usually wants. Stated explicitly it wins, so a caller serving a chosen
        format is not obliged to encode that choice in a filename.

    Returns ``(dataset, StepReport)``. The dataset is returned unchanged — this
    reads and writes files.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    path = Path(path)
    driver = _driver_for(path, format)
    is_shapefile = driver == "ESRI Shapefile"
    # Only GeoPackage holds several layers in one file. Shapefile allows one
    # geometry type; the GeoJSON driver allows neither layers nor appending.
    holds_layers = driver == "GPKG"

    built, notes = build_layers(dataset, crs, source_crs, columns, lines,
                                unclassified)
    report.notes.extend(notes)
    if not built:
        report.notes.append("There is nothing to export; no file was written.")
        return dataset, report

    points = built["soundings"]
    attributes = points.drop(columns=points.geometry.name)

    if is_shapefile:
        collisions = truncation_collisions(attributes.columns)
        if collisions:
            detail = "; ".join(f"{short!r} <- {', '.join(names)}"
                               for short, names in collisions.items())
            raise ValueError(
                f"A shapefile truncates field names to {SHAPEFILE_NAME_LIMIT} "
                f"characters, and these would collide: {detail}. One would "
                f"silently replace the other. Rename them, select fewer with "
                f"'columns:', or write GeoPackage instead."
            )
        long_names = [c for c in attributes.columns
                      if len(str(c)) > SHAPEFILE_NAME_LIMIT]
        if long_names:
            report.notes.append(
                f"{len(long_names)} field name(s) will be truncated to "
                f"{SHAPEFILE_NAME_LIMIT} characters: "
                f"{', '.join(map(str, long_names))}."
            )

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
            if is_shapefile:
                _write_encoding_sidecar(path)
            written.append(str(path))
        else:
            # One geometry type per file, so the lines go beside the points.
            beside = path.with_name(f"{path.stem}_lines{path.suffix}")
            layer.to_file(beside, driver=driver, engine="pyogrio")
            if is_shapefile:
                _write_encoding_sidecar(beside)
            written.append(str(beside))
            report.notes.append(
                f"{driver} holds one layer per file, so the flightlines "
                f"were written beside the soundings as {beside.name}."
            )
        layers[name] = len(layer)

    _warn_on_size(path, report)

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



def _write_encoding_sidecar(path: Path) -> None:
    """Declare UTF-8 beside a shapefile.

    DBF's encoding is ambiguous and readers guess differently, so a ``.cpg``
    stating it is the difference between a client seeing a degree sign and
    seeing mojibake.
    """
    path.with_suffix(".cpg").write_text("UTF-8")


def _warn_on_size(path: Path, report: StepReport) -> None:
    """Note when a shapefile component approaches the size the format allows.

    Each of .shp and .dbf is capped at 2 GB by 32-bit offsets. Writing past it
    produces a file that opens and is wrong, so it is worth saying before that
    rather than after.
    """
    for component in path.parent.glob(f"{path.stem}.*"):
        size = component.stat().st_size
        if size > SHAPEFILE_SIZE_LIMIT * 0.9:
            report.notes.append(
                f"{component.name} is {size / 1024 ** 3:.1f} GB, near the 2 GB "
                "limit a shapefile component allows. Write GeoPackage instead."
            )
