# Drape clearance, line adherence and coverage

**Status:** spec, not started. Decisions settled 2026-11-14 to 2026-11-16, the later ones after a review against a survey's daily QC.

**Tracking:** #13. Reference files on the platform: YmerFlow/Ymerflow#69.

## What this is

Three acquisition-QC checks, and the route by which their reference files reach them on the platform.

| step | check | reference input |
|---|---|---|
| `drape_clearance` | new; not in `building_plans/03` | one drape grid per flight direction; optionally a terrain grid |
| `line_adherence` | qc_06 | planned lines |
| `line_coverage` | qc_07 | planned lines |

On the platform a reference file is imported once into the project as a dataset, by one of two import processes planned separately, and every step form picks it with the dataset selector the platform already has.

Report pages (PDF, extra GeoPackage layers) are planned separately. This plan puts what they need into columns and step reports and stops there.

## Why it is needed

A survey flown to a drape needs drape compliance graded in its daily QC, and its flight plan checked against what was flown. qc_06 and qc_07 have waited on flight-plan input since the QC package was built (#13), because the platform has no way to hand a step a file that is not survey data. `apply_terrain_model` has the same gap: it takes a raster path, and no platform run has ever given it one.

## The shed

What ships first: the three checks, the shared planned-line reader and matcher, and dataset pickers for raster and vector step inputs. Everything under "Later, once the shed stands" waits until these run on real deliveries.

## Reference files reach the steps as datasets

### In the library

Steps take external inputs typed, as R8 asks and as `apply_terrain_model` and `cull_near_infrastructure` already do: `str | rasterio.io.DatasetReader` for a grid, `str | geopandas.GeoDataFrame` for vector geometry. A path is accepted because a standalone run has nothing else to hand over. Every input defaults to `None`, and a check given none is skipped with a note and marked `OPTIONAL`, so a survey with no drape or no plan reads as not having one, not as ungraded.

No new reader and no new dependency:

- Grids are sampled with `sample_terrain` (`filters/terrain.py`): nearest neighbor, positions reprojected into the grid's CRS when both are declared, `NaN` outside coverage or on nodata. Nearest neighbor for the same reason it is used for terrain: a cell is the surface the grid asserts.
- Vector files are read with `cull_evidence._open`, made public as `read_features`. It reads anything geopandas reads (pyogrio, the engine `export_vector` writes with), refuses geometry with no declared CRS, and reprojects to the survey's CRS.

CRS rules, the same as the existing readers:

- The survey's CRS comes from the dataset (`model_info["projection"]`), which every platform import sets. Channel labels in a delivery are never read for it: one delivery labels its northings with a UTM zone its values are not in.
- Planned lines must declare a CRS and are reprojected to the survey's. The survey's CRS must be known and projected, because deviation and gap length are distances in CRS units. Either missing is refused, as `cull_near_infrastructure` refuses it.
- Grids follow `apply_terrain_model`: with no survey CRS the positions are sampled as they are, and the report says so.

### On the platform

Reference files enter a project through two import processes. They are not AEM-specific, so they go in a new generic package, with a plan of their own:

- A raster import takes anything GDAL reads, converts it to a Cloud-Optimized GeoTIFF and saves it as a dataset. The plot's map tile layer already takes a COG URL.
- A vector import takes anything OGR reads (GeoPackage, KML, a zipped shapefile), converts it to GeoJSON and saves it as a dataset. GeoJSON is the vector form the frontend uses.

Until datasets can be tagged, which is built separately, each import puts `raster` or `vector` at the front of its dataset's name, so typing it in the dataset selector lists only that kind.

Datasets already give a reference file most of what YmerFlow/Ymerflow#69 asks for. They are project-scoped and searchable, and each import is a process version, so a corrected drape is a new version and earlier runs keep pointing at the file they used.

What the QC steps rely on from the imports, which their plan has to keep true:

- The dataset's URL, which is what the dataset selector stores, names the converted file itself, so a step given it downloads one file with the right extension. It is the file `info.json` maps the dataset's own `mime_type` to, as for the gridding process's non-XYZ output.
- A file that declares no CRS is refused at import unless the import is given a CRS to assign, which is then written into the converted file. Drape grids exported as ESRI float grids carry their CRS only in vendor sidecar files GDAL does not read, so assigning one at import is the normal path for them. A multi-file grid (`.flt` with `.hdr`) arrives as a zip.
- A COG keeps the source CRS and nodata. GeoJSON is WGS84 longitude and latitude (RFC 7946), and `read_features` reprojects it to the survey's CRS.
- GeoJSON coordinates keep enough precision that distances survive that round trip to well under a meter. Seven decimal places of a degree is about a centimeter.
- A flight plan's attributes (line id, block) survive the conversion.

In aem-processes, `steps_schema` renders each step parameter that `data_parameters` reports as data, and whose loaded type is a raster or a GeoDataFrame, as a dataset picker: `{"type": "string", "format": "uri", "x-format": "dataset"}` with the docstring's title and description, replacing swaggerspect's `anyOf`. One rule, so `apply_terrain_model.terrain`, `cull_near_infrastructure.infrastructure` and `cull_by_misfit.within` get pickers too. XYZ inputs (`apply_manual_edits.diff`, `cull_by_misfit.result`) are left as they are.

`drape_clearance.drapes` is a list of `DrapeGrid` entries, each holding a grid and its azimuth. `data_parameters` and `steps_schema` are extended to see a data field inside the items of a `list[TypedDict]`: the form renders it as a picker within each item, and `fetch_step_inputs` fetches it like a top-level one. swaggerspect's handling of a `TypedDict` is probed before relying on it; if it does not produce a usable array schema, `steps_schema` writes that one schema itself.

`fetch_step_inputs` (aem-processes #6, merged 2026-10-28) is otherwise unchanged. It already downloads a URL given to a data parameter before the run; the step reads the temporary copy, and the report and the processing history record the URL. The backend translates every URL in a process's parameters to a storage URL, nested ones included, so a dataset picked inside a step reaches the pod as `input_data` does. That covers `apply_terrain_model.terrain` as well as the drapes.

Nothing changes in Ymerflow for the steps: the dataset selector exists, and the custom string field renders it at any depth of a form.

## drape_clearance

A drape is the elevation surface the system should fly at or under. The check grades how far each sounding flew above it.

### Which drape grid

A drape is built for a flight direction, so a survey with production and tie lines has one grid per direction. The check takes them as a list, each with its azimuth, and each flown line uses the grid whose azimuth is closest to the line's own axis, compared modulo 180 degrees. The line's axis is the principal axis of its sounding positions. One grid serves a whole line, so `drape_z` never steps mid-line where two grids disagree. With a single grid the azimuth can be left out.

### Which instrument elevation

The instrument elevation the drape is compared with comes from one of three sources, in this order:

1. A named GPS elevation column (`elevation_column`), the system's own. Never discovered: no spelling reliably separates a GPS elevation from a height, which is the lesson of the `Alt` column graded as height in August. A helicopter system that records the aircraft's GPS and altimeter as well as the transmitter's needs the transmitter's named; the aircraft antenna sits tens of meters higher.
2. Otherwise ground plus altitude, the instrument position libaarhusxyz defines, with ground sampled from a terrain grid (`terrain`) when one is given.
3. Otherwise ground plus altitude with the delivery's own ground column.

The choice matters by meters. On one survey the delivered ground read a median of about 3 m high against a national elevation model, while the transmitter's GPS minus its laser altimeter agreed with that model to under a meter. Ground plus altitude then put every sounding about 3 m higher above the drape than it flew. A terrain grid, ideally the one the drape was built on, removes the delivery's ground error; the named GPS removes the ground from the comparison entirely. Over canopy a laser altimeter reads the treetops, which biases ground plus altitude low by the canopy height, and the GPS frame avoids that too.

When a GPS column is named and ground plus altitude is also available, the report gives the median and robust spread of GPS minus ground plus altitude. A large median with a small spread is a datum or antenna offset; a large spread is ground or canopy error. The verdict uses the GPS either way.

In the third frame, `apply_terrain_model` (default `preserve="elevation"`) holds ground plus altitude fixed, so running it first does not move the verdict.

```python
class DrapeGrid(typing.TypedDict, total=False):
    """One drape grid and the flight direction it was built for.

    ``grid`` is required; ``azimuth`` may be left out when it is the only
    grid. Plain keys rather than a class, so a pipeline document spells it
    as a mapping.
    """
    grid: str | DatasetReader   # drape elevation [m]
    azimuth: float              # flight direction [degrees clockwise from north]


def drape_clearance(
    dataset,
    good_max: float,
    marginal_max: float,
    drapes: list[DrapeGrid] | None = None,
    terrain: str | DatasetReader | None = None,
    elevation_column: str | None = None,
    altitude_column: str | None = None,
    ground_column: str | None = None,
    band: int = 1,
    output_column: str = "qs_drape",
    line_column: str | None = None,
    unit: str = "m",
    gap_tolerance: float = 0.0,
):
    """Classify how far each sounding flew above the drape.

    Parameters
    ----------
    dataset : Dataset
        The dataset to grade. Columns are added; no measurement changes.
    good_max : float
        Height above the drape at or below which a sounding meets spec, in
        ``unit`` [m]. Below the drape is always in spec. A contract term, so
        no default.
    marginal_max : float
        Height above the drape beyond which a sounding is Out of Spec [m].
        Between the two is Marginal.
    drapes : list of DrapeGrid | None
        The drape grids, each an elevation surface [m] in the vertical datum
        and units of the instrument elevation, with the flight azimuth it was
        built for [degrees]. Each flown line uses the grid whose azimuth is
        closest to its axis, modulo 180. Unset or empty, the check is skipped
        with a note.
    terrain : str | rasterio.io.DatasetReader | None
        A terrain grid [m] to take ground elevation from instead of the
        delivery's ground column, ideally the one the drapes were built on.
        Ignored for the comparison when ``elevation_column`` is named.
    elevation_column : str | None
        A column holding the instrument's own GPS elevation [m]. When named,
        the drape is compared with it directly. Never discovered.
    altitude_column : str | None
        Which column holds the instrument's altitude above ground [m].
        Discovered as ``height_above_ground`` discovers it if unset.
    ground_column : str | None
        Which column holds ground elevation [m]. The survey's own if unset.
    band : int
        Which raster band holds elevation, in every grid.
    output_column : str
        Where the per-sounding verdict is written.
    line_column : str | None
        Which column identifies the line. Discovered if unset.
    unit : str
        Vertical unit of the grids and the columns, for report text only.
    gap_tolerance : float
        In-spec stretches shorter than this, in CRS units [m], do not break a
        run when contiguous runs are measured. ``0`` measures runs raw.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
```

It writes five columns:

| column | meaning |
|---|---|
| `drape_az` | azimuth of the grid the sounding's line used [degrees] |
| `drape_z` | drape elevation sampled at the sounding [m] |
| `drape_clr` | the drape's clearance over the ground, `drape_z - ground` [m], with ground from the terrain grid or the delivery |
| `drape_dz` | height above the drape, instrument elevation minus `drape_z` [m]; positive is above |
| `qs_drape` | verdict, `ascending` on `drape_dz` |

So a plot can show altitude against the drape's clearance, or the instrument elevation against the drape.

Per line, the report gives the standard tier counts and out-of-spec runs from `aggregate.per_line`, and adds to each line's entry the grid it used, the angle between the line's axis and that grid's azimuth, and two threshold-free measures: the fraction of soundings above the drape (`drape_dz > 0`) and the longest contiguous run above it, in soundings and on the ground. The run is the same measure, from the same function, applied to the above-the-drape flag. Kept in `per_line` so anything that reads per-line measures from step reports finds them there. The report also names the instrument-elevation frame and the columns it used.

Altitude is found by the discovery `height_above_ground` uses, and the same guard warns when the column found looks like elevation. Both are imported from `qc/height.py`, not repeated.

Absent inputs: no drapes is skipped with a note and marked `OPTIONAL`. Drapes with no way to form an instrument elevation (no GPS named, and no altitude or no ground) are skipped with a note and marked `MISSING`, because then the input is there and the delivery is not gradable. A column the caller named and that is absent raises, as everywhere else. Soundings outside their grid or on nodata get `NaN` and are counted in the report.

The grids' vertical unit is not checked, because rasters rarely state one reliably. The report gives the median and spread of `drape_clr`, where a feet-against-meters mix shows as an implausible clearance.

## Planned lines and line matching

Shared by qc_06 and qc_07, in `qc/planned_lines.py`.

```python
def read_planned_lines(source, crs, line_attribute=None, block_attribute=None):
    """Planned lines in ``crs``, each with a numeric line id and its spacing.

    Parameters
    ----------
    source : str | geopandas.GeoDataFrame
        The flight plan: line geometry with a line-id attribute. A path is
        read with ``read_features``.
    crs : str
        The survey's CRS. Must be projected; distances are in its units [m
        for a metric CRS].
    line_attribute : str | None
        Which attribute holds the planned line id. Discovered
        case-insensitively from common spellings if unset.
    block_attribute : str | None
        Which attribute names each line's block, so spacing is measured per
        block. Unset, the whole plan is one block.

    Returns
    -------
    tuple
        ``(GeoDataFrame, notes)``. The frame gains a float ``plan_id``, a
        ``block``, and ``spacing``, the line spacing of its block [m].

    Raises
    ------
    ValueError
        If ``crs`` is unknown or geographic, or the lines declare no CRS.
    """


def match_lines(easting, northing, lines, planned, max_offset=None):
    """The planned line each flown line follows.

    Parameters
    ----------
    easting, northing : array_like
        Sounding positions in the planned lines' CRS [m].
    lines : array_like
        Flown line id per sounding.
    planned : geopandas.GeoDataFrame
        From ``read_planned_lines``.
    max_offset : float | None
        Largest median distance at which a flown line is matched [m]. Unset,
        half the spacing of the candidate planned line's block.

    Returns
    -------
    tuple
        ``(plan_id per sounding, NaN where unmatched; per-flown-line table)``.
    """
```

Matching is geometric and per flown line, so it is untouched by numbering that differs between plan and delivery, and it has no direction test:

1. For each flown line, the median distance from a subsample of its soundings to each planned line is measured.
2. The planned line with the smallest median distance wins. A line crossing the flown line is close to it only near the crossing, so its median distance is large; that is what keeps a tie line from matching the production lines it crosses, without comparing directions.
3. A flown line whose best median distance exceeds `max_offset` is unmatched: not graded, and listed in the report. That is what keeps a test or calibration line flown off the block from being graded against the nearest edge line.

There is no direction test because a planned line need not be straight. A tie line that bends partway has a whole-line axis well away from either of its stretches, and comparing that axis with the flown line's left one such line unmatched in a review against real data. Median distance alone matched every line of that survey.

Several flown lines may match one planned line (a line flown in parts, or reflown); each flown line matches at most one. The report's table lists each flown line with its matched planned line, both ids and the median offset, so a wrong match is visible.

Line spacing is measured from the plan: for each planned line, the median distance from points sampled along it to its nearest other planned line by that same median measure, so a crossing line never counts as a neighbor. A block's spacing is the median of its lines', which its production lines dominate. The report gives each block's spacing.

The planned line id is stored as a float, because the plot layer shows only numeric columns and the XYZ format cannot carry text with spaces. A numeric attribute is used as it is. A text attribute uses its trailing integer when every id ends in one and those integers are unique (a block name, an underscore and a four-digit number, say), and the feature's index otherwise. Either way the report carries the table from the float back to the original text.

The line checks need a flown line id. Without one they are skipped with a note and marked `MISSING`.

## line_adherence (qc_06)

```python
def line_adherence(
    dataset,
    good_max: float,
    marginal_max: float,
    planned_lines: str | GeoDataFrame | None = None,
    per_spacing: bool = False,
    line_attribute: str | None = None,
    block_attribute: str | None = None,
    max_offset: float | None = None,
    output_column: str = "qs_adhere",
    deviation_column: str = "plan_dev",
    fraction_column: str = "plan_devf",
    plan_column: str = "plan_line",
    line_column: str | None = None,
    gap_tolerance: float = 0.0,
):
    """Classify each sounding's distance from its planned line.

    Parameters
    ----------
    dataset : Dataset
        The dataset to grade. Columns are added; no measurement changes.
    good_max : float
        Distance from the planned line at or below which a sounding meets
        spec: in CRS units [m], or with ``per_spacing`` a fraction of the
        block's line spacing [-]. A contract term, so no default.
    marginal_max : float
        Distance beyond which a sounding is Out of Spec, in the same terms.
    planned_lines : str | geopandas.GeoDataFrame | None
        The flight plan. Unset, the check is skipped with a note.
    per_spacing : bool
        Read the thresholds as fractions of the planned line spacing of each
        line's block, measured from the plan.
    line_attribute, block_attribute, max_offset
        As for ``read_planned_lines`` and ``match_lines``.
    output_column : str
        Where the per-sounding verdict is written.
    deviation_column : str
        Where the distance to the matched planned line is written [m].
    fraction_column : str
        Where that distance as a fraction of the block's spacing is written
        [-]. Written only with ``per_spacing``.
    plan_column : str
        Where the matched planned line id is written.
    line_column : str | None
        Which column identifies the flown line. Discovered if unset.
    gap_tolerance : float
        In-spec stretches shorter than this [m] do not break a run.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
```

Deviation is the distance from each sounding to its matched planned line, unsigned. A sounding whose nearest point on the planned line is one of its ends, lying before the start or past the end (run-in and run-out), is not graded: deviation means nothing off the end of a line. Such soundings get `NaN` and are counted. Unmatched lines get `NaN` throughout.

With `per_spacing`, the thresholds scale with each block's spacing, so one step serves a plan whose blocks are flown at different spacings. The report states the thresholds in meters as applied in each block.

Per line, the report gives the tier counts and out-of-spec runs from `aggregate.per_line`, and the matching table above.

## line_coverage (qc_07)

```python
def line_coverage(
    dataset,
    good_max: float,
    marginal_max: float,
    planned_lines: str | GeoDataFrame | None = None,
    corridor: float | None = None,
    line_attribute: str | None = None,
    block_attribute: str | None = None,
    max_offset: float | None = None,
    output_column: str = "qs_cover",
    gap_column: str = "plan_gap",
    line_column: str | None = None,
):
    """Classify each planned line by the longest hole in its flown coverage.

    Parameters
    ----------
    dataset : Dataset
        The dataset to grade. Columns are added; no measurement changes.
    good_max : float
        Longest interior gap along a planned line at or below which it meets
        spec, in CRS units [m]. Also the shortest stretch listed as a gap. A
        contract term, so no default.
    marginal_max : float
        Longest interior gap beyond which the planned line is Out of Spec [m].
    planned_lines : str | geopandas.GeoDataFrame | None
        The flight plan. Unset, the check is skipped with a note.
    corridor : float | None
        Largest distance from the planned line [m] at which a matched
        sounding counts as covering it. Unset, every matched sounding counts.
    line_attribute, block_attribute, max_offset
        As for ``read_planned_lines`` and ``match_lines``.
    output_column : str
        Where the verdict is written, on every sounding of the planned line.
    gap_column : str
        Where the planned line's longest interior gap is written [m].
    line_column : str | None
        Which column identifies the flown line. Discovered if unset.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
```

How it measures:

1. Only planned lines matched by at least one flown line in this dataset are graded. The rest are listed as not flown here, with their count and total length. On a daily delivery most of the plan is unflown and that is no defect. Run on the survey to date, the list is the work remaining.
2. Every covering sounding (every matched one, or with `corridor` those within it of the planned line) is projected onto its planned line, giving its distance along the line, clipped to the line's length. Soundings of every flown line matched to one planned line count together, so a line flown in two parts is covered by both.
3. The interior gaps are the stretches between consecutive along-line positions. The longest is graded, `ascending`.
4. The start and end gaps, from the start of the planned line to the first covering sounding and from the last to its end, are listed and counted in the covered fraction, and not graded. A contractor often flies a shortened version of the client's plan, starting kilometers in from its ends, and that is a difference of plans; an unfinished line still shows, in the gap list and the covered fraction.
5. The report lists every gap longer than `good_max`: planned line, from and to along the line and as coordinates, length, and whether it sits at the start, in the interior or at the end. It also gives each planned line's covered fraction.
6. A matched planned line with no covering sounding (all of it outside the corridor) gets no verdict and is listed as such.

The verdict is constant along a planned line and written on every sounding matched to it, because a gap has no soundings to carry it. It reaches `qs_verdict` and the exit code like any per-line verdict denormalized onto soundings. Where on the line the hole is comes from the gap list.

This departs from `building_plans/03`, which gives qc_07 a `descending` threshold on coverage. A contract states a maximum gap, and one long hole in an otherwise covered line is what triggers infill, while a coverage fraction can pass it.

Without a corridor, lateral offset plays no part here: a stretch flown far off the line still covers it, and is out of spec in qc_06. The corridor is for a survey that does not count such a stretch as coverage.

qc_10 (`sounding_spacing`) and this check overlap on gaps inside one flown line, at different grains: spacing grades each sounding's distance to its neighbor along the flown line, and coverage grades the planned line, including the joins between separately flown parts.

## Rollup and the bundled pipeline

`DEFAULT_WEIGHTS` gains three entries, all three `flight` findings:

| step | weight | why |
|---|---|---|
| `drape_clearance` | 2.0 | as `height_above_ground`: flying high costs signal at depth and widens the footprint |
| `line_adherence` | 0.25 | as speed and spacing: it costs coverage and leaves what a sounding means unchanged |
| `line_coverage` | 0.25 | likewise |

A drape survey can weight `height_above_ground` down in its own pipeline's `rollup:` block, since the two checks grade the same concern against different surfaces.

`acquisition-qc.yaml` lists all three under "Flown as contracted", with `drapes` and `planned_lines` unset and representative thresholds under the file's existing warning: `drape_clearance` 15 and 30 m above the drape; `line_adherence` with `per_spacing: true`, 1/8 and 1/4 of the line spacing; `line_coverage` 100 and 200 m of interior gap. With no input they skip as optional, and the platform form opens with the dataset pickers in place.

## Decisions

1. Drape grids are per flight direction: the check takes a list of grids with their azimuths, and each flown line uses the grid closest to its axis, modulo 180. One grid per line keeps `drape_z` continuous along it.
2. The instrument elevation is a named GPS column when one is given, and otherwise ground plus altitude, with ground from a terrain grid when one is given and from the delivery otherwise. A delivery's ground can be meters off, and that error lands in the verdict.
3. One check samples the drape and grades it, the shape every existing check has.
4. The input is the drape elevation grid; clearance is computed against the ground in use.
5. Flown lines are matched to the planned line with the smallest median distance, per flown line, with no direction test, because numbering differs between plan and delivery and planned lines can bend.
6. qc_07 grades the longest interior gap per planned line, `ascending`. Start and end gaps are listed and counted in the covered fraction. An optional corridor limits which soundings count as covering.
7. Deviation thresholds are in meters, or with `per_spacing` fractions of the block's line spacing measured from the plan.
8. Reference files are project datasets made by two import processes, raster to COG and vector to GeoJSON, in a new generic package with a plan of their own. Step forms pick them with the dataset selector. The library reads whatever rasterio and geopandas read from a path.
9. A file with no declared CRS is refused at import unless the import is given one to assign. Vectors are reprojected to the survey's CRS, which must be projected.
10. Default weights are 2.0, 0.25 and 0.25, and all three are flight findings.
11. The bundled pipeline lists the three checks with their inputs unset.

## Later, once the shed stands

- Drape grid chosen per stretch of line rather than per line, by the local heading over a window, for a line that bends between two flight directions. The cost is a step in `drape_z` where the grid switches.
- A drape clearance grid, or a drape column the delivery carries, as the input instead of the elevation grids.
- Signed deviation, left or right of the planned direction, so a systematic offset (wind, a GPS bias) shows as a sign.
- Dataset pickers for XYZ step inputs such as `cull_by_misfit.result`.

## Dependencies outside this library

- The two import processes, in a new generic package, planned on their own, including assigning a CRS to a file that declares none. The checks run from the command line without them; on the platform they are how the drapes, the terrain grid and the flight plan arrive.
- aem-processes: the dataset pickers in `steps_schema`, including data fields inside list items.
- Dataset tagging, built separately. The `raster` and `vector` name prefix stands in until then.
- A platform environment built with these versions.

## Work (this library)

1. `read_features`: `cull_evidence._open` renamed and made public in place; `cull_near_infrastructure` uses it unchanged. Fix `sample_terrain`'s docstring in passing: it returns `(elevation, crs)`, where the docstring says an array.
2. `qc/planned_lines.py`: `read_planned_lines` (line ids, blocks, spacing), `match_lines`, and the along-line projection both line checks use (shapely `line_locate_point`).
3. `qc/height.py`: export the altitude discovery and the looks-like-elevation guard for `drape_clearance` to import.
4. `qc/drape.py`: `DrapeGrid` and `drape_clearance`.
5. `qc/adherence.py`: `line_adherence`.
6. `qc/coverage.py`: `line_coverage`.
7. `qc/rollup.py`: the three `DEFAULT_WEIGHTS` entries with their reasons in the comment block.
8. `setup.py`: the three entry points under `ymerflow.processing.qc`.
9. `pipelines/acquisition-qc.yaml`: the three steps as above.
10. Regenerate `docs/reference/qc.md` and `llms.txt`; README entries; a dated addition to `building_plans/03` recording the drape check and qc_07's graded quantity.

## Work (aem-processes)

11. `steps_schema`: raster and vector data parameters rendered as dataset pickers, and `data_parameters`, `steps_schema` and `fetch_step_inputs` extended to data fields inside the items of a `list[TypedDict]`.

## Tests

Library, on synthetic surveys, grids and plans written to a temporary directory:

12. `drape_clearance`, grids: with two grids at different azimuths each line uses the closer one, modulo 180, and `drape_az` says which; a single grid with no azimuth serves every line; `NaN` outside a grid, counted; a grid in another CRS is sampled at the right place; no drapes skips as `OPTIONAL` and writes nothing.
13. `drape_clearance`, frames: with a named GPS column `drape_dz` is GPS minus the drape; with a terrain grid it is the grid's ground plus altitude minus the drape; otherwise the delivery's ground plus altitude minus the drape, unchanged by running `apply_terrain_model` first; with GPS and ground plus altitude both available the report gives their difference, and a GPS column 30 m high reads as a large median with a small spread; no GPS and no altitude skips as `MISSING`; tiers at the thresholds, and the per-line fraction above and longest run above, raw and with `gap_tolerance`.
14. Matching: renumbered flown lines match their planned lines; a tie line does not match the production lines it crosses; a bent flown line matches its bent planned line; an off-block line beyond `max_offset` is unmatched and listed; spacing per block is the median nearest-neighbor distance and ignores crossing lines; text ids use their trailing integers, falling back to the index when those are missing or repeated.
15. `line_adherence`: a known offset is recovered; run-in and run-out soundings are not graded; with `per_spacing` two blocks at different spacings get thresholds scaled to each, and `plan_devf` is written; a geographic survey CRS and planned lines with no CRS are refused.
16. `line_coverage`: interior gaps of known length set the verdict; start and end gaps are listed, counted in the covered fraction and not graded; a planned line flown in two parts is covered by both; a corridor drops soundings outside it from coverage, and a line with nothing inside it gets no verdict; planned lines not flown are listed and not graded.
17. The same plan read from GeoPackage and from KML (where the bundled GDAL reads it, skipped otherwise) gives the same matches, and written as RFC 7946 GeoJSON (WGS84, seven decimals) it gives the same matches with deviations within a centimeter.
18. The bundled pipeline runs on a delivery with neither input, marks the three checks optional and leaves the run's ungraded list empty; the rollup carries the three weights.
19. The generated step forms, probed through swaggerspect, validate sample values for every new parameter, the list of drape grids included.

aem-processes:

20. `steps_schema`: `terrain`, `planned_lines`, `infrastructure` and `within` are dataset pickers, and so is `grid` in each item of `drapes`; XYZ inputs are unchanged.
21. A QC pipeline given dataset URLs naming two drape COGs and a GeoJSON plan fetches all three before the run, grades with them, and records the URLs in the report.
