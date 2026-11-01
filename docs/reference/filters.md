# filters

*Generated from the entry-point registry — do not edit by hand.*

dataset in, dataset out. These change the data.

33 step(s).

## selection

### `classify_lines`

Label flightlines, without removing anything.

```python
classify_lines(criterion: str, line_type: str, threshold: float | None = None, lines: list[int | str] | None = None, altitude_column: str | None = None, line_column: str | None = None, default_line_type: str = 'Production')
```

| parameter | default | |
|---|---|---|
| `criterion` | `required` | ``height_threshold`` — label lines whose *mean* height exceeds ``threshold``, which is how high-altitude test lines are found; or ``explicit`` — label the lines named in ``lines``. |
| `line_type` | `required` | The label to apply. Free-form, but see the module docstring: no whitespace. |
| `threshold` | `None` | Height above ground, for ``height_threshold``. |
| `lines` | `None` | Line identifiers, for ``explicit``. |
| `altitude_column` | `None` | Which column holds height. Discovered if unset. |
| `line_column` | `None` | Which column identifies the line. Discovered if unset. |
| `default_line_type` | `'Production'` | What unclassified lines are called. |

### `select_lines`

Reduce the dataset to the flightlines wanted. **Drops soundings.**

```python
select_lines(criterion: str, line_types: list[str] | None = None, lines: list[int | str] | None = None, line_column: str | None = None)
```

| parameter | default | |
|---|---|---|
| `criterion` | `required` | ``line_type`` — keep lines carrying one of ``line_types``; or ``explicit_lines`` — keep the lines named in ``lines``. |
| `line_types` | `None` | Classifications to keep, for ``line_type``. |
| `lines` | `None` | Line identifiers to keep, for ``explicit_lines``. |
| `line_column` | `None` | Which column identifies the line. Discovered if unset. |

### `select_moments`

Keep only the channels whose ``criterion`` value is in ``keep``.

```python
select_moments(criterion: str, keep: list[str], drop_system_blocks: bool = True, drop_unclaimed: bool = False)
```

| parameter | default | |
|---|---|---|
| `criterion` | `required` | How channels are chosen — ``moment`` (the transmitter moment name, such as ``LM`` or ``HM``) or ``component`` (``x``/``y``/``z``, from the receiver polarization). Both are read from the system description, so a dataset without one cannot be selected on. |
| `keep` | `required` | Which values to keep. Matched case-insensitively, so ``Z`` and ``z`` are the same request. |
| `drop_system_blocks` | `True` | Also remove the discarded channels' blocks from the system description, so it does not go on declaring channels whose data has gone. Set false to leave the system description alone. |
| `drop_unclaimed` | `False` | Also remove gate-shaped arrays that no channel claims — see :func:`unclaimed_arrays`. Off by default: such an array is undeclared rather than empty, and at least one of them is real coherent signal that the incumbent toolchain merely discards, so deleting it unasked would be worse than leaving it. It is always reported. |

## corrections

### `apply_nominal_current`

Overwrite the per-sounding transmitter current with the declared nominal.

```python
apply_nominal_current(columns: str | dict[str, str] | None = None, stash: bool = True)
```

| parameter | default | |
|---|---|---|
| `columns` | `None` | Which ``flightlines`` column holds each channel's current, per channel key or moment name. Discovered if unset. |
| `stash` | `True` | Keep the measured values alongside, under the column's name with ``orig_`` in front. |

### `apply_terrain_model`

Re-express altitude against a terrain model.

```python
apply_terrain_model(terrain: str | rasterio.io.DatasetReader, crs: str | None = None, band: int = 1, preserve: str = 'elevation', altitude_column: str | None = None, elevation_column: str | None = None, stash: bool = True)
```

| parameter | default | |
|---|---|---|
| `terrain` | `required` | The terrain model — an open ``rasterio`` dataset or a path. Its values are ground elevation in the same vertical units as the delivery. |
| `crs` | `None` | The survey's horizontal CRS, needed only if it differs from the raster's. Taken from the dataset when it carries one. |
| `band` | `1` | Which raster band holds elevation. |
| `preserve` | `'elevation'` | ``elevation`` — hold the instrument's elevation and recompute altitude, which is the default and almost always right; or ``altitude`` — trust the altimeter and move the instrument instead. |
| `altitude_column` | `None` | Which column holds altitude. Discovered if unset. |
| `elevation_column` | `None` | Which holds ground elevation. Discovered if unset. |
| `stash` | `True` | Keep the delivered values under the same column names with ``orig_`` in front. |

### `correct_tilt_1d`

Rescale amplitudes to what a horizontal transmitter would have measured, by 1/cos².

```python
correct_tilt_1d(pitch_column: str | None = None, roll_column: str | None = None, max_correction: float = 4.0, again: bool = False, stash: bool = True)
```

| parameter | default | |
|---|---|---|
| `pitch_column` | `None` | Which column holds pitch, along the flight direction. Discovered if unset; a delivery with none is skipped with a note. |
| `roll_column` | `None` | Which holds roll, across it. Likewise. |
| `max_correction` | `4.0` | Largest factor to apply. Beyond this the first-order treatment has stopped being defensible, so the sounding is left uncorrected and reported rather than scaled by a number the physics does not support. ``4.0`` is 60° of total tilt (``1/cos² 60° = 4``). |
| `again` | `False` | Permit correcting a dataset that already carries a correction. |
| `stash` | `True` | Keep the delivered values and the measured attitude alongside, under their own names with ``orig_`` in front. |

### `zero_attitude`

Force the recorded attitude to zero, keeping the originals.

```python
zero_attitude(pitch_column: str | None = None, roll_column: str | None = None, stash: bool = True)
```

| parameter | default | |
|---|---|---|
| `pitch_column` | `None` | Which column holds pitch. Discovered if unset; created under the canonical spelling if absent. |
| `roll_column` | `None` | Which holds roll. Likewise. |
| `stash` | `True` | Keep the delivered attitude alongside, under the column names with ``orig_`` in front. Where the columns did not exist there is nothing to keep, and that is reported. |

## noise

### `assign_uncertainty`

Replace uncertainties with the noise floor and a proportional error.

```python
assign_uncertainty(relative_fraction: float | None = None, stash: bool = True)
```

| parameter | default | |
|---|---|---|
| `relative_fraction` | `None` | The proportional error as a fraction of signal — the "3 %" of an inversion's error model, combined with the floor in quadrature. Absent takes each channel's ``UniformDataSTD`` from the system description, and a description that declares none gives a floor-only uncertainty, which is reported. **Never estimated** — it has no separable signature in the data, and estimating it would invent a number with a measurement's authority. |
| `stash` | `True` | Keep the delivered uncertainties alongside, under the same key with ``orig_`` in front. This is the first step that overwrites something an inversion weights by, so what was delivered is kept. |

### `compute_misfit`

Write misfit between the data and a forward synthetic as columns.

```python
compute_misfit(result: str | libaarhusxyz.xyz.XYZ | ymerflow_processing.dataset.Dataset, rms_column: str = 'resdata')
```

| parameter | default | |
|---|---|---|
| `result` | `required` | A previous inversion's or forward model's output — an XYZ, a Dataset, or a path to one. |
| `rms_column` | `'resdata'` | Where the per-sounding root-mean-square misfit goes. |

### `inflate_uncertainty`

Add ``fraction`` to the uncertainty of the nominated gates.

```python
inflate_uncertainty(fraction: float | dict[str, float], gates: int | str | list[int | str] | dict | None = None)
```

| parameter | default | |
|---|---|---|
| `fraction` | `required` | How much to add, as a fraction — ``0.02`` is two percentage points. Negative reduces. Per channel key or moment name (``{"ch1": 0.02}``), or one value for every channel. |
| `gates` | `None` | Which gates, per channel or one nomination for all. A single index, a list, or an inclusive ``"first-last"`` range. Every gate if unset. |

### `set_noise_floor`

Set the dataset's noise floor — supplied or estimated — for every consumer.

```python
set_noise_floor(scope: str = 'per_dataset', noise_level_1ms: float | None = None, noise_exponent: float = -0.5, fit_exponent: bool = False, late_gates: int | None = None)
```

| parameter | default | |
|---|---|---|
| `scope` | `'per_dataset'` | ``per_line``, ``per_flight`` or ``per_dataset`` — the grain an estimate is made over. Not "per survey": this library has no concept of one, ``Survey`` is already the data object's class, and a campaign may map to one dataset, part of one, or several. |
| `noise_level_1ms` | `None` | The absolute floor at 1 ms in physical units (V/Am², before moment normalization) — the instrument's own dB/dt noise, the same number SimPEG's ``uncertainties__noise_level_1ms`` takes. Applied to the data it is divided by each channel's dipole moment and compared against the physical form ``dbdt × scalefactor × GateFactor``. Absent means estimate it from the late-gate scatter, and an estimated floor is in the raw arrays' own units instead. |
| `noise_exponent` | `-0.5` | The power-law slope of the floor in time. ``-0.5`` is what ambient noise does for every airborne TEM system in use, so it is a fixed number rather than an estimate. |
| `fit_exponent` | `False` | Fit the slope from the data instead of using ``noise_exponent``. For checking the assumption: a fitted value far from ``-0.5`` usually means the gates fitted over are not ambient noise. The fitted value is reported under ``measured`` whether or not it is applied. |
| `late_gates` | `None` | How many trailing gates to estimate from. Enough to fit a slope, few enough to be noise-dominated; the last third by default. The per-gate scatter is reported so the choice is checkable rather than trusted. |

## cull soundings

### `cull_by_altitude_tilt`

Disable soundings flown outside the geometry the forward model assumes.

```python
cull_by_altitude_tilt(max_pitch: float | None = None, max_roll: float | None = None, max_altitude: float | None = None, min_altitude: float | None = None, pitch_column: str | None = None, roll_column: str | None = None, altitude_column: str | None = None)
```

| parameter | default | |
|---|---|---|
| `max_pitch` | `None` | Beyond ±this, in the attitude column's units, the sounding is disabled. Defaults to ``system_attitude``'s out-of-spec boundary if that check has run. |
| `max_roll` | `None` | The same, across the flight direction. |
| `max_altitude` | `None` | Above this the sounding is disabled. Defaults to ``height_above_ground``'s out-of-spec boundary if recorded. |
| `min_altitude` | `None` | Below this the sounding is disabled — a reading too low to be real usually means the altimeter, not the aircraft. **No default:** qc_01 is one-sided and has no minimum to borrow. |
| `pitch_column` | `None` | Which ``flightlines`` column holds pitch. Discovered if unset, under the same spellings ``system_attitude`` looks for. |
| `roll_column` | `None` | Which holds roll. Discovered likewise. |
| `altitude_column` | `None` | Which holds height above ground. Discovered if unset. |

### `cull_sparse_soundings`

Disable soundings whose usable gate count has fallen below ``minimum``.

```python
cull_sparse_soundings(minimum_gates: int)
```

| parameter | default | |
|---|---|---|
| `minimum_gates` | `required` | Fewest usable gates a sounding may have and still be kept, per channel. Inclusive — a sounding with exactly this many is kept, since "at least N gates" is how the requirement is stated. **No default:** how many gates constrain a model depends on how many layers the inversion solves for. |

## cull gates

### `apply_manual_edits`

Apply a hand-authored in-use diff at the soundings it names.

```python
apply_manual_edits(diff: str | libaarhusxyz.xyz.XYZ, checkpoint: str | None = None)
```

| parameter | default | |
|---|---|---|
| `diff` | `required` | The edit set — a ``libaarhusxyz`` XYZ, or a path to one. |
| `checkpoint` | `None` | A name for the state this diff is a difference from. Recorded on the run so the pairing survives; nothing is written here. |

### `cull_below_noise_floor`

Disable gates at or below the dataset's noise floor.

```python
cull_below_noise_floor(margin: float = 1.0, start_gate: int = 0, spread_earlier: int = 0, disable_tail: bool = False)
```

| parameter | default | |
|---|---|---|
| `margin` | `1.0` | How many times the floor a datum must exceed to be kept. ``1.0`` culls at the floor itself; ``2.0`` demands twice it. This is a multiplier on the shared floor rather than a floor of its own, so it cannot drift away from what filter_12 assigned uncertainty with. |
| `start_gate` | `0` | Where evaluation begins, 0-based. |
| `spread_earlier` | `0` | Also disable this many gates before each hit. |
| `disable_tail` | `False` | Also disable everything after the earliest hit in that sounding. Well justified here — the floor is crossed once and the signal keeps decaying, so later gates are further below it. |

### `cull_by_coupling`

Mark, and optionally cull, gates the survey's basis cannot reproduce.

```python
cull_by_coupling(action: str = 'flag', max_reconstruction_error: float = 2.5, min_confidence: float = 0.86, cull_below_confidence: float | None = None, window: float = 15.0, max_uncertainty: float = 0.2, n_components: int = 4, confidence_column: str = 'qs_coupling', start_gate: int = 0, spread_earlier: int = 1, disable_tail: bool = True)
```

| parameter | default | |
|---|---|---|
| `action` | `'flag'` | ``flag`` writes the coupling table and confidence only; ``cull`` also applies the table to the mask. |
| `max_reconstruction_error` | `2.5` | A gate reconstructs when its error is below this many standard deviations of its assigned uncertainty. Asif et al.'s ``α``, 2.5. Lower is stricter. |
| `min_confidence` | `0.86` | A sounding whose smoothed confidence, the fraction of its gates that reconstruct, exceeds this keeps every gate. Asif et al.'s ``β_max``, 0.86 for their airborne survey (0.99 on the ground). Higher is stricter. A fraction, 0..1. |
| `cull_below_confidence` | `None` | Below this smoothed confidence every gate of the sounding is culled; between it and ``min_confidence`` only gates within one standard deviation survive. ``0.80 × min_confidence`` if unset, as in Asif et al. A fraction, at most ``min_confidence``. |
| `window` | `15.0` | The along-line length the data are averaged over before they are judged, in the survey's horizontal units (meters for a projected CRS). The confidence is averaged over twice this. Asif et al. average 7 raw stacks; 15 m is that at a 10 Hz SkyTEM spacing of about 2.2 m. Set by the sounding spacing, not a strictness dial: a larger window smears the coupling. |
| `max_uncertainty` | `0.2` | From the first gate whose relative uncertainty, after averaging, exceeds this, the transient is not judged: that is noise, not coupling. Asif et al.'s ``Δ_max``, 20%. A fraction. |
| `n_components` | `4` | Components kept in the reconstruction basis. Too many and the basis learns the coupling as well as the decay. |
| `confidence_column` | `'qs_coupling'` | Prefix of the per-sounding confidence column written for each channel, suffixed with the channel key (``qs_coupling_ch1``). The smoothed fraction of gates that reconstruct, 0..1; ``NaN`` where nothing could be judged. |
| `start_gate` | `0` | Where the culling policy begins, 0-based. |
| `spread_earlier` | `1` | Also cull this many gates before each hit. |
| `disable_tail` | `True` | Also cull everything after the earliest hit in that sounding. |

### `cull_by_curvature`

Disable data whose local curvature is implausible.

```python
cull_by_curvature(max_curvature: float | None = None, min_curvature: float | None = None, start_gate: int = 0, spread_earlier: int = 0, disable_tail: bool = False)
```

| parameter | default | |
|---|---|---|
| `max_curvature` | `None` | Curvature above this is culled (filter_27). |
| `min_curvature` | `None` | Curvature below this is culled (filter_28). |
| `start_gate` | `0` | Where evaluation begins, 0-based. |
| `spread_earlier` | `0` | Also disable this many gates before each hit. |
| `disable_tail` | `False` | Also disable everything after the earliest hit. |

### `cull_by_misfit`

Disable gates a previous inversion fit poorly, optionally only near mapped features.

```python
cull_by_misfit(max_misfit: float, result: str | libaarhusxyz.xyz.XYZ | ymerflow_processing.dataset.Dataset | None = None, within: str | geopandas.geodataframe.GeoDataFrame | None = None, distance_first: float | dict[str, float] | None = None, distance_last: float | dict[str, float] | None = None, crs: str | None = None, start_gate: int = 0, spread_earlier: int = 0, disable_tail: bool = False)
```

| parameter | default | |
|---|---|---|
| `max_misfit` | `required` | Largest acceptable gatewise misfit, **in standard deviations**. Data missed by more than this is suspect. |
| `result` | `None` | A previous inversion's or forward model's output. **Only needed when the dataset has no misfit column yet** — this reads the column first and computes one from the result only if there is none, writing it so nothing recomputes it. Running ``compute_misfit`` beforehand makes this an ordinary filter thresholding a column, which is what ``01`` asks for. |
| `within` | `None` | Restrict the cull to soundings within ``distance_first`` / ``distance_last`` of these features. Poorly fit **and** near a feature is more selective than either alone (``01``): a poor fit away from anything mapped may be geology a 1D model cannot represent, and removing it removes the evidence that the model was wrong. |
| `distance_first` | `None` | Exclusion distance at the first gate, in the survey's CRS units, when ``within`` is given. Per channel key or moment name, or one value. |
| `distance_last` | `None` | The same at the last gate. Constant if unset. |
| `crs` | `None` | The survey's CRS, if the dataset does not carry one. |
| `start_gate` | `0` | Where evaluation begins, 0-based. |
| `spread_earlier` | `0` | Also disable this many gates before each hit. |
| `disable_tail` | `False` | Also disable everything after the earliest hit. |

### `cull_by_slope`

Disable data whose local decay rate is implausible.

```python
cull_by_slope(shallowest: float | None = None, steepest: float | None = None, start_gate: int = 0, spread_earlier: int = 0, disable_tail: bool = False)
```

| parameter | default | |
|---|---|---|
| `shallowest` | `None` | Slopes above this — closer to flat — are culled (filter_25). Signed and negative for a decay. |
| `steepest` | `None` | Slopes below this — falling faster — are culled (filter_26). |
| `start_gate` | `0` | Where evaluation begins, 0-based. |
| `spread_earlier` | `0` | Also disable this many gates before each hit. |
| `disable_tail` | `False` | Also disable everything after the earliest hit. |

### `cull_by_uncertainty`

Disable gates whose relative uncertainty exceeds ``max_uncertainty``.

```python
cull_by_uncertainty(max_uncertainty: float, start_gate: int = 0, spread_earlier: int = 0, disable_tail: bool = False)
```

| parameter | default | |
|---|---|---|
| `max_uncertainty` | `required` | Largest relative uncertainty to keep, as a fraction — ``0.15`` is fifteen percent. Inclusive: a point exactly at the limit is kept. **No default:** what is tolerable depends on the inversion. |
| `start_gate` | `0` | Where evaluation begins, 0-based. |
| `spread_earlier` | `0` | Also disable this many gates before each hit. |
| `disable_tail` | `False` | Also disable everything after the earliest hit in that sounding. |

### `cull_near_infrastructure`

Disable gates whose sounding lies within an exclusion distance of a mapped feature.

```python
cull_near_infrastructure(infrastructure: str | geopandas.geodataframe.GeoDataFrame, distance_first: float | dict[str, float], distance_last: float | dict[str, float] | None = None, crs: str | None = None, start_gate: int = 0, spread_earlier: int = 0, disable_tail: bool = False)
```

| parameter | default | |
|---|---|---|
| `infrastructure` | `required` | Vector geometry of the features to stay clear of — powerlines, fences, pipelines — or a path geopandas can read. |
| `distance_first` | `required` | Exclusion distance at the first gate, in the survey's CRS units. Per channel key or moment name, or one value for all. |
| `distance_last` | `None` | The same at the last gate; interpolated in log time between the two. Constant if unset. |
| `crs` | `None` | The survey's CRS, if the dataset does not carry one. |
| `start_gate` | `0` | Where evaluation begins, 0-based. |
| `spread_earlier` | `0` | Also disable this many gates before each hit. |
| `disable_tail` | `False` | Also disable everything after the earliest hit. |

### `cull_negatives`

Disable gates whose measured value is negative.

```python
cull_negatives(start_gate: int = 0, spread_earlier: int = 0, disable_tail: bool = False)
```

| parameter | default | |
|---|---|---|
| `start_gate` | `0` | Where evaluation begins, 0-based. Early gates sit in the transmitter turn-off and legitimately change sign there. |
| `spread_earlier` | `0` | Also disable this many gates before each negative — the contamination precedes its visible onset. |
| `disable_tail` | `False` | Also disable everything after the earliest negative in that sounding, since a transient does not recover. |

### `save_intermediate_and_apply_diff`

Save the state reached here under a name, then replay its manual edit.

```python
save_intermediate_and_apply_diff(name: str, diff: pydantic.networks.AnyUrl = None)
```

| parameter | default | |
|---|---|---|
| `name` | `required` | A name for the intermediate state saved here. It appears in the plot editor under Dataset, and is the state this point's manual edits are a difference from. |
| `diff` | `None` | An in-use edit saved from a plot of this save point. Filled in by the plot editor; replayed after the save point is written, so re-running keeps it. Leave empty for a pure save point. |

### `set_inuse_gates`

Set named gates in the mask, either direction, per channel.

```python
set_inuse_gates(enable_gates: int | str | list[int | str] | dict | None = None, disable_gates: int | str | list[int | str] | dict | None = None)
```

| parameter | default | |
|---|---|---|
| `enable_gates` | `None` | Gates to turn back on, per channel key or moment name (``{"ch1": "0-5"}``), or one nomination for every channel. |
| `disable_gates` | `None` | Gates to turn off, in the same forms. |

## averaging

### `average_along_line`

Average each sounding with its along-line neighbours.

```python
average_along_line(width_first: int, width_last: int | None = None, method: str = 'hann', min_valid_fraction: float = 0.5, decimate_to: float | None = None, line_column: str | None = None, stash: bool = True, alpha: float = 0.1, max_weight_factor: float = 10.0, independent_fraction: float = 0.0)
```

| parameter | default | |
|---|---|---|
| `width_first` | `required` | Window width in soundings at the first gate in use. |
| `width_last` | `None` | Width at the last gate in use. Equal to ``width_first`` if unset, giving a constant window. |
| `method` | `'hann'` | ``hann`` (default), ``gaussian``, ``boxcar`` or ``robust_stack``. ``robust_stack`` is an alpha-trimmed, weight-capped inverse-variance mean whose error is the window's total spread, with odd widths; see the module docstring. ``hybrid`` is accepted as its earlier name. ``boxcar`` is a flat kernel kept for reproducing a previous run. |
| `min_valid_fraction` | `0.5` | For the tapered methods, the **effective** sample size a window must reach, as a fraction of its width. For ``robust_stack``, the **count** of valid neighbours, ``max(2, ceil(fraction × width))``. |
| `decimate_to` | `None` | Keep only soundings this far apart along the line, in CRS units. Drops soundings, changing the dataset's extent. |
| `line_column` | `None` | Which column identifies the line. Discovered if unset. |
| `stash` | `True` | Keep the unaveraged values and uncertainties alongside. |
| `alpha` | `0.1` | ``robust_stack`` only: the fraction trimmed from each end of the sorted values in a window. |
| `max_weight_factor` | `10.0` | ``robust_stack`` only: the cap on any weight, as a multiple of the window's median weight. |
| `independent_fraction` | `0.0` | ``robust_stack`` only, 0 to 1: how independent neighbouring soundings are. The error variance is divided by ``max(1, fraction × n)``. 0 keeps the window's total spread as the error; 1 is the standard error of the mean for fully independent soundings. |

## housekeeping

### `copy_array`

Copy a per-gate array, leaving the original (filter_15).

```python
copy_array(source: str, destination: str)
```

| parameter | default | |
|---|---|---|
| `source` | `required` | The array to copy. |
| `destination` | `required` | The key to copy it under. |

### `copy_attribute`

Copy a per-sounding attribute, leaving the original (filter_14).

```python
copy_attribute(source: str, destination: str)
```

| parameter | default | |
|---|---|---|
| `source` | `required` | The column to copy. |
| `destination` | `required` | The name to copy it under. |

### `export_vector`

Write the dataset's per-sounding attributes out as vector geometry.

```python
export_vector(path: str, crs: str, source_crs: str | None = None, columns: list[str] | None = None, lines: bool = True, unclassified: str = 'Unclassified', format: str | None = None)
```

| parameter | default | |
|---|---|---|
| `path` | `required` | Where to write. |
| `crs` | `required` | **Output** CRS, e.g. ``"EPSG:32613"``. Required and never guessed: it is a project decision, and a contractor's delivery CRS and a client's output CRS are routinely different. |
| `source_crs` | `None` | What the dataset's coordinates are in. Taken from the dataset if it knows; an error if neither says, since reprojecting from an assumed CRS produces plausible, wrong positions. |
| `columns` | `None` | Which attributes to write. All of ``flightlines`` by default. |
| `lines` | `True` | Also write one feature per flightline. GeoPackage takes both layers in one file; GeoJSON holds one, so this writes a second file alongside. |
| `unclassified` | `'Unclassified'` | What a verdict with no classification is written as. |
| `format` | `None` | Which format to write — ``gpkg`` or ``geojson``. Taken from the extension if unset, which is what a standalone run usually wants. Stated explicitly it wins, so a caller serving a chosen format is not obliged to encode that choice in a filename. Shapefile is refused. |

### `remove_array`

Remove a per-gate array (filter_19).

```python
remove_array(name: str)
```

| parameter | default | |
|---|---|---|
| `name` | `required` | The array to remove. |

### `remove_attribute`

Remove a per-sounding attribute (filter_18).

```python
remove_attribute(name: str)
```

| parameter | default | |
|---|---|---|
| `name` | `required` | The column to remove. |

### `rename_array`

Rename a per-gate array (filter_17).

```python
rename_array(source: str, destination: str)
```

| parameter | default | |
|---|---|---|
| `source` | `required` | The array to rename. |
| `destination` | `required` | Its new key. |

### `rename_attribute`

Rename a per-sounding attribute (filter_16).

```python
rename_attribute(source: str, destination: str)
```

| parameter | default | |
|---|---|---|
| `source` | `required` | The column to rename. |
| `destination` | `required` | Its new name. |

## synthetic

### `add_noise`

Add noise drawn from each datum's assigned uncertainty.

```python
add_noise(seed: int = 42, channels: list[str] | None = None, again: bool = False, stash: bool = True)
```

| parameter | default | |
|---|---|---|
| `seed` | `42` | The draw to make. Defaults to 42, which is a legible marker for "not chosen" rather than a better number — no seed is. **Independent realizations need different seeds**; the default gives every one of them the same noise. |
| `channels` | `None` | Which channels to perturb, by key or moment name. All of them if unset. |
| `again` | `False` | Permit perturbing a dataset that already carries noise. Off by default, because doing it unknowingly is the silent error this filter is written to prevent. |
| `stash` | `True` | Keep the noise-free values under the same key with ``orig_`` in front. |
