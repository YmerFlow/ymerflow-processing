# filters

*Generated from the entry-point registry — do not edit by hand.*

dataset in, dataset out. These change the data.

33 step(s).

## `apply_manual_edits`

Apply a hand-authored in-use diff at the soundings it names.

```python
apply_manual_edits(diff: str | libaarhusxyz.xyz.XYZ, checkpoint: str | None = None)
```

| parameter | default | |
|---|---|---|
| `diff` | `required` | The edit set — a ``libaarhusxyz`` XYZ, or a path to one. |
| `checkpoint` | `None` | A name for the state this diff is a difference from. Recorded on the run so the pairing survives; nothing is written here. |

## `apply_nominal_current`

Overwrite the per-sounding transmitter current with the declared nominal.

```python
apply_nominal_current(columns: str | dict[str, str] | None = None, stash: bool = True)
```

| parameter | default | |
|---|---|---|
| `columns` | `None` | Which ``flightlines`` column holds each channel's current, per channel key or moment name. Discovered if unset. |
| `stash` | `True` | Keep the measured values alongside. |

## `apply_terrain_model`

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
| `stash` | `True` | Keep the delivered values under ``tx_alt_raw`` and ``topo_raw``. |

## `assign_uncertainty`

Replace uncertainties with the dataset's recorded noise model.

```python
assign_uncertainty(stash: bool = True)
```

| parameter | default | |
|---|---|---|
| `stash` | `True` | Keep the delivered uncertainties alongside, under the same key with ``_raw`` appended. |

## `average_along_line`

Average each sounding with its along-line neighbours.

```python
average_along_line(width_first: int, width_last: int | None = None, method: str = 'hann', min_valid_fraction: float = 0.5, decimate_to: float | None = None, line_column: str | None = None, stash: bool = True)
```

| parameter | default | |
|---|---|---|
| `width_first` | `required` | Window width in soundings at the first gate. |
| `width_last` | `None` | Width at the last gate. Equal to ``width_first`` if unset, giving a constant window. |
| `method` | `'hann'` | ``hann`` (default), ``gaussian`` or ``boxcar``. The last is a flat kernel and is kept for reproducing a previous run, not because it is a good choice. |
| `min_valid_fraction` | `0.5` | The **effective** sample size a window must reach, as a fraction of its width, before an output is produced. Effective, not counted — see the module docstring. |
| `decimate_to` | `None` | Keep only soundings this far apart along the line, in CRS units. Drops soundings, changing the dataset's extent. |
| `line_column` | `None` | Which column identifies the line. Discovered if unset. |
| `stash` | `True` | Keep the unaveraged values and uncertainties alongside. |

## `classify_lines`

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

## `compute_misfit`

Write misfit between the data and a forward synthetic as columns.

```python
compute_misfit(result: str | libaarhusxyz.xyz.XYZ | ymerflow_processing.dataset.Dataset, rms_column: str = 'resdata')
```

| parameter | default | |
|---|---|---|
| `result` | `required` | A previous inversion's or forward model's output — an XYZ, a Dataset, or a path to one. |
| `rms_column` | `'resdata'` | Where the per-sounding root-mean-square misfit goes. |

## `convert_normalization`

Convert amplitudes into ``to``, from whatever the dataset declares.

```python
convert_normalization(to: str = 'rx_tx', stash: bool = True)
```

| parameter | default | |
|---|---|---|
| `to` | `'rx_tx'` | The convention to produce. Defaults to ``rx_tx``, which is what the forward code expects. |
| `stash` | `True` | Keep the values as they were alongside. |

## `copy_array`

Copy a per-gate array, leaving the original (filter_15).

```python
copy_array(source: str, destination: str)
```

| parameter | default | |
|---|---|---|
| `source` | `required` | The array to copy. |
| `destination` | `required` | The key to copy it under. |

## `copy_attribute`

Copy a per-sounding attribute, leaving the original (filter_14).

```python
copy_attribute(source: str, destination: str)
```

| parameter | default | |
|---|---|---|
| `source` | `required` | The column to copy. |
| `destination` | `required` | The name to copy it under. |

## `correct_tilt`

Rescale amplitudes to what a horizontal transmitter would have measured.

```python
correct_tilt(pitch_column: str | None = None, roll_column: str | None = None, max_correction: float = 4.0, again: bool = False, stash: bool = True)
```

| parameter | default | |
|---|---|---|
| `pitch_column` | `None` | Which column holds pitch, along the flight direction. |
| `roll_column` | `None` | Which holds roll, across it. |
| `max_correction` | `4.0` | Largest factor to apply. Beyond this the first-order treatment has stopped being defensible, so the sounding is left uncorrected and reported rather than scaled by a number the physics does not support. ``4.0`` corresponds to about 45° of total tilt. |
| `again` | `False` | Permit correcting a dataset that already carries a correction. |
| `stash` | `True` | Keep the delivered values alongside. |

## `cull_below_noise_floor`

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

## `cull_by_curvature`

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

## `cull_by_decay_rate`

Disable data whose local decay rate is implausible.

```python
cull_by_decay_rate(shallowest: float | None = None, steepest: float | None = None, start_gate: int = 0, spread_earlier: int = 0, disable_tail: bool = False)
```

| parameter | default | |
|---|---|---|
| `shallowest` | `None` | Slopes above this — closer to flat — are culled (filter_25). Signed and negative for a decay. |
| `steepest` | `None` | Slopes below this — falling faster — are culled (filter_26). |
| `start_gate` | `0` | Where evaluation begins, 0-based. |
| `spread_earlier` | `0` | Also disable this many gates before each hit. |
| `disable_tail` | `False` | Also disable everything after the earliest hit. |

## `cull_by_external_evidence`

Disable data that evidence from outside the transient says is suspect.

```python
cull_by_external_evidence(criterion: str, infrastructure: str | geopandas.geodataframe.GeoDataFrame | None = None, distance_first: float | dict[str, float] | None = None, distance_last: float | dict[str, float] | None = None, result: str | libaarhusxyz.xyz.XYZ | ymerflow_processing.dataset.Dataset | None = None, max_misfit: float | None = None, crs: str | None = None, start_gate: int = 0, spread_earlier: int = 0, disable_tail: bool = False)
```

| parameter | default | |
|---|---|---|
| `criterion` | `required` | ``proximity`` — near a mapped feature (filter_30); ``misfit`` — poorly fit by a previous inversion; or ``both`` — near a feature **and** poorly fit, which is filter_31 and is more selective than either. |
| `infrastructure` | `None` | Vector geometry, for ``proximity`` and ``both``. |
| `distance_first` | `None` | Exclusion distance at the first gate, in the survey's CRS units. Per channel key or moment name, or one value for all. |
| `distance_last` | `None` | The same at the last gate. Constant if unset. |
| `result` | `None` | A previous inversion's or forward model's output, for ``misfit`` and ``both``. **Only needed when the dataset has no misfit column yet** — this reads the column first and computes one from the result only if there is none, writing it so nothing recomputes it. Running ``compute_misfit`` beforehand makes this an ordinary filter thresholding a column, which is what ``01`` asks for. |
| `max_misfit` | `None` | Largest acceptable gatewise misfit, **in standard deviations**. Data missed by more than this is suspect. |
| `crs` | `None` | The survey's CRS, if the dataset does not carry one. |
| `start_gate` | `0` | Where evaluation begins, 0-based. |
| `spread_earlier` | `0` | Also disable this many gates before each hit. |
| `disable_tail` | `False` | Also disable everything after the earliest hit. |

## `cull_by_geometry`

Disable soundings flown outside the geometry the forward model assumes.

```python
cull_by_geometry(max_pitch: float | None = None, max_roll: float | None = None, max_altitude: float | None = None, min_altitude: float | None = None, pitch_column: str | None = None, roll_column: str | None = None, altitude_column: str | None = None)
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

## `cull_by_uncertainty`

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

## `cull_negatives`

Disable gates whose measured value is negative.

```python
cull_negatives(start_gate: int = 0, spread_earlier: int = 0, disable_tail: bool = False)
```

| parameter | default | |
|---|---|---|
| `start_gate` | `0` | Where evaluation begins, 0-based. Early gates sit in the transmitter turn-off and legitimately change sign there. |
| `spread_earlier` | `0` | Also disable this many gates before each negative — the contamination precedes its visible onset. |
| `disable_tail` | `False` | Also disable everything after the earliest negative in that sounding, since a transient does not recover. |

## `cull_sparse_soundings`

Disable soundings whose usable gate count has fallen below ``minimum``.

```python
cull_sparse_soundings(minimum_gates: int)
```

| parameter | default | |
|---|---|---|
| `minimum_gates` | `required` | Fewest usable gates a sounding may have and still be kept, per channel. Inclusive — a sounding with exactly this many is kept, since "at least N gates" is how the requirement is stated. **No default:** how many gates constrain a model depends on how many layers the inversion solves for. |

## `declare_normalization`

State which convention the amplitudes are already in. Changes no value.

```python
declare_normalization(convention: str, override: bool = False)
```

| parameter | default | |
|---|---|---|
| `convention` | `required` | ``raw``, ``rx`` or ``rx_tx``. Platform imports of SkyTEM and VTEM deliveries are ``rx_tx`` — V/(A·m⁴), normalized by receiver area and transmitter moment. |
| `override` | `False` | Permit contradicting a convention the dataset already records. Refused by default: the record was written by a step that knew, and a declaration that disagrees with it is the undetectable error ``01`` names. |

## `disable_initial_gates`

Disable each channel's declared initial gates in the in-use mask.

```python
disable_initial_gates(gates: dict | int | None = None)
```

| parameter | default | |
|---|---|---|
| `gates` | `None` | Override the system description, per channel key or moment name (``{"ch1": 6}``), or one count for every channel. Left unset — the normal case — each channel's own declaration is used. An override is for a delivery whose description is known to be wrong, and it is recorded in the report alongside what was declared, so the two never get confused. |

## `estimate_noise_model`

Estimate the dataset's noise model, and record it for every consumer.

```python
estimate_noise_model(scope: str = 'per_dataset', noise_level_1ms: float | None = None, noise_exponent: float | None = None, relative_fraction: float | None = None, late_gates: int | None = None)
```

| parameter | default | |
|---|---|---|
| `scope` | `'per_dataset'` | ``per_line``, ``per_flight`` or ``per_dataset`` — the grain the estimate is made over. Not "per survey": this library has no concept of one, ``Survey`` is already the data object's class, and a campaign may map to one dataset, part of one, or several. |
| `noise_level_1ms` | `None` | The absolute floor at 1 ms, before moment normalization — the instrument's own dB/dt noise. Applied to the data, it is divided by each channel's dipole moment when the dataset declares ``rx_tx``, and left as it is otherwise; so the dataset's declaration must be right (see ``declare_normalization``). Absent means estimate. |
| `noise_exponent` | `None` | The power-law slope. Absent means fit it; ``-0.5`` is the usual value and a fitted number far from it usually means the gates involved are not ambient noise. |
| `relative_fraction` | `None` | Systematic error as a fraction of signal. Absent takes it from the system description's ``UniformDataSTD``. **Never estimated** — it has no separable signature in the data, and estimating it would invent a number with a measurement's authority. |
| `late_gates` | `None` | How many trailing gates to estimate from. Enough to fit a slope, few enough to be noise-dominated; the last third by default. The per-gate scatter is reported so the choice is checkable rather than trusted. |

## `export_vector`

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
| `lines` | `True` | Also write one feature per flightline. GeoPackage takes both layers in one file; a shapefile holds one geometry type, so this writes a second file alongside. |
| `unclassified` | `'Unclassified'` | What a verdict with no classification is written as. |
| `format` | `None` | Which format to write — ``gpkg``, ``shp`` or ``geojson``. Taken from the extension if unset, which is what a standalone run usually wants. Stated explicitly it wins, so a caller serving a chosen format is not obliged to encode that choice in a filename. |

## `inflate_uncertainty`

Add ``fraction`` to the uncertainty of the nominated gates.

```python
inflate_uncertainty(fraction: float | dict[str, float], gates: int | str | list[int | str] | dict | None = None)
```

| parameter | default | |
|---|---|---|
| `fraction` | `required` | How much to add, as a fraction — ``0.02`` is two percentage points. Negative reduces. Per channel key or moment name (``{"ch1": 0.02}``), or one value for every channel. |
| `gates` | `None` | Which gates, per channel or one nomination for all. A single index, a list, or an inclusive ``"first-last"`` range. Every gate if unset. |

## `nominate_gates`

Set named gates in the mask, either direction, per channel.

```python
nominate_gates(enable_gates: int | str | list[int | str] | dict | None = None, disable_gates: int | str | list[int | str] | dict | None = None)
```

| parameter | default | |
|---|---|---|
| `enable_gates` | `None` | Gates to turn back on, per channel key or moment name (``{"ch1": "0-5"}``), or one nomination for every channel. |
| `disable_gates` | `None` | Gates to turn off, in the same forms. |

## `perturb_from_uncertainty`

Add noise drawn from each datum's assigned uncertainty.

```python
perturb_from_uncertainty(seed: int = 42, channels: list[str] | None = None, again: bool = False, stash: bool = True)
```

| parameter | default | |
|---|---|---|
| `seed` | `42` | The draw to make. Defaults to 42, which is a legible marker for "not chosen" rather than a better number — no seed is. **Independent realizations need different seeds**; the default gives every one of them the same noise. |
| `channels` | `None` | Which channels to perturb, by key or moment name. All of them if unset. |
| `again` | `False` | Permit perturbing a dataset that already carries noise. Off by default, because doing it unknowingly is the silent error this filter is written to prevent. |
| `stash` | `True` | Keep the noise-free values under the same key with ``_clean`` appended. |

## `remove_array`

Remove a per-gate array (filter_19).

```python
remove_array(name: str)
```

| parameter | default | |
|---|---|---|
| `name` | `required` | The array to remove. |

## `remove_attribute`

Remove a per-sounding attribute (filter_18).

```python
remove_attribute(name: str)
```

| parameter | default | |
|---|---|---|
| `name` | `required` | The column to remove. |

## `rename_array`

Rename a per-gate array (filter_17).

```python
rename_array(source: str, destination: str)
```

| parameter | default | |
|---|---|---|
| `source` | `required` | The array to rename. |
| `destination` | `required` | Its new key. |

## `rename_attribute`

Rename a per-sounding attribute (filter_16).

```python
rename_attribute(source: str, destination: str)
```

| parameter | default | |
|---|---|---|
| `source` | `required` | The column to rename. |
| `destination` | `required` | Its new name. |

## `select_channels`

Keep only the channels whose ``criterion`` value is in ``keep``.

```python
select_channels(criterion: str, keep: list[str], drop_system_blocks: bool = True, drop_unclaimed: bool = False)
```

| parameter | default | |
|---|---|---|
| `criterion` | `required` | How channels are chosen — ``component`` (``x``/``y``/``z``, from the receiver polarization), ``moment`` (the transmitter moment name, such as ``LM``), or ``key`` (``ch1``, always available). |
| `keep` | `required` | Which values to keep. Matched case-insensitively, so ``Z`` and ``z`` are the same request. |
| `drop_system_blocks` | `True` | Also remove the discarded channels' blocks from the system description, so it does not go on declaring channels whose data has gone. Set false to leave the system description alone. |
| `drop_unclaimed` | `False` | Also remove gate-shaped arrays that no channel claims — see :func:`unclaimed_arrays`. Off by default: such an array is undeclared rather than empty, and at least one of them is real coherent signal that the incumbent toolchain merely discards, so deleting it unasked would be worse than leaving it. It is always reported. |

## `select_lines`

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

## `zero_attitude`

Force the recorded attitude to zero, keeping the originals.

```python
zero_attitude(pitch_column: str | None = None, roll_column: str | None = None, stash: bool = True)
```

| parameter | default | |
|---|---|---|
| `pitch_column` | `None` | Which column holds pitch. Discovered if unset; created under the canonical spelling if absent. |
| `roll_column` | `None` | Which holds roll. Likewise. |
| `stash` | `True` | Keep the delivered attitude alongside. Where the columns did not exist there is nothing to keep, and that is reported. |
