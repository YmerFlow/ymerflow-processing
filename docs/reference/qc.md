# qc

*Generated from the entry-point registry — do not edit by hand.*

dataset in, dataset plus a report out. These grade it and change no measurement.

9 step(s).

## `altimeter_agreement`

Classify the disagreement between independent altimeters.

```python
altimeter_agreement(columns, good_max: float, marginal_max: float, output_column: str = 'qs_altim', spread_column: str = 'alt_spread', line_column: str | None = None, unit: str = 'm', gap_tolerance: float = 0.0)
```

| parameter | default | |
|---|---|---|
| `columns` | `required` | The altimeter columns to compare — at least two. Required and never guessed: which sensors a survey carries, and what they are called, is a property of the aircraft rather than something to infer. |
| `good_max` | `required` | Spread at or below which the altimeters agree acceptably, in ``unit``. A contract term, so no default. |
| `marginal_max` | `required` | Spread above which the disagreement is Out of Spec. |
| `output_column` | `'qs_altim'` | Where the per-sounding verdict is written. |
| `spread_column` | `'alt_spread'` | Where the numeric spread is written. Kept because the verdict alone does not say *how far* apart they were, and because a later step may want to threshold it differently. |
| `line_column` | `None` | Which column identifies the line. Discovered if unset. |
| `unit` | `'m'` | Unit of the altimeter columns, for report text. |
| `gap_tolerance` | `0.0` | In-spec stretches shorter than this, in CRS units, do not break an out-of-spec run when the contiguous-run length is measured. A contract saying "out of spec for 1 km or more" describes a stretch of flying that one recovered sounding does not end. ``0`` measures runs raw; the raw figure is reported either way. |

## `culled_fraction`

Measure how much of each sounding culling disabled.

```python
culled_fraction(good_max: float | None = None, marginal_max: float | None = None, output_column: str = 'qs_culled', fraction_column: str = 'cull_frac', line_column: str | None = None, gap_tolerance: float = 0.0)
```

| parameter | default | |
|---|---|---|
| `good_max` | `None` | Culled fraction at or below which a sounding is unremarkable. **Left unset, no verdict is produced** and only the measurement is written. |
| `marginal_max` | `None` | Fraction above which the sounding is Out of Spec. Required if ``good_max`` is given. |
| `output_column` | `'qs_culled'` | Where the per-sounding verdict is written. |
| `fraction_column` | `'cull_frac'` | Where the numeric fraction is written. |
| `line_column` | `None` | Which column identifies the line. Discovered if unset. |
| `gap_tolerance` | `0.0` | In-spec stretches shorter than this, in CRS units, do not break an out-of-spec run when contiguous runs are measured. |

## `data_misfit`

Classify how well a previous inversion explained each sounding.

```python
data_misfit(good_max: float | None = None, marginal_max: float | None = None, column: str = 'resdata', output_column: str = 'qs_misfit', line_column: str | None = None, gap_tolerance: float = 0.0)
```

| parameter | default | |
|---|---|---|
| `good_max` | `None` | Root-mean-square misfit, **in standard deviations**, at or below which a sounding is unremarkable. **Left unset, no verdict is produced** and the distribution is reported alone. |
| `marginal_max` | `None` | Misfit above which the sounding is Out of Spec. Required if ``good_max`` is given. |
| `column` | `'resdata'` | Which column holds the per-sounding misfit. Written by ``compute_misfit``, or by an inversion under the same name. |
| `output_column` | `'qs_misfit'` | Where the per-sounding verdict is written. |
| `line_column` | `None` | Which column identifies the line. Discovered if unset. |
| `gap_tolerance` | `0.0` | In-spec stretches shorter than this, in CRS units, do not break an out-of-spec run when contiguous runs are measured. |

## `ground_speed`

Classify each sounding's ground speed against the contracted window.

```python
ground_speed(good_min: float, good_max: float, marginal_min: float, marginal_max: float, column: str | None = None, time_column: str | None = None, output_column: str = 'qs_speed', line_column: str | None = None, unit: str = 'km/h', scale: float = 3.6, gap_tolerance: float = 0.0)
```

| parameter | default | |
|---|---|---|
| `good_min` | `required` | Slowest speed meeting spec, in ``unit``. A contract term, so no default. |
| `good_max` | `required` | Fastest speed meeting spec. |
| `marginal_min` | `required` | Below this is Out of Spec; between it and ``good_min`` is Marginal. |
| `marginal_max` | `required` | Above this is Out of Spec. |
| `column` | `None` | Which ``flightlines`` column holds ground speed. Discovered if unset; derived from position and time if the survey records none. |
| `time_column` | `None` | Which column holds acquisition time, used only when deriving. Discovered if unset. |
| `output_column` | `'qs_speed'` | Where the per-sounding verdict is written. |
| `line_column` | `None` | Which column identifies the line. Discovered if unset. |
| `unit` | `'km/h'` | Unit the thresholds are expressed in, for report text. |
| `scale` | `3.6` | Multiplier from derived units — distance-units per second, so m/s for a projected CRS in metres — to ``unit``. Only applied to derived speed; a recorded column is assumed to be in ``unit`` already. |
| `gap_tolerance` | `0.0` | In-spec stretches shorter than this, in CRS units, do not break an out-of-spec run when the contiguous-run length is measured. A contract saying "out of spec for 1 km or more" describes a stretch of flying that one recovered sounding does not end. ``0`` measures runs raw; the raw figure is reported either way. |

## `height_above_ground`

Classify each sounding's height above ground against the contracted spec.

```python
height_above_ground(good_max: float, marginal_max: float, column: str | None = None, output_column: str = 'qs_height', line_column: str | None = None, unit: str = 'm', gap_tolerance: float = 0.0)
```

| parameter | default | |
|---|---|---|
| `good_max` | `required` | Height at or below which a sounding meets spec, in ``unit``. A contract term — no default, because inventing one would put a number with no authority into a client deliverable. |
| `marginal_max` | `required` | Height above which a sounding is Out of Spec. Between the two is Marginal. |
| `column` | `None` | Which ``flightlines`` column holds height above ground. Left unset, it is discovered from the survey's own altitude column. |
| `output_column` | `'qs_height'` | Where the per-sounding verdict is written. |
| `line_column` | `None` | Which column identifies the line. Discovered if unset. |
| `unit` | `'m'` | Unit of the height column, used in report text only. |
| `gap_tolerance` | `0.0` | In-spec stretches shorter than this, in CRS units, do not break an out-of-spec run when the contiguous-run length is measured. A contract saying "out of spec for 1 km or more" describes a stretch of flying that one recovered sounding does not end. ``0`` measures runs raw; the raw figure is reported either way. |

## `powerline_noise`

Measure how far the powerline monitor rose above its local background.

```python
powerline_noise(good_max: float | None = None, marginal_max: float | None = None, window: int = 101, column: str | None = None, output_column: str = 'qs_pline', excess_column: str = 'pl_excess', line_column: str | None = None, gap_tolerance: float = 0.0)
```

| parameter | default | |
|---|---|---|
| `good_max` | `None` | Excess above the local baseline, in robust sigmas, at or below which a sounding is unremarkable. **Left unset, no verdict is produced at all** and only the measurement is written — which is the usual way to read this check. |
| `marginal_max` | `None` | Excess above which the sounding is Out of Spec. Required if ``good_max`` is given. |
| `window` | `101` | How many soundings the rolling baseline spans. Must be comfortably wider than the feature being detected, or a broad anomaly becomes its own baseline; the default is about a kilometre at typical spacing. |
| `column` | `None` | Which ``flightlines`` column holds the monitor. Discovered if unset. |
| `output_column` | `'qs_pline'` | Where the per-sounding verdict is written. |
| `excess_column` | `'pl_excess'` | Where the numeric excess is written. Kept because the verdict says a sounding was noisy without saying how noisy, and because a later culling step may want to threshold it differently. |
| `line_column` | `None` | Which column identifies the line — the grain the baseline is derived over. Discovered if unset. |
| `gap_tolerance` | `0.0` | In-spec stretches shorter than this, in CRS units, do not break an out-of-spec run when the contiguous-run length is measured. |

## `sounding_spacing`

Classify local sounding density against the contracted minimum.

```python
sounding_spacing(good_max: float, marginal_max: float, output_column: str = 'qs_spacing', spacing_column: str = 'spacing', line_column: str | None = None, units_per_km: float = 1000.0, gap_tolerance: float = 0.0)
```

| parameter | default | |
|---|---|---|
| `good_max` | `required` | Spacing at or below which the line meets spec, in CRS units. A contract term, so no default. |
| `marginal_max` | `required` | Spacing above which the line is Out of Spec. Between the two is Marginal. |
| `output_column` | `'qs_spacing'` | Where the per-sounding verdict is written. |
| `spacing_column` | `'spacing'` | Where the along-line spacing is written, in CRS units. Reported alongside the verdict because spacing is the number a geophysicist pictures, even though density is what is classified. |
| `line_column` | `None` | Which column identifies the line. Discovered if unset. |
| `units_per_km` | `1000.0` | CRS units in a kilometre — 1000 for a metric CRS. |
| `gap_tolerance` | `0.0` | In-spec stretches shorter than this, in CRS units, do not break an out-of-spec run when the contiguous-run length is measured. A contract saying "out of spec for 1 km or more" describes a stretch of flying that one recovered sounding does not end. ``0`` measures runs raw; the raw figure is reported either way. |

## `system_attitude`

Classify each sounding's pitch and roll against the contracted limits.

```python
system_attitude(good: float, marginal: float, pitch_column: str | None = None, roll_column: str | None = None, pitch_output: str = 'qs_pitch', roll_output: str = 'qs_roll', output_column: str = 'qs_tilt', line_column: str | None = None, unit: str = 'deg', gap_tolerance: float = 0.0)
```

| parameter | default | |
|---|---|---|
| `good` | `required` | Attitude within ±this, in ``unit``, meets spec. A contract term, so it has no default. |
| `marginal` | `required` | Beyond ±this is Out of Spec. Between the two is Marginal. |
| `pitch_column` | `None` | Which ``flightlines`` column holds pitch. Discovered from the survey if unset. |
| `roll_column` | `None` | Likewise for roll. |
| `pitch_output` | `'qs_pitch'` | Where the per-sounding pitch verdict is written. |
| `roll_output` | `'qs_roll'` | Likewise for roll. |
| `output_column` | `'qs_tilt'` | Where the combined verdict — the worse of the two — is written, so a later culling step has one column to threshold on. |
| `line_column` | `None` | Which column identifies the line. Discovered if unset. |
| `unit` | `'deg'` | Unit of the attitude columns, used in report text only. |
| `gap_tolerance` | `0.0` | In-spec stretches shorter than this, in CRS units, do not break an out-of-spec run when the contiguous-run length is measured. A contract saying "out of spec for 1 km or more" describes a stretch of flying that one recovered sounding does not end. ``0`` measures runs raw; the raw figure is reported either way. |

## `transmitter_stability`

Classify transmitter output stability, and agreement with nominal.

```python
transmitter_stability(good_fraction: float, marginal_fraction: float, nominal_good_fraction: float | None = None, nominal_marginal_fraction: float | None = None, jump_fraction: float = 0.01, columns: dict | None = None, output_column: str = 'qs_cur', nominal_column: str | None = 'qs_nom', nominal_offset_prefix: str | None = 'nom_', per_moment_prefix: str | None = 'qs_cur_', line_column: str | None = None, flight_column: str | None = None, gap_tolerance: float = 0.0)
```

| parameter | default | |
|---|---|---|
| `good_fraction` | `required` | Departure from the moment's own mean current, as a fraction of it, within which a sounding meets spec — e.g. ``0.01`` for 1%. No default: what counts as steady is an operational judgement. |
| `marginal_fraction` | `required` | Departure beyond which a sounding is Out of Spec. |
| `nominal_good_fraction` | `None` | Departure of the *mean* current from the system description's declared nominal within which the description is considered to describe this instrument. Both directions. **Left unset, the offset is measured and no verdict is produced** — what counts as too far from nominal is an operational judgement, not something to guess. |
| `nominal_marginal_fraction` | `None` | Beyond which the disagreement is Out of Spec. Required if ``nominal_good_fraction`` is given. |
| `jump_fraction` | `0.01` | Sounding-to-sounding change, as a fraction of the previous value, counted as a sudden step. |
| `columns` | `None` | ``{channel key or moment name: column}``, when the current columns cannot be discovered. Discovered from the moment names otherwise. |
| `output_column` | `'qs_cur'` | Where the combined verdict is written — the worst of stability and nominal agreement, across every moment. |
| `nominal_column` | `'qs_nom'` | Where the nominal-agreement verdict alone is written. ``None`` to skip. |
| `nominal_offset_prefix` | `'nom_'` | Prefix for one column per moment holding its offset from the declared nominal current, as a fraction — ``nom_ch1``. ``None`` writes none. |
| `per_moment_prefix` | `'qs_cur_'` | Prefix for the per-moment stability verdicts, suffixed with the channel key so the name stays inside the ten characters a shapefile field allows. ``None`` to skip them. |
| `line_column` | `None` | Which column identifies the line. Discovered if unset. |
| `flight_column` | `None` | Which column identifies the flight, used to keep jump detection from flagging each flight's first sample. Discovered if unset. |
| `gap_tolerance` | `0.0` | In-spec stretches shorter than this, in CRS units, do not break an out-of-spec run when the contiguous-run length is measured. |
