# compare

*Generated from the entry-point registry — do not edit by hand.*

N reports or datasets in, one report out. These answer questions no single run can.

4 step(s).

## `implementation_equivalence`

Compare a candidate implementation against a baseline, per datum.

**Consumes:** datasets

```python
implementation_equivalence(reference: ymerflow_processing.dataset.Dataset | None = None, relative: float = 1e-06, absolute: float = 0.0, justification: str | None = None, expected_differences: list[dict] = (), quantities: list[str] = ('data', 'uncertainty', 'mask'), position_tolerance: float = 0.5)
```

| parameter | default | |
|---|---|---|
| `reference` | `None` | The baseline's output over the same input. Required. |
| `relative` | `1e-06` | Fractional agreement required, against the baseline value. |
| `absolute` | `0.0` | Absolute agreement required, added to the relative part. Needed where the baseline can be zero, since a fraction of zero is zero. |
| `justification` | `None` | Why that tolerance is the right one. Carried into the report because a tolerance without a reason is a number someone tuned until the comparison passed. |
| `expected_differences` | `()` | Declarations of where the two are *meant* to differ. Each is ``{quantity, channel?, reason, direction?}``, where ``direction`` is ``"smaller"``, ``"larger"`` or ``"any"``. A covered difference is a pass; one whose direction is violated is not. |
| `quantities` | `('data', 'uncertainty', 'mask')` | Which of ``data``, ``uncertainty``, ``mask`` to compare. |
| `position_tolerance` | `0.5` | How close two soundings must be, in CRS units, to be the same sounding when there is no timestamp. |

## `noise_floor_drift`

Trend the estimated noise floor across a sequence of deliveries.

```python
noise_floor_drift(reference: dict | None = None, source_step: str = 'estimate_noise_model', max_ratio: float | None = None)
```

| parameter | default | |
|---|---|---|
| `reference` | `None` | A run to measure everything against. Without one the baseline is the series median, which is robust to the outlier being looked for. |
| `source_step` | `'estimate_noise_model'` | Which step's report carries the model. |
| `max_ratio` | `None` | Largest acceptable factor between a delivery's floor and the baseline. **Unset, this measures and says nothing** — what counts as too much drift depends on the instrument and the survey. |

## `parameter_sweep`

Trace how the output responds to one varied parameter.

**Consumes:** datasets

```python
parameter_sweep(reference: dict | None = None, response_column: str | None = None, flat_fraction: float = 0.1, position_tolerance: float = 0.5)
```

| parameter | default | |
|---|---|---|
| `reference` | `None` | The control every run is measured against. Without one the first run is used, which is the conventional sweep — a series starting from the current setting. |
| `response_column` | `None` | A per-sounding column to trace alongside the divergence, such as a misfit or a noise estimate. Optional: divergence from the control exists for every sweep, while a better response measure exists only sometimes. |
| `flat_fraction` | `0.1` | A step below this fraction of the largest counts as flat, when looking for where the response levels off. |
| `position_tolerance` | `0.5` | How close two soundings must be, in CRS units, to be the same sounding when there is no timestamp. |

## `version_diff`

What changed between two processing versions of the same data.

**Consumes:** datasets

```python
version_diff(reference: ymerflow_processing.dataset.Dataset | None = None, quantities: list[str] = ('data', 'uncertainty', 'mask'), relative: float = 1e-09, absolute: float = 0.0, position_tolerance: float = 0.5)
```

| parameter | default | |
|---|---|---|
| `reference` | `None` | The older version. Required: which of two versions is the newer one cannot be inferred, and getting it backwards reverses the sign of every finding while reporting the same magnitudes. |
| `quantities` | `('data', 'uncertainty', 'mask')` | Which of ``data``, ``uncertainty``, ``mask`` to diff. |
| `relative` | `1e-09` | Below this fractional change a value counts as unmoved. Tight by default — unlike an equivalence tolerance this is not a judgement about acceptability, only about float noise. |
| `absolute` | `0.0` | Absolute companion to ``relative``, for values that pass through zero. |
| `position_tolerance` | `0.5` | How close two soundings must be, in CRS units, to be the same sounding when there is no timestamp. |
