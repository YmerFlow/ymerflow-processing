# ymerflow-processing

[![tests](https://github.com/YmerFlow/ymerflow-processing/actions/workflows/tests.yml/badge.svg)](https://github.com/YmerFlow/ymerflow-processing/actions/workflows/tests.yml)

Airborne geophysical data processing, QC and comparison for YmerFlow.

Time-domain electromagnetics today; frequency-domain and magnetics are the stated
direction and the layout is built for them.

---

## Read this before writing code here

**This library is built from a specification, not from an existing implementation.**

`building_plans/` describes the behaviors needed — what each step does to a dataset,
in functional terms, with no implementation detail and no names carried across. That
is the input.

**Write from the mathematics and the published literature.** Do not work from the
source of any predecessor library, and do not have it open while writing. Cite the
source for each method — paper, textbook, or derivation — in the docstring.

Algorithms are not copyrightable and convergent implementations of a standard method
are expected; a moving average looks like a moving average. What must not happen is
reproducing another implementation's code or its structure. The specification exists
precisely so that this can be built by someone who has never read one.

Where a method genuinely has one obvious implementation, that is fine.

---

## Layout

```
ymerflow_processing/
├── dataset.py pipeline.py mask.py …   the data model, the runners, the in-use mask
├── filters/       dataset in, dataset out — all 32, flat; cull_* write the mask
├── qc/            dataset in, dataset plus verdicts out
├── compare/       N reports or datasets in, one report out
└── methods/       one adapter per measurement method — tem.py today
```

Role at the top, mirroring the three entry-point groups the runners see. The
specialization axis — which of this is EM, which is time-domain — is deliberately
not a directory yet: it was drawn once before a second method existed and produced
an empty package. `docs/plans/done/package-layout.md` has the measurement and the
decision; it is redrawn when a second method registers.

---

## Three runners

Three operations with genuinely different shapes, each with its own entry-point group,
all sharing the pipeline-as-YAML form and the CLI.

| runner | shape | entry-point group |
|---|---|---|
| processing | dataset → dataset | `ymerflow.processing.filters` |
| QC | dataset → dataset + report | `ymerflow.processing.qc` |
| comparison | N reports (or datasets) → one report | `ymerflow.processing.compare` |

A pipeline is a YAML document — storable, diffable, versionable, transmissible, and
not executable. A runner walks it.

Any installed package can contribute steps through the entry-point groups, so nothing
needs a central registry.

---

## The rules that shape everything

Full statements in `building_plans/01_filter_inventory.md`. In brief:

**R1 — one to N moments.** The number of moments is data, not structure. A single-moment
system and a three-moment system both work with no special casing. No default names a
specific moment; a scalar per-moment setting means "apply to all."

**R2 — readability is a requirement.** One substantial filter per file. A docstring that
says what the step does to the dataset. The mathematical source cited.

**R3 — every step returns a log of what it did.** A structured value, not a printed line:
counts, the distribution of whatever was thresholded on before and after, and the
parameters actually used. Diagnostics go to `logging`; what a step *did* goes in the
report.

**R6 — who writes what.** *QC writes attributes. Culling writes the mask. Manual edits
override the mask and survive re-runs because they replay as a diff.* This is the rule
that tells you which runner a new step belongs to.

**R7 — one selection idiom.** Wherever a step needs to know *how* something was chosen it
takes a `criterion` naming the method, carrying only that method's parameters. Never a
boolean.

**R8 — external inputs are typed, not paths.** A step declares `VectorGeometry`, `Raster`
or `Dataset` and reads through a standard loader. The platform materialises; the library
reads a path. This is what keeps the library runnable standalone.

**R9 — one noise model.** Held by the dataset, set by an explicit estimation step, read by
every consumer. No step carries its own copy of the floor or the slope.

---

## Status

Interface layer, channel discovery, seven QC checks, and the command line.
**333 tests**, all synthetic — no survey files, no platform, no web stack.

**Done**

- **Channel discovery (R1).** The number of channels is read from the data, never assumed.
  One moment, two, three or four take the same path, and a survey with no array families at
  all — a scalar magnetometer — correctly reports none.
- **`qc/`** — Good / Marginal / OOS with four threshold kinds, per-line and
  per-flight aggregation with gap-tolerant contiguous runs, and a cross-check rollup.
- **Nine checks**: qc_01 height, qc_02 attitude, qc_03 speed, qc_04 transmitter stability,
  qc_05 powerline, qc_08 altimeter agreement, qc_10 sounding spacing, qc_12 culled fraction —
  which reads the mask culling wrote, and measures only what *this* pipeline removed rather
  than what arrived already disabled — and qc_11 data misfit, which grades the column
  `compute_misfit` writes. qc_11 reports as a `model` finding at zero weight: a poor fit may
  be geology a 1D model cannot represent, so it names itself among the reasons without
  grading a survey down for the earth being complicated.
- **`inventory.py`** — the implicit first step of every run: what arrived, what was
  resolved into channels, and what was not. An array no channel claims is invisible rather
  than absent, and every channel-based step steps over it silently.
- **`mask.py`** — the in-use mask (R6's second clause) and the culling policy every
  predicate shares: start gate, spread-earlier, tail disabling, applied once rather than
  thirteen times. Per-datum provenance rides alongside, recording which step removed each
  point and whether it was a direct hit or collateral.
- **Filters**: `select_channels`, which reduces a delivery to the channels an inversion
  accepts by the R7 `criterion` idiom; `export_vector`; and `disable_initial_gates`
  (filter_20), which is **applied automatically** by `load()` and at the top of every run
  rather than listed in a pipeline — it reads the instrument's own statement about which of its
  samples can be inverted, which is part of reading the file correctly rather than a judgement.
- **`cull_by_geometry`** (filter_22) — the first real predicate through the mask. Disables
  whole soundings whose attitude or altitude broke the forward model's geometry assumption.
  Its limits are restated rather than shared with QC, because a contract limit and a physical
  one are different numbers — but each defaults to the corresponding QC out-of-spec boundary
  where one was recorded.
- **`cull_sparse_soundings`** (filter_23) — disables soundings left with too few usable gates
  to constrain a model. The first step whose *answer* depends on pipeline order, so it belongs
  last: run first it has nothing to find.
- **`nominate_gates`** (filter_21) — sets named gates either direction, per channel. One of
  only two steps permitted to turn data back on, which is what gives the early gates back to
  anyone doing full primary-field modelling.
- **`cull_negatives`** (filter_32) and **`cull_by_uncertainty`** (filter_24) — point-local
  predicates through the shared policy. Negatives need no threshold at all, since zero is not
  a number anyone chose.
- **`filters/noise.py`** (R9) — one noise model, estimated from the sounding-to-sounding scatter
  at late gates, held by the dataset and read by both consumers. `assign_uncertainty`
  (filter_12) and `cull_below_noise_floor` (filter_29) carry no floor of their own and refuse
  to run without one, which is what makes the defect R9 exists for unrepresentable.
- **`cull_by_external_evidence`** (filter_30 + filter_31) — one filter, three criteria:
  `proximity` to mapped features, gatewise `misfit` against a previous inversion, or `both`,
  which `01` says is more selective than either.
- **`filters/misfit.py`** (filter, per `01`'s resolution of filter_31) — writes measured-minus-
  synthetic as a column, gatewise and per sounding, in standard deviations. Once it is a column
  misfit is plottable, exportable and thresholdable by an ordinary culling filter.
- **`compare/align.py`** — which sounding is which, which channel is which, which
  gate is which. Shared by every dataset-consuming comparison rather than copied into each,
  since three copies drift while each looks correct alone.
- **`compare/version_diff.py`** — what changed between two processing versions, joining
  the parameter that moved to the step whose culling moved with it. No verdict: the two are
  meant to differ.
- **`compare/sweep.py`** — one parameter varied across a series, with the varied
  parameter discovered from the runs rather than restated by the caller, and the point where
  the response stops moving reported as an observation rather than a recommendation.
- **`compare/equivalence.py`** — two implementations over one input, per datum.
  Four outcomes rather than two, because bit-for-bit agreement with a defective baseline
  would mean reproducing its defects: a *declared* difference is a pass, and a declared
  difference that went the wrong way is still a finding. Aligns by timestamp, gate time and
  semantic quantity — never by column name, since two implementations are two schemas.
- **`compare/`** — the fourth contract: N reports to one report. `noise_floor_drift`
  (qc_09) trends the estimated floor across deliveries, which no per-run check can see. It
  invents no state — R3 already stores what it reads.
- **`filters/export_vector.py`** — `build_layers` produces named `GeoDataFrame`s and writes
  nothing; `export_vector` is a thin writer over it. R8's "format is a loader concern" applied
  in reverse: the layers are the product, a file is one delivery of them, and one build can
  serve several formats.
- **`transient.py`** — local slope and curvature in log-log space, measured over the gates
  that are still in use, at their real times. Shared by `cull_by_decay_rate` (filter_25/26) and
  `cull_by_curvature` (filter_27/28), because a culled gate is not a neighbour and getting that
  wrong yields a plausible number rather than an error.
- **`filters/attitude.py`** (filter_06, filter_07) — the tilt correction and its counterpart. One
  rescales amplitudes to what a level transmitter would have measured; the other declares there
  is no trustworthy attitude, so the correction becomes exactly one.
- **`filters/normalize.py`** (filter_08, filter_09) — conversion between normalization conventions,
  reading what the dataset declares rather than what a caller asserts, and the nominal current.
- **`average_along_line`** (filter_10) — the along-line weighted average, which `01` calls the
  most consequential filter in the library. Tapered kernel, widths that grow with gate time,
  validity by effective sample size, and a ratio of convolutions rather than a loop per datum.
- **`inflate_uncertainty`** (filter_11) and **`perturb_from_uncertainty`** (filter_13) — the
  operator's knob on uncertainty, and drawing synthetic noise from it. The second enforces its
  own ordering hazard: it refuses to perturb twice and calls out uncertainties that look like
  an untouched default.
- **`filters/terrain.py`** (filter_05) — samples a terrain model and re-expresses altitude
  against it, holding the instrument's elevation fixed. Reports the offset's median and spread
  together, because a vertical datum mismatch and genuine terrain error have opposite
  signatures in those two numbers.
- **`apply_manual_edits`** (filter_33) — replays a hand-authored in-use diff, which is what
  makes the chain reproducible in the presence of human judgement. The second and last step
  permitted to turn data back on.
- **`filters/lines.py`** (filter_01-04, collapsed to two by `01`'s decision) — `classify_lines`
  labels flightlines by height threshold or by name; `select_lines` reduces the dataset to the
  ones wanted, moving every array with the soundings so nothing can misalign.
- **`filters/manage.py`** (filter_14-19) — copy, rename and remove, for attributes and arrays.
  Six entry points over one symmetry. They exist to work around import problems, not to do
  geophysics, and a rename that takes a channel out of the model says so.
- **The CLI.** `yf-process run` walks a pipeline against a delivery and writes the enriched
  dataset plus a JSON report; `compare`, `list` and `validate` alongside it.

```bash
yf-process run qc.yaml --xyz 20260815.xyz --gex sys.gex --out QC/
```

Exit codes carry both questions a scheduled job asks — *did it run*, and *was the flight any
good*: `0` in spec, `1` marginal, `2` out of spec, `3` never started, `4` a step raised. The
split falls at 2, so `0-2` means the run completed. A finding is not a failure: an
out-of-spec report is the deliverable, and still gets written.

Which registry a step name resolves in comes from `--group`, else a `kind:` in the document
— validated against the steps rather than trusted — else derived from which registry
resolves them.

### Column names

Four grains, one letter each, so the scheme reads itself:

```
qs_  this sounding          qf_  its flight
ql_  its line               qd_  the delivery it arrived in
```

A bare name is a *measurement* rather than a verdict — `spacing`, `alt_spread`, `pl_excess`.

**Every name stays inside ten characters**, because that is what a shapefile field allows and
the vector export is what reaches a client. Two names truncating to the same ten would
collide silently, so this is a constraint on new checks rather than a stylistic preference.

---

## NumPy

**Both majors are supported, deliberately.** Development happens against numpy 2, but the
inversion stack this library feeds still runs on 1.x, so `numpy` carries no floor and the code
uses no numpy-2-only API. The full suite is run against both before anything is committed —
the aliases numpy 2.0 removed (`np.NaN`, `np.Inf`, `np.infty` and the rest) appear nowhere, and
that is worth re-checking rather than assuming, since the failure is silent on whichever major
you happen to be developing on.

---

## Deviations from `building_plans/`

The plans are the clean-room input and are not revised. Where the build diverged, it is
recorded here — "the plan said X, we did Y, here is why X could not hold" is more useful
than a corrected plan.

**`Moment` on the data object** — `01` and `02` put it there, while the data model is
specified as knowing nothing about gates or moments. Both cannot hold. Split on structure
versus physics: identity and array addressing are method-agnostic and live in `dataset.py`
as `Channel` and `Axis`; gate times and dipole moment are time-domain and live in
`methods/tem.py`. This produced a fourth entry-point group, `ymerflow.processing.methods`,
so adding FEM, mag, rad or gravity later is a new adapter with nothing in the data model to
change.

**qc_10 classifies spacing, not density** — `03` asks for "along-line spacing" with kind
`descending`, which cannot both hold: they are inverses, so wider spacing is `ascending` and
lower density is `descending`. Spacing carries the verdict because it is what a contract
states; density is reported alongside.

**qc_01 takes no terrain model** — `03` lists one among its inputs. Producing a correct
height-above-ground column from a DTM is filter_05's job, and duplicating it here would put
one physical statement in two places with two chances to disagree, which is the defect R9
exists to prevent.

**qc_04 judges stability, not agreement with nominal** — `03` asks for a window derived from
survey-wide statistics. A steady current away from the declared nominal is not a data
problem, since the current flown is recorded per sounding and carried into the
normalization; what costs quality is the current *moving*. Both are measured, and a large
nominal offset earns its own separate verdict.

**qc_04's combined verdict was penalizing nominal disagreement** — the module docstring said
stability carries the verdict and a nominal offset earns a separate one, and the code took the
worse of the two, so a steady transmitter flying away from its declared nominal failed. Fixed
to what the docstring said. The nominal thresholds also lost their defaults, since they were
the only tier thresholds in the library carrying invented round numbers.

**Per-line results are measures, not verdicts** — an earlier form wrote the worst verdict
found anywhere on a line, which made one bad sounding in five hundred read as a failed line.
Replaced by the fraction and the longest contiguous run at every grain, with a verdict only
where a contract supplies a rule. `03` left this open; this resolves it by measuring and not
deciding.

**The Out of Spec tier is written as `OOS`** — `03` names the tier "Out of Spec", and the
concept keeps that name in prose and in reports. What lands in the *data* cannot: the survey
xyz format is whitespace-delimited with no quoting, so a value containing a space becomes
three fields and the file will not parse back. A sounding with no reasons has the mirror
problem — an empty string occupies no field at all — and is written `NONE`. Both are the
format dictating the token, not a change of meaning.

**Verdicts are stored as numeric codes** — recorded as a dated addition in `03`. A label column
fails the plot layer's typed-array filter and is invisible to it entirely, so verdicts are
`float32` on the same 0-2 scale as the grade, with `NaN` for unevaluated. The legend lives in the
step report and the vector exports translate back to names. Because that removed the dtype that
identified a verdict column, the dataset now records the set explicitly and round-trips it.

**A weighted grade beside the verdict** — recorded as a dated addition in `03` rather than a
divergence, since it is new scope rather than a departure. Tiers are worth 0/1/2 points and
each check carries a weight, so failing lines can be ordered rather than tying at Out of Spec.
The grade never gates: a mean dilutes and aliases, so the worst-verdict column stays the
pass/fail and the per-tier fractions are written beside the grade.

**Findings are typed** — not in `03`. A check declares `flight`, `instrument` or
`environment`, so a failing altimeter does not read as a badly flown line and a powerline
reads as neither. Free-form rather than an enum, since more kinds will follow.

---

## Getting started

Four pipeline documents ship with the package, so the common case does not begin
with a list of 46 steps:

```bash
yf-process pipelines                      # what ships
yf-process show standard-processing       # read it — the comments are most of it
yf-process run acquisition-qc --xyz DELIVERY.xyz --gex SYSTEM.gex --out qc/
```

`acquisition-qc` grades a delivery and changes no measurement.
`standard-processing` takes delivered data to inversion-ready and **does** change
it. `manual-processing` declares the convention and assigns uncertainties, and
leaves every removal to a person in the plot — no culling, no averaging.
`synthetic-data` turns a clean forward model into something that behaves like
a real delivery.

To adjust one, copy it rather than editing in place — `yf-process show NAME >
mine.yaml` — since the installed copy is what every other project on the machine
runs. A path always beats a bundled name, so a local file is never shadowed.

Thresholds in the shipped documents are **representative defaults, not contract
values**.

## Next

The filter, QC and comparison groups are built, and the command line runs all three.
What remains:

1. **qc_06 and qc_07** — line adherence and coverage, both blocked on flight-plan input
   (`Ymerflow#69`). Nothing else in `03` is outstanding.
2. **filter_32's defaults** ([#5](https://github.com/YmerFlow/ymerflow-processing/issues/5)) —
   built, but whether a sign reversal is one phenomenon or three is unsettled.
3. **Altitude processing** ([#4](https://github.com/YmerFlow/ymerflow-processing/issues/4)) —
   filter_05 did the mechanics; the judgement calls are unopened.
4. **Packaging** ([#3](https://github.com/YmerFlow/ymerflow-processing/issues/3)) — the
   libaarhusxyz pin by URL blocks development against a local checkout.

Specifications live in `building_plans/`. `04_cli_runner.md` records every decision behind
the command line and why the alternatives lost.

---

## License

Copyright (c) 2026 Benjamin R. Bloss

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program in the [LICENSE file](LICENSE). If not, see <https://www.gnu.org/licenses/>.
