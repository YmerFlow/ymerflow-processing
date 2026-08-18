# ymerflow-processing

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
├── core/          any airborne survey — no method assumptions
│   └── qc/        acquisition QC
├── em/            shared by TEM and FEM
│   └── tem/       gates, moments, transient decay
└── (mag/, em/fem/ when there is code for them)
```

`em/` is a sibling of `core/` rather than nested inside it, so that "core" keeps
meaning *method-agnostic* and other methods have an obvious home. Packages are created
when they have contents, not before.

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
- **`core/qc/`** — Good / Marginal / OOS with four threshold kinds, per-line and
  per-flight aggregation with gap-tolerant contiguous runs, and a cross-check rollup.
- **Seven checks**: qc_01 height, qc_02 attitude, qc_03 speed, qc_04 transmitter stability,
  qc_05 powerline, qc_08 altimeter agreement, qc_10 sounding spacing.
- **`core/inventory.py`** — the implicit first step of every run: what arrived, what was
  resolved into channels, and what was not. An array no channel claims is invisible rather
  than absent, and every channel-based step steps over it silently.
- **`core/mask.py`** — the in-use mask (R6's second clause) and the culling policy every
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

**`Moment` in `core/`** — `01` and `02` put it on the data object, while `core/` is specified
as knowing nothing about gates or moments. Both cannot hold. Split on structure versus
physics: identity and array addressing are method-agnostic and live in `core/` as `Channel`
and `Axis`; gate times and dipole moment are time-domain and live in `em/tem/`. This
produced a fourth entry-point group, `ymerflow.processing.methods`, so adding FEM, mag, rad
or gravity later is a new package plus one adapter with nothing in `core/` to change.

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

## Next

1. **The vector export** — `03`'s third output, and the one that leaves the building.
   GeoPackage by default, shapefile when a contract names it, points and lines both.
2. **The remaining QC checks** — qc_12 culled fraction is buildable; qc_11 needs an
   inversion, qc_09 needs the comparison registry, qc_06 and qc_07 need flight-plan input
   (`Ymerflow#69`).
3. **filter_33**, the manual-edit checkpoint — the only thing that makes re-running safe once
   a human has edited, and now fully specified.
4. **The shared culling policy**, then the culling predicates as small functions over it.
5. **The noise group**, which is where most of the value is.

Specifications live in `building_plans/`. `04_cli_runner.md` records every decision behind
the command line and why the alternatives lost.

---

## License

GPL-3.0. See `LICENSE`.
