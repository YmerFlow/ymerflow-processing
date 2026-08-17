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

Interface layer, channel discovery, and the first QC check.

**Done**

- **Channel discovery (R1).** The number of channels is read from the data, never assumed.
  One moment, two, three or four all take the same path, and a survey with no array
  families at all — a scalar magnetometer — correctly reports none.
- **`core/qc/`** — the Good / Marginal / Out of Spec model with four threshold kinds, the
  per-line and per-flight aggregation, and **qc_01 height above ground** end to end: verdict
  column, denormalized line verdict, structured report, registered under
  `ymerflow.processing.qc`.
- **The CLI.** `yf-process run` walks a pipeline against a delivery and writes the enriched
  dataset plus a JSON report; `compare`, `list` and `validate` alongside it. No platform, no
  web stack, no database — which is what acquisition QC on a schedule actually needs.
- 149 tests, all synthetic, no survey files and no platform required.

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

**A fourth entry-point group: `ymerflow.processing.methods`**

`core/` owns the *vocabulary* (`Channel`, `Axis`) and the *mechanism* (pattern-driven
discovery over `layer_data`). It does not own the array-naming patterns or what the second
axis means — those belong to the method package, which registers an adapter. TEM's adapter
says its arrays are named `dbdt_ch*` / `Gate_Ch*` and its axis is gate time in seconds.

This is what keeps `core/` genuinely method-agnostic rather than holding TEM naming under a
general-sounding label, and it is why adding FEM, mag, rad or gravity later is a new package
plus one adapter, with nothing in `core/` to change.

> **Deviation from `building_plans/`.** `01` and `02` place `Moment` — carrying `gate_times`
> and `dipole_moment` — on the data object in `core/`, while `core/` is specified as knowing
> nothing about gates or moments. The two cannot both hold. Resolved by splitting on
> structure versus physics: identity and array addressing are agnostic and live in `core/`;
> gate times and dipole moment are time-domain and live in `em/tem/`. The plans should be
> updated to match.

Next, in order:

1. **The remaining QC checks** — qc_02 attitude, qc_03 speed, qc_04 transmitter stability,
   and the rest. The `centered`, `descending` and `window` threshold kinds are built and
   tested but not yet used by any check. qc_06 and qc_07 wait on flight-plan input
   (`Ymerflow#69`).
2. **The shared culling policy**, then the culling predicates as small functions over it.
3. **The noise group**, which is where most of the value is.

Specifications live in `building_plans/`. `04_cli_runner.md` records every decision behind
the command line and why the alternatives lost.

---

## License

GPL-3.0. See `LICENSE`.
