# Filter inventory — functional specification

Seed document for the `ymerflow-processing` rebuild.

Every filter that the existing processing library exposes is described here by **what it does
to a dataset**, not by how it does it and not by what it is called. Each is given a stable
reference (`filter_01` … `filter_33`) so it can be discussed, scoped and ticketed without
carrying the old naming across.

Read this as a requirements list: *these are the behaviors the new library needs.* Nothing
here prescribes an implementation, and several entries note where the behavior should
change rather than be reproduced.

**Provenance note.** This is a functional specification written from observed behavior and
parameter surfaces. Implementations should be written from the mathematics and the
literature, not from the existing source.

---

## Contents

| group | filters | what the group is for |
|---|---|---|
| [1 · Line selection](#1--line-selection) | 01–04 | deciding which flightlines enter the pipeline at all |
| [2 · Geometry and corrections](#2--geometry-and-corrections) | 05–09 | making the data physically correct and consistent |
| [3 · Noise and uncertainty](#3--noise-and-uncertainty) | 10–13 | assigning, propagating and simulating error |
| [4 · Data management](#4--data-management) | 14–19 | moving and reshaping columns and arrays |
| [5 · Culling and QC](#5--culling-and-qc) | 20–32 | disabling data that should not reach the inversion |
| [6 · Manual editing](#6--manual-editing) | 33 | checkpointing and human-in-the-loop edits |
| [7 · Gaps](#7--gaps-filters-that-do-not-exist-yet) | — | what is missing and needs building |

Grouping starts from the functional split Egil used for the AEM26 processing-steps slide, so
the documentation, the product and the talk describe the chain the same way.

**Treat it as proposed, not settled.** It is well thought through, but it describes what exists
rather than what should, and writing the inventory surfaced several places where it strains:

- **"Culling (QC)" now collides with actual QC.** Acquisition QC is a separate thing with its
  own runner and its own outputs. Culling *disables* data; QC *judges* it. The parenthetical
  muddies exactly the distinction R6 draws.
- **filter_13 is in the wrong group.** It sits with the error-assignment filters, which is right
  by subject and wrong by purpose — it fabricates noise for synthetic data and would never run
  on a measured survey. It belongs to the forward-modeling chain.
- **Data management is not a workflow stage.** All of its filters exist because something
  arrived in the wrong shape at import. It is an escape hatch, and placing it inline with
  genuine stages implies a pipeline position it does not have.
- **filter_20** applies the instrument's own statement about unusable gates, which is arguably a
  correction rather than a quality judgment.
- **filter_10** is grouped under noise but also decimates, and decimation changes dataset
  extent — closer to selection. It is two operations bundled.
- **Nothing covers export.** The chain ends at manual editing, but getting data out — vector,
  XYZ, whatever the deliverable is — is a real stage with real decisions in it, including the
  output CRS.

The grouping feeds three things: the step picker, these documents, and the talk. If it changes,
the slide changes too.

---

## Cross-cutting design requirements

These apply to every filter and are the main reasons this is a rebuild rather than a port.

### R1 — One to N moments

The existing library assumes a dual-moment instrument almost everywhere. Channels appear as
hardcoded pairs, defaults name two of them explicitly, and several filters accept a
dictionary keyed by two channel names.

**The new library must treat the number of moments as data, not structure.** A single-moment
system (HeliTEM, VTEM, Xcite) and a three-moment system must both work with no special
casing. Concretely:

- a filter operating on "the data" operates on **every** moment present unless told otherwise
- a filter accepting per-moment settings accepts a mapping of arbitrary length, and a scalar
  means "apply to all"
- no default may name a specific moment
- discovering which moments exist is a property of the dataset, asked once, not assumed

This also opens the door to multi-**component** data (X/Y/Z), which several instruments
deliver and which is a different axis from moment count. The abstraction should not
accidentally foreclose it.

### R2 — Readability is a requirement, not a preference

This library is the part of the product a user will read when they want to know what a step
actually did. It is also what a reviewer reads when asking where the code came from.

- one filter per file where the filter is non-trivial
- a docstring that states what the filter does to the dataset, in the same voice as this
  document
- the mathematical source cited in the docstring — paper, textbook, or derivation
- no clever vectorisation without a comment explaining the plain-language equivalent
- parameters named for what they mean, not for what they are indexed by

### R3 — Every filter returns a log of what it did

**Each filter must return a structured record of its own effect**, alongside the data. Not a
printed line — a value, which the runner collects, the process record stores, and the UI can
render.

Minimum content for every filter:

- how many soundings and how many individual data points it changed, out of how many
- for anything with a threshold: the **distribution of the quantity that was thresholded on**,
  before and after. This is what turns "disabled 4,182 gates" into "here is the spread of
  uncertainty values, here is where you cut, here is what that removed."
- for value-modifying filters: the magnitude of the change, not just that it happened
- parameters actually used, including defaults that were not supplied

Two of the four outstanding upstream requests are this same request, arriving separately —
one asking for uncertainty statistics before and after culling, one asking for the
distribution of usable gates per sounding. They are not two features. They are one missing
capability observed twice.

#### Why not a verbose flag

The current library has **336 `print()` calls**, of which roughly **46** sit behind a
`verbose` check — so about **86% print unconditionally**. There is **no logging framework at
all**; not one `import logging` in the package. Only about ten of the thirty-three filters
expose a `verbose` parameter, so it cannot be set consistently even where it exists.

`culling.py` alone accounts for 110 of those prints.

Beyond the inconsistency, `print()` writes to stdout — which, for a job running in a
container, lands in container logs. Not in the process record, not in the UI, not queryable,
and gone when the pod is reaped. Those 336 lines produce output almost nobody can act on.

Adding a `verbose` parameter to the remaining twenty-three filters would make that
consistent without making it useful. The two needs are genuinely different and should be
built differently:

| need | mechanism |
|---|---|
| **what the filter did** — counts, distributions, parameters | structured value returned by the filter |
| **diagnostics while developing or debugging** | standard `logging`, one level set globally |

The first is data and belongs to the run. The second is text and belongs to the operator.
Neither is a per-filter boolean.

### R4 — Idempotency and provenance

Re-running a filter on already-filtered data must be safe and detectable. Several existing
filters manage this by stashing original values under a suffixed name; whatever mechanism
replaces that should be uniform rather than per-filter.

### R5 — Build against current pandas

The upstream library declares no pandas upper bound and has a large `inplace=True` surface
that is unsafe under Copy-on-Write. The new library should assume pandas 3.x semantics from
the first commit: no `inplace=True`, no reliance on chained assignment, and no dependence on
grouping columns surviving inside `groupby().apply()`.

### R6 — Who writes what

The governing rule for the whole library. Easier to hold than any grouping label, and it tells
you immediately which runner a new step belongs to.

> **QC writes attributes. Culling writes the mask. Manual edits override the mask, and survive
> re-runs because they replay as a diff.**

**QC adds information; culling acts on it.** A QC check writes a per-sounding verdict as a
column in `flightlines`. Adding it changes nothing about what the inversion sees — you can run
every check and the data going into processing is byte-identical. Culling writes the in-use
mask, which directly changes what is used.

That separation buys three things:

- QC can run without committing to any culling decision.
- The verdict is auditable independently of the action. Someone can disagree with the culling
  and re-derive a different decision from the same columns, because the judgment is still there.
- The client receives the QC — classification, exported geometry — whether or not anything was
  culled.

**Verdicts are facts; decisions are overrideable.** A QC column records what happened: this
sounding was flown at 78 m against a 65 m spec. That is a measurement, and it is what the
client sees in the exported file. Hand-editing it would falsify the record. What is legitimately
changeable is the *threshold*, which is fully recorded since R3 puts the parameters actually
used into the structured record.

**Ordering matters for overrides.** Automated culling runs first, the manual edit replays last.
Because the manual edit is stored as a diff rather than baked into state, re-running the
pipeline does not silently undo it — which is the standard failure mode when automated and
manual edits mix.

**Unresolved:** there is nowhere to attach a *reason* to an override — "line 401 flew high over
the ridge, accepted by the client." That is not a verdict change and not a mask change; it is a
third thing, an annotation. It is what someone will want the first time a client asks why a
flagged line was kept.

### R7 — One selection idiom: `criterion`

Wherever a filter needs to know *how* something was chosen, it takes a **`criterion`** naming
the method, and only the parameters belonging to that criterion. A user who learns the idiom
once knows it everywhere.

It appears in at least three places:

```yaml
- Classify lines:   {criterion: height_threshold, threshold: 500, line_type: High-Altitude}
- Select lines:     {criterion: line_type, line_types: [Production]}
- Cull soundings:   {criterion: qc_verdict, column: qc_height, reject: [Out of Spec]}
- Cull soundings:   {criterion: threshold, column: height_agl, kind: ascending, max: 75}
```

**Never a boolean.** A flag like `auto_high_altitude: true` bakes one specific rule into a
parameter *name*, so a third option forces either a second flag that can contradict the first
or a rename that breaks existing pipelines. It also produces the failure where half the
parameters are dead depending on the flag and nothing in the signature says which half.

#### Why culling keeps its own threshold criterion

Culling can consume a QC verdict, but it must not be limited to that, because the two
thresholds answer different questions:

| | source of the number | question it answers |
|---|---|---|
| **QC threshold** | the survey contract | did the contractor meet spec? |
| **culling threshold** | the physics and the target depth | is this data usable? |

These legitimately differ. A contractor flying at 70 m against a 65 m spec is Out of Spec — the
client cares, it may be a payment conversation — but the data is very likely still fine to
invert. Forcing one number would mean either discarding usable data to register a breach, or
softening the contract threshold and losing the breach from the record.

With both criteria available, the QC column still records 70 m against a 65 m spec regardless of
what was culled. The breach stays on the record and the data stays in the inversion.

### R8 — External inputs are typed, not paths

Eight steps take an input that is not the dataset being processed:

| step | external input |
|---|---|
| filter_05 | terrain raster |
| filter_30 | infrastructure geometry |
| filter_31 | infrastructure geometry **and** inversion results |
| qc_01 | terrain model |
| qc_06, qc_07 | planned line geometry |

Today these are bare filesystem paths. filter_30's default parameter value is literally the
string `'/path/to/my/shape/file.shp'`, which summarizes the problem — designed for a desktop.

**The filter declares a *type*, not a format, and reads through a standard loader.**

Not `shapefile: str`, but something like `infrastructure: VectorGeometry` — "a path to
something readable as vector geometry." The filter opens it and does not care whether it was a
shapefile, a GeoPackage, or a KML. Three types cover every case above: **vector geometry**,
**raster**, and **dataset**.

Consequences:

- **No placeholder-path defaults.** A required external input has no default.
- **Format is a loader concern**, not a filter concern. Adding a format touches the loader, not
  eight filters.
- **The library stays standalone.** The parameter is a path to a readable file, so the CLI
  runner works with local files and no platform. That is what the scheduled field QC case and the
  federal-deployment case both need.

#### What is deliberately *not* here

Versioned asset storage — upload, versioning, normalization on ingest, resolving a reference to
a concrete file — is a **platform** capability, not a library one. If a filter's parameter were
an asset identifier, the library could not resolve it without the platform and standalone use
would die.

The layering: **the platform materialises, the library reads.** The pipeline document the
platform stores may name an asset; the invocation passes a path. Tracked as `Ymerflow#69`.

### R9 — One noise model, held by the dataset, estimated from the data

The noise model is defined **once**, as a property of the dataset, by an explicit estimation
step. Every consumer reads it. No filter carries its own copy of the floor or the slope.

#### Why this is not merely tidier

Today two filters encode the same physical statement independently, and their defaults
disagree: the error-assignment filter defaults the floor to `None` (falling back to the system
description) while the noise-floor culling filter defaults it to `1e-8`. The former's own
docstring gives empirical values of ~5e-10 and ~1.5e-9 for a SkyTEM 304 — so the culling
default is roughly seven to twenty times higher than what the sibling documents as realistic.

Nothing detects the disagreement. You can assign uncertainty using one floor and cull using
another, and the run completes.

A single definition makes that unrepresentable rather than merely discouraged.

#### The three parameters

| parameter | source | overrideable | estimated |
|---|---|---|---|
| **absolute floor** at 1 ms | estimated from late gates | yes | yes |
| **slope** | `-0.5` default | yes | yes, fittable |
| **relative fraction** | system description (`UniformDataSTD`) | yes | **no** |

**The relative fraction is never estimated.** It is systematic — calibration, geometry,
primary-field removal — and has no separable signature in the data the way the ambient floor
does at late gates. Estimating it would be inventing a number with a measurement's authority.

**Overrides are optional parameters, not a mode switch.** Absent means estimate, supplied means
use. This matters because the common middle case — estimate the floor, hold the slope at −0.5 —
cannot be expressed by an either/or:

```yaml
- Estimate noise model:
    scope: per_flight          # per_line | per_flight | per_dataset
    noise_level_1ms: null      # null -> estimate
    noise_exponent: -0.5       # supplied -> use, do not fit
    relative_fraction: 0.03
```

This does not conflict with R7. R7 governs *how something is selected*; here no parameter ever
becomes meaningless — each is either the value to use or absent meaning "estimate."

#### An override changes what is applied, never what is measured

Even when a value is supplied, estimate it anyway and report both. "You set 1e-9; the data
suggests 3e-9" is itself a finding — either the number came from a different survey, or
something is noisier than expected. Suppressing the measurement because a value was given
discards the most useful diagnostic in the step, and R3 already requires reporting the
distribution the step worked from.

Same for slope: a fitted value far from −0.5 usually means the late gates are not ambient noise
at all.

#### Scope: estimate fine, apply coarse

`per_line | per_flight | per_dataset`. **Not** "per survey" — the library has no concept of a
survey, `Survey` is already the data object's class name, and a real-world campaign may map to
one dataset, part of one, or several.

**Estimate per flight**, because that is where the physics changes — instrument state and the
day's conditions — and it is the granularity acquisition QC needs for drift detection.

**Apply per dataset by default.** A spatially varying error model is a spatially varying
weighting, and the inversion will honour it: different floors on adjacent lines flown on
different days can produce visible seams at flight boundaries. That is an artifact created by
the processing.

**Report the spread.** If per-flight estimates agree within noise, applying the dataset value
costs nothing. If one flight is 3× the rest, that is a QC finding — and only then is a
per-flight floor, or a reflight, the right response.

**The scope trap worth knowing:** if flights are processed as separate datasets and inverted
together, then `per_dataset` *is* per-flight and the seams return without anyone selecting a
per-flight scope. This is a workflow property, not a parameter, and it will not announce
itself.

#### Where the step sits: before averaging

1. **Estimate** the raw floor
2. **Assign** uncertainty from it
3. **Average**, which propagates uncertainty and correctly shrinks the ambient part by √N
4. **Cull**

Estimation must precede assignment, and assignment must precede averaging so that averaging has
uncertainties to propagate.

**A consequence worth taking seriously:** by step 4 the data already carries its correct
post-averaging uncertainty. Culling should threshold *that*, not re-derive a floor and compare
against it — re-deriving would use the pre-averaging number and over-cull.

Which means "below the noise floor" and "fractional uncertainty ≥ 1.0" become the same test
whenever the error model is trustworthy.

**DECIDED: the noise-floor culling filter is kept**, as a distinct step rather than folded into
the STD-threshold filter. It makes a direct physical statement — *this datum is below the
ambient noise* — that does not depend on the error model being right. That independence is the
point: it is what you reach for when uncertainties have not been assigned yet, when the
assigned model is suspect, or when you want a check that cannot be argued with by adjusting a
different parameter.

Document it as such. Under a correct error model the two tests agree, and that agreement is
itself a useful signal — a large divergence between them means the error model and the physics
disagree, which is worth surfacing rather than hiding behind one filter.

#### Known defects this design removes

- the disagreeing defaults described above
- the system description's uniform value being interpreted as a dimensionless fraction in one
  code path and an absolute amplitude in another, which in one branch assigns ~100% uncertainty
  to every datum (`Ymerflow#70`)
- a docstring documenting a geometric mean while the code performs quadrature (`Ymerflow#63`)

Three defects in one function, and it is the function that decides what the inversion is
allowed to believe. A reasonable argument for rebuilding the noise group immediately after the
interface, ahead of the culling predicates.

This also gives `Ymerflow#29` — "no way to inspect data for the applicable noise floor,
currently guesswork" — a real answer.

### filter_31 specifically

With R8 in place, filter_31's remaining oddity is that it depends on a **prior run of the same
chain** rather than on a static asset — process, invert, cull on misfit, invert again. A loop
expressed inside a linear pipeline.

**Resolution:** the comparison runner computes misfit between the inversion and the measured
data and writes it as a column. filter_31 then stops being special — it becomes
`cull where near_infrastructure AND misfit > threshold`, both columns on the dataset, thresholded
like any other culling filter.

That makes the loop explicit as steps rather than one filter reaching backwards, and misfit
becomes generally available — plottable, exportable in the vector output, usable by QC — instead
of being trapped inside one culling function.

The cost is that it is no longer a single step. That friction is absorbed by shipping the
sequence as a standard workflow rather than expecting a user to assemble it.

---

## Method tier — PROPOSED, not settled

Which filters are method-agnostic, which are shared across EM methods, and which need
transients. This is a reading of the behaviors below, not an established fact — worth
disagreeing with, and it feeds the package-layout decision in `02_pipeline_and_entry_points.md`.

| tier | filters | count |
|---|---|---|
| **agnostic** — any airborne survey | 01–05, 11, 13, 14–19, 22, 24, 30, 31, 33 | 18 |
| **EM-shared** — TEM *and* FEM | 06, 07, 08, 09, 20 | 5 |
| **TEM** — needs gates, moments or transient decay | 12, 21, 23, 25–29, 32 | 9 |
| **splits** | 10 | 1 |

The agnostic group is larger than expected because the whole data-management group is
structural, line selection applies to anything flown, and the two most broadly useful culling
filters are method-blind: attitude and altitude limits, and cultural-feature exclusion.
Powerlines do not care what is being measured.

**filter_10 splits.** The mechanism — along-line weighted averaging with uncertainty
propagation — is entirely generic. Only the parameterisation is TEM-specific, in that the
window width tapers with gate time. The engine belongs with the agnostic filters and the
per-gate widths belong with TEM. Any other filter that looks method-specific only because of
how it is configured should be treated the same way.

> **Added 2026-08-15.** On reflection at least two of the five below are misassigned, and the
> reasoning is physics rather than institutional memory:
>
> - **filter_20** applies the instrument's declaration of unusable *gates*. Frequency-domain
>   data has frequencies, not gates. That is time-domain.
> - **filter_08** normalises to what the forward code expects — response per unit transmitter
>   dipole moment for TEM, whereas frequency-domain work conventionally uses ppm of the primary
>   field. The *idea* is shared; the filter is not.
>
> So the EM-shared tier is probably three, not five. Left as originally written above rather
> than corrected, since the tier table is explicitly a proposal — and the assignment is
> **reversible**: moving a file from `em/` to `em/tem/` is a file move. Decide when the first
> filter in each tier is written, revise when frequency-domain code actually lands.

**The five EM-shared filters are the reason the layout question exists.** Transmitter attitude
correction, the horizontal-transmitter assumption, forward-model normalization, filling
current from the system description, and applying the system mask are all needed unchanged by
frequency-domain work. If they sit inside a TEM package, FEM either imports across from a
sibling or duplicates them.

---

## 1 · Line selection

*Deciding which flightlines enter the pipeline at all. These run first, because everything
downstream is cheaper on less data.*

### filter_01
**Reads:** attributes (height above terrain) · **Writes:** attributes (line classification)

Takes an input dataset and labels flightlines whose mean height above the terrain exceeds a
threshold, marking them as high-altitude test lines rather than production lines. Nothing is
removed — this only classifies, so a later selection step can act on the label.

**Parameters:** a height threshold in meters above ground.

### filter_02
**Reads:** — · **Writes:** attributes (line classification)

Takes an input dataset and assigns a caller-supplied classification to a caller-supplied set
of flightlines. The classification is free-form rather than drawn from a fixed vocabulary, so
a survey can define its own line categories. Lines carry a default classification until this
sets otherwise.

**Parameters:** a list of line identifiers, and the label to apply to them.

### filter_03
**Reads:** attributes (line id) · **Writes:** **drops soundings** — changes dataset extent

Takes an input dataset and reduces it to only the flightlines named by the caller. This is a
subsetting operation — everything not named is dropped from the working dataset.

**Parameters:** the list of lines to keep.

### filter_04
**Reads:** attributes (line classification) · **Writes:** **drops soundings** — changes dataset extent

Takes an input dataset and reduces it to only those flightlines carrying one of the
classifications named by the caller. Complements filter_02, which assigns the labels this
one selects on.

**Parameters:** the list of classifications to keep.

> ### DECIDED — these four collapse to two
>
> **`Classify lines`** (was filter_01 + filter_02) and **`Select lines`** (was filter_03 +
> filter_04). Each takes a **`criterion`** naming how the lines are chosen, and only the
> parameters belonging to that criterion.
>
> ```yaml
> - Classify lines:
>     criterion: height_threshold     # or: explicit
>     threshold: 500
>     line_type: High-Altitude
>
> - Select lines:
>     criterion: line_type            # or: explicit_lines
>     line_types: [Production, Reflight]
> ```
>
> **A named criterion rather than a boolean.** With exactly two options a bool is equivalent,
> but `auto_high_altitude: true` bakes one specific rule into a *parameter name* — so a third
> criterion, such as classifying by date flown or by flight number, forces either a second
> boolean that can contradict the first, or a rename that breaks existing pipelines. A
> criterion field takes a new value and nothing else changes.
>
> It also avoids the boolean-that-switches-which-other-parameters-matter problem, where half
> the parameters are dead depending on a flag and nothing in the signature says which half.
> Each criterion carries only its own parameters, so the form shows what applies and the
> docstring documents two clear modes.
>
> **Terminology: `line_type`, not `tag`.** `tag` is already taken — `ProcessTag` is a
> first-class model with its own table, CRUD router and `TagSelector` widget, meaning labels on
> process versions. Reusing the word for flightline classification would collide inside the same
> UI. `line_type` is also more accurate than the existing `flight_type`, since a flight contains
> many lines and the classification applies per line.
>
> **Values stay free-form**, not a fixed vocabulary — a survey can define its own line types,
> with Production as the default until something sets otherwise. That flexibility exists today
> and should survive the consolidation.

---

## 2 · Geometry and corrections

*Making the data physically correct and internally consistent before anything is derived
from it.*

### filter_05
**Reads:** attributes (altitude), external terrain raster · **Writes:** attributes (altitude, elevation)

Takes an input dataset and a digital terrain model, and corrects instrument altitude and
topographic elevation so that heights are expressed consistently relative to the ground
surface rather than to whatever datum the raw data used.

**Parameters:** path to a terrain raster in meters above mean sea level.

### filter_06
**Reads:** values, attributes (attitude) · **Writes:** **values**, attributes (stashes originals)

Takes an input dataset in which the transmitter was not perfectly level and rescales the
measured amplitudes to what a horizontal transmitter would have produced, using the recorded
attitude angles. This exists because the 1D forward model assumes a horizontal loop; without
it, tilt is indistinguishable from geology.

Must be idempotent — applying it twice would apply the correction twice.

### filter_07
**Reads:** attributes (attitude) · **Writes:** attributes (attitude, stashes originals)

Takes an input dataset and forces the recorded transmitter attitude angles to zero, retaining
the originals. Used when measured attitude is unreliable or was never recorded, so that a
downstream tilt correction becomes a no-op rather than applying garbage. If the attitude
fields are absent entirely, they are created as zero.

### filter_08
**Reads:** values, system description · **Writes:** **values**

Takes an input dataset and converts its amplitudes into the normalization convention the
forward modeling code expects — response per unit transmitter dipole moment. Data arrives in
one of several conventions (raw instrument units, receiver-area normalized, or fully
normalized by both receiver area and transmitter moment), and the inversion misfit is
meaningless unless observed and predicted data share one convention.

**Parameters:** which convention the input is currently in.

> This filter is where unit confusion concentrates. The three conventions differ by orders of
> magnitude and are easy to mistake for one another. The rebuild should make the current
> convention an explicit property of the dataset that travels with it, rather than something
> the user asserts at this step and hopes is right.

### filter_09
**Reads:** system description · **Writes:** attributes (transmitter current)

Takes an input dataset and populates the per-sounding transmitter current from the nominal
value in the system description, overwriting whatever was measured. Used when measured current
is missing, noisy, or known to be wrong.

---

## 3 · Noise and uncertainty

*Assigning, propagating and simulating error. This group determines what the inversion is
allowed to believe.*

### filter_10
**Reads:** values, uncertainty, mask · **Writes:** **values, uncertainty** — and optionally drops soundings if decimating

Takes an input dataset and replaces each sounding's values with a weighted average over its
along-line neighbours, propagating uncertainty through the average. The averaging width can
vary with gate time — narrow at early gates, wider at late gates — so that late-time data,
which is noisier, is smoothed more.

Where per-datum uncertainties exist they inform the weighting; where they do not, the spread
within the window is used to estimate one. Output always carries uncertainty as a fraction of
the value.

**Parameters:** per-moment widths at the first and last gate; an averaging method; a minimum
fraction of the window that must contain valid data before an output is produced; optional
decimation of the result to a target along-line spacing.

> **This is the most consequential filter in the library and the one most worth rebuilding
> rather than porting.** Known issues to design out:
> - a width of one silently produces all-invalid output instead of being a no-op
> - the validity gate counts samples rather than effective sample size, which is wrong for
>   any non-flat weighting
> - the weighting kernel is flat, so a neighbour at the edge of the window counts as much as
>   the sounding itself; a distance taper preserves narrow targets far better
> - the implementation is a nested per-gate, per-sounding loop over row slices, which is
>   orders of magnitude slower than a strided or convolution-based formulation

### filter_11
**Reads:** uncertainty · **Writes:** uncertainty

Takes an input dataset and adds a fixed fractional uncertainty to a specified range of gates —
for example, inflating uncertainty by 2% across all gates, or over a chosen subrange. Negative
values reduce it.

**Parameters:** which moment and gate range; the fraction to add.

### filter_12
**Reads:** values, system description · **Writes:** uncertainty

Takes an input dataset and replaces its uncertainty estimates with a noise model, rather than
adjusting whatever was already there. The model has two components which combine in
quadrature:

- an **absolute floor** — ambient environmental noise, independent of signal strength, decaying
  with time as a power law. Expressed before division by transmitter moment, so it must be
  normalized consistently with the data before use.
- a **relative fraction** — calibration, geometry and system error, proportional to signal.

Supplying neither uses a uniform value from the system description. Supplying either or both
selects the corresponding model.

**Parameters:** which moment; the floor amplitude at a reference time; the power-law exponent;
the relative fraction.

> The distinction matters more than it looks. A relative-only model never lets data die — the
> fractional uncertainty stays flat forever and every gate looks usable, including gates that
> are pure noise. The absolute floor is what makes late gates fail honestly.
>
> Note also that the floor is a property of **how much stacking produced the sounding**, not
> of the instrument alone. Data exported at a coarser along-line interval has a proportionally
> lower floor. The rebuild should ideally derive this from the data rather than requiring the
> user to supply a number they got from someone else's survey.

### filter_13
**Reads:** values, uncertainty · **Writes:** **values**

Takes an input dataset that has uncertainties assigned and adds a random perturbation to each
datum drawn from those uncertainties. Used to turn a noise-free forward model into something
resembling measured data, so that an inversion of it is a fair test.

**Parameters:** which moment; a random seed for reproducibility.

> Order matters and is easy to get wrong: uncertainties must be assigned before noise is
> drawn from them, and any along-line averaging belongs after both. The rebuild should make
> a wrong ordering detectable rather than silently producing plausible nonsense.

---

## 4 · Data management

*Moving and reshaping the dataset's own structure. These exist to work around import
problems and format mismatches, not to do geophysics.*

Two kinds of thing are addressed: **per-sounding attributes** (one value per sounding —
position, altitude, attitude, current) and **per-gate arrays** (a value per sounding per
gate — the measurements themselves, their uncertainties, and their in-use flags).

### filter_14
**Reads:** attributes · **Writes:** attributes (structural)

Takes an input dataset and copies a per-sounding attribute to a new name, leaving the
original in place. Overwrites the destination if it exists.

### filter_15
**Reads:** arrays · **Writes:** arrays (structural)

Takes an input dataset and copies a per-gate array to a new name, leaving the original in
place. Overwrites the destination if it exists.

### filter_16
**Reads:** attributes · **Writes:** attributes (structural)

Takes an input dataset and renames a per-sounding attribute.

### filter_17
**Reads:** arrays · **Writes:** arrays (structural)

Takes an input dataset and renames a per-gate array.

### filter_18
**Reads:** attributes · **Writes:** attributes (structural)

Takes an input dataset and removes a per-sounding attribute.

### filter_19
**Reads:** arrays · **Writes:** arrays (structural)

Takes an input dataset and removes a per-gate array.

> ### DECIDED — these six collapse to three
>
> **`Copy`**, **`Rename`**, **`Drop`**. Each resolves attribute-versus-array from the name
> rather than making the user assert it.
>
> The six read word for word identically in pairs, differing only in whether the target is a
> per-sounding attribute or a per-gate array — a distinction the dataset already knows, since
> attributes live in `flightlines` and arrays in `layer_data`. Asking the user to restate it at
> every call is asking them to compensate for the library not looking it up.
>
> **A name present in both is an explicit error** — "ambiguous: present in both attributes and
> arrays" — not a silent preference. It is rare, it means something is wrong upstream, and
> reporting it is more useful than guessing.
>
> **Not collapsed further to one.** A single `Restructure` step taking an operation parameter
> was considered and rejected: the step name stops carrying meaning (`- Drop: {target: X}` reads;
> `- Restructure: {operation: drop, target: X}` needs unpacking, and three identically-labeled
> boxes in a flow view are worse than three named ones), copy/rename/drop are already universal
> verbs needing no explanation, and the parameter shapes differ — copy and rename take a source
> and a destination, drop takes only a target — which reintroduces the dead-parameter problem
> that `criterion` was chosen to avoid.
>
> **Grouping is a display concern, not a filter-identity concern.** The conceptual grouping is
> achieved by presenting these under a "Data management" heading, not by merging them. That
> keeps the UI regroupable without touching the library, and the same argument says the culling
> predicates stay separate predicates under a heading rather than becoming one filter with a
> thirteen-valued mode.
>
> **Carry over the overwrite warning.** All six currently warn when they will overwrite an
> existing destination. Under R3 that belongs in the structured record — "overwrote existing X"
> — rather than a printed line.
>
> **Note why these exist.** Every one is a workaround for an import problem. If import improves,
> demand for them falls. Worth stating so they are not mistaken for core capability.

---

## 5 · Culling and QC

*Disabling data that should not reach the inversion. Nothing here deletes — each marks data
as not-in-use, which is reversible and auditable.*

Most filters in this group share three behaviors worth implementing once rather than
thirteen times:

- **a starting gate** — evaluation begins at a nominated gate and considers everything after it
- **spread-earlier** — when a problem is found, optionally also disable some number of gates
  before it, since a bad gate usually means its neighbours are suspect
- **tail disabling** — when a problem is found, optionally disable every later gate in that
  sounding, on the reasoning that a transient does not recover once it has gone bad

In the rebuild these should be a shared concern — a policy applied around a predicate — not
copy-pasted parameters. **Each culling filter is then just a predicate over the data.**

### filter_20
**Reads:** system description · **Writes:** mask

Takes an input dataset and disables the early gates that the system description declares
unusable — the instrument's own statement about which gates are contaminated by the
transmitter turn-off.

### filter_21
**Reads:** — · **Writes:** mask

Takes an input dataset and enables or disables one nominated gate outright, by index. A manual
override rather than a criterion.

### filter_22
**Reads:** attributes (attitude, altitude) · **Writes:** mask

Takes an input dataset and disables entire soundings whose recorded attitude or altitude falls
outside acceptable bounds — too much roll, too much pitch, too high, or too low. These are
soundings where the geometry assumption underlying the forward model has broken down.

**Parameters:** maximum roll, maximum pitch, maximum and minimum altitude.

### filter_23
**Reads:** mask · **Writes:** mask

Takes an input dataset and disables any sounding left with fewer than a minimum number of
usable gates. A sounding reduced to a handful of points cannot constrain a model, and leaving
it in produces confident nonsense.

Runs late, since it depends on what earlier culling removed.

### filter_24
**Reads:** uncertainty · **Writes:** mask

Takes an input dataset and disables individual data points whose assigned uncertainty exceeds
a threshold, expressed as a fraction of the value.

### filter_25 / filter_26
**Reads:** values, mask · **Writes:** mask

Take an input dataset and disable points where the local decay rate of the transient — the
slope in log-time, log-amplitude space — is shallower than a maximum (filter_25) or steeper
than a minimum (filter_26).

This is a physical plausibility test. A transient decaying far too steeply or far too shallowly
is not describing an earth; it is describing noise, coupling, or an instrument problem. For
reference, a homogeneous half-space and ambient noise have characteristically different slopes,
and real data lies between them.

### filter_27 / filter_28
**Reads:** values, mask · **Writes:** mask

Take an input dataset and disable points where the local curvature of the transient — the
second difference in log-log space — exceeds a maximum (filter_27) or falls below a minimum
(filter_28).

Where slope tests catch data decaying at an impossible rate, curvature tests catch data
*changing* its decay rate impossibly fast. Curvature is undefined at the first and last gate.

### filter_29
**Reads:** values · **Writes:** mask

Takes an input dataset and disables data falling below an assumed ambient noise floor — a
power law in time with a specified amplitude at a reference time. Below the floor, the
instrument is measuring the environment rather than the ground.

**Parameters:** the floor amplitude at a reference time; the power-law exponent.

> Shares its noise model with filter_12, and the two should share an implementation. It is the
> same physical statement used once to assign uncertainty and once to cull.

### filter_30
**Reads:** attributes (position), external geometry file · **Writes:** mask

Takes an input dataset and a set of geographic features, and disables data within a distance
of those features — powerlines, pipelines, fences, railways. Cultural conductors couple into
the receiver and produce anomalies that are real measurements of the wrong thing.

The exclusion distance can widen with gate time, since late-time data senses a larger volume
and is affected from further away.

**Parameters:** a geometry file; per-moment distances at the first and last gate.

### filter_31
**Reads:** attributes (position), external geometry file, external inversion result · **Writes:** mask

Takes an input dataset, a set of geographic features, and the results of a previous inversion,
and disables data that is both near a feature **and** poorly fit by the inversion. Proximity
alone over-culls, since much data near infrastructure is fine; misfit alone over-culls, since
a poor fit may indicate genuine geology the model cannot represent. The conjunction is more
selective than either.

**Parameters:** a geometry file; a search distance; a misfit threshold; the inversion inputs
and outputs to compare.

> This filter depends on an inversion having been run, which makes it the only culling step
> that is not a pure function of the input dataset. Worth deciding whether the rebuild keeps
> that coupling or expresses it as a separate feedback stage.

### filter_32
**Reads:** values · **Writes:** mask

Takes an input dataset and disables negative measurements after a nominated gate. For a
standard transient the response should be positive throughout; a sign change indicates
coupling, an IP effect, or noise dominating.

---

## 6 · Manual editing

### filter_33
**Reads:** everything · **Writes:** mask, plus a named snapshot

Takes an input dataset, saves it as a named intermediate result, and applies a manually
authored set of edits to it. The edits are expressed as a difference against that saved state
rather than as a replacement of it, so a human edit made in a viewer can be replayed as part
of an automated pipeline without discarding everything computed since.

This is what makes the whole chain reproducible in the presence of human judgment. Without
it, any manual editing forces the pipeline to be abandoned in favor of a hand-edited file.

**Parameters:** a name for the intermediate state; the edit set to apply.

> The manual step is not necessarily last. It commonly happens mid-chain and again at the end,
> which is why it is expressed as a checkpoint that can appear anywhere rather than a final
> stage.

#### Added 2026-08-15 — the diff format already exists

Not a revision of the above; new information, found by reading the platform rather than the
old library.

**The edit set is a sparse dataset, not a bespoke file format.** The plot editor
(`InUseEditor.jsx` → `saveAllDiffs` in `ProcessContext.jsx`) already produces one, and it is
already wired end to end:

- a `libaarhusxyz` msgpack XYZ, named `inuse_diff_{dataset}.msgpack`
- `flightlines.apply_idx` — the global sounding indices the edit touches, so the diff carries
  its own addressing rather than depending on row order surviving
- `layer_data["InUse_{channel}"]` — one array per gate key, with **`NaN` meaning "no
  override"**
- `model_info.diff_dummy` as a marker

The file is uploaded and a **new process version is created with `diff` as a parameter**, so
the edit becomes an input to a re-run. That is the replay-as-diff behaviour R6 describes,
already implemented on the platform side.

**So this filter's job is narrower than the description above suggests:** read that sparse
XYZ, and apply its non-NaN values at the soundings named in `apply_idx`. No format to design.

**It carries only `InUse_*`.** The mask, nothing else — which agrees exactly with R6. Values
are not overridable by hand, and that is coherent rather than a gap: a hand-edited *value*
would be unattributable, whereas a hand-disabled *datum* is a judgement with a reason.

**DECIDED 2026-08-15 — one step**, as the old library had it.

The coupling is real: a diff is meaningless without knowing which state it is a difference
from, so splitting them would require something to name the checkpoint each diff belongs to —
inventing a reference to replace one that is implicit in the pairing. Saving a checkpoint
alone is a plausible want, but it is a smaller want than the complexity of that reference.

**Not to be confused with** the inversion process's "save intermediate models and synthetic
data for every inversion iteration", which already exists and is unrelated — that is
per-iteration inversion output, not a processing checkpoint.

---

## 7 · Gaps — filters that do not exist yet

The inventory above is what exists. These are known to be missing.

### During-acquisition QC

**Now specified separately in `03_acquisition_qc.md`.** Twelve checks, the workflow, and the
three output kinds — profile plots, map views, and vector export.

Summary of what changed since this was a one-paragraph gap: the design largely exists already
in our own prior flight QC work, which is a far better starting point than the old library's
QC modules — those turn out to be nine files of plotters with no thresholds, no pass/fail, and
one instrument-locked scoring function. Its Good/Marginal/Out-of-Spec classification, its
threshold kinds, and its derive-rather-than-hardcode approach to instrument-specific ranges
all carry over directly.

Distinct from the culling group above: the output is a decision with a stated reason, not a
modified dataset. Tolerances are contract parameters and differ per job.

Tracked as `Ymerflow#64`.

### Frequency-domain support

The inventory above is entirely time-domain. Frequency-domain data needs its own import,
system description, and processing steps. Some utilities exist as scripts — format readers,
attitude and geometry culling — but nothing is exposed as a pipeline step.

Needed for the upcoming frequency-domain program. Several filters above are method-agnostic and should be
shared rather than duplicated: line selection, altitude and terrain correction, the data
management group, geometry culling, and the manual edit checkpoint.

### Consolidations requested but not made

Four requests are outstanding against the existing library and should be designed in rather
than retrofitted:

1. **Merge filter_20 and filter_12** into a single system-description step with two switches —
   reset in-use flags, and replace uncertainties — since both read the same file and are
   almost always run together.
2. **Default disabled gates to a neutral value** rather than whatever they currently carry.
3. **Report uncertainty distributions** before and after culling, so a user can see what a
   threshold actually did rather than guessing at it.
4. **Report the distribution of usable gates per sounding** in filter_23, for the same reason.

Requests 3 and 4 are the same request, and generalize to R3 above: filters should report what
they changed.

### Distance-weighted averaging

filter_10 uses a flat kernel. A tapered kernel preserves narrow targets substantially better
for a modest cost in noise reduction, and is not offered by the incumbent desktop tool.
Tracked as `Ymerflow#60`.

---

## Addition, 2026-08-18 — every run opens with an inventory

`01` describes filters as steps a pipeline lists. It assumes, without saying so, that the
channel model is a complete account of the delivery. It is not, and the gap is silent.

**Steps operate through `Channel.array_keys`.** An array no channel claims is not skipped —
it is *unreachable*. Every channel-based step steps over it, which looks exactly like
success. `select_channels(criterion="component", keep=["z"])` reported "nothing removed"
while a non-Z gate array sat untouched in `layer_data`: true of channels, false of the file,
and the docstring's promise that a forward code "will not merely ignore X and Y, it will fail
on them" was not being kept.

**How an array ends up claimed by nobody.** SkyTEM's data files carry a horizontal component,
but **the GEX and the ALC are built for the Z component only**. Nothing declares the horizontal
one, so discovery never registers it. A
dual-moment delivery then carries three gate arrays against two declared channels.

Undeclared is not unusable. That array is real, coherently signed, and decays into noise like
a response rather than like junk. The incumbent toolchain discards it; that is a limitation of
the toolchain, not a property of the data.

So **every run opens with an inventory**, implicitly, before any step acts: what arrived, what
was resolved into channels, and what was not. Implicit rather than something a pipeline lists,
because the thing it exists to surface is precisely what nobody thinks to ask about — a step
that ran only when requested would never run on the delivery that needed it.

It reads and changes nothing, which is what makes it safe at the top of a QC pipeline where R6
promises the data comes out byte-identical.

**Unresolved arrays are reported, not removed.** `select_channels` gained `drop_unclaimed` for
when an operator has looked and wants them gone. Refusing outright was considered and rejected:
the hazard is unconfirmed — the consumers we actually use read specific keys by channel and
would most likely ignore an extra array — and refusing on an unconfirmed hazard would block
real deliveries. Deleting by default was rejected for the stronger reason that it destroys data
we do not understand.

The recovery path for such an array is to **describe it** — amend the system description so
discovery can claim it — rather than to delete it. Tested against a hypothetical GEX channel
declaring `ReceiverPolarizationXYZ=X` sharing the high moment's dipole moment: the existing
component-versus-moment split handles it correctly with no change to the channel model. The gap
was only ever in what happens to arrays outside the model.

## Addition, 2026-08-18 — filter_20 is a mandate, not a step

`01` lists filter_20 among the culling filters, and above it observes that it is "arguably a
correction rather than a quality judgment." That instinct was right, and it has a consequence
the inventory made obvious: **it is applied automatically, not listed in a pipeline.**

Every other culling filter decides something. A threshold was chosen, a predicate objected, an
operator could reasonably disagree. filter_20 reads the instrument's own statement about which
of its samples can be inverted at all. Data outside that window is not data anyone decided to
drop — it is samples the vendor already said are contaminated by the transmitter turn-off.
Applying it is part of reading the file correctly, in the same way the gate times are.

Left optional it is the single easiest thing in the library to forget, and forgetting it is
**silent**: the inversion runs, converges, and is confidently wrong about the shallow section.
Nothing fails, so nothing prompts anyone to look.

So the declaration is applied in `load()` — the moment a survey and a system description become
one thing, which is exactly where "what is in the system description is what you get" becomes a
true statement rather than an instruction someone has to follow — and again at the top of every
run, for a dataset assembled some other way. Both are no-ops on a dataset that already has it,
because culling is subtractive.

It stays registered as an explicit step for the one case the automatic pass cannot serve: a
delivery whose description is known to be wrong, where the declared count is overridden. And
filter_21 puts the gates back for anyone doing full primary-field modeling, which is the reason
this writes a mask rather than removing anything.

**Two consequences worth stating.** A channel declaring *zero* initial gates must not
materialize an in-use array to say so, or the mandate would write arrays into a delivery that
had none — breaking exactly the promise lazy materialization exists to keep. And a description
that disagrees with the delivery raises from the explicit step, where someone asked and is
watching, but only warns from a load: raising there would make an unreadable file out of a
readable one.

## Addition, 2026-08-18 — culling limits are restated, and default from QC

`01` observes that filter_22 "currently applies its own attitude and altitude thresholds" and
suggests sharing them with the QC checks would be "a real simplification". The build restates
them instead, and the reason is worth recording because the suggestion is tempting.

**A contract limit and a physical limit are different numbers.** qc_01 asks whether the
aircraft was flown as agreed; filter_22 asks whether the forward model's geometry assumption
still holds. A sounding at 70 m against a 65 m contract is out of spec and still perfectly
invertible. Forcing the two to be equal makes one of them wrong, and it is not obvious which —
so the operator would be culling to a contract number or reporting compliance against a
physics number, without either being stated.

**But they are related**, and "where did QC stop calling this acceptable" is a sensible
starting point for "where has the geometry broken down". So each limit defaults to the
corresponding QC threshold where one was recorded — the **out-of-spec boundary**, not the Good
one, since Marginal is where QC stopped accepting the sounding and Good is where it stopped
being unremarkable. An explicit value always wins.

That requires a threshold to outlive the run that chose it, since QC and culling are separate
pipeline documents and usually separate invocations. So the QC checks now record what they
judged against on the dataset itself, alongside the verdict-column registry.

**No built-in limits.** Unset, with nothing recorded to default from, filter_22 raises. What
counts as too tilted or too high is an operational judgement about an instrument, and a house
number here would silently cull a client's data.

`min_altitude` has no QC threshold to borrow, because qc_01 is one-sided. It stays unapplied
rather than invented.

## Addition, 2026-08-18 — filter_21 takes a set or a range, per channel

`01` describes filter_21 as enabling or disabling "one nominated gate outright, by index."
Two things were left open, and both are cheap to close.

**A set or a range, not one gate.** One-at-a-time is a limit nobody wants, and R1's argument
applies: the number of gates is data, so the number a person might nominate is too. Accepted
forms are a single index, a list, or a `"first-last"` range, mixable.

**A range is inclusive.** `"0-4"` names five gates. That sits one filter away from
`RemoveInitialGates: 5`, which removes the same five — they agree on the outcome and differ in
what they are, a span against a quantity, which is why one is inclusive at the top and the
other is not. Worth stating because getting it wrong costs exactly one good gate, silently.

**Indices are positions in the delivered window, 0-based** — the same origin as the gate-time
axis, since the two are 1:1. They are *not* the vendor's own gate numbers: `libaarhusxyz`
normalizes the origin away at parse time, so the vendor's numbering is not recoverable from a
loaded dataset and cannot be what this addresses.

**Two parameters, not one plus a direction flag.** R7's argument against booleans applies —
`enable: false` reads as "do not enable" rather than "disable", and a reader should not have to
work out which.

**Enabling is a guarded exception, not a general capability.** `core/mask.py` grew an `enable`
beside `cull`, callable only by this filter and filter_33. Everything else remains subtractive,
which is what keeps idempotency nearly free and stops an automated step quietly undoing a
human's judgement.

The mask disambiguates a provenance code rather than a third encoding doing it: a non-zero code
against a **culled** entry says why the datum is off, and the same code against an **in-use**
entry says it was deliberately restored. The two arrays are read together anyway.

## Addition, 2026-08-19 — the uncertainty array is already a fraction

`01` describes filter_24 as disabling points "whose assigned uncertainty exceeds a threshold,
**expressed as a fraction of the value**." That phrasing invites dividing the uncertainty by
the value. Doing so is wrong, and wrong silently.

In this format the uncertainty array already *is* the fraction — AarhusInv's relative standard
deviation. The forward code confirms it by inverting the operation when it writes uncertainties
back out:

    std = np.abs(1 / self.inv.invProb.dmisfit.W.diagonal()
                 / self.inv.invProb.dmisfit.data.dobs)

Dividing by `dobs` to recover the stored value means the stored value was already relative. The
system description agrees: `UniformDataSTD` is a plain `0.03` — three percent, not an amplitude.

So the comparison is direct, and a threshold of `0.15` means "more than fifteen percent
uncertainty", which is what an operator means when they say it. Dividing again against a
response of order 1e-9 would produce a number around 1e7, or its inverse, and cull everything
or nothing — with no error either way.

The distribution of the compared values is reported for exactly this reason: a delivery
carrying *absolute* uncertainties would show magnitudes nothing like a percentage, and be
visible rather than quietly under-culled.

**Uncertainty is not noise.** filter_24 culls what the processing claims to know about a point;
filter_29 culls where the measurement sits below the ambient noise floor. Both can be true of
the same datum and neither implies the other.

## Addition, 2026-08-19 — two constraints on a free-form line label

`01` decides filter_01–04 collapse into `Classify lines` and `Select lines`, and that
`line_type` values stay **free-form** so a survey can define its own categories. That
flexibility is kept. Two limits on it were not foreseeable when the decision was made.

**A label may not contain whitespace.** The survey xyz format is whitespace-delimited with no
quoting, so `"High Altitude"` becomes two fields and the file it is written to will not parse
back. This is the same defect that made every enriched export unreadable when a verdict tier
contained a space, and it is refused at the point the label is *set* rather than discovered at
the point the file is read — the error suggests the hyphenated form.

**A label is a string, so the plot layer cannot draw it.** YmerFlow's plot layer enumerates
columns that are typed arrays; a label column is not one, so it never appears as a plottable
channel. Verdicts solved this by being numeric with a fixed legend, and `line_type` cannot
follow, because the vocabulary is deliberately open — a per-dataset code registry would have to
invent stable codes for values nobody has declared yet.

That is an acceptable loss rather than a problem to solve: `line_type` exports to GeoPackage and
shapefile, which is where someone actually wants to symbolise on it, and it is readable in the
xyz. It is simply not available as a plot channel, and that is worth knowing rather than
discovering.

**Selecting drops soundings from every array together.** `flightlines` and each `layer_data`
array are subset positionally in one pass and reindexed as one, so nothing can end up
addressing a different sounding than its neighbour. An array whose row count does not match the
sounding count is left alone and reported — it is not indexed by sounding, and guessing would
be worse than saying so.

## Addition, 2026-08-19 — filter_33's checkpoint is a name, not a file

`01` decides the checkpoint and the edit stay one step, because "a diff is meaningless without
knowing which state it is a difference from" and splitting them would require inventing a
reference to replace one that is implicit in the pairing. That decision stands.

What the checkpoint *is* needed clarifying. `01` says the step "saves it as a named intermediate
result", which reads as writing a snapshot file. Here it is a **name recorded on the run**, and
nothing is written.

Materialising and versioning intermediate state is the platform's job — it already creates a
process version per edit, which is how the diff reaches the pipeline in the first place. A
library that wrote snapshot files as a side effect of filtering would be doing that job badly
and in a second place. What this owes is the name, so the pairing survives into the record; a
run without one is noted, since a diff whose reference state is unnamed cannot be checked
against the state it was authored on.

**The addressing is what makes replay safe, so it is validated rather than trusted.** An edit
naming a sounding index outside the dataset was authored against a different state, and applying
it would edit the wrong soundings silently. That raises. A row-count disagreement between
`apply_idx` and an override array raises for the same reason. A *gate*-count disagreement is
noted and skipped rather than raising, since one mismatched channel need not stop the others.

## Addition, 2026-08-19 — filter_05 holds elevation, not altitude

`01` says filter_05 "corrects instrument altitude and topographic elevation so that heights are
expressed consistently relative to the ground surface". It does not say which of the two is
held fixed while the other moves, and the two readings give different answers for every
sounding in a survey.

**Altitude is distance above the ground. Elevation is distance above (or below) the geoid.**
`libaarhusxyz` states the relation between them by plotting `alt_column + z_column` as the
instrument's position, so:

    instrument elevation = ground elevation + altitude

Read as a definition, that settles it. **The instrument's elevation is held fixed and altitude
is recomputed.** The aircraft was where it was — that is GPS, an absolute measurement against a
datum. What a better terrain model changes is not where the aircraft flew but how far above the
ground it flew.

The alternative is available as `preserve="altitude"` and is the wrong default. In a SkyTEM
delivery the normalizer maps `dem_lidar` to `topo` and `height_lidar` to `tx_altitude` — **both
come from the same lidar**, so the delivered ground elevation already carries whatever the
altimeter got wrong. Holding altitude fixed would preserve exactly the error this filter exists
to remove.

**A vertical datum mismatch looks like terrain error and is not.** GPS reports height above the
ellipsoid; terrain models are usually above the geoid, and the separation is tens of metres in
places. This library has no geoid model and cannot resolve it — but it can be *seen*, because
the two have opposite signatures. A datum mismatch is a large median offset with a **small**
spread: the whole survey shifted by nearly the same amount. Genuine terrain error is a small
median with a **large** spread. Both figures are reported and neither is judged, because where
the boundary lies depends on the terrain and a house number would either cry wolf on real
relief or wave through an error that puts every sounding tens of metres wrong.

**Outside the model's coverage, both columns go to NaN** rather than keeping delivered values.
Keeping them would leave one dataset carrying two definitions of ground with nothing marking
which is which, and a later check grading that column would be comparing against a datum it
cannot know. The delivered values are stashed either way, so nothing is lost.

## Addition, 2026-08-19 — how the floor is estimated, and one filter that may be redundant

R9 says the floor is "estimated from late gates" and leaves the method open. Two choices had to
be made.

**Estimated from the scatter, not from the level.** At late gates the measured value is signal
*plus* noise, and taking its magnitude as the noise counts whatever response remains. What
separates them is that **geology varies smoothly along a line and noise does not** — so the
sounding-to-sounding difference at a late gate is noise-dominated even where signal is still
present. The floor is the robust scatter of consecutive differences, divided by `sqrt(2)`
because differencing two independent samples adds their variances, and taken as a median
absolute deviation so a few spikes do not inflate the floor that is meant to detect them.

Verified by recovery rather than by inspection: against a synthetic transient with a known
5e-10 floor and −0.5 slope, the estimator returns 4.7e-10 and −0.49, and a hundredfold stronger
signal over the same ambient noise does not move it.

**Which gates are "late" is a parameter with a stated default** — the last third, with a floor
of three so a slope can be fitted at all. There is no principled number here, so the per-gate
scatter is reported and the choice is checkable rather than trusted.

**filter_29 may be redundant with filter_12, and that is worth saying out loud.** On a dataset
whose uncertainties were assigned from the model, a gate below the floor already carries a
relative uncertainty past one, and any inversion that weights by uncertainty has already
discounted it. filter_29 removes it outright instead, which matters for a consumer that
thresholds on the mask rather than weighting, and for anyone reading how much of a sounding was
real. Both are kept, and the overlap is stated rather than left for someone to discover.

Its `margin` is a multiplier on the shared floor rather than a floor of its own — which is the
whole point of R9, since a second number could drift away from the one filter_12 assigned
uncertainty with.

## Addition, 2026-08-19 — filter_11 adds arithmetically, and filter_13 refuses to run twice

**filter_11 adds arithmetically, not in quadrature.** Combining two *independent* uncertainties
is a quadrature sum, and this is not that: `01` says "adds" and "negative values reduce it", and
a negative cannot be subtracted in quadrature without going imaginary at the crossing point.

Arithmetic is also what makes the parameter mean what an operator says aloud — "add two
percent" moves 3% to 5%, not to 3.6%. The trade is that it does not compose the way independent
errors do, which is correct: a number someone chose is not an independent error. That is the
same distinction that keeps filter_11 and filter_12 apart at all — one adjusts, the other
asserts a physical model.

Uncertainty is clamped at zero rather than allowed negative. A negative uncertainty travelling
downstream would be squared by something and come back as a plausible positive, which is worse
than an error.

**filter_13's ordering hazard is enforced, not documented.** `01` asks that "the rebuild should
make a wrong ordering detectable rather than silently producing plausible nonsense". Three
things do that:

- **It refuses to run twice.** Perturbing an already-perturbed dataset makes it noisier by
  roughly √2 and the result looks exactly like data. The application is recorded on the dataset
  with its seed, and a second pass raises unless explicitly asked for.
- **It says so when every uncertainty in a channel is identical.** That is the signature of a
  uniform default from the system description rather than an assigned model, which usually
  means filter_12 has not run. Noise drawn from it is flat in relative terms, so late gates come
  out as clean as early ones — precisely what the absolute floor exists to prevent. There is a
  test that this stays quiet on a properly assigned model, since a warning firing on correct use
  is worse than none.
- **The seed is required.** "A fair test" is not a fair test if it cannot be repeated.

The perturbation's spread is `uncertainty × |value|`, because the stored uncertainty is
relative. Drawing from the relative figure directly would put noise of order one on every datum
whatever its magnitude. A gate whose assigned uncertainty has passed one receives a perturbation
larger than the datum itself — correct, and what makes a synthetic look like real late-time data.
