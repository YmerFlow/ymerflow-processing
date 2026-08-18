# Acquisition QC — functional specification

Third document in the `ymerflow-processing` seed set. `01` says what the processing filters
do, `02` says what shape to build them in, this one covers QC.

Lives in `core/qc/`. Almost everything here is method-agnostic — altitude, attitude, speed and
line adherence apply to any airborne survey — which is why this is also the work that will
prove whether `core/` is genuinely agnostic before the TEM filters depend on it.

**This is the first funded deliverable of an upcoming acquisition program**, required before the first
production flight.

---

## The design already exists

An in-house flight QC panel is a working implementation of most of this. It is our own work, so
there is no provenance question, and it is a better starting point than anything in the old
library — which on inspection is nine modules of matplotlib plotters with no thresholds, no
pass/fail, and one instrument-locked scoring function.

What it already solves, and should be carried over directly:

**A three-state classification.** Good / Marginal / Out of Spec, with two threshold kinds:

- `centered` — for a quantity that should sit near zero. Good within ±good, Marginal within
  ±marginal, Out of Spec beyond. Used for attitude.
- `ascending` — for a one-sided quantity where higher is worse. Good below `good_max`,
  Marginal up to `marginal_max`. Used for height.

A metric with no threshold entry gets **no classification at all** rather than a default. That
graceful absence is the right behavior and should be preserved.

**Portability by derivation rather than hardcoding.** Transmitter current levels differ
completely between systems, so it does not encode a value. It computes survey-wide mean and
standard deviation from the loaded data, then applies a multiplier window — consistent across
the survey, entirely derived from whatever was loaded. This is the direct answer to the old
library's instrument-locked scoring, and the pattern should be reused anywhere an absolute
value would otherwise be baked in.

**Operational knowledge that is not guessable.** Sudden-jump detection on current that
excludes the first sample of each flight, because current legitimately steps there. A
match-distance tolerance for joining two sources. Line renumbering between delivery and
processing. These were earned and should not be rediscovered.

**Declarative series definitions** — an ordered list, each with a name, label, unit, plot kind,
and which threshold applies. Same config-driven shape `02` argues for throughout.

### Two threshold kinds to add

- `descending` — lower is worse. Needed for coverage and sounding density.
- `window` — both bounds are meaningful and asymmetric, which `centered` cannot express.

---

## Workflow

### Inversion is optional enrichment, never a dependency

**The QC is valid on its own.** Everything in the check list below except two items runs on a
single flight's data with no processing history and no inversion. That is the whole point —
the answer has to arrive while the aircraft is still on site, which is long before anything
has been inverted.

**If an inversion exists, it should be included.** Data misfit and culled/edited fraction are
genuinely useful and are already in that panel. They enrich the picture; they do not gate
it.

So the series list must be **resilient to absent inputs**: a series whose source data is not
present is dropped from the output with a note, not raised as an error. Concretely, the same
configuration should produce:

| context | what runs |
|---|---|
| single flight, mid-survey | all flight-parameter checks; no misfit, no culled |
| survey to date, mid-survey | above, plus cross-flight drift |
| completed survey, post-processing | above, plus culled fraction |
| completed survey, post-inversion | everything, including misfit — the fully-processed case |

One configuration, four contexts, no separate code paths.

### Cadence

- **Per flight** — the primary mode. Run against one flight's archive as soon as it lands.
- **Per day** — a rollup, which is the natural unit for a reflight decision.
- **Cumulative** — drift only appears as a series across flights. Per-flight results must
  accumulate so that a slowly failing altimeter or a drifting transmitter is visible before
  it becomes a hole in the survey.

### The decision, not just the classification

Classification happens per sounding. A reflight decision is per line. Something has to bridge
them, and that rule is a genuine design question rather than an obvious one:

- what fraction of a line Out of Spec fails the line?
- does a contiguous run matter more than a scattered total? Three hundred scattered bad
  soundings and one continuous 3 km stretch are very different problems.
- do Marginal soundings count fractionally toward the threshold?

**Open decision.** Recommend starting with both a fraction rule and a longest-contiguous-run
rule, reporting both, and letting the survey contract set the numbers — since it usually
specifies them.

---

## Outputs — three kinds, all first-class

Not one output with two afterthoughts. Each answers a different question and each has a
different audience.

### 1. Profile / strip plots

Per line, series stacked vertically in a fixed order, classification shown as banding. This is
what a geophysicist reads to understand *what happened along the line*.

Axis handling matters and that panel settled it deliberately: fix the range for anything with a
threshold, so lines are comparable at a glance; leave genuinely variable quantities
auto-scaled per line — ground speed varies with terrain and wind, and a powerline trace exists
to show a spike against its own baseline.

### 2. Map views

Per series, spatially, colored by classification. This is what answers *where* the problem is —
which is the question that actually determines whether a reflight is worth it. A line failing
at one end near infrastructure is a different decision from one failing throughout.

### 3. Shapefile / vector export

**The one that leaves the building.** Classification attached to geometry, exported as a
standard vector format.

This matters more than it looks:

- it is how QC results reach people who do not use our platform — the client, the acquisition
  contractor, the regulator
- for a funded program it is plausibly a **contract deliverable**, not just an internal
  artifact
- it is what lets someone overlay QC against their own GIS — planned lines, infrastructure,
  land access, previous surveys

Export must carry the CRS explicitly. The output CRS is a project decision and cannot be
derived from the input.

---

## Checks

Same convention as the filter inventory: a stable reference, what it consumes, what it
produces.

### qc_01 — Height above ground
**Consumes:** altimeter, terrain model · **Produces:** per-sounding classification
**Threshold kind:** ascending — higher is worse

Flags soundings flown above the contracted maximum height. The single most common spec
violation, and the one most directly tied to depth of investigation.

### qc_02 — System attitude
**Consumes:** attitude angles · **Produces:** per-sounding classification
**Threshold kind:** centered — should sit near zero

Flags excessive tilt in either axis. Beyond the spec limit the geometry assumption behind the
1D forward model has broken down, so the data is not merely noisy but wrong.

### qc_03 — Ground speed
**Consumes:** position and time · **Produces:** per-sounding classification
**Threshold kind:** window

Flying too fast reduces stacking and widens effective sounding spacing; too slow wastes flight
hours. Bears directly on the noise floor, since the floor is a property of how much stacking
produced each sounding.

### qc_04 — Transmitter output stability
**Consumes:** transmitter current, flight identifier · **Produces:** per-sounding
classification, plus discrete jump events
**Threshold kind:** centered on the survey-derived mean

Two distinct failures: slow drift across a flight, and sudden steps. Jump detection must
exclude the first sample of each flight, where a step is legitimate.

Window derived from survey-wide statistics, never hardcoded — current levels differ entirely
between systems.

### qc_05 — Powerline and cultural noise
**Consumes:** powerline monitor · **Produces:** per-sounding classification

Baseline varies enormously by region, so the useful signal is a spike against the line's own
baseline rather than an absolute level. Auto-scaled per line by design.

### qc_06 — Line adherence
**Consumes:** position, planned line geometry · **Produces:** per-sounding deviation and
classification
**Threshold kind:** ascending

Lateral deviation from the planned line. Requires the flight plan as an input, which is a
new input the platform does not currently take.

### qc_07 — Coverage and gaps
**Consumes:** position, planned survey grid · **Produces:** per-line gap list
**Threshold kind:** descending — less coverage is worse

Identifies unflown segments and grid holes. The check most likely to trigger a reflight, and
the one that is most expensive to discover late.

### qc_08 — Altimeter agreement
**Consumes:** two or more independent altimeters · **Produces:** per-sounding disagreement
and classification
**Threshold kind:** ascending

Disagreement between laser and radar altimeters indicates a failing sensor rather than a
flying problem. Distinct from qc_01, which asks whether the height was right; this asks
whether the height is *known*.

### qc_09 — Noise floor drift
**Consumes:** late-gate amplitudes, flight identifier · **Produces:** per-flight noise
estimate, trended across the survey

The floor is a property of the environment and the instrument together. A trend across flights
means something is changing — a failing component, a seasonal shift, or a region genuinely
noisier than the rest. Only visible cumulatively, which is why per-flight results have to
accumulate.

### qc_10 — Sounding density
**Consumes:** position · **Produces:** per-line along-line spacing
**Threshold kind:** descending

Effective spacing against what was contracted. Partly a consequence of qc_03, but worth
reporting directly since it is usually what the contract actually specifies.

### qc_11 — Data misfit *(optional — requires an inversion)*
**Consumes:** inversion residuals · **Produces:** per-sounding classification

Included when an inversion exists. Absent otherwise, with no error.

### qc_12 — Culled fraction *(optional — requires processing history)*
**Consumes:** in-use mask · **Produces:** per-sounding and per-line culled fraction

How much of the line survived processing. Included when a processing history exists.

### Terrain and elevation context
Not a check. A display panel carrying the terrain model alongside the elevation traces, for
reading the other panels against. No classification.

---

## Decisions

### QC gets its own runner — DECIDED

QC checks consume a dataset and produce a report; processing filters consume a dataset and
produce a modified dataset. Rather than stretch one contract to cover both, QC gets a parallel
runner sharing the same mechanism.

- **Its own entry-point group** — `ymerflow.processing.qc`, alongside
  `ymerflow.processing.filters`. Checks register the same way filters do.
- **The same pipeline-as-YAML form** and the same CLI treatment, so a QC run is as
  reproducible and as scriptable as a processing run.
- **Runs standalone.** No processing pipeline required, no platform required. This is what
  acquisition QC actually needs — a scheduled run against one flight's archive on whatever
  machine is available, with no web stack.
- **The runner aggregates.** Individual checks report per sounding; the runner assembles those
  into the per-line and per-flight summary that a reflight decision is made from.

### Classification is an attribute on the dataset — DECIDED

Each check writes its per-sounding Good / Marginal / Out of Spec verdict as an attribute on
the dataset, rather than into a separate artifact.

Concretely, `flightlines` is already the per-sounding attribute table and already carries
position, so a verdict is simply another column on it — `qc_height`, `qc_attitude`,
`qc_speed` — riding along with the dataset exactly as `TxRoll` does. Nothing new to store,
nothing new to version.

**Two different things get called "the report", and they live in different places:**

| what | where | why |
|---|---|---|
| per-sounding classification | column in `flightlines` | travels with the data, exports for free |
| per-line verdict | column in `flightlines`, denormalized | redundant, but it is what makes the exported vector file useful to a client — they see which lines failed without needing a second table |
| aggregate statistics | separate structured record (R3) | distributions before and after, counts, parameters, contiguous-run measurements. Not per-sounding, so it does not fit in a column |

The first two export. The third accumulates, and is what the comparison runner reads for
cross-flight drift.

Three consequences, all good, and worth stating because they simplify things elsewhere:

**1. QC checks are no longer read-only, and that is consistent.** A check writes an attribute,
which is exactly what filter_01 and filter_02 already do when they classify flightlines. The
QC runner therefore returns both an enriched dataset and a report. Same shape as a filter,
different emphasis.

**2. The vector export stops being a QC concern.** If classification is just attributes, then
exporting classified geometry is a general dataset-export capability — export any dataset with
its attributes to a vector format. That serves QC, but it also serves anything else that wants
geometry out, and it means a contracted shapefile deliverable is not a bespoke code path.

**3. Classification becomes available to the processing pipeline.** The attributes travel with
the dataset, so a culling filter can consume a QC verdict instead of recomputing it. That is a
real simplification: filter_22 currently applies its own attitude and altitude thresholds,
duplicating what qc_01 and qc_02 compute. With classification present, culling on
"Out of Spec" is a one-line predicate over an attribute, and the thresholds live in exactly
one place.

Worth designing toward deliberately — it collapses a duplicated concern rather than adding a
feature.

---

## Open decisions

1. **Line-level decision rule** — fraction Out of Spec, longest contiguous run, or both, and
   whether Marginal counts fractionally. The numbers come from the survey contract. Recommend
   computing and reporting both rules, since a scattered total and a single continuous stretch
   are different problems and a fraction alone cannot distinguish them.

2. **Where the flight plan enters.** qc_06 (line adherence) and qc_07 (coverage) both need
   planned line geometry, which is an input the platform does not currently accept. This is a
   new input type rather than a new check, so it is probably its own piece of work.

3. **How per-flight results accumulate** — *resolved, see `02`.* qc_09 needs a view across
   flights, which initially looked like QC state persisting between runs. It is not: a third
   **comparison runner** consumes the structured reports that R3 already requires and the
   process record already stores, so cross-flight drift is a query over existing data rather
   than a new state mechanism. What remains open is only whether that runner reads reports or
   datasets by default.

---

## Addition, 2026-08-17 — column names are constrained by the export

Not a revision of anything above; a constraint on new checks that was not visible when this
was written, because the reason lives in a file format rather than in a check.

**Every column a check writes must be at most ten characters, and no two may share their
first ten.** A shapefile field name is limited to ten, and the vector export named above is
the artifact that reaches a client. An eleven-character name is silently truncated on
export; two names truncating to the same ten collide without complaint, and one quietly
replaces the other in the deliverable. That has already happened once — `qc_attitude` and
`qc_attitude_line` both became `qc_attitud` — and was caught by a test rather than by review.

The scheme adopted, which keeps names inside the limit and says what grain a column belongs
to:

```
qs_  this sounding          qf_  its flight
ql_  its line               qd_  the delivery it arrived in
```

A bare name with no prefix is a *measurement* rather than a verdict.

GeoPackage has no such limit, and is the better default for exactly this kind of reason.
The constraint is kept anyway, because a contract that names shapefile should not force a
column rename.

## Addition, 2026-08-17 — unclassified must survive the export

`03` says above that a metric with no threshold entry gets **no classification at all**
rather than a default, and calls that graceful absence the right behaviour.

Shapefile cannot express it. DBF does not distinguish NULL from empty, so "not measured" and
"measured and fine" risk arriving at a client as the same thing — losing the distinction in
the one artifact they read.

So the export writes unclassified as an **explicit label** rather than as a blank, and states
in its report what that label was. GeoPackage could carry a true NULL, but both formats write
the same label so that two exports of one dataset do not disagree.

---

## Addition, 2026-08-17 — a weighted grade beside the verdict

`03` gives every grain a verdict and, above, the fraction and contiguous run that a
contract rule turns into one. That answers *does this pass*. It cannot answer *which of
these failing lines is worst*, because once two lines are both Out of Spec they tie, and
a reflight conversation has to order them.

So each tier is also worth points — **Out of Spec 0, Marginal 1, in spec 2** — and a
sounding's grade is the weighted mean across the checks that classified it. Line, flight
and dataset grades are the mean of their soundings, taken over soundings rather than over
lines so a two-sounding line does not weigh as much as a thousand-sounding one.

**Weights are per check, and they have defaults**, unlike thresholds, which never do. The
distinction is what the number decides: a threshold decides whether a sounding passes and
so goes to a client as a contract deliverable, while a weight only orders which problems
matter more. A wrong weight sorts a list badly; a wrong threshold mis-states a delivery.

Height and attitude dominate because they change what the measurement *means* — a sounding
flown too high or at a bad attitude couples to the ground differently, and no processing
recovers it. Speed, spacing and transmitter stability are a quarter of that: they degrade
coverage or repeatability without invalidating the physics. Altimeter agreement is weighted
to zero — flagged, not graded, since one bad sensor should not mark down data that was flown
correctly — and powerline noise likewise, since the ground's character is not the
contractor's doing.

Zero is not the same as absent. A weighted-out check still runs, still writes its verdict,
still names itself among the reasons, and still feeds the instrument-health rollup.

**The grade is not the gate, and this is the point that matters.** A mean dilutes: a
sounding failing one lightly weighted check still grades well, which is exactly what an
average does and exactly why it cannot decide anything. The worst-verdict column stays the
pass/fail. A mean also aliases — a grade of 1.0 is either uniformly Marginal or half in spec
and half out — so the per-tier fractions are written beside it rather than replaced by it.

The 0/1/2 scale treats an ordinal thing as an interval one, asserting that Out of Spec to
Marginal is worth the same as Marginal to in spec. That is a choice, not a measurement. It is
kept even because no contract states an exchange rate, and inventing one would be the only
unstated number in the QC path.

---

## Addition, 2026-08-17 — agreement with nominal is measured, not judged

`03` treats transmitter current as one check. It is two questions, and only one of them is
about data quality.

**Stability** — is the current steady against its own mean — carries the verdict. What costs
quality is the current *moving*, because then the excitation differs between soundings that
are later stacked and compared.

**Agreement with the declared nominal** does not. The current actually flown is recorded per
sounding and carried into the normalization, so a transmitter holding steady away from its
declared nominal produces perfectly self-consistent data. What is wrong is the system
description, and everything downstream trusts that description — so it is worth knowing, and
worth knowing separately.

So the two never combine. The offset from nominal is always measured and written; a verdict
appears only where thresholds were supplied, and it never joins the combined verdict. Like
the powerline monitor, it is a reference measurement by default. What counts as too far from
nominal is an operational judgement, and a default here would put a house number into a
client deliverable.

---

## Addition, 2026-08-18 — verdicts are stored as numbers

`03` treats a verdict as a classification and says nothing about how it is stored, which left
the obvious reading: write the tier name into the column. That reading does not survive
contact with the platform this library feeds.

**A label column is invisible, not merely awkward.** YmerFlow's plot layer enumerates what it
can draw with `Object.keys(fl).filter(k => ArrayBuffer.isView(fl[k]))` and coerces the result
through `toFloat32Array`. A column of names is not a typed array after the msgpack round-trip,
so it fails that filter and never appears in the list of plottable columns at all. A QC map
built on it would have no QC on it, with nothing reporting a fault in either direction.

The good half of the same mechanism: the enumeration has no allow-list, so **any numeric
column is available to every plot the moment it is written** — no schema to register, no
backend change, nothing to coordinate between repositories.

So verdicts are `float32` codes. `float32` and not an integer type because "not evaluated" has
to be representable and integers cannot carry `NaN`; float32 holds integers exactly far past
the range in use, so the codes are never approximate.

**The scale is the one the grade already uses** — Out of Spec 0, Marginal 1, in spec 2. The
alternative ordering was proposed and declined: a file carrying verdict codes ascending toward
bad *and* grades ascending toward good would have two numeric scales pointing opposite ways,
and a colorscale wired to the wrong one renders out-of-spec soundings as good. Confidently
wrong is worse than absent. One scale, higher is better, and the domain is fixed at 0..2 rather
than derived from the data — a flight containing no out-of-spec soundings must not rescale and
paint Marginal in the failing color.

**Unevaluated is `NaN`**, not a sentinel. A sentinel invites arithmetic mistakes and has to be
known about by every reader; `NaN` is already handled by anything that handles floats, and it
is in no tier at all rather than being quietly counted as one.

**The labels are not lost.** The legend goes in the step report, which R3 already provides, so
a fourth tier would never become a change in another repository. The vector exports translate
back on the way out, because a GeoPackage or shapefile is opened by a person and `0` in a
legend is worse than useless where `OOS` explains itself.

**What this costs: the dataset now has to record which of its columns are verdicts.** Storing
them as numbers removed the dtype that used to answer that question, and no naming rule
replaces it — `ql_verdict` is a verdict and `ql_score`, `ql_oos` and `ql_n` are not. So the
set is recorded on the dataset and round-trips with it, since the export runs on a file loaded
back from disk with no report to consult.

Read off YmerFlow's own frontend, not off any predecessor library.
