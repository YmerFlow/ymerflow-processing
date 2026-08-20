# Pipeline structure and entry points

How filters register, how a pipeline is expressed, and how it runs. Companion to
`01_filter_inventory.md`, which says *what* to build; this says *what shape* to build it in.

---

## The pattern to follow: AirMagTools

`AirMagTools` already solves this cleanly and is ours. Use it as the template rather than
inheriting the existing processing library's arrangement.

Its three moving parts:

**1. Filters register through a named entry-point group.**

```python
filters = {ep.name: ep.load()
           for ep in importlib.metadata.entry_points(group="mag_pipeline.filters")}
```

Declared in `setup.py` under that group name. Any installed package can contribute filters —
which is what makes the plugin story work — and nothing needs a central registry file.

**2. A pipeline is a document, not code.**

`MagPipeline` parses YAML, from a string or a file. A step is either a bare name (no
arguments) or a name with keyword arguments. That means a pipeline is data: storable,
diffable, versionable, and transmissible, without being executable.

**3. A runner walks the steps against a data object.**

`MagPipeline.run(data)` iterates and applies. There is also a `main()` behind a CLI, so a
pipeline can be executed from the command line against an input file, independent of the
platform.

### What to carry over

- entry-point group registration, under our own group name — `ymerflow.processing.filters`
- pipeline-as-YAML, with the string-or-mapping step form
- a runner that takes a pipeline and a dataset
- **the CLI.** The existing library has no equivalent, and it is the difference between a
  library you can test in isolation and one you can only exercise through the platform.

### What to change

- **Filters must not mutate in place silently.** The existing convention is that a step
  receives the dataset and modifies it, returning nothing. That makes provenance, dry-running
  and reasoning about failure all harder than they need to be. Prefer returning the modified
  dataset, even if the implementation is in-place under the hood.
- **A step should be able to report.** Per R3 in the inventory: a filter returns not only data
  but a structured summary of what it changed. The runner collects these into a record of the
  run.
- **Steps should be introspectable without executing.** The parameter surface is already
  derived from signatures and docstrings for the UI; keep that, and make it part of the
  library's contract rather than something the platform reaches in and extracts.

---

## Expose the pipeline runner

The runner should be a public, documented part of the library, not an internal detail of the
platform.

Reasons:

- **Testing.** A filter is testable in isolation; a pipeline needs a runner. Without an exposed
  one, every test either duplicates the walk or spins up the platform.
- **Scheduled field QC.** Acquisition QC needs to run per flight, likely on a schedule, plausibly
  on a machine that is not the platform. A CLI runner against a YAML pipeline is exactly the
  right shape for that, and needs no web stack.
- **Reproducibility as a claim.** "Here is the pipeline document and the command that ran it"
  is a much stronger statement than "here is a screenshot of our UI."
- **Federal deployment.** If the platform is ever deployed into someone else's environment,
  a library plus a runner travels far more easily than a platform.

---

## The data object

The existing library's dataset type is a thin wrapper — roughly a hundred lines — over
`libaarhusxyz.Survey`, adding construction from files and a `process(steps)` method.
`libaarhusxyz` is MIT licensed, so it is available to us without qualification and should
remain the foundation.

The new wrapper should:

- hold the survey data and the system description
- **know how many moments it has, and their properties**, as data rather than as structure
  (R1) — this is the single most important difference from what exists
- know its own normalization convention, so filter_08 can check rather than trust
- carry a record of the filters applied to it
- not know anything about pipelines — the runner composes them, the data does not run itself

Separating "the data" from "the pipeline that ran over it" is the main structural
improvement available here, and AirMagTools already does it this way.

---

## Three runners, not one

Three operations with genuinely different shapes. Each gets its own entry-point group and its
own runner, all sharing the pipeline-as-YAML form and the CLI treatment.

| runner | shape | entry-point group |
|---|---|---|
| **processing** | dataset → dataset | `ymerflow.processing.filters` |
| **QC** | dataset → dataset + report | `ymerflow.processing.qc` |
| **comparison** | N reports (or N datasets) → one report | `ymerflow.processing.compare` |

### Why comparison is a separate shape

It consumes many things and produces one, which neither other contract expresses. Stretching
the filter contract to cover it would make every filter's signature worse to serve one case.

**It also resolves the cumulative-QC problem without inventing persistent state.** R3 requires
every filter and check to return a structured record, and those records are already stored in
the process record. Cross-flight drift is therefore not "QC state carried between runs" — it is
*reading prior runs' reports*, which is a query over data that already exists.

### What it is for, beyond drift

The scope is considerably wider than the noise-floor case that prompted it:

- **What changed between two processing versions** — the versioning story the product already
  claims, currently demonstrated by eye
- **Recovered model against true model** — done by hand for the fault study
- **One parameter choice against another** — the averaging-width comparison, also by hand
- **Regression checking a reimplementation against its predecessor**, which is precisely the
  acceptance criterion for the rebuild itself

That last one is worth noting: the comparison runner is a tool we need in order to *build*
this library, not only a feature the library offers. Running the old and new implementations
over the same input and diffing the outputs is legitimate — comparing behavior is not copying
expression — and having a runner for it makes that systematic rather than ad hoc.

### Input type — DECIDED: both, declared per check

A comparison check declares what it consumes; the runner loads accordingly.

```yaml
- name: noise_floor_drift
  consumes: reports          # default
- name: implementation_equivalence
  consumes: datasets
```

Neither input type alone is sufficient. Cross-flight drift reads twenty small stored records —
loading twenty full datasets to recompute numbers already computed would take minutes for
something that should take milliseconds. Implementation equivalence needs per-datum comparison
across the actual arrays, which no report contains.

**Reports are the default** when a check does not declare. It is the cheap path, and a check
that silently loads gigabytes because a line was omitted is a worse failure than one that
errors saying it needs data.

**Reports are also the primary abstraction.** A dataset can always produce a report; a report
can never reconstruct a dataset. So build the report path first — that is what cross-flight
drift and per-flight QC need, both needed in the field — and add the dataset path when the rebuild's
equivalence testing requires it.

A reports-only runner is a pure function over small records: no storage backend, no
credentials, trivially testable, runs anywhere. Keeping that property for the common case is
worth the one branch in the loader.

### Check signature — DECIDED: ordered sequence plus optional reference

```
check(items, reference=None, **params) -> report
```

- **`items`** — an ordered sequence. A pair is simply a two-item sequence, so this subsumes
  pairwise comparison without a separate shape.
- **`reference`** — the thing everything is compared *against*, when the comparison is
  asymmetric. Optional.

The reference slot earns its place: "compared against what" appears in three of the four uses
worth writing. Without it the baseline is `items[0]` by positional convention, which produces
subtly wrong answers when arguments arrive in the wrong order and nothing complains.

| use | items | reference |
|---|---|---|
| noise-floor drift across flights | flights, in time order | none |
| old vs new implementation | the candidate | the baseline |
| recovered vs true model | the recovered model | the true model |
| parameter A vs parameter B | both runs | whichever is the control |

**The sequence is ordered, and a check may rely on that.** Stated in the contract rather than
left to chance — drift depends on ordering, equivalence does not care, and a check should be
able to assume it rather than sort defensively.

**Items carry identity, not just data.** Each knows which flight or version it is, so a report
can say "flight 6 is the outlier" rather than "item 4 is the outlier." Cheap to build in,
awkward to retrofit.

---

## Keeping the step list navigable

A long flat list of filters was a real usability problem in the previous library, and
consolidation does not fix it — the collapses decided in `01` take 33 filters to 26, which is
still a wall of names, and QC checks add a dozen more.

Five levers, in order of payoff:

**1. Most users should never see the list.** The standard-workflow story — load a standard chain
and adjust it — means someone doing ordinary work never browses filters at all. The full list
becomes the advanced path rather than the entry point. This reframes the problem from "make 26
items navigable" to "make the common case not need them", which is a much easier problem.

**2. Group, and order the groups by workflow position.** Six groups of three to six items reads
differently from 26 flat. Order them the way the pipeline runs — selection, corrections, noise,
culling, manual — rather than alphabetically, so "where would this step go" and "where is it in
the list" become the same question.

**3. Hide what cannot apply to this dataset.** Newly cheap, because R1 already requires the
dataset to know its own properties. The picker can show only what could actually run:

- no inversion present → the misfit-dependent steps do not appear
- single-moment data → per-moment parameters collapse to one
- no attitude columns → attitude-dependent filters do not appear

This is not a UI trick; it is the dataset answering a question about itself. On any given
dataset it plausibly removes a third of the list.

**4. Recently-used and most-used first**, within groups. Cheap, and it adapts to how a
particular user works rather than to an average one.

**5. Search.** Table stakes, but it only helps someone who already knows what they are looking
for, so it complements the others rather than replacing them.

**Be careful with progressive disclosure.** A "show advanced" toggle reduces the visible list,
but a filter someone cannot find is worse than one in a long list — and collapsing filters has
already spent some discoverability.

Levers 1 and 3 change the experience. Lever 2 is free and should happen regardless.

---

## Package layout — DECIDED

```
ymerflow_processing/
├── core/          # any airborne survey        — 18 filters
├── em/            # shared by TEM and FEM      —  5 filters
│   ├── tem/       # gates, moments, transients —  9 filters
│   └── fem/       # later
└── mag/           # later
```

The hierarchy states the real relationship: EM specializes general airborne processing, and
TEM and FEM specialize EM. `em/` sits as a sibling of `core/` rather than inside it, so that
`core/` continues to mean *method-agnostic* and other methods have an obvious home.

**Create `em/` now** — its five filters are already identified, so it is not an empty package.
**Create `em/fem/` and `mag/` when there is code for them**, not before.

Filter tiers are in `01_filter_inventory.md`. The one filter that splits across tiers is
filter_10: its averaging engine is generic and belongs in `core/`, while the gate-time-varying
window widths belong in `em/tem/`.

### Considered and rejected

Recorded so the question does not reopen.

**Nesting the method branches under `core/`** — `core/em/tem/` — mirrors the specialization
relationship directly, but "core" stops meaning method-agnostic once the most method-specific
code in the library lives inside it. Magnetics would become `core/mag/`, at which point
`core/` contains everything and the distinction it was drawn for is gone. Import paths also
reach four levels before any module.

**Staying flat** — `core/` and `tem/` only, deferring until FEM needs something — is
consistent with not creating packages before there is code, but the five EM-shared filters are
already identified and frequency-domain work is months away rather than hypothetical. That
defers a decision we already have the information to make, and converts it into a later
refactor.

**Package names are lowercase.** `core.EM.TEM` is unconventional in Python, and
case-sensitivity differences across filesystems cause avoidable trouble.

---

## Added 2026-08-15 — what may be read, and what may not

The provenance rule is narrow and worth stating positively rather than only as a prohibition,
because an over-broad reading of it wastes good prior art.

### Off limits

**The source of the library this one replaces.** Do not read it, grep it, or import it —
importing and then introspecting docstrings is reading.

That is the whole list.

### Free to read and use

| | why |
|---|---|
| `libaarhusxyz` | **MIT.** The survey data model. We pin our own fork; the fix in it is ours (`YmerFlow/libaarhusxyz#3`). |
| `simpeg` / `simpleem3` | **MIT.** The inversion layer, including `static_instrument`. We have made changes here — see `YmerFlow/simpeg#2` and `#3`. |
| `AirMagTools` | Ours. The template for the runner, entry-point registration and pipeline-as-YAML. |
| `GeoWrangler` | Ours. |
| An in-house flight QC panel and its config | Ours, and the best reference for the QC design — `03` leans on it deliberately. |
| The platform itself — `backend/`, `frontend/`, `docker/` | Ours. Reading it is how the diff format in `01` was established. |

The rule is about *one library's expression*, not about avoiding all prior work. Anything we
wrote, and anything permissively licensed, is available — and reading the platform is often the
fastest way to find that something already exists.

---

## Sequencing

1. **The data object and the runner.** Roughly 150 lines. Nothing else can land until these
   exist, and every day they do not exist is a day new work has nowhere to go.
2. **Acquisition QC in `core/qc/`** — new code, displaces nothing, exercises the interface,
   and is the first funded deliverable of an upcoming acquisition program.
3. **The shared culling policy**, then the culling predicates. Thirteen filters collapse into
   one policy plus thirteen small predicates, which is where most of the readability win is.
4. **The noise group**, which is where most of the *value* is — filter_10 in particular is
   both the most consequential filter and the one with the most known defects.
5. **Everything else**, by how often a real pipeline actually uses it.

Frequency-domain support enters at step 2 or 3 depending on that program's schedule, and is
new development rather than a port.

---

## Addition, 2026-08-20 — the first comparison check, and what orders a series

`noise_floor_drift` implements qc_09, which `03` specifies and deliberately cannot deliver as a
QC check: *"Only visible cumulatively, which is why per-flight results have to accumulate."* One
delivery's floor is a number; twenty deliveries' floors are a diagnosis, and no per-run check
can see the second.

It invents no state, which is `02`'s argument for the shape. `estimate_noise_model` already
records the floor, the exponent, and which values were supplied rather than measured; R3 means
those reports are stored. So this reads prior runs rather than carrying anything between them.

**Two numbers, because a rising floor and a jumping floor are different faults.** A component
degrading gives a trend; a change of survey area or a swapped receiver gives a step. The trend is
fitted in **log space** — a floor is a scale, so 5e-10 to 1e-9 is the same event as 1e-9 to 2e-9,
and a linear fit would call the second twenty times larger. The largest single step is reported
separately, because a trend can be small while one step is enormous and averaging hides exactly
the delivery someone needs to look at.

### What orders the series, which is where this goes silently wrong

A series trended in the wrong order produces a plausible slope with nothing to give it away. So
the ordering key is chosen most-trustworthy-first and the choice is reported:

**Time range**, read from the soundings' own timestamps — chronological for any instrument.

**Then flight number, ordered numerically.** Numbering *schemes* vary by contractor — SkyTEM
writes a date with an index on the end, like `26051201`; another may number sequentially from
one — but whatever the scheme, flights are numbered in the order they were flown. Ordering the
value as **text** rather than as a number would put `"10"` before `"9"`, trending a sequentially
numbered survey's tenth flight before its ninth and shifting every delivery after it.

**Then the order given.** `02` makes `items` ordered and says a check may rely on it.

**Never the identity.** Delivery names routinely embed an export date, so ordering by one would
enter a single delivery into the series twice after a re-export.

### The assumption is tested, not asserted

"Numeric and increasing" is a safe bet and not a guarantee, so rather than stating it in a
docstring the check verifies it and says when it does not hold: a scheme that does not parse as
a number, the same flight number appearing in two deliveries, or a flight ordering that
disagrees with the order the reports were given in. That last one is the real catch — `items` is
ordered by contract, so if the caller passed deliveries chronologically and the numbers
disagree, the numbering is not chronological.

Nothing is judged unless asked. `max_ratio` compares each delivery against the baseline — the
`reference` where one was given, otherwise the series **median**, which is robust to the outlier
being looked for — and a delivery far *below* the baseline is flagged as readily as one above,
since a floor that collapses is as much a change as one that rises.

## Addition, 2026-08-20 — implementation_equivalence: four outcomes, not two

`02` names this check, gives it `consumes: datasets`, and calls it *"precisely the
acceptance criterion for the rebuild itself"*. What it does not settle is the shape
of the answer, and that turns out to be the whole design.

**Bit-for-bit is the wrong bar.** A predecessor has defects. An implementation that
agrees with it everywhere has reproduced them, which is a failure dressed as a pass.
The bar is *equivalent where the baseline is right, better where it is not,
documented wherever they deliberately differ* — and that third clause needs
somewhere to live. A report whose vocabulary is only **matches** and **does not
match** has nowhere to put a deliberate improvement, so someone widens a tolerance
until the improvement disappears into it and the comparison stops meaning anything.

So four outcomes: `agrees`, `expected` (differs, declared, **a pass**), `differs`,
`absent` (one side has no such quantity — a gap in coverage, not a difference).

**A declared difference can still fail.** An expectation may name a `direction`, and
that is what makes "better" testable rather than assertable: declaring a new
uncertainty should be *smaller* than a baseline that clipped its denominator means a
candidate coming out larger is a finding, even though the difference was declared.
Without the direction, a declaration would excuse a regression as readily as an
improvement. An expectation nothing exercised is reported too — stale, or the
comparison never reached what it describes, and neither should read as agreement.

**Alignment assumes no shared schema**, because two implementations are two schemas.
Soundings match by timestamp, else by position, else — only when the counts agree —
by order, which is stated in the report because it is an assumption rather than a
match. Channels match by their gate times, since a channel *is* its gate times
whatever the file calls it. Gates match by center time within a tolerance, so a side
that dropped three gates is compared over the ones it kept rather than shifted and
declared wrong everywhere.

**Values are compared only where both sides kept the datum**, since underneath a
mask an implementation may leave anything at all. The masks are compared as their
own quantity, which is where "this one culled more" belongs.

**The tolerance carries a justification** into the report. A tolerance without a
reason is a number someone tuned until the comparison passed, and the report is
written to be read rather than to return a boolean.

`consumes` is now honored: declared as a module-level `CONSUMES`, so the declaration
sits beside the code depending on it, and reports remain the default.
