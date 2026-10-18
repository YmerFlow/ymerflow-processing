# The CLI runner — functional specification

Fourth document in the `ymerflow-processing` seed set. `01` says what the filters do, `02`
says what shape to build them in, `03` covers QC, this one covers **running a pipeline from
the command line**.

**Status: BUILT.** Everything below is implemented in `cli.py`, with `RunReport`'s
identity fields and `resolve_group` in `pipeline.py`, and covered by `tests/test_cli.py`.

Two deviations from what is described below, both noted in place: the CLI lives in its own
module rather than inside `pipeline.py`, and `compare` is registered but has no checks
to run yet, since no package registers under `ymerflow.processing.compare`.

---

## Why this is missing and why it matters

`02` names the CLI as the single most important thing to carry over from AirMagTools:

> **the CLI.** The existing library has no equivalent, and it is the difference between a
> library you can test in isolation and one you can only exercise through the platform.

What exists today is `yf-process list` and `yf-process validate`. Both are introspection.
**There is no `run`.** So a QC pipeline can be listed and validated from the command line
but not executed, and every one of the four arguments `02` gives for exposing the runner is
only half-served:

| the claim | today |
|---|---|
| **Testing** — a pipeline needs a runner | met, but only from Python |
| **Scheduled field QC** — per flight, on a schedule, on a machine that is not the platform | **not met** |
| **Reproducibility** — "here is the pipeline document and the command that ran it" | **not met**, there is no command |
| **Federal deployment** — a library plus a runner travels further than a platform | **not met** |

The second is the pressing one. Acquisition QC is the first funded deliverable for the New
Mexico program, it has to run per flight while the aircraft is still on site, and `03` is
explicit that it must run *"against one flight's archive on whatever machine is available,
with no web stack."* That is a cron job invoking a command. It does not exist yet.

---

## The shape — DECIDED: two commands, not one and not three

```
yf-process run     PIPELINE --xyz DELIVERY.xyz --gex SYSTEM.gex --out OUT/ [--group qc]
yf-process compare PIPELINE --report A.report.json --report B.report.json --out drift.json
```

The three runners do not divide evenly, and the command surface should follow the division
that actually exists:

| group | reads | writes | command |
|---|---|---|---|
| `filters` | one dataset | a dataset **and** a report | `run --group filters` |
| `qc` | one dataset | a dataset (verdict columns) **and** a report | `run --group qc` |
| `compare` | N reports, plus an optional reference | a report only | `compare` |

**`filters` and `qc` share a shape** — one dataset in, dataset plus report out — and differ
only in which entry-point group is resolved. One command with `--group` serves both.

**`compare` does not share it.** Folding it in would leave most options dead or
reinterpreted depending on `--group`:

| option | `run --group filters` | `run --group qc` | `run --group compare` |
|---|---|---|---|
| `--xyz`, `--gex` | required | required | **dead** |
| `--label` | delivery identity | delivery identity | **dead** |
| `--report`, `--reference` | **dead** | **dead** | required / optional |
| `--out` | a **directory** | a **directory** | a **file** |

That is the failure `01` describes when arguing for `criterion` over a boolean — *"half the
parameters are dead depending on a flag and nothing in the signature says which half."*

**The consistency argument was considered and rejected.** `list` and `validate` already take
`--group`, so `run` differing looks inconsistent. But `list --group qc` and
`list --group filters` have *identical* option surfaces; the group only selects which
registry to read. `run --group compare` would not. The analogy breaks exactly where it
matters.

**Three commands was also rejected** — splitting `filters` and `qc` apart would duplicate a
command whose inputs, outputs and options are the same in both cases.

---

## Loading a dataset — the one real decision

The runner needs a `Dataset`, and `Dataset` wraps `libaarhusxyz.Survey(xyz, gex)`. So the
command needs at minimum a survey file and a system description.

**Proposed: `--xyz` and `--gex` as separate required options**, because that is what the
data model actually takes, it is what a delivered archive looks like on disk, and it keeps the
loader honest — no format sniffing, no guessing which of two files is which.

Three things follow, and each is a decision rather than an obvious default:

### DECIDED — `--gex` is required

Both options are required. The dataset is always fully formed and there is no degraded mode
to reason about.

**The case that was made for optional, and why it lost.** Of the twelve checks in `03`,
only qc_09 (noise-floor drift) reads the system description at all — the other eleven need a
flight plan, an inversion, or nothing but `flightlines`. Height, attitude, speed, adherence,
coverage and density are properties of *how the aircraft flew*, not of the instrument. Our
own prior flight QC panel ran with no `.gex` anywhere, off a Geosoft CSV and an SCI `.xyz`.
Channel discovery already degrades correctly without one, and that is tested.

**Overruled because real QA/QC is not QC alone.** A complete acceptance workflow inverts the
data, and everything downstream of that — qc_11 misfit, and every processing filter — needs
the system description. Treating the `.gex` as optional optimises for the subset of the job
that happens to run first, and makes the input set of a *complete* run inconsistent with the
input set of a partial one.

The residual risk is worth stating: requiring an input that most checks never open invites
someone to pass a placeholder or the wrong instrument's file, and the checks that ignore it
cannot detect that it is wrong. If that bites, the answer is validation that the system
description matches the data — not making the option optional again.

### Output is a directory, not a file

A run produces **two** artifacts and they are not the same kind of thing:

- the **dataset**, enriched — QC verdict columns, or filtered data
- the **report**, structured — R3's record of what each step did

`--out DIR` writing `DIR/<stem>.xyz` and `DIR/<stem>.report.json` keeps them together,
names them predictably, and leaves room for the third artifact (`03`'s vector export) without
another option. `Survey.dump()` already writes xyz/gex/geojson, so the writing side is
largely solved.

### DECIDED — flat, one set of files per delivery

```
QC/
  20260815.xyz            <- dataset, verdict columns added
  20260815.report.json    <- the run record
  20260816.xyz
  20260816.report.json
```

Named from the run's identity, one set per delivered file, and a re-run of the same delivery
replaces its own output.

**There is no versioning here, and that is not an oversight.** Versioning is a **platform**
capability — `process.versions[]` backed by a database — and this command exists precisely
to run without the platform. `--out` writes plain files into a folder. Anyone needing an
audit trail of successive re-runs should run it through the platform, which is what that
machinery is for.

**Why overwriting is safe, having first looked like it was not.** An earlier draft of this
document claimed that overwriting made qc_09 drift impossible, since `03` requires
per-flight results to accumulate. That was wrong, and the error is worth recording so it is
not repeated: **the accumulation drift needs is across deliveries, not across re-runs of
one.** Each delivery owns its own file, so the series builds up naturally as deliveries
arrive, and `QC/*.report.json` is the whole input to a drift comparison.

Keeping re-runs would in fact make drift *worse*: re-running one delivery with a corrected
threshold would put it into the series twice under two different thresholds, and something
would then have to decide which one counts. One record per delivery keeps that unambiguous.

**A delivery is several flights, and that is the point.** Deliveries commonly arrive as
three or four flights in one file, so a single run produces per-flight results for all of
them. `03`'s two cadences — per flight, and the per-day rollup — therefore come out of one
invocation rather than needing separate ones, provided the report carries both grains. It
does: `per_line` and `per_flight` are both in `distributions`.

A worked report for a three-flight delivery of 120 soundings is about 4 KB, so a survey's
worth of them is a few hundred kilobytes. That is another reason drift reads reports rather
than datasets.

### The report is JSON

`StepReport.to_dict()` and `RunReport.to_dict()` already exist and already produce plain
Python. `tests/test_height.py` pins that a report survives a JSON round trip. So the
report writer is `json.dump(report.to_dict())` and nothing more.

**Why JSON and not YAML**, given pipelines are YAML: the pipeline is written by a human and
read by a human; the report is written by a machine and read by a machine — the comparison
runner consumes stored reports for cross-flight drift. Different audiences, different
formats, and JSON is what the process record and the web client already speak.

---

## The comparison runner

`ComparisonRunner.run(items, reference=None)` takes a sequence, so the command does too:

```
yf-process compare PIPELINE --report FLIGHT_01.report.json \
                            --report FLIGHT_02.report.json \
                            --reference BASELINE.report.json \
                            --out DRIFT.json
```

- **`--report` repeats**, and **order is significant** — `02` states the sequence is ordered
  and a check may rely on it. Drift across flights depends on it; equivalence does not care.
  A repeated option preserves the order the user typed, which a glob would not.
- **`--reference` is optional**, matching the check signature.
- **`--dataset` is the other input kind**, for checks declaring `consumes: datasets`. `02`
  says build the report path first, so **`--dataset` is deferred** until the rebuild's
  equivalence testing needs it.

**Its own command**, not `run --group compare` — see the option table under *The shape*
above for why.

### DECIDED — identity belongs on the report, not on the command line

`02` requires a comparison report to be able to say "flight 6 is the outlier" rather than
"item 4", and notes it is *"cheap to build in, awkward to retrofit."* But `RunReport`
currently carries only `pipeline` and `steps` — **no identity at all** — so a stored report
does not know which flight produced it.

Deriving it in the CLI from each `--report` path was considered and rejected: it makes
identity a property of where a file happens to sit, lost the moment it is moved, and pairing
repeated `--report` and `--label` options by position is fragile.

**So `RunReport` gains an `identity` field**, set when the run happens — from an explicit
`--label`, defaulting to the input filename stem. The comparison runner then reads identity
out of the report itself and the command-line question disappears.

This is an interface change in `pipeline.py`, and the only one this document proposes.

**Identity names the delivery, not a flight.** A delivered file usually holds three or four
flights, so one report covers several. Per-flight identity is not the report's identity — it
lives *inside* the report, in the `per_flight` rollup, keyed by whatever the data's flight
column says. A drift check therefore reads N delivery reports and flattens them into a
per-flight series, rather than expecting one report per flight.

### DECIDED — identity is a label; the facts drift reads come from the data

`--label` sets it. Absent, it defaults to the input filename stem, and the report records
which of the two happened:

```json
"identity":        "20260815",
"identity_source": "filename",
"flights":         [6, 7, 8],
"time_range":      ["2026-08-15T09:12:04", "2026-08-15T17:48:31"]
```

**Why the filename stem is a default and not the answer.** Real delivery names are not
identities. A representative one, of the shape contractors actually deliver:

```
<acquired>_<client>_<project>_<region>_<system>_prj-<number>_<exported>export.csv
```

— client, project, region, system, project number, and *two different dates*. The second,
`<exported>`, is the **export** date, so re-exporting the same delivery changes the stem.
An identity taken from the filename alone would then enter the same delivery into a drift
series twice under two names, which is exactly the duplication the one-record-per-delivery
layout exists to prevent.

**The resolution is to stop asking identity to do two jobs.** Ordering a drift series needs
*time, per flight*, and that is data — `libaarhusxyz` already has a `timestamp` column
convention built from `date` and `time` in `normalize_dates`, and the flight column is
discovered by `qc/aggregate.find_flight_column`. So the report carries `flights` and
`time_range` read from the data, drift orders by those, and `identity` is left to be what a
human calls the run.

With that split, a messy or unstable filename costs a poor *label*, not a corrupted series.

`identity_source` is R3 applied — *"parameters actually used, including defaults that were
not supplied"* — so a reader can distinguish a deliberate label from a fallback, the same way
`NoiseModel.provenance` distinguishes an estimated value from a supplied one.

---

## DECIDED — a pipeline may declare its own `kind`

```yaml
kind: qc            # optional; validated, not trusted
steps:
  - height_above_ground:
      good_max: 45
      marginal_max: 65
```

Resolution order for which registry a step name is looked up in:

1. **`--group` on the command line**, when given — an explicit override, and the escape
   hatch if a step name is ever ambiguous across two groups.
2. **`kind:` in the document**, when present. **Validated against the steps, not trusted:**
   if it says `filters` but the steps only resolve under `qc`, that is an error. A document
   cannot silently misdescribe itself.
3. **Derived**, by finding which registry resolves every step.

### Why it earns a new key

**It fixes an actively misleading error.** Today, `yf-process validate qc.yaml` defaults to
`filters` and reports:

```
KeyError: 'Unknown step(s): height_above_ground. Available: '
```

The step exists and is perfectly valid; the group was wrong. The message blames the step,
and the empty `Available:` makes it worse. Under any of the three resolution paths above,
that is unrepresentable.

**It makes the document self-describing.** `02` calls a pipeline *"storable, diffable,
versionable, and transmissible"*. A bare list of steps is not quite that — a recipient
cannot tell which runner it needs without already knowing the step names, and the platform
would need a separate column to record something the document should carry itself.

**Derivation alone was considered and kept as the fallback, not the answer.** It cannot
disagree with the content, which is its real virtue, and it is why it is retained for
documents with no `kind`. But under derivation the kind of a *stored* document depends on
what happens to be installed when the question is asked — so a pipeline whose steps were
later renamed becomes unclassifiable, while `kind: qc` stays true. Validating a declared
kind against the steps keeps derivation's guarantee without giving up durability.

### What this does not change

Nothing about what any step does. This is dispatch — which dictionary a name is looked up in
— and it sits well below R6. QC and culling remain separate steps in separate registries
with separate runners, writing different things: qc_01 writes the `qc_height` **column**,
filter_22 writes the **mask**, and their thresholds come from different places (the survey
contract, and the physics). Running a QC pipeline still leaves the data going into the
inversion byte-identical.

**Out of scope:** whether one document could mix filters and QC steps. Those are two runners
with different contracts, so that remains two runs.

---

## What `run` does, step by step

1. Resolve the registry — `--group`, else the document's `kind`, else derived — and load the
   registered steps.
2. Parse the pipeline document and **`validate()` it before touching any data.** This
   already exists and matters more here than in-process: `Pipeline.validate` was written
   because *"a job reserves budget up front, so an unknown step found at step nine is a
   wasted reservation."* On the command line the equivalent is a scheduled overnight job
   that dies at 3 a.m. on a typo.
3. Load the survey, construct the `Dataset`.
4. Walk it with the runner for that group.
5. Write the dataset and the report to `--out`.
6. Exit non-zero if anything raised.

### DECIDED — exit codes mirror the classification

A scheduled job is read by its exit status before anyone reads its output. One number has to
answer two different questions — *did the QC run?* and *was the flight any good?* — so the
codes are split to carry both.

| code | meaning | ran? | flight |
|---|---|---|---|
| **0** | completed, everything Good | yes | clean |
| **1** | completed, some Marginal, nothing Out of Spec | yes | within tolerance |
| **2** | completed, some Out of Spec | yes | **breach** |
| **3** | never started — bad options, missing file, unknown step | **no** | unknown |
| **4** | started, then a step raised | **no** | unknown |

Three properties make this worth the extra codes:

- **The split falls at 2.** `0-2` means the run completed; `3-4` means it did not. So
  `[ $? -le 2 ]` is "did the QC happen?", which is the question monitoring actually asks.
- **The result codes mirror Good / Marginal / Out of Spec**, so no information is thrown
  away and a caller gates at whatever severity it cares about: `-eq 0` for spotless,
  `-le 1` for no breach.
- **3 is your mistake; 4 is our bug.** A typo in the pipeline document and a crash inside a
  check need completely different responses, and collapsing them would make the two
  indistinguishable.

**A finding is not a failure.** A check that reports a line flown at 78 m against a 65 m
spec has *succeeded* — that report is the deliverable, and it should still reach the client.
That is why an out-of-spec result is a distinct code rather than lumped in with the errors.

**No `--fail-on` flag.** An earlier draft proposed one to let a caller opt into failing on
out-of-spec data. Distinct codes make it unnecessary: the caller reads the number and
decides for itself, with no extra parameter surface.

The consequence to document for users: `&&` treats every non-zero as failure, so it is the
wrong idiom here. The shape that works is

```bash
yf-process run qc.yaml --xyz flight_12.xyz --gex sys.gex --out QC/
case $? in
  0|1) ./upload.sh ;;                                        # deliver
  2)   ./upload.sh; ./notify.sh "flight 12 out of spec" ;;   # deliver AND flag
  3)   ./notify.sh "QC misconfigured — check the pipeline" ;;
  4)   ./notify.sh "QC crashed — this is a bug" ;;
esac
```

which is the behavior actually wanted: a bad flight still delivers its report *and* raises
an alarm, while a crash raises a different alarm and delivers nothing.

---

## Logging

R3 already splits these: *what a step did* goes in the report, *diagnostics* go to
`logging`. The CLI already has `-v` / `-vv` wired to the root logger, so `run` needs nothing
new — but it is worth stating that **the report is not written to stdout.** stdout gets a
human-readable summary; the report goes to a file. A scheduled job's stdout lands in a mail
spool or a container log, which `01` already identifies as where 336 `print()` calls went to
die.

---

## Deliberately not here

- **Reading from object storage.** `--xyz` is a local path. R8's layering is *"the platform
  materialises, the library reads"*; a CLI that grew an S3 client would be re-implementing
  the platform inside the library and would break the standalone property this exists for.
- **A `--param` override surface.** The pipeline document is the parameter surface. Allowing
  overrides on the command line means the document no longer describes the run, which
  forfeits *"here is the pipeline document and the command that ran it."*
- **Parallelism across flights.** A shell loop or the scheduler already does this, and doing
  it inside the CLI would mean owning process management for no gain.

---

## Decisions and what is still open

### Decided

| | decision |
|---|---|
| **Commands** | two — `run` for `filters` and `qc` (they share a shape), `compare` separately (it does not) |
| **Which registry** | `--group`, else a `kind:` in the document (validated against the steps, not trusted), else derived from which registry resolves them |
| **`--xyz` / `--gex`** | both **required**. No degraded mode; a complete acceptance workflow inverts, and everything downstream of that needs the system description |
| **Exit codes** | five: `0` Good, `1` Marginal, `2` Out of Spec, `3` never started, `4` step raised. Split at 2 — `0-2` ran, `3-4` did not. No `--fail-on` flag |
| **Comparison identity** | `RunReport` gains `identity` (from `--label`, defaulting to the filename stem) plus `identity_source`, `flights` and `time_range` read from the data. Identity is a human label; drift orders by the data-derived time. The only interface change proposed here |
| **Output layout** | flat, one set of files per delivery, named from identity; a re-run replaces its own output. No versioning — that is the platform's job |
| **Report grains** | every report carries per-sounding verdicts, a `per_line` rollup **and** a `per_flight` rollup, since a delivery holds several flights |
| **Report format** | JSON. Written by a machine, read by a machine — unlike the pipeline, which is YAML because a human writes it |

### Still open

Nothing. Everything raised in this document has been decided; what remains is building it.

## Addition, 2026-08-21 — `--dataset` built, and history had to become durable

`04` deferred `--dataset` with a condition: *"`02` says build the report path first,
so `--dataset` is deferred until the rebuild's equivalence testing needs it."* It
needs it now — three of the four registered comparison checks consume datasets, and
without this they were registered, tested and **unreachable from a command line**.
Passing an XYZ to `--report` produced a raw `JSONDecodeError` naming a column.

**The pipeline is asked, not the user.** Each check declares `CONSUMES` on its
module, `ComparisonRunner.consumes_all()` reads the pipeline, and the command loads
accordingly — which is what `02` meant by *"the runner loads accordingly"*. A
pipeline whose steps disagree is refused with which steps want which, since one run
loads one set of inputs and letting the odd steps fail on the wrong kind is worse
than not starting.

`--gex` is required with `--dataset`, per this document's own rule: a survey file
does not describe its own gates, and the channels a comparison aligns on come from
the description. Given to a report comparison it is noted rather than refused —
harmless, but silently ignoring a typed option is how someone comes to believe it
did something.

### The real finding — `dataset.history` did not survive a file

Building the CLI path exposed it: a sweep over three datasets **read from disk**
discovered no parameter, because `history` lived on the Python object and the dump
dropped it.

That is worse for `version_diff`, which would have reported *"the two versions
record the same steps with the same parameters"* for two files processed
differently — a **confident wrong answer** rather than a missing one, and the
report even says the difference must have come from outside the pipeline.

So history moved into `model_info`, which round-trips, joining the provenance
legend and the verdict registry there. `mask` already made the argument: *the file
has to carry its own legend because whoever opens it will not have the run report.*
It holds at least as well for how a delivery was processed, and a delivered file
that records its own processing is worth having independently of any comparison.

`dataset.history.append(...)` would now append to a list rebuilt from the file and
thrown away, so it is replaced by `record_step()`, and the runner calls that.

## Addition, 2026-08-27 — a sixth exit code, for a run that graded nothing

`04` decided five codes and argued the split well: `0-2` ran, `3-4` did not, and
`[ $? -le 2 ]` is *"did the QC happen?"*, which is the question monitoring asks.

Real data found a question those five cannot answer: **did the checks actually
grade anything?**

Six of nine checks on a real delivery found no columns. Each said so politely in
a note. The run wrote its report and exited **0** — *"completed, everything
Good"* — because `_VERDICT_EXIT[None]` mapped no-verdict to `EXIT_GOOD`. That is
the most dangerous answer available: a scheduler gates on it, and a reader
believes it. Nine notes in a run nobody reads is indistinguishable from a clean
run.

```
5   ran, wrote a report, and could not grade — the flight is UNGRADED, not clean
```

**It sits outside the original split, deliberately.** `-le 2` still means "the
run completed", and 5 is excluded from it. That is the cost and it is the right
way round: a run that graded nothing should not pass a test for having happened
usefully.

**A breach outranks it.** A line flown out of spec is a finding someone must act
on; "some checks did not run" is a finding about the run. Both are printed, so
reporting the breach loses nothing — but reporting *Good* would be a false
statement, which is what this fixes.

### Three reasons for no verdict, and only one is a problem

The code needs a check to say *why* it produced nothing, so `StepReport` gains
`ungraded`:

| | meaning |
|---|---|
| `unjudged` | measured, and no thresholds were supplied. The caller's choice. |
| `optional` | input absent and expected to be — `03` requires data_misfit and culled_fraction be absent *"with no error"*. |
| `missing` | the input should have been there. **Only this sets exit 5.** |

The classification is a judgement per check, not a rule. A missing **height**
column is `missing` — every airborne survey has one. A missing **powerline
monitor** is `optional`, because not every system carries one, and treating an
absent instrument as a fault would make the code fire on correct use — which is
the failure mode this repository has objected to everywhere else.
