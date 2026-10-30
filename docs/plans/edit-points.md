# A save point that also replays a manual edit

**Status:** spec, not started.

## What this is

One pipeline step, `save_intermediate_and_apply_diff`, that does two things at the point in
the chain where it sits:

1. **Saves a named intermediate state** — a *save point* — so the data as the steps above it
   left it becomes a dataset a person can open in the plot editor.
2. **Replays a manual in-use edit** drawn against that state, so a person's gate- and
   sounding-level decisions survive a re-run of the whole chain.

The two belong in one step because a manual edit is a *difference from a state*, and the state
it is a difference from is exactly the save point. Splitting them would force inventing a
reference to stand in for the one the pairing already carries. This is the same reasoning
behind `apply_manual_edits` keeping its `checkpoint` name beside its `diff`; this step makes
the checkpoint a thing the editor can actually open, not just a name recorded on the run.

## Why it is needed

A real chain wants a person's edits at more than one stage — at full rate after the automatic
culls but before averaging, and again on the averaged result. Today a step can replay a saved
edit (`apply_manual_edits`), but nothing in the library materialises the intermediate state
the edit is drawn on, and the platform only lets the editor open a process's *final* output.
So a two-edit-point chain has to be split into a chain of processes, one per edit point, and
every Save means re-running the downstream processes by hand.

With this step, one pipeline document carries named save points. The platform writes the state
at each as a dataset the editor opens; a Save on that dataset lands at that point's edit list
and a re-run applies it before the steps that follow.

## What the step does, precisely

```
save_intermediate_and_apply_diff(dataset, name, diff=None)  ->  (dataset, StepReport)
```

In order:

1. **Emit the save point first, before touching the data.** The step hands the current
   `dataset` to a sink the runner provides (below), under the dataset name `name`. This is the
   state the steps above produced — *not* the edited state. The editor opens this snapshot and
   overlays the point's own edit on top, so it renders as an override that can be toggled back
   off, exactly as the plot editor already shows the final output's edits.

2. **Replay the edit.** With `diff` given, the step applies it to the dataset's in-use masks —
   identical behaviour to `apply_manual_edits`, with `checkpoint = name`. With no `diff`, the
   step is a pure save point and the data passes through unchanged.

3. **Return `(dataset, StepReport)`** like every other filter. The report records the save
   point's name, whether the snapshot was emitted or skipped, and — delegated from the replay —
   how many soundings were edited and how many gates were disabled or re-enabled per channel.

The order is the whole point: the snapshot is the pre-edit state, so the edit is recoverable;
and because the edit is stored as a diff rather than baked in, re-running the chain does not
silently undo it.

## The sink: the library emits, the runner writes

The library **writes no snapshot files.** Materialising and versioning intermediate state is
the runner's job — the platform already creates a process version per edit, and a filter that
dropped `.xyz`/`.gex`/`.msgpack` files as a side effect would be doing that job badly and
unportably. This is the same decision `apply_manual_edits` documents for its checkpoint.

So the runner gives the pipeline a **sink**: a callable `(name, dataset) -> None`. The step
calls it with the save point's name and the current dataset. Each runner supplies its own:

- **The platform wrapper's sink** writes an output dataset named `name` beside the pipeline's
  usual output (`processed_data`), using the same `write_dataset` path every process output
  takes — the survey's XYZ and GEX serialised through libaarhusxyz.
- **The CLI's sink** writes `<label>.<name>.xyz` (and its companion GEX/msgpack) beside the
  final output on disk.
- **No sink** (a library caller with no runner): the save point is skipped and the report says
  so. Such a caller loses only the materialised file; the edit still replays.

**Snapshots stream out.** The sink writes immediately when called, mid-pipeline. Nothing is
held until the end, because a full-rate snapshot of a large survey is gigabytes.

### How the sink reaches the step

The runner (`ProcessingRunner`) gains an optional `sink` argument. When it walks the pipeline,
it passes `sink=` only to steps whose signature accepts a `sink` parameter (decided by
`inspect.signature`), so every other step's calling convention is unchanged. `sink` is a
runner-injected argument, **not a pipeline parameter** — like `dataset`, it is excluded from
the generated form (the platform wrapper already drops `dataset` from each step's schema; it
drops `sink` the same way). The step's full signature is therefore:

```
save_intermediate_and_apply_diff(dataset, name, diff=None, sink=None)
```

## Parameters and their JSON Schema

The form is generated by swaggerspect from the NumPy docstring and the type hints, as for
every other filter. The two pipeline-visible parameters and the schema each must yield:

### `name` — the save point's name (required)

```json
{
  "type": "string",
  "title": "Save point name",
  "description": "A name for the intermediate state saved here. It appears in the plot editor under Dataset, and is the state this point's manual edits are a difference from."
}
```

- Must be **unique** within the pipeline.
- Must not be a reserved name — `output` or `processed_data` — which name the pipeline's final
  output.
- Validated **before any data is read**, so a clash or a reserved name fails at submission,
  when a job reserves its budget, rather than part-way through a run.

### `diff` — the manual edit to replay (optional)

A single URL to a libaarhusxyz msgpack diff saved by the plot editor. One save point carries
one edit, drawn against the state this save point writes — the same shape as
`apply_manual_edits`, which takes one `diff`.

The field **must carry the `x-url-media-type` annotation that identifies a saved in-use edit**,
so the editor offers the right saved artifact for it. Declare it on the parameter's type with a
`typing.Annotated` hint whose `{"json_schema": {...}}` metadata swaggerspect merges into the
field:

```python
ManualEditUrl = typing.Annotated[
    pydantic.AnyUrl,
    {"json_schema": {"x-url-media-type": "application/x-geophysics-xyz-model"}},
]

def save_intermediate_and_apply_diff(
    dataset,
    name: str,
    diff: ManualEditUrl | None = None,
    sink=None,
):
    ...
```

The media type **`application/x-geophysics-xyz-model`** and the surrounding
`Annotated[pydantic.AnyUrl, {"json_schema": {...}}]` shape are fixed — they are the convention
the plot editor already matches saved in-use edits against, so the annotation on this field has
to be byte-for-byte that one for the editor to populate it. The resulting field schema is a
string URL carrying that media type:

```json
{
  "type": "string",
  "format": "uri",
  "x-url-media-type": "application/x-geophysics-xyz-model",
  "title": "Manual edit",
  "description": "An in-use edit saved from a plot of this save point. Filled in by the plot editor; replayed after the save point is written, so re-running keeps it. Leave empty for a pure save point."
}
```

(`type`/`format` come from `pydantic.AnyUrl`; `x-url-media-type` from the annotation.)
`apply_manual_edits` already accepts the diff as a string (or an already-loaded XYZ), so this
step coerces the `AnyUrl` to `str` and passes it straight through.

## How it uses libaarhusxyz

The **replay half reuses `apply_manual_edits` wholesale** — it should delegate to that step's
function rather than re-implement it — so the libaarhusxyz usage is the usage already proven
there:

- **Load the diff.** A `.msgpack` diff is read with `libaarhusxyz.export.msgpack.load(path)`;
  a `.xyz`/`.xyzd` diff with `libaarhusxyz.XYZ(path)`. The msgpack path is the plot editor's,
  and the one that preserves gate-index column labels (the text reader would renumber them).
- **The diff's own addressing.** The diff carries `flightlines.apply_idx` — the global
  sounding indices it touches — so replay does not depend on row order surviving the steps
  above it. An index outside the current dataset fails the run: the diff was authored against a
  different state.
- **The sentinel.** Each `layer_data["<inuse>"]` array uses `NaN` for "no override", marked by
  `model_info.diff_dummy`. It is read out and discarded; no `NaN` is ever written into a main
  in-use array.
- **Naming.** The diff's channel naming is normalised to the dataset's naming standard
  (`"alc"` for every importer) so the edit's in-use arrays line up with the dataset's channels.
- **Applying.** The override is applied with `libaarhusxyz.xyz.df_apply(scratch, array,
  apply_idx, dummy_value=diff_dummy)` onto a scratch copy of the mask; the difference is then
  routed through the mask's `cull`/`enable` so provenance records who changed what. An edit may
  turn data back **on** — a person overruling an automated cull — which this step is permitted
  to do.

The **save-point half uses no libaarhusxyz directly.** The step hands the `dataset` (whose
`survey` holds the libaarhusxyz XYZ and GEX) to the sink; whichever runner's sink serialises
it does so through the same libaarhusxyz export path every dataset output already uses. The
library never calls `xyz.dump` itself.

## The shipped end-of-chain edit replay is removed

The `YFProcessTEM` process type in the platform wrapper (aem-processes,
`yf_processing_process.py`) already replays saved in-use edits — but only one implicit point,
after the *whole* pipeline. It does this with a process-level `diff` **array** (one entry per
save), a `takes_edits` flag, and a `_replay_edits` helper that calls `apply_manual_edits` with
`checkpoint = "output of the steps above"` once per entry after the run.

That is the same job this step does, done less well, so it is **removed** rather than kept —
two ways to replay the same edits is exactly the duplicate functionality to avoid, and the old
one has a defect the new one does not: replaying after every step means an edit can leave a
sounding with fewer gates than `cull_sparse_soundings` requires, with nothing left to cull it.

**What removal entails in the wrapper:**

- Delete the `takes_edits` class flag — the default `False` on `_PipelineProcess` and the
  `True` override on `YFProcessTEM` (the only subclass that sets it; `YFQCTEM` leaves it
  `False`) — and the `properties["diff"]` schema block in `_PipelineProcess.schema()`.
- Delete `_replay_edits`, `EDIT_CHECKPOINT`, and the `diffs` loop (and its validation) in
  `_PipelineProcess.run()`.
- The only edit replay left is an explicit `save_intermediate_and_apply_diff` step inside the
  pipeline document.

**This plan lands first; the editor follows.** The pipeline has to be able to produce the
save-point data before an editor can be built against it, so the order is: ship the step, the
sink and this removal, then implement the editor against the named save points the pipeline now
emits. Today a Save appends a URL to the process's `diff` array; once the editor is rebuilt, a
Save will instead set the `diff` of the `save_intermediate_and_apply_diff` step whose save
point is being viewed, in a new process version, and the chain re-runs. For the final output to
stay editable, a pipeline must carry a terminal save point.

**Existing saved processes** carry a populated process-level `diff` array that the new schema
no longer defines. Handling those — migrate into a terminal `save_intermediate_and_apply_diff`
step, or accept that their edits need re-saving — is a platform decision to settle with the
editor work, not here.

## Decisions

1. **One combined step.** The editor finds a point by name, and a diff cannot be separated
   from the state it was drawn on.
2. **The snapshot is the state before the point's own edits**, so the editor overlays them as
   reversible overrides.
3. **The library emits to a runner sink and writes no files.**
4. **`name` is unique and not `output`/`processed_data`; validated before any data is read.**
5. **The replay delegates to `apply_manual_edits`** rather than duplicating its libaarhusxyz
   handling.
6. **The shipped end-of-chain `diff` replay is removed, not kept**, so there is one way to
   replay an edit — an explicit save point in the pipeline.

## Dependencies outside this library

- **aem-processes** supplies a sink that writes a named output dataset per save point, and
  removes the end-of-chain `diff` replay (`takes_edits`, `properties["diff"]`, `_replay_edits`,
  `EDIT_CHECKPOINT`).
- **The editor (Ymerflow)** is a **follow-on**, built after this plan ships, against the named
  save points the pipeline then emits. It moves from appending to the process-level `diff`
  array to setting a named `save_intermediate_and_apply_diff` step's `diff` in a new process
  version. For the final output to stay editable, a pipeline must carry a terminal save point.
  Existing saved processes' `diff` arrays are migrated or their edits re-saved — a platform
  decision settled with that editor work.
- Separately, a Save does not carry the run's cluster and resource settings today, so a large
  re-run can fall back to platform defaults and run out of memory — that must be fixed for
  one-process pipelines on large surveys.
- A platform environment built with these versions.

## Work (this library)

1. `save_intermediate_and_apply_diff`: the step, its `ManualEditUrl`-annotated `diff` field,
   its NumPy docstring, its `StepReport` (save point name, snapshot emitted or skipped, and the
   replay's per-channel counts), and its entry-point registration under
   `ymerflow.processing.filters`. Replay delegates to `apply_manual_edits`.
2. `ProcessingRunner`'s `sink` argument, passed only to steps whose signature accepts it; and
   the CLI's file sink.
3. Validation of save-point names (unique, not reserved), before data is read.

## Work (aem-processes)

4. Add the sink to the process wrapper (writes a named output dataset per save point), and
   **remove** the end-of-chain edit replay: the `takes_edits` flag (base default and the
   `YFProcessTEM` override), the `properties["diff"]` schema block, `_replay_edits`,
   `EDIT_CHECKPOINT`, and the `diffs` loop and its validation in `run()`.

## Tests

5. The snapshot is the pre-edit state; the diff replays onto it; no sink means the save
   point is skipped and the report says so; duplicate or reserved names fail before any data is
   read; the CLI writes one snapshot per point; a pipeline with a
   `save_intermediate_and_apply_diff` step both writes its named dataset and applies its edit,
   and no process-level `diff` path remains in the wrapper.
