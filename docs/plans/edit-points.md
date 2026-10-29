# Manual edit points inside one pipeline

**Status:** decided 2026-10-29. Not started.

## Motivation

A real processing chain needs a person's edits at more than one stage: at full rate after the
automatic culls but before averaging, and again on the averaged data. Today a platform
processing job replays the editor's saved edits only after its last step, and the editor can
only open a process's final output. So a two-edit-point chain has to run as a chain of
processes, one per edit point, and every Save means rerunning the downstream processes by
hand.

What's wanted: one pipeline document with named edit points. The platform writes the state
at each edit point as a dataset the editor can open. A Save on that dataset lands at that edit
point, and the rerun applies it before the steps that follow.

## What exists

- `apply_manual_edits(dataset, diff, checkpoint)` replays a diff and records the name of the
  state it was drawn on. Its docstring settles one design point already: the checkpoint and the
  edit stay one step, and the library never writes snapshot files. Materializing intermediate
  state is the platform's job.
- The platform wrapper (aem-processes) already downloads any step parameter that is an input
  dataset before the run, so a diff URL can sit anywhere in the step list.
- The platform's earlier process type had a "compound filter" step that saved a named
  intermediate and replayed a diff at that point, which is the same idea.

## Proposed design

**One step: `manual_edit_point`.**

    - manual_edit_point: {name: pre_average, diff: [<url>, <url>, ...]}

At this point in the chain the step:

1. hands the current state to the runner as a snapshot named `name`, *before* applying any of
   its own diffs. That snapshot is what the editor opens. It shows the data as the steps above
   left it, and the editor overlays this point's diffs, as it does today for the final output;
2. replays its diffs in order, as `apply_manual_edits` does (later saves override earlier
   ones where they overlap), with `checkpoint = name`.

**The library never writes the snapshot.** The runner gives the pipeline a sink, a callable
`(name, dataset) -> None`, and the step calls it. The CLI's sink writes `<label>.<name>.xyz`
beside the final output. The platform wrapper's sink writes an output dataset named `name`
next to `processed_data`. No sink means the snapshot is skipped and said so in the report, so
a library user without a runner loses nothing but the file.

**Snapshots stream out.** The sink writes immediately. Nothing is held in memory until the
end, because a full-rate snapshot of a large survey is gigabytes.

**The end-of-pipeline `diff` list stays for now, deprecated.** It is the implicit last edit
point, named `output`, so existing processes and workspaces behave exactly as now. It is
deprecated because it replays edits after every step, including the final clean-up steps. An
edit there can leave a sounding with fewer gates than `cull_sparse_soundings` requires, and
nothing culls it afterwards. An explicit `manual_edit_point` placed before the clean-up steps
doesn't have that problem.

**Names are unique in a pipeline** and must not be `output` or `processed_data`. Validation
fails before any data is read if they clash.

**The editor (platform side, its own plan).** Editing the snapshot named `name`, Save appends
the new diff's URL to that `manual_edit_point`'s `diff` list in a new process version, and the
whole pipeline reruns. Editing the final output keeps today's behavior.

## Decisions (2026-10-29)

1. **One combined step, `manual_edit_point`.** The editor finds a point by name, and a diff
   can't be separated from the state it was drawn on.
2. **The snapshot is the state before the point's own edits.** The editor overlays the edits,
   so they show as overrides and can be undone.
3. **The library hands snapshots to a runner sink** and writes no files.
4. **The end-of-pipeline `diff` list stays, deprecated.** Using it adds a deprecation warning
   to the run report, which names the problem above and suggests a `manual_edit_point` before
   the clean-up steps. The docs and example pipelines use explicit points only.

## Open: retiring the implicit point

When to remove it, and what the editor does with a pipeline that has no edit point at its end,
are left for a later plan. Removing it means the editor can't save edits on a process's final
output and has to send the user to a named point. That is a platform change, so it is decided
with the platform's editor plan.

## Dependencies outside this library

- aem-processes: the wrapper supplies a sink that writes named output datasets.
- Ymerflow: the editor saves into a named edit point. Separately, a Save does not carry
  the cluster and resource settings today, so a large rerun falls back to platform defaults
  and can run out of memory. That has to be fixed for one-process pipelines on large surveys.
- A platform environment built with these versions.

## Work (this library)

1. `manual_edit_point`: step, schema, report (snapshot sent or skipped, diffs replayed, gates
   changed), docstring, and registration.
2. The runner's `sink` argument (`ProcessingRunner`) and the CLI's file sink.
3. Validation of edit-point names.
4. The deprecation warning in the run report when the end-of-pipeline `diff` list is used.
5. Tests: the snapshot is the pre-edit state; diffs replay in order; no sink means skipped and
   reported; duplicate or reserved names fail before data is read; the CLI writes one file per
   point; an end-of-pipeline diff still applies and adds the warning.
