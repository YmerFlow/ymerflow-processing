# Package layout — role at the top, and the specialization axis deferred

**Status:** done 2026-10-18, in two commits on one branch — the filename numbers
first, then the directory move. Analysis done 2026-08-22. One amendment to the
layout below: the culling filters are flat in `filters/` with a `cull_` prefix
(`filters/cull_geometry.py`), so the family that writes the mask stays visible in a
listing without a sub-package. The names in the text describe the layout before.
**Scope:** moves and renames only. No behavior change, no interface change.

## Why

The layout has a *declared* architecture that the code does not follow.

`em/__init__.py` states that `em/` holds five things:

> transmitter attitude correction, the horizontal transmitter assumption,
> forward-model normalization, filling current from the system description, and
> applying the system's own gate mask

All five are in `core/`:

| `em/` claims | actually lives in |
|---|---|
| transmitter attitude correction | `core/attitude.py` |
| the horizontal transmitter assumption | `core/attitude.py` |
| forward-model normalization | `core/normalize.py` |
| filling current from the system description | `core/normalize.py` |
| applying the system's own gate mask | `core/cull/filter_20_initial_gates.py` |

`em/` contains nothing but `tem/method.py` — 237 lines under two `__init__` files.
Meanwhile `core/__init__.py` says *"Nothing in this package may assume
electromagnetic data, gates, moments, or transient decay"*, and `core/transient.py`
opens *"The shape of a transient… a decay rate is a difference between
neighbours."*

Two further problems in the same area:

**Filters live in two places with no rule.** Twelve at `core/*.py`, eleven in
`core/cull/`. `filter_10_average.py` is top level and `filter_20_initial_gates.py`
is not, because the culling filters happened to be built as a group. That is build
history, not structure.

**Naming is inconsistent within one role.** `filter_10_average.py` sits beside
`attitude.py`, `noise.py` and `manage.py`. A reader cannot tell a filter from
machinery without opening the file.

## What is *not* wrong

**The 45 public step names are already clean** — no digits, all descriptive:
`average_along_line`, `disable_initial_gates`, `cull_by_curvature`,
`height_above_ground`, `implementation_equivalence`. The `filter_20` a reader sees
is a *filename*. Removing numbers from filenames removes them from the repo
entirely; there is no separate interface rename to do.

## The measurement that shaped this plan

Every module was parsed with `ast`, docstrings and comments discarded, and the
remaining executable code searched for method-specific signals — the axis, gate
times, GEX, dB/dt, transmitter current, attitude. Prose was excluded deliberately:
docstring vocabulary is what produced the current layout, and it is not evidence
about what the code depends on.

Genuinely method-agnostic in executable code:

| group | agnostic | of |
|---|---|---|
| filters | `lines`, `manage`, `terrain`, `export_vector` | 32 |
| qc | `height`, `speed`, `altimeter`, `spacing`, `misfit` | 9 |
| compare | `drift` | 4 |

**Everything else touches gates, the axis, GEX or dB/dt — including `mask.py`,
whose arrays are `(soundings, gates)`.** The central data structure of the library
is gate-shaped. A directory tree asserting a clean method-agnostic core would be
asserting something the code does not currently support.

## DECIDED — role is the top-level axis

Chosen over specialization-first. It mirrors `02`'s three entry-point groups,
which is the structure that actually drives the runtime, and it serves the common
reader: someone looking for one of 45 steps.

```
ymerflow_processing/
  __init__.py
  dataset.py          the data model, MethodAdapter, discovery
  pipeline.py         runners, StepReport, RunReport
  cli.py
  mask.py             the in-use mask and its provenance
  distribution.py     shared summarizer
  inventory.py        the implicit first step
  transient.py        shared shape machinery for the decay-rate family
  filters/            all 32, flat
  qc/                 all 9, plus thresholds, aggregate, rollup
  compare/            all 4, plus align
  methods/
    tem.py            was em/tem/method.py
```

## DECIDED — filenames lose their spec numbers

`filter_10_average.py` becomes `filters/average.py`; `filter_20_initial_gates.py`
becomes `filters/initial_gates.py`; `qc_01_height.py` becomes `qc/height.py`. The
spec number is already in each module's docstring, which is where the tie back to
`building_plans/01` belongs. Test files follow: `test_filter_20_initial_gates.py`
becomes `test_initial_gates.py`.

## DEFERRED — the `em/` and `em/tem/` split inside each group

The user's preference is both axes: role at top, specialization inside — e.g.
`filters/em/attitude.py`. That is the right end state and it is **deliberately not
in this pass**.

The current layout is broken *because the `em`/`tem` boundary was drawn before a
second method existed*. Drawing it again now, one level deeper, repeats the
mistake with more confidence — and the measurement above says the boundary would
be drawn through code that is gate-shaped almost everywhere, so most of it would
land on one side and the tier would be as empty as `em/` is today.

**Revisit when a second method registers** — FEM is the likely first, and it is the
only thing that can test where the line actually falls. At that point the move is
cheap, because the modules will already be gathered by role rather than scattered
across two directories.

Until then `core/__init__.py`'s promise is deleted rather than left standing and
false: a docstring claiming an invariant nothing enforces is worse than no
docstring, and this one has already misled once.

## Why now, or not

**For:** nothing outside this repo imports `ymerflow_processing` — it appears
nowhere in `docker/` — so downstream breakage is zero and this is the cheapest it
will ever be. 1087 tests on two numpy baselines make a pure-move refactor about as
safe as refactors get. The moment the platform migration (`Ymerflow#79`) starts,
module paths become load-bearing outside this repo.

**Against:** it is a large mechanical diff across nearly every file and it adds no
capability. The work that actually de-risks the project is running against real
survey data and starting the platform migration.

## Execution

Mechanical, and verifiable as a no-op:

1. `git mv` every module to its new home; no edits to module contents beyond
   imports.
2. Rewrite imports across `ymerflow_processing/` and `tests/`.
3. Rewrite all 45 entry-point module paths in `setup.py`. **Names do not change**,
   so every pipeline document keeps working.
4. Rename test files to match.
5. Delete `em/__init__.py` and `em/tem/__init__.py`, and the false invariant in
   `core/__init__.py`.
6. `pip install -e .` — entry points are cached by path and will not resolve until
   this is rerun.
7. Full suite on numpy 1.26.4 and 2.2.6. **1087 passing before, 1087 after** is the
   acceptance criterion; a changed count means something moved that should not
   have.

Step 3 is the one that fails silently: an entry point pointing at a moved module
raises only when that step is loaded, which no import-time check catches. The
`test_registered` test in each step's test file is what covers it.
