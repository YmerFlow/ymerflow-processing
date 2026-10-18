# Documentation — the goal

**Status: this states an intention, not a status.** Almost none of what follows
exists. It is written down so the shape is agreed before anyone starts, and so
that "we should document this" stops being a feeling and becomes a set of
tickets.

What exists today: `plans/`, holding design plans for work not yet done.

## What is already written, and where it is

Two things are commonly mistaken for documentation here, and neither is:

**`building_plans/`** is the **clean-room specification** — the input this library
was written from. It records what each filter must do, which decisions were made,
and which alternatives lost. It is preserved unrevised, additions only. Hand it to
someone deciding *whether the library is right*; never to someone trying to *use*
it.

**The docstrings** are the detailed record. Every step has one; most module
docstrings run to eighty or a hundred lines and carry the mathematics, the
citation, and the reasoning behind each judgement call. That is deliberate — R2
asks for it — and it means a generated reference would be unusually good rather
than a wall of stubs. **45 of 45 steps have docstrings; 39 carry `:param:`
entries; every signature is fully annotated.**

So the documentation problem here is mostly one of **extraction and arrangement**,
not of writing from nothing.

## Three audiences, and they want different things

### 1. Someone who has a delivery to process

They do not want an API. They want to know which steps to run, in what order, and
what the result means. Two documents:

**A worked example.** One delivery from load to graded output — the commands, the
report, and what to look at in it. Prose.

**How the geophysics works.** The document this library is missing most, and the
one hardest to get from anywhere else: *why* you cull negatives, what a decay-rate
anomaly actually is, why a sounding flown too high couples differently, what the
noise floor is a property of. The steps are named throughout, but the subject is
the physics, not the function signatures.

**Every filter in it should carry a figure**, and the figures should be
**generated from the test suite** rather than drawn. A before-and-after transient
with the culled gates marked is exactly what the library already computes, so a
figure can be produced by the same fixture that tests the behavior. Generated
figures cannot drift from the code: a filter whose behavior changes gets a changed
figure in the same commit, and a filter with no figure fails a check.

### 2. Someone extending the library

**Architecture** — the three runner shapes, the entry-point groups, the data
model, and the invariants R1–R9 stated compactly in one place instead of scattered
through module docstrings.

**How to add a step**, and **how to add a method** — the second being the one with
no worked example, since only TEM is registered.

Much of this exists in `building_plans/02` and in the package docstrings, and
needs collecting rather than composing.

### 3. An agent or LLM using the library

Distinct from the human reference, and worth stating as its own goal rather than
hoping the human docs serve it.

What an agent needs and a person mostly does not:

- **A machine-readable index of every step** — name, group, full signature with
  types and defaults, what it consumes, what it writes to the dataset, and which
  columns or arrays it creates. Derivable today from the entry points, the
  annotations and the `:param:` entries.
- **The ordering constraints.** Which steps must precede which — the noise model
  before the steps that read it, culling before averaging — because that is the
  part a plausible-looking wrong pipeline gets wrong.
- **The invariants**, especially the ones that are easy to violate silently:
  culling is subtractive, QC writes attributes rather than the mask, verdicts are
  numeric codes, external inputs arrive typed rather than as paths.
- **Working pipeline documents** as examples, which is the same artifact issue #8
  asks for.

The natural form is generated, not hand-written — a structured index built from
the registry at release time, so it cannot fall out of step with what is actually
installed. `llms.txt` is the emerging convention for the entry point.

## Intended structure

```
docs/
  README.md         this file
  plans/            design plans for work not yet done
  guide/            audience 1 — worked example, and how the geophysics works
    figures/        generated from the test suite, never drawn by hand
  architecture/     audience 2 — the runners, the data model, R1-R9, adding a step
  reference/        generated from the registry and the docstrings
  llms.txt          audience 3 — the machine-readable index
```

## Tickets

Nothing here is scheduled. It is tracked so it can be:

- [#9](https://github.com/YmerFlow/ymerflow-processing/issues/9) — user
  documentation: worked example, concepts, step reference
- [#14](https://github.com/YmerFlow/ymerflow-processing/issues/14) — how the
  geophysics works, with a generated figure per filter
- [#15](https://github.com/YmerFlow/ymerflow-processing/issues/15) — the
  machine-readable reference for agents
- [#8](https://github.com/YmerFlow/ymerflow-processing/issues/8) — shipped
  pipeline documents, which both the guide and the agent index depend on

The figures in #14 get materially better once [#10](https://github.com/YmerFlow/ymerflow-processing/issues/10)
lands — a real transient shows what a filter does; a synthetic one shows what it
was told to do.
