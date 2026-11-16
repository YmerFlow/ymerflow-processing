"""The step reference, generated from the registry rather than written.

Issue #9's reference half, and most of #15's agent index.

**This is extraction, not authorship.** Every registered step has a docstring,
nearly all describe their parameters, and every signature is annotated — so the
reference already exists, scattered across thirty-odd modules. A hand-written
copy would duplicate eighty to a hundred lines of reasoning per module and start
drifting the same afternoon.

NumPy-style ``Parameters`` sections are the standard. The Sphinx field lists
this library was first written in are still read, so the conversion can proceed
one module at a time; that branch goes once the last one is converted.

Generated from the entry-point registry specifically, not from a list of
modules: the registry is **open**, so a step contributed by any installed package
appears without anyone remembering to write it down. That is the property that
makes this stay true.

Two outputs, because two readers want different things
-------------------------------------------------------
**Markdown**, for a person: grouped by runner, each step with its signature, its
parameters and their defaults, and its prose.

**``llms.txt``**, for an agent: the same facts without the prose, plus the things
an agent needs and a person mostly infers — what a step consumes, which steps
must precede which, and the invariants that are easy to violate silently.

Kept honest by a test
---------------------
The generated files are committed, so a fresh clone and GitHub both have them.
A test regenerates and diffs: documentation that can drift from the code will,
and the only reliable fix is to make drift a failing test rather than a habit.
"""

import inspect
import logging
import typing

from numpydoc.docscrape import NumpyDocString

from .pipeline import (
    CATEGORIES, COMPARE_GROUP, FILTER_GROUP, QC_GROUP, SINK, category, load_steps,
)

log = logging.getLogger(__name__)

#: The runner groups, in the order a pipeline runs them, with what each is for.
GROUPS = [
    ("filters", FILTER_GROUP,
     "dataset in, dataset out. These change the data."),
    ("qc", QC_GROUP,
     "dataset in, dataset plus a report out. These grade it and change no "
     "measurement."),
    ("compare", COMPARE_GROUP,
     "N reports or datasets in, one report out. These answer questions no "
     "single run can."),
]

#: Ordering constraints, stated once here because they are the part a
#: plausible-looking wrong pipeline gets wrong. Each is recorded in
#: ``building_plans/01``; none is discoverable from a signature.
ORDERING = [
    ("set_noise_floor", "assign_uncertainty",
     "estimation precedes assignment: there is nothing to assign from"),
    ("assign_uncertainty", "add_noise",
     "noise is drawn from the assigned uncertainty, not from a system default"),
    ("assign_uncertainty", "average_along_line",
     "averaging propagates the uncertainty, so it must exist first"),
    ("cull_negatives", "average_along_line",
     "cull before averaging, or a bad gate is smeared into its neighbors"),
    ("assign_uncertainty", "cull_by_coupling",
     "the reconstruction error is in sigmas of the assigned uncertainty"),
    ("cull_by_coupling", "average_along_line",
     "judge before averaging, or the coupling is smeared into its neighbors"),
    ("apply_manual_edits", None,
     "replays last, so a human decision is never overwritten by an automated "
     "one"),
    ("save_intermediate_and_apply_diff", "cull_sparse_soundings",
     "an edit can leave a sounding with too few gates, and only a later "
     "sparse cull can catch it"),
]

#: The invariants that are easy to violate silently. Stated for an agent, which
#: has no way to infer them from a signature and every reason to guess wrong.
INVARIANTS = [
    "Culling is subtractive. Data already culled stays culled, and provenance "
    "names the step that FIRST removed each datum.",
    "QC writes attributes, never the mask. Deciding what to remove is a "
    "separate job from deciding what is wrong.",
    "Verdicts are numeric codes on a fixed scale (0 out of spec, 1 marginal, "
    "2 good), not labels. NaN means not evaluated.",
    "External inputs arrive typed, not as paths. Format is a loader's concern.",
    "A step returns (result, StepReport) and never prints.",
    "The number of channels comes from the data. No default may name a "
    "specific moment.",
    "One noise model per dataset, so uncertainty cannot be assigned with one "
    "floor and data culled with another.",
    "The system description's RemoveInitialGates is applied automatically when "
    "a survey and its description meet. It is not a step and cannot be listed.",
]


def _listed(function) -> list[inspect.Parameter]:
    """The parameters a pipeline document can set: not the leading dataset or
    items argument, and not the runner's sink."""
    parameters = list(inspect.signature(function).parameters.values())[1:]
    return [p for p in parameters if p.name != SINK]


def _signature(function) -> str:
    """The call signature, without the arguments a document cannot set."""
    parameters = _listed(function)
    rendered = []
    for parameter in parameters:
        # Built piece by piece rather than from str(parameter), which quotes
        # string DEFAULTS away — `output_column: str = 'qs_altim'` reading as a
        # name rather than a string. Only the annotation is rendered bare.
        #
        # An annotation arrives either as a real object or, under
        # `from __future__ import annotations`, as the source text. Both are
        # accepted: a string is already what we want to print, and
        # `formatannotation` would wrap it in quotes.
        #
        # An Annotated type is printed as the type it annotates. Its metadata
        # is for the platform's form, and Python 3.10 and 3.12 print it
        # differently, so the committed reference could match only one.
        text = parameter.name
        if parameter.annotation is not inspect.Parameter.empty:
            annotation = parameter.annotation
            if typing.get_origin(annotation) is typing.Annotated:
                annotation = typing.get_args(annotation)[0]
            text += ": " + (annotation if isinstance(annotation, str)
                            else inspect.formatannotation(annotation))
        if parameter.default is not inspect.Parameter.empty:
            text += f" = {parameter.default!r}"
        rendered.append(text)
    return ", ".join(rendered)


def _is_underline(line: str) -> bool:
    """A NumPy section underline: dashes, and nothing else."""
    stripped = line.strip()
    return bool(stripped) and set(stripped) == {"-"}


def _sphinx_parameters(doc: str) -> dict[str, str]:
    """Descriptions from ``:param name: text`` field lists."""
    described: dict[str, str] = {}
    current = None
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped.startswith(":param "):
            name, _, text = stripped[len(":param "):].partition(":")
            current = name.strip()
            described[current] = text.strip()
        elif not stripped:
            # A blank line ends the entry. Without this the description runs on
            # into whatever follows — every step's last parameter was carrying
            # "Returns (dataset, StepReport)" glued to the end of its text.
            current = None
        elif current and not stripped.startswith(":"):
            described[current] += " " + stripped
        elif stripped.startswith(":"):
            current = None
    return described


def _numpy_parameters(doc: str) -> dict[str, str]:
    """Descriptions from a NumPy-style ``Parameters`` section.

    Read with numpydoc's own parser — the same one swaggerspect uses to build
    the platform's step forms, so the reference and the forms cannot disagree
    about what a docstring says. The type is ignored: it is already in the
    signature, and printing it twice is how the two drift apart.
    """
    parsed = NumpyDocString(doc)
    return {entry.name: " ".join(line.strip() for line in entry.desc).strip()
            for entry in parsed["Parameters"]}


def _parameters(function) -> list[tuple[str, str, str]]:
    """``(name, default, description)`` for each parameter a document can set.

    Both docstring styles are read. NumPy is the standard; the Sphinx field
    lists this library was first written in are still accepted, so that the
    conversion can proceed one module at a time without the generated
    reference changing underneath it.
    """
    doc = inspect.getdoc(function) or ""
    described = _sphinx_parameters(doc)
    described.update({k: v for k, v in _numpy_parameters(doc).items() if v})

    rows = []
    for parameter in _listed(function):
        default = ("required" if parameter.default is inspect.Parameter.empty
                   else repr(parameter.default))
        rows.append((parameter.name, default,
                     described.get(parameter.name, "")))
    return rows


def _summary(function) -> str:
    """The docstring's first paragraph — what the step does, in prose."""
    doc = inspect.getdoc(function) or ""
    lines = doc.splitlines()
    out = []
    for index, line in enumerate(lines):
        if not line.strip():
            if out:
                break
            continue
        if line.strip().startswith(":"):
            break
        # A heading, even with no blank line before it, ends the summary —
        # otherwise "Parameters" is read as the last word of the prose.
        if index + 1 < len(lines) and _is_underline(lines[index + 1]):
            break
        out.append(line.strip())
    return " ".join(out)


def _consumes(function) -> str | None:
    """What a comparison check declares it needs, if it declares anything."""
    import sys
    module = sys.modules.get(getattr(function, "__module__", ""), None)
    return getattr(module, "CONSUMES", None)


def markdown(group: str) -> str:
    """The reference for one runner group, as markdown.

    Parameters
    ----------
    group : str
        ``"filters"``, ``"qc"`` or ``"compare"`` — a name from :data:`GROUPS`.

    Returns
    -------
    str
        The document, ending in one newline.
    """
    name, entry_point, purpose = next(g for g in GROUPS if g[0] == group)
    steps = load_steps(entry_point)

    out = [f"# {name}", "",
           "*Generated from the entry-point registry — do not edit by hand.*",
           "", purpose, "",
           f"{len(steps)} step(s).", ""]

    for heading, names in _sections(steps):
        if heading:
            out += [f"## {heading}", ""]
        for step in names:
            function = steps[step]
            out += [f"### `{step}`" if heading else f"## `{step}`", ""]
            summary = _summary(function)
            if summary:
                out += [summary, ""]

            consumes = _consumes(function)
            if consumes:
                out += [f"**Consumes:** {consumes}", ""]

            out += ["```python", f"{step}({_signature(function)})", "```", ""]

            rows = _parameters(function)
            if rows:
                out += ["| parameter | default | |", "|---|---|---|"]
                out += [f"| `{n}` | `{d}` | {t} |" for n, d, t in rows]
                out += [""]
    return "\n".join(out).rstrip() + "\n"


def _sections(steps: dict) -> list[tuple[str | None, list[str]]]:
    """Steps grouped by category in :data:`CATEGORIES` order, sorted within.

    A group whose steps declare no category — QC, compare — is one unheaded
    section, so those documents read as they always did.
    """
    by_category: dict = {}
    for step, function in steps.items():
        by_category.setdefault(category(function), []).append(step)
    if list(by_category) == [None]:
        return [(None, sorted(steps))]
    ordered = [(c, sorted(by_category[c])) for c in CATEGORIES if c in by_category]
    # Anything declaring a category outside the list, or none, still appears —
    # the test is what refuses it, not the renderer.
    for other in sorted(k for k in by_category if k not in CATEGORIES):
        ordered.append((other or "uncategorized", sorted(by_category[other])))
    return ordered


def llms_txt() -> str:
    """The machine-readable index: facts an agent needs, without the prose.

    Returns
    -------
    str
        The ``llms.txt`` document.
    """
    out = ["# ymerflow-processing",
           "",
           "Airborne geophysical data processing, QC and comparison.",
           "Generated from the entry-point registry — do not edit by hand.",
           "",
           "## Invariants",
           "",
           "These are easy to violate silently and cannot be inferred from a "
           "signature.",
           ""]
    out += [f"- {line}" for line in INVARIANTS]

    out += ["", "## Ordering", "",
            "A wrong order here does not fail. It produces plausible nonsense.",
            ""]
    for first, second, why in ORDERING:
        where = f"`{first}` before `{second}`" if second else f"`{first}` last"
        out.append(f"- {where} — {why}")

    for name, entry_point, purpose in GROUPS:
        steps = load_steps(entry_point)
        out += ["", f"## {name}", "", purpose, ""]
        for step in sorted(steps):
            function = steps[step]
            consumes = _consumes(function)
            kind = category(function)
            suffix = (f"  [consumes: {consumes}]" if consumes
                      else f"  [{kind}]" if kind else "")
            out.append(f"### {step}{suffix}")
            out.append(f"{step}({_signature(function)})")
            summary = _summary(function)
            if summary:
                out.append(summary)
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def generate(directory) -> dict[str, str]:
    """Every reference file, as ``{relative path: content}``.

    Returned rather than written, so a test can compare against what is on disk
    without a temporary directory — the check that keeps this from drifting.

    Parameters
    ----------
    directory : object
        Unused; accepted so the call reads as "generate for this directory".

    Returns
    -------
    dict[str, str]
        Relative path to content, for every reference file.
    """
    files = {f"{name}.md": markdown(name) for name, _, _ in GROUPS}
    files["llms.txt"] = llms_txt()
    return files
