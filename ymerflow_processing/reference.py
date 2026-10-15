"""The step reference, generated from the registry rather than written.

Issue #9's reference half, and most of #15's agent index.

**This is extraction, not authorship.** Every registered step has a docstring,
nearly all carry ``:param:`` entries, and every signature is annotated — so the
reference already exists, scattered across thirty-odd modules. A hand-written
copy would duplicate eighty to a hundred lines of reasoning per module and start
drifting the same afternoon.

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

from __future__ import annotations

import inspect
import logging

from .core.pipeline import (
    COMPARE_GROUP, FILTER_GROUP, QC_GROUP, load_steps,
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
    ("estimate_noise_model", "assign_uncertainty",
     "estimation precedes assignment: there is nothing to assign from"),
    ("assign_uncertainty", "perturb_from_uncertainty",
     "noise is drawn from the assigned uncertainty, not from a system default"),
    ("assign_uncertainty", "average_along_line",
     "averaging propagates the uncertainty, so it must exist first"),
    ("cull_negatives", "average_along_line",
     "cull before averaging, or a bad gate is smeared into its neighbours"),
    ("apply_manual_edits", None,
     "replays last, so a human decision is never overwritten by an automated "
     "one"),
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
    "disable_initial_gates is applied automatically when a survey and its "
    "system description meet. Do not list it in a pipeline.",
]


def _signature(function) -> str:
    """The call signature, without the leading dataset or items argument."""
    parameters = list(inspect.signature(function).parameters.values())[1:]
    rendered = []
    for parameter in parameters:
        # Built piece by piece rather than from str(parameter), because
        # `from __future__ import annotations` makes every annotation a string
        # and renders it quoted — `good_max: 'float'`. Stripping quotes from the
        # whole thing also strips them from string DEFAULTS, turning
        # `output_column: str = 'qs_altim'` into something that reads as a name
        # rather than a string. Only the annotation is unquoted.
        text = parameter.name
        if parameter.annotation is not inspect.Parameter.empty:
            annotation = parameter.annotation
            text += f": {annotation if isinstance(annotation, str) else annotation}"
        if parameter.default is not inspect.Parameter.empty:
            text += f" = {parameter.default!r}"
        rendered.append(text)
    return ", ".join(rendered)


def parameter_descriptions(function) -> dict[str, str]:
    """Each ``:param name:`` entry in a step's docstring, by parameter name.

    Public because a caller building a form for a step needs exactly this: the
    platform's process types show it under each field.
    """
    doc = inspect.getdoc(function) or ""
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


def _parameters(function) -> list[tuple[str, str, str]]:
    """``(name, default, description)`` for each parameter after the first."""
    described = parameter_descriptions(function)
    rows = []
    for parameter in list(inspect.signature(function).parameters.values())[1:]:
        default = ("required" if parameter.default is inspect.Parameter.empty
                   else repr(parameter.default))
        rows.append((parameter.name, default,
                     described.get(parameter.name, "")))
    return rows


def summary(function) -> str:
    """The docstring's first paragraph — what the step does, in prose."""
    doc = inspect.getdoc(function) or ""
    out = []
    for line in doc.splitlines():
        if not line.strip():
            if out:
                break
            continue
        if line.strip().startswith(":"):
            break
        out.append(line.strip())
    return " ".join(out)


def _consumes(function) -> str | None:
    """What a comparison check declares it needs, if it declares anything."""
    import sys
    module = sys.modules.get(getattr(function, "__module__", ""), None)
    return getattr(module, "CONSUMES", None)


def markdown(group: str) -> str:
    """The reference for one runner group, as markdown."""
    name, entry_point, purpose = next(g for g in GROUPS if g[0] == group)
    steps = load_steps(entry_point)

    out = [f"# {name}", "",
           "*Generated from the entry-point registry — do not edit by hand.*",
           "", purpose, "",
           f"{len(steps)} step(s).", ""]

    for step in sorted(steps):
        function = steps[step]
        out += [f"## `{step}`", ""]
        summary_text = summary(function)
        if summary_text:
            out += [summary_text, ""]

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


def llms_txt() -> str:
    """The machine-readable index: facts an agent needs, without the prose."""
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
            suffix = f"  [consumes: {consumes}]" if consumes else ""
            out.append(f"### {step}{suffix}")
            out.append(f"{step}({_signature(function)})")
            summary_text = summary(function)
            if summary_text:
                out.append(summary_text)
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def generate(directory) -> dict[str, str]:
    """Every reference file, as ``{relative path: content}``.

    Returned rather than written, so a test can compare against what is on disk
    without a temporary directory — the check that keeps this from drifting.
    """
    files = {f"{name}.md": markdown(name) for name, _, _ in GROUPS}
    files["llms.txt"] = llms_txt()
    return files
