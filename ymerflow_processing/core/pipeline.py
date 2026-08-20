"""Runners.

A pipeline is a YAML document, not code — storable, diffable, versionable and
transmissible without being executable. A runner walks it against data.

Three runners, because there are three genuinely different shapes
(building_plans/02):

===================  =====================================  =========================
runner               shape                                  entry-point group
===================  =====================================  =========================
processing           dataset -> dataset                     ymerflow.processing.filters
QC                   dataset -> dataset + report            ymerflow.processing.qc
comparison           N reports (or datasets) -> one report   ymerflow.processing.compare
===================  =====================================  =========================

Every step returns a :class:`StepReport` alongside its result. Not a printed line —
a value the runner collects, the process record stores and the UI renders (R3).
Diagnostics go to ``logging``; what a step *did* goes in the report.
"""

from __future__ import annotations

import importlib.metadata
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import yaml

log = logging.getLogger(__name__)

FILTER_GROUP = "ymerflow.processing.filters"
QC_GROUP = "ymerflow.processing.qc"
COMPARE_GROUP = "ymerflow.processing.compare"

#: A fourth group, which is not a runner: method packages register here to say
#: how their arrays are named and what the second axis means. Defined with the
#: data object, since that is the concept it belongs to, and re-exported here so
#: the CLI can list it alongside the runner groups.
from .dataset import METHOD_GROUP  # noqa: E402  (grouped with its siblings)

#: Short names for the CLI, so a user types ``--group qc`` rather than the
#: fully-qualified entry-point group.
GROUPS = {
    "filters": FILTER_GROUP,
    "qc": QC_GROUP,
    "compare": COMPARE_GROUP,
    "methods": METHOD_GROUP,
}


def load_steps(group: str) -> dict[str, Callable]:
    """Every step registered under an entry-point group.

    Any installed package can contribute steps, so nothing needs a central
    registry file.
    """
    return {ep.name: ep.load() for ep in importlib.metadata.entry_points(group=group)}


# ── What a step reports ──────────────────────────────────────────────────────

@dataclass
class StepReport:
    """A structured record of what a step did (R3).

    The minimum every step should fill in: how much it touched out of how much,
    and — for anything with a threshold — the distribution of the quantity it
    thresholded on, before and after. That is what turns "disabled 4,182 gates"
    into something a user can judge.
    """

    step: str
    params: dict[str, Any] = field(default_factory=dict)

    soundings_total: int | None = None
    soundings_affected: int | None = None
    points_total: int | None = None
    points_affected: int | None = None

    #: Which column this step wrote its per-sounding verdict to, if it wrote
    #: one. Declared rather than discovered, so a caller can tell one check's
    #: verdict from another's without inspecting column dtypes.
    verdict_column: str | None = None

    #: What kind of problem the verdict describes. Not every Out of Spec means
    #: the same thing, and conflating them is how a dead sensor comes to read as
    #: a badly flown line:
    #:
    #: ``"flight"``
    #:     The aircraft was not flown as contracted — height, attitude, speed,
    #:     coverage. This is what a reflight conversation is about.
    #: ``"instrument"``
    #:     Equipment is misbehaving — altimeters disagreeing, transmitter
    #:     current drifting. The data may be perfectly usable; what needs
    #:     fixing is the kit, before the next flight.
    #:
    #: Free-form rather than an enum, because more kinds are coming — cultural
    #: noise is environmental, culled fraction is a processing outcome — and
    #: ``01`` makes the same choice for ``line_type``.
    finding: str | None = None

    #: Distribution of whatever was thresholded on, before and after.
    distributions: dict[str, Any] = field(default_factory=dict)
    #: Anything worth surfacing that is not a failure — an overwritten column,
    #: a series dropped because its input was absent.
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "params": self.params,
            "verdict_column": self.verdict_column,
            "finding": self.finding,
            "soundings": {"total": self.soundings_total,
                          "affected": self.soundings_affected},
            "points": {"total": self.points_total,
                       "affected": self.points_affected},
            "distributions": self.distributions,
            "notes": self.notes,
        }


@dataclass
class RunReport:
    """Every step's report, in order, plus what the run was and what it ran on.

    The identity fields exist because the comparison runner reads stored reports
    and has to be able to say "flight 6 is the outlier" rather than "item 4".

    Two different things are recorded, and the distinction matters:

    ``identity`` is a **human label** for the run — normally the delivered
    file's name. It is not required to be meaningful, parseable or stable,
    because nothing computes with it.

    ``flights`` and ``time_range`` are **read from the data**. These are what a
    drift check orders its series by, and being derived from the soundings
    themselves they survive a file being renamed or re-exported. Delivery names
    routinely embed an export date, so an identity taken from a filename is not
    stable across re-exports — which would enter one delivery into a series
    twice. Keeping the ordering key separate from the label removes that.
    """

    steps: list[StepReport] = field(default_factory=list)
    pipeline: dict | None = None

    #: What to call this run. A delivery, not a flight — one delivered file
    #: commonly holds three or four flights.
    identity: str | None = None
    #: "label" if supplied, "filename" if defaulted. R3 asks for the parameters
    #: actually used, including the defaults nobody supplied.
    identity_source: str | None = None

    #: Read from the data, not from the invocation.
    flights: list | None = None
    time_range: list[str] | None = None

    #: What the cross-check rollup wrote, and which checks fed it. Empty for a
    #: run whose checks wrote no verdicts. See ``core/qc/rollup.py``.
    rollup: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "identity": self.identity,
            "identity_source": self.identity_source,
            "flights": self.flights,
            "time_range": self.time_range,
            "pipeline": self.pipeline,
            "rollup": self.rollup,
            "steps": [s.to_dict() for s in self.steps],
        }


# ── Reading a pipeline document ──────────────────────────────────────────────

def _normalize_step(step) -> tuple[str, dict]:
    """A step is either a bare name or a single-key mapping of name to params."""
    if isinstance(step, str):
        return step, {}
    if isinstance(step, dict) and len(step) == 1:
        name, params = next(iter(step.items()))
        return name, dict(params or {})
    raise ValueError(
        f"A step must be a name or a single-key mapping of name to parameters; got {step!r}"
    )


class Pipeline:
    """A pipeline document."""

    def __init__(self, document: dict):
        self.document = dict(document)

    @classmethod
    def parse(cls, text: str) -> "Pipeline":
        return cls(yaml.safe_load(text))

    @classmethod
    def load(cls, path: str | Path) -> "Pipeline":
        return cls(yaml.safe_load(Path(path).read_text()))

    @property
    def steps(self) -> list[tuple[str, dict]]:
        return [_normalize_step(s) for s in self.document.get("steps", [])]

    @property
    def kind(self) -> str | None:
        """Which runner this document declares itself for, if it declares one.

        Optional. Where present it is **validated against the steps rather than
        trusted** (see :func:`resolve_group`), so a document cannot quietly
        misdescribe itself.
        """
        return self.document.get("kind")

    def validate(self, available: dict[str, Callable]) -> None:
        """Fail before running rather than part-way through.

        A job reserves budget up front, so an unknown step found at step nine is
        a wasted reservation. Checked at submission instead.
        """
        unknown = [name for name, _ in self.steps if name not in available]
        if unknown:
            raise KeyError(
                f"Unknown step(s): {', '.join(unknown)}. "
                f"Available: {', '.join(sorted(available))}"
            )


# ── Which registry a step name is looked up in ───────────────────────────────

class PipelineKindError(ValueError):
    """The runner for a pipeline could not be settled, or was declared wrongly."""


def resolve_group(pipeline: Pipeline, candidates: Sequence[str],
                  group: str | None = None) -> tuple[str, dict[str, Callable]]:
    """Decide which entry-point group ``pipeline``'s steps belong to.

    Three sources, in order of authority:

    1. **``group``**, when the caller states it — an explicit override, and the
       way out if a step name is ever ambiguous across two registries.
    2. **``pipeline.kind``**, when the document declares one. Checked against the
       steps, not taken on trust: a document claiming ``filters`` whose steps
       only exist under ``qc`` is an error rather than a silent mismatch.
    3. **Derived**, by finding which registry resolves every step.

    This exists to remove a genuinely misleading failure. Looking a QC pipeline
    up in the filters registry reports ``Unknown step: height_above_ground`` —
    blaming a step that is perfectly valid, when the real problem is the
    registry. Under all three paths above that cannot happen.

    Returns ``(group name, the steps registered under it)``.
    """
    wanted = [name for name, _ in pipeline.steps]

    if group is not None:
        if group not in GROUPS:
            raise PipelineKindError(
                f"Unknown group {group!r}. Available: {', '.join(sorted(GROUPS))}")
        return group, load_steps(GROUPS[group])

    resolvable = {}
    for candidate in candidates:
        registered = load_steps(GROUPS[candidate])
        if wanted and all(name in registered for name in wanted):
            resolvable[candidate] = registered

    declared = pipeline.kind
    if declared is not None:
        if declared not in candidates:
            raise PipelineKindError(
                f"This pipeline declares kind: {declared!r}, which is not one of "
                f"{', '.join(candidates)}."
            )
        if declared not in resolvable:
            registered = load_steps(GROUPS[declared])
            missing = [name for name in wanted if name not in registered]
            found_in = [c for c in resolvable]
            hint = (f" They are registered under {', '.join(found_in)}."
                    if found_in else "")
            raise PipelineKindError(
                f"This pipeline declares kind: {declared!r}, but "
                f"{', '.join(missing)} is not registered there.{hint}"
            )
        return declared, resolvable[declared]

    if len(resolvable) == 1:
        [(name, registered)] = resolvable.items()
        log.info("resolved pipeline as %s", name)
        return name, registered

    if not resolvable:
        available = sorted(
            {n for c in candidates for n in load_steps(GROUPS[c])})
        raise PipelineKindError(
            f"No registry resolves every step of this pipeline. "
            f"Wanted: {', '.join(wanted) or '(none)'}. "
            f"Available across {', '.join(candidates)}: "
            f"{', '.join(available) or '(nothing registered)'}"
        )

    raise PipelineKindError(
        f"These steps resolve under more than one registry "
        f"({', '.join(sorted(resolvable))}). Say which with a 'kind:' in the "
        f"document or --group on the command line."
    )


# ── Runners ──────────────────────────────────────────────────────────────────

class _Runner:
    """Shared walk. Subclasses say which entry-point group they draw from."""

    group: str = ""

    def __init__(self, pipeline: Pipeline, steps: dict[str, Callable] | None = None,
                 inventory: bool = True):
        self.pipeline = pipeline
        self.steps = steps if steps is not None else load_steps(self.group)
        self.pipeline.validate(self.steps)
        self.inventory = inventory

    def _open(self, dataset, report):
        """Account for the delivery before any step acts on it.

        Implicit rather than something a pipeline has to remember, because the
        thing it exists to surface — data the channel model cannot account for —
        is exactly what nobody thinks to ask about. A step that only runs when
        requested would never run on the delivery that needed it.

        Reads only, so it is safe in a QC pipeline where R6 promises the data
        comes out byte-identical.
        """
        # Imported here rather than at module scope: both import StepReport
        # from this module.
        from .cull.filter_20_initial_gates import honor_system_description
        from .inventory import inventory as take_inventory

        # A dataset built without load() — a fixture, or one the platform
        # assembled — has not had the declaration applied yet. Subtractive, so
        # this is a no-op on one that has.
        honor_system_description(dataset)

        if not self.inventory:
            return dataset
        dataset, step_report = take_inventory(dataset)
        step_report.params = {"implicit": True}
        report.steps.append(step_report)
        return dataset


class ProcessingRunner(_Runner):
    """dataset -> dataset."""

    group = FILTER_GROUP

    def run(self, dataset):
        report = RunReport(pipeline=self.pipeline.document)
        dataset = self._open(dataset, report)
        for index, (name, params) in enumerate(self.pipeline.steps, start=1):
            log.info("step %d/%d: %s", index, len(self.pipeline.steps), name)
            dataset, step_report = self.steps[name](dataset, **params)
            step_report.params = params
            report.steps.append(step_report)
            dataset.history.append({"step": name, "params": params})
        return dataset, report


class QCRunner(_Runner):
    """dataset -> dataset + report.

    A check writes its verdict as a column on ``flightlines`` and returns the
    aggregate statistics separately — the classification exports with the data,
    the statistics accumulate (R6).
    """

    group = QC_GROUP

    def run(self, dataset):
        report = RunReport(pipeline=self.pipeline.document)
        dataset = self._open(dataset, report)
        for index, (name, params) in enumerate(self.pipeline.steps, start=1):
            log.info("check %d/%d: %s", index, len(self.pipeline.steps), name)
            dataset, step_report = self.steps[name](dataset, **params)
            step_report.params = params
            report.steps.append(step_report)

        # Imported here rather than at module scope: the checks import
        # StepReport from this module, so a top-level import back into the qc
        # package would be a cycle waiting to happen as that package grows.
        from .qc.rollup import summarize_run
        report.rollup = summarize_run(dataset, report.steps,
                                      self.pipeline.document.get("rollup"))
        return dataset, report


class ComparisonRunner(_Runner):
    """N reports (or N datasets) -> one report.

    ``items`` is **ordered**, and a check may rely on that — drift depends on it,
    equivalence does not care. Each item carries its identity, so a report can
    say "flight 6 is the outlier" rather than "item 4".

    ``reference`` names the thing everything is compared against, where the
    comparison is asymmetric.
    """

    group = COMPARE_GROUP

    #: What a check consumes when it does not say. Per ``02``: reports are the
    #: cheap path — small records, no storage backend, no credentials — and a
    #: check that silently loaded gigabytes because a line was omitted would be
    #: a worse failure than one that errors saying it needs data.
    default_consumes = "reports"

    def consumes(self, name: str) -> str:
        """What a check declares it needs: ``"reports"`` or ``"datasets"``.

        Declared as a ``CONSUMES`` attribute on the check's module, so the
        declaration lives beside the code that depends on it rather than in a
        registry someone has to remember to update.
        """
        check = self.steps[name]
        module = sys.modules.get(getattr(check, "__module__", ""), None)
        return getattr(module, "CONSUMES", self.default_consumes)

    def run(self, items: Sequence, reference=None):
        report = RunReport(pipeline=self.pipeline.document)
        for name, params in self.pipeline.steps:
            log.info("comparison: %s over %d items (consumes %s)",
                     name, len(items), self.consumes(name))
            step_report = self.steps[name](items, reference=reference, **params)
            step_report.params = params
            report.steps.append(step_report)
        return report
