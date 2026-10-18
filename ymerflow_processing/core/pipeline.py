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

import importlib.metadata
import importlib.resources
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import yaml

log = logging.getLogger(__name__)

#: Why a check produced no verdict — see :attr:`StepReport.ungraded`.
UNJUDGED = "unjudged"      # no thresholds supplied; the caller's choice
OPTIONAL = "optional"      # input absent, and expected to be
MISSING = "missing"        # input absent, and it should not have been

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


#: Where the bundled pipeline documents live. ``02``'s first lever against the
#: step-list problem is that most users should never see the list — *"load a
#: standard chain and adjust it"* — which only works if the chains ship with the
#: package rather than sitting in a repository the user does not have.
BUNDLED_DIR = "pipelines"


def bundled() -> dict[str, Path]:
    """Every pipeline document shipped with the package, by name.

    Read through ``importlib.resources`` rather than by walking ``__file__``, so
    it keeps working for an installed wheel and not only for a checkout.

    Returns
    -------
    dict[str, Path]
        Document name (the filename without ``.yaml``) to its path.
    """
    try:
        root = importlib.resources.files("ymerflow_processing") / BUNDLED_DIR
        return {p.name.removesuffix(".yaml"): Path(str(p))
                for p in root.iterdir() if p.name.endswith(".yaml")}
    except (FileNotFoundError, ModuleNotFoundError, NotADirectoryError):
        # An install without package data. Worth continuing without: every
        # bundled document is a convenience, and a path still works.
        log.warning("no bundled pipelines found; paths still work")
        return {}


def describe_bundled() -> dict[str, str]:
    """Bundled pipeline names to their one-line description.

    Returns
    -------
    dict[str, str]
        Name to the document's ``description``, or an empty string.
    """
    out = {}
    for name, path in sorted(bundled().items()):
        document = yaml.safe_load(path.read_text()) or {}
        out[name] = " ".join((document.get("description") or "").split())
    return out


def load_steps(group: str) -> dict[str, Callable]:
    """Every step registered under an entry-point group.

    Any installed package can contribute steps, so nothing needs a central
    registry file.

    Parameters
    ----------
    group : str
        The entry-point group — ``FILTER_GROUP``, ``QC_GROUP`` or
        ``COMPARE_GROUP``.

    Returns
    -------
    dict[str, Callable]
        Step name to the function registered under it.
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

    Parameters
    ----------
    step : str
        The step's registered name.
    params : dict
        The parameters it ran with.
    soundings_total, soundings_affected : int | None
        How many soundings there were, and how many this step touched.
    points_total, points_affected : int | None
        The same per gate-sample, for a step that works below the sounding.
    verdict_column : str | None
        Which column holds this step's per-sounding verdict, if it wrote one.
    finding : str | None
        What kind of problem the verdict describes — ``"flight"`` or
        ``"instrument"``; see the attribute's note.
    ungraded : str | None
        Why no verdict was produced — ``"unjudged"``, ``"optional"`` or
        ``"missing"``; ``None`` means one was.
    distributions : dict
        Distribution of whatever was thresholded on, before and after.
    notes : list[str]
        Anything worth surfacing that is not a failure.
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

    #: Why no verdict was produced, when none was. ``None`` means one was.
    #:
    #: Three things look identical in a report and are not, and conflating them
    #: is how six checks failed to find their columns on a real delivery while
    #: the run exited 0 and read as a clean survey:
    #:
    #: ``"unjudged"``
    #:     Measured, and no thresholds were supplied to judge it against. The
    #:     caller's choice, and the distribution is the deliverable.
    #: ``"optional"``
    #:     The input is absent and its absence is expected — data_misfit needs
    #:     an inversion, culled_fraction needs processing history, and a raw
    #:     delivery has neither. ``03`` requires these be absent *"with no
    #:     error"*.
    #: ``"missing"``
    #:     The input should have been there and was not. **This is a finding
    #:     about the run**, not about the flight: a check that was asked for and
    #:     could not act leaves the flight ungraded rather than clean.
    ungraded: str | None = None

    #: Distribution of whatever was thresholded on, before and after.
    distributions: dict[str, Any] = field(default_factory=dict)
    #: Anything worth surfacing that is not a failure — an overwritten column,
    #: a series dropped because its input was absent.
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """The report as plain data, grouped for a reader, ready for JSON.

        Returns
        -------
        dict
            The step, its parameters, the sounding and point counts grouped
            under ``soundings`` and ``points``, and the rest as they are.
        """
        return {
            "step": self.step,
            "params": self.params,
            "verdict_column": self.verdict_column,
            "ungraded": self.ungraded,
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

    Parameters
    ----------
    steps : list[StepReport]
        One report per step, in the order run.
    pipeline : dict | None
        The document that was run.
    identity : str | None
        The human label — a delivery, not a flight.
    identity_source : str | None
        ``"label"`` if supplied, ``"filename"`` if defaulted.
    flights : list | None
        The flights present, read from the data.
    time_range : list[str] | None
        First and last timestamp, read from the data.
    rollup : dict
        What the cross-check rollup wrote, and which checks fed it.
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
        """The run as plain data, every step included, ready for JSON.

        Returns
        -------
        dict
            The identity fields, the pipeline, each step's ``to_dict()``, and
            the rollup.
        """
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
    """A pipeline document.

    Parameters
    ----------
    document : dict
        The parsed YAML: ``steps``, and optionally ``kind`` and
        ``description``.
    """

    def __init__(self, document: dict):
        self.document = dict(document)

    @classmethod
    def parse(cls, text: str) -> "Pipeline":
        return cls(yaml.safe_load(text))

    @classmethod
    def load(cls, reference: str | Path) -> "Pipeline":
        """A pipeline, from a path or from a bundled document's name.

        **A real file always wins.** A name is only consulted when nothing
        exists at that path, so a local ``standard-processing.yaml`` can never
        be shadowed by the bundled document of the same name — the surprising
        direction, and the one that would have someone editing a file that is
        not the one running.
        """
        path = Path(reference)
        if not path.is_file():
            found = bundled().get(str(reference))
            if found is None:
                raise FileNotFoundError(
                    f"No pipeline at {reference!r}, and no bundled pipeline by "
                    f"that name. Bundled: {', '.join(sorted(bundled())) or 'none'}."
                )
            path = found
        return cls(yaml.safe_load(path.read_text()))

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

        Parameters
        ----------
        available : dict[str, Callable]
            The steps the runner can resolve.

        Raises
        ------
        KeyError
            Naming the first step that is not available.
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

    Parameters
    ----------
    pipeline : Pipeline
        The document whose steps are resolved.
    candidates : sequence of str
        The groups to consider, by name.
    group : str | None
        The caller's explicit choice, if any.

    Returns
    -------
    tuple[str, dict[str, Callable]]
        ``(group name, the steps registered under it)``.

    Raises
    ------
    PipelineKindError
        If the stated or declared group does not resolve every step, or no
        candidate does.
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
        from .cull.initial_gates import honor_system_description
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
    """Dataset in, dataset out.

    Parameters
    ----------
    pipeline : Pipeline
        The document to run.
    steps : dict[str, Callable] | None
        The registry to resolve step names in; the ``filters`` entry-point
        group if unset.
    inventory : bool
        Take the implicit inventory before the first step.
    """

    group = FILTER_GROUP

    def run(self, dataset):
        """Apply every step in order, recording each on the dataset.

        Parameters
        ----------
        dataset : Dataset
            The delivery to process.

        Returns
        -------
        tuple
            ``(dataset, RunReport)`` — the processed dataset, and one
            StepReport per step after the implicit inventory.
        """
        report = RunReport(pipeline=self.pipeline.document)
        dataset = self._open(dataset, report)
        for index, (name, params) in enumerate(self.pipeline.steps, start=1):
            log.info("step %d/%d: %s", index, len(self.pipeline.steps), name)
            dataset, step_report = self.steps[name](dataset, **params)
            step_report.params = params
            report.steps.append(step_report)
            dataset.record_step(name, params)
        return dataset, report


class QCRunner(_Runner):
    """Dataset in, dataset plus report out.

    A check writes its verdict as a column on ``flightlines`` and returns the
    aggregate statistics separately — the classification exports with the data,
    the statistics accumulate (R6).

    Parameters
    ----------
    pipeline : Pipeline
        The document to run.
    steps : dict[str, Callable] | None
        The registry to resolve step names in; the ``qc`` entry-point group if
        unset.
    inventory : bool
        Take the implicit inventory before the first check.
    """

    group = QC_GROUP

    def run(self, dataset):
        """Run every check in order, then roll their verdicts up.

        Parameters
        ----------
        dataset : Dataset
            The delivery to grade. Its measurements are not changed; verdict
            columns are added to ``flightlines``.

        Returns
        -------
        tuple
            ``(dataset, RunReport)`` — with one StepReport per check and the
            cross-check rollup filled in.
        """
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

    Parameters
    ----------
    pipeline : Pipeline
        The document to run.
    steps : dict[str, Callable] | None
        The registry to resolve step names in; the ``compare`` entry-point
        group if unset.
    inventory : bool
        Accepted for symmetry with the other runners; a comparison takes no
        inventory.
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

        Parameters
        ----------
        name : str
            The step's registered name.

        Returns
        -------
        str
            ``"reports"`` or ``"datasets"``; ``"reports"`` when the module
            declares nothing.
        """
        check = self.steps[name]
        module = sys.modules.get(getattr(check, "__module__", ""), None)
        return getattr(module, "CONSUMES", self.default_consumes)

    def consumes_all(self) -> str:
        """What this whole pipeline needs, or a raise if its steps disagree.

        One run loads one set of inputs, so a pipeline whose steps want
        different kinds cannot be served — and it is better to say which steps
        want which than to load one kind and let the other steps fail on it.

        Returns
        -------
        str
            ``"reports"`` or ``"datasets"``.

        Raises
        ------
        ValueError
            If the steps disagree, naming which want which.
        """
        wanted = {}
        for name, _ in self.pipeline.steps:
            wanted.setdefault(self.consumes(name), []).append(name)
        if len(wanted) > 1:
            detail = "; ".join(f"{kind}: {', '.join(names)}"
                               for kind, names in sorted(wanted.items()))
            raise ValueError(
                f"This pipeline mixes checks that consume different things "
                f"({detail}). One run loads one set of inputs — split it into "
                f"one pipeline per kind."
            )
        return next(iter(wanted), self.default_consumes)

    def run(self, items: Sequence, reference=None):
        """Run every comparison over the same items.

        Parameters
        ----------
        items : sequence
            The reports or datasets compared, in order; see
            :meth:`consumes_all` for which.
        reference : object | None
            What everything is compared against, for an asymmetric check.

        Returns
        -------
        RunReport
            One StepReport per comparison.
        """
        report = RunReport(pipeline=self.pipeline.document)
        for name, params in self.pipeline.steps:
            log.info("comparison: %s over %d items (consumes %s)",
                     name, len(items), self.consumes(name))
            step_report = self.steps[name](items, reference=reference, **params)
            step_report.params = params
            report.steps.append(step_report)
        return report
