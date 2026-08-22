"""The command line.

What makes this a library you can run rather than a component of a platform.
Acquisition QC has to run per delivery, on a schedule, on whatever machine is
available while the aircraft is still on site — no web stack, no database, no
browser. That is one command in a cron entry, and this is it.

Two commands, because the three runners do not divide evenly. ``filters`` and
``qc`` share a shape — one dataset in, dataset plus report out — so one ``run``
serves both. ``compare`` reads N reports and writes one, which is different
enough that folding it in would leave most options dead or reinterpreted
depending on a flag.

Specification: ``building_plans/04_cli_runner.md``.

On output: ``click.echo`` here is the user interface, not diagnostics. R3's rule
that steps must not ``print()`` is about *steps*, whose job is to return a
structured record. A command-line program's job is to talk to a person. Anything
diagnostic still goes to ``logging``, behind ``-v``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from .dataset import load
from .pipeline import (
    COMPARE_GROUP, GROUPS, ComparisonRunner, Pipeline, PipelineKindError,
    ProcessingRunner, QCRunner, RunReport, load_steps, resolve_group,
)
from .qc.thresholds import GOOD, MARGINAL, OUT_OF_SPEC, worst_verdict

log = logging.getLogger(__name__)

# ── Exit codes ───────────────────────────────────────────────────────────────
# One number has to answer two questions — did the run happen, and was the
# flight any good — so the codes carry both. The split falls at 2: 0-2 mean the
# run completed, 3-4 mean it did not. A finding is not a failure; a check that
# reports a line flown out of spec has succeeded, and its report is the
# deliverable.

EXIT_GOOD = 0
EXIT_MARGINAL = 1
EXIT_OUT_OF_SPEC = 2
EXIT_NOT_STARTED = 3        # your mistake: bad options, missing file, unknown step
EXIT_STEP_RAISED = 4        # our bug: a step blew up mid-run

_VERDICT_EXIT = {None: EXIT_GOOD, GOOD: EXIT_GOOD,
                 MARGINAL: EXIT_MARGINAL, OUT_OF_SPEC: EXIT_OUT_OF_SPEC}

#: Which runner walks which group. ``compare`` is absent because it has its own
#: command; its inputs are not a dataset.
RUNNERS = {"filters": ProcessingRunner, "qc": QCRunner}


def _column_verdict(frame, column):
    """The worst verdict in one column, or ``None`` if it is not there."""
    if frame is None or not column or column not in frame.columns:
        return None
    return worst_verdict(frame[[column]])


class InputError(Exception):
    """An input the user can fix, as opposed to a bug in the loader.

    The distinction decides whether a traceback is printed. A clear sentence
    about the wrong file is help; the same sentence under twelve frames of
    stack reads as a crash, and the reader stops trusting the sentence.
    """


def _fail(message: str, code: int = EXIT_NOT_STARTED) -> int:
    click.secho(f"error: {message}", fg="red", err=True)
    return code


# ── run ──────────────────────────────────────────────────────────────────────

def _write_report(report: RunReport, path: Path) -> None:
    path.write_text(json.dumps(report.to_dict(), indent=2, default=str))


def _summarize(report: RunReport, dataset) -> None:
    """A few readable lines on stdout. The detail is in the report file."""
    for step in report.steps:
        counts = step.distributions.get("verdicts")
        if counts:
            click.echo(
                f"  {step.step}: "
                + ", ".join(f"{n} {tier}" for tier, n in counts.items() if n)
            )
        else:
            click.echo(f"  {step.step}")
        for note in step.notes:
            click.secho(f"    note: {note}", fg="yellow")


@click.command()
@click.argument("pipeline_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--xyz", required=True, type=click.Path(),
              help="The survey file to run against.")
@click.option("--gex", required=True, type=click.Path(),
              help="The system description.")
@click.option("--out", required=True, type=click.Path(file_okay=False),
              help="Directory for the resulting dataset and report.")
@click.option("--label", default=None,
              help="What to call this run. Defaults to the survey file's name.")
@click.option("--group", type=click.Choice(sorted(RUNNERS)), default=None,
              help="Override which registry the steps are looked up in.")
def run(pipeline_path, xyz, gex, out, label, group):
    """Run a processing or QC pipeline against a delivery.

    Exit codes: 0 in spec, 1 marginal, 2 out of spec, 3 never started,
    4 a step raised.
    """
    pipeline = Pipeline.load(pipeline_path)

    # Everything that can be checked without touching data is checked first. A
    # scheduled overnight run should not die at 3 a.m. on a typo after loading
    # gigabytes.
    try:
        group, steps = resolve_group(pipeline, list(RUNNERS), group)
        pipeline.validate(steps)
    except (PipelineKindError, KeyError) as exc:
        return _fail(str(exc).strip("'"))

    try:
        dataset = load(xyz, gex)
    except FileNotFoundError as exc:
        return _fail(str(exc))

    identity = label or Path(xyz).stem
    report = RunReport(
        pipeline=pipeline.document,
        identity=identity,
        identity_source="label" if label else "filename",
        **dataset.describe(),
    )

    click.echo(f"{identity}: {dataset.n_soundings} soundings, "
               f"{len(pipeline.steps)} step(s) [{group}]")
    try:
        dataset, walked = RUNNERS[group](pipeline, steps).run(dataset)
    except Exception as exc:                      # noqa: BLE001 — reported, not swallowed
        log.exception("step raised")
        return _fail(f"a step raised: {exc}", EXIT_STEP_RAISED)
    report.steps = walked.steps
    # The runner's cross-check rollup, which the exit code below reads.
    report.rollup = walked.rollup

    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    dataset.survey.dump(xyzfile=str(directory / f"{identity}.xyz"))
    _write_report(report, directory / f"{identity}.report.json")

    _summarize(report, dataset)
    click.echo(f"wrote {directory}/{identity}.xyz and {identity}.report.json")

    # The exit code answers "was the flight any good", so it is driven by
    # flight-quality findings alone. A failing altimeter is urgent and gets
    # said loudly, but it is not a badly flown line and must not read as one.
    columns = report.rollup.get("columns", {})
    flight = _column_verdict(dataset.flightlines, columns.get("verdict"))
    sensor = _column_verdict(dataset.flightlines, columns.get("sensor"))

    if flight is not None:
        click.echo(f"worst verdict: {flight}")
    if sensor is not None and sensor != GOOD:
        click.secho(
            f"instrument: {sensor} — equipment fault, not a flying problem. "
            "The data may still be usable; see the report.", fg="yellow")
    return _VERDICT_EXIT[flight]


# ── compare ──────────────────────────────────────────────────────────────────

@click.command()
@click.argument("pipeline_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--report", "report_paths", multiple=True,
              type=click.Path(exists=True, dir_okay=False),
              help="A stored report. Repeat it; order is significant.")
@click.option("--dataset", "dataset_paths", multiple=True,
              type=click.Path(exists=True, dir_okay=False),
              help="A survey file, for checks that consume datasets. Repeat "
                   "it; order is significant. Needs --gex.")
@click.option("--gex", default=None, type=click.Path(exists=True,
                                                     dir_okay=False),
              help="The system description, required with --dataset.")
@click.option("--reference", default=None,
              type=click.Path(exists=True, dir_okay=False),
              help="What everything is compared against, where the comparison "
                   "is asymmetric. Whichever kind the pipeline consumes.")
@click.option("--out", required=True, type=click.Path(dir_okay=False),
              help="File to write the comparison report to.")
def compare(pipeline_path, report_paths, dataset_paths, gex, reference, out):
    """Compare deliveries — drift across flights, two versions, or two implementations.

    ``--report`` and ``--dataset`` are repeatable and **ordered**; a check may
    rely on that, since drift depends on sequence while equivalence does not.

    Which one to pass is not a preference: each check declares what it consumes,
    and the pipeline is asked before anything is read. Reports are the cheap
    path — small records, no system description, runs anywhere — and a check
    needing per-datum arrays says so.
    """
    pipeline = Pipeline.load(pipeline_path)
    try:
        steps = load_steps(COMPARE_GROUP)
        pipeline.validate(steps)
    except KeyError as exc:
        return _fail(str(exc).strip("'"))

    runner = ComparisonRunner(pipeline, steps)
    try:
        wanted = runner.consumes_all()
    except ValueError as exc:
        return _fail(str(exc))

    given = {"reports": list(report_paths), "datasets": list(dataset_paths)}
    if given["reports"] and given["datasets"]:
        return _fail(
            "--report and --dataset were both given. One run loads one set of "
            f"inputs, and this pipeline consumes {wanted}.")

    paths = given[wanted]
    if not paths:
        other = "datasets" if wanted == "reports" else "reports"
        flag = "--dataset" if wanted == "datasets" else "--report"
        extra = (f" Pass {flag}, and --gex with it."
                 if wanted == "datasets" else f" Pass {flag}.")
        if given[other]:
            return _fail(
                f"This pipeline consumes {wanted}, but only "
                f"{other.rstrip('s')} paths were given.{extra}")
        return _fail(f"Nothing to compare.{extra}")

    if wanted == "datasets" and gex is None:
        return _fail(
            "--gex is required with --dataset: a survey file does not describe "
            "its own gates, and the channels a comparison aligns on come from "
            "the system description.")
    if wanted == "reports" and gex is not None:
        click.secho("note: --gex is ignored; this pipeline consumes reports.",
                    fg="yellow", err=True)

    try:
        items, baseline = _load_items(wanted, paths, reference, gex)
    except InputError as exc:
        return _fail(str(exc))
    except Exception as exc:                      # noqa: BLE001 — reported, not swallowed
        log.exception("loading failed")
        return _fail(f"could not read the inputs: {exc}")

    click.echo(f"comparing {len(items)} {wanted[:-1]}(s)"
               + (f" against {Path(reference).name}" if reference else ""))
    try:
        result = runner.run(items, reference=baseline)
    except Exception as exc:                      # noqa: BLE001 — reported, not swallowed
        log.exception("comparison raised")
        return _fail(f"a step raised: {exc}", EXIT_STEP_RAISED)

    _write_report(result, Path(out))
    click.echo(f"wrote {out}")
    return EXIT_GOOD


def _load_items(wanted, paths, reference, gex):
    """The inputs, read as whatever the pipeline said it consumes."""
    if wanted == "datasets":
        items = [load(path, gex) for path in paths]
        return items, (load(reference, gex) if reference else None)

    def read(path):
        text = Path(path).read_text()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # The likeliest mistake by far, and a raw decoder error names the
            # column it choked on rather than the thing that is wrong.
            raise InputError(
                f"{Path(path).name} is not a stored report. If it is a survey "
                f"file, this pipeline's checks consume reports rather than "
                f"datasets — check which steps it names."
            ) from None

    return [read(p) for p in paths], (read(reference) if reference else None)


# ── introspection ────────────────────────────────────────────────────────────

@click.command("list")
@click.option("--group", type=click.Choice(sorted(GROUPS)), default="filters",
              show_default=True, help="Which entry-point group to list.")
def list_steps(group):
    """List the steps registered for a group."""
    registered = load_steps(GROUPS[group])
    if not registered:
        click.echo(f"No steps registered under the {group} group yet.")
        return EXIT_GOOD
    for name in sorted(registered):
        click.echo(name)
    return EXIT_GOOD


@click.command()
@click.argument("pipeline_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--group", type=click.Choice(sorted(GROUPS)), default=None,
              help="Override which registry the steps are looked up in.")
def validate(pipeline_path, group):
    """Check a pipeline document without running it."""
    pipeline = Pipeline.load(pipeline_path)
    candidates = [group] if group else list(RUNNERS) + ["compare"]
    try:
        resolved, steps = resolve_group(pipeline, candidates, group)
        pipeline.validate(steps)
    except (PipelineKindError, KeyError) as exc:
        return _fail(str(exc).strip("'"))
    click.echo(f"{len(pipeline.steps)} step(s), all resolvable [{resolved}].")
    return EXIT_GOOD


# ── entry point ──────────────────────────────────────────────────────────────

@click.group()
@click.option("-v", "--verbose", count=True, help="-v for info, -vv for debug")
def cli(verbose):
    logging.basicConfig(
        level={0: logging.WARNING, 1: logging.INFO}.get(verbose, logging.DEBUG),
        format="%(levelname)s %(name)s: %(message)s",
    )


for _command in (run, compare, list_steps, validate):
    cli.add_command(_command)


def main() -> int:
    try:
        return cli(standalone_mode=False) or EXIT_GOOD
    except click.ClickException as exc:
        exc.show()
        return EXIT_NOT_STARTED
    except click.Abort:
        return EXIT_NOT_STARTED


if __name__ == "__main__":
    raise SystemExit(main())
