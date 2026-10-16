"""Figures, and the generated step reference.

Both exist to stop documentation drifting from code, and both are only worth
having if that property is enforced rather than intended — so the interesting
tests here are the two that fail when someone adds a filter and forgets.
"""

import pathlib

import numpy as np
import pytest

from conftest import make_dataset
from ymerflow_processing import figures, reference
from ymerflow_processing.core.pipeline import (
    FILTER_GROUP, Pipeline, ProcessingRunner, load_steps,
)

REFERENCE_DIR = pathlib.Path(__file__).parent.parent / "docs" / "reference"

needs_matplotlib = pytest.mark.skipif(
    not figures.available(),
    reason="figures are an optional extra: pip install .[figures]")


# ── The property worth having ────────────────────────────────────────────────

def test_every_registered_filter_declares_a_figure_shape():
    """The test this module exists for. A filter added without a shape has no
    figure and nothing says so — which is how the teaching document would come
    to describe thirty-one of thirty-two filters while looking complete."""
    assert figures.missing_shapes() == []


def test_declaring_nothing_is_a_claim_rather_than_an_omission():
    """Renaming an array changes no measurement, so a before-and-after transient
    would imply something untrue. That has to be sayable."""
    assert figures.SHAPES["rename_array"] == figures.NOTHING
    assert figures.SHAPES["cull_negatives"] == figures.CULLING


def test_an_unknown_step_is_refused_rather_than_drawn_blank():
    with pytest.raises(KeyError, match="No figure shape declared"):
        figures.figure("not_a_step", make_dataset(n=2, n_gates=3),
                       make_dataset(n=2, n_gates=3))


# ── matplotlib is optional, and says so ──────────────────────────────────────

def test_asking_without_matplotlib_names_the_extra(monkeypatch):
    """A bare ImportError three frames down does not tell anyone what to
    install."""
    import builtins
    real = builtins.__import__

    def refuse(name, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("no matplotlib")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", refuse)
    with pytest.raises(figures.FiguresUnavailable, match=r"\[figures\]"):
        figures._figure_class()


def test_matplotlib_is_not_a_required_dependency():
    """A runner that only processes data should not pull a plotting stack."""
    setup = (pathlib.Path(__file__).parent.parent / "setup.py").read_text()
    required = setup.split("install_requires")[1].split("extras_require")[0]
    assert "matplotlib" not in required
    assert '"figures": ["matplotlib"]' in setup


# ── They actually draw ───────────────────────────────────────────────────────

def _ran(document, negatives=False, **kw):
    """A before/after pair and the report, from a real run.

    ``negatives`` puts a real sign reversal in the late gates: the synthetic
    fixture is uniformly positive, so a culling filter run against it correctly
    culls nothing and a test asserting it did would be asserting a bug.
    """
    before = make_dataset(n=20, n_lines=1, n_gates=8, **kw)
    if negatives:
        for channel in before.channels:
            before.layer_data[channel.data_key].iloc[:, -2:] *= -1
    after, report = ProcessingRunner(Pipeline.parse(document)).run(before.copy())
    return before, after, report


@needs_matplotlib
def test_a_culling_figure_marks_what_went():
    before, after, report = _ran("""
        kind: filters
        steps:
          - cull_negatives: {start_gate: 2}
    """, negatives=True)
    step = next(s for s in report.steps if s.step == "cull_negatives")
    fig = figures.figure("cull_negatives", before, after, step)

    assert fig is not None
    # Two panels only when a threshold was recorded; cull_negatives has none.
    assert len(fig.axes) >= 1
    labels = [t.get_text() for ax in fig.axes for t in ax.get_legend().texts]
    assert any("culled" in label for label in labels)


@needs_matplotlib
def test_sign_is_visible_rather_than_folded_into_a_magnitude():
    """The transient is drawn as |dB/dt| on a log axis, so a negative gate reads
    as a bounce. On the one filter whose subject IS negatives that hid the
    thing being filtered — and worse, a gate that went negative and was NOT
    culled was indistinguishable from a good one.

    Colour carries sign, marker carries fate, so all four combinations read."""
    before = make_dataset(n=6, n_lines=1, n_gates=6)
    channel = before.channels[0]
    before.layer_data[channel.data_key].iloc[:, 4] = -1e-9      # a real negative
    after = before.copy()

    fig = figures.figure("cull_negatives", before, after)
    colours = {line.get_color() for line in fig.axes[0].get_lines()
               if line.get_linestyle() == "None"}
    assert figures.NEGATIVE in colours, "a negative gate is not drawn as negative"
    assert figures.POSITIVE in colours


@needs_matplotlib
def test_a_sounding_with_no_negatives_does_not_claim_any():
    """A legend entry for negatives where there are none is noise."""
    before = make_dataset(n=6, n_lines=1, n_gates=6)
    fig = figures.figure("cull_negatives", before, before.copy())
    labels = [t.get_text() for t in fig.axes[0].get_legend().texts]
    assert not any("negative" in label for label in labels)


@needs_matplotlib
def test_a_threshold_gets_its_own_panel():
    """The panel that turns "disabled 4,182 gates" into something judgeable."""
    before, after, report = _ran("""
        kind: filters
        steps:
          - estimate_noise_model: {noise_level_1ms: 5.0e-10, noise_exponent: -0.5,
                                   relative_fraction: 0.03}
          - assign_uncertainty: {}
          - cull_by_uncertainty: {max_uncertainty: 0.5}
    """)
    step = next(s for s in report.steps if s.step == "cull_by_uncertainty")
    fig = figures.figure("cull_by_uncertainty", before, after, step)

    assert len(fig.axes) == 2
    drawn = [line.get_label() for line in fig.axes[1].get_lines()]
    assert any("threshold" in str(label) for label in drawn)


@needs_matplotlib
def test_a_transform_figure_shows_both_states():
    before, after, report = _ran("""
        kind: filters
        steps:
          - average_along_line: {width_first: 3}
    """)
    fig = figures.figure("average_along_line", before, after)
    labels = [t.get_text() for t in fig.axes[0].get_legend().texts]
    assert labels == ["before", "after"]


@needs_matplotlib
def test_axes_are_logarithmic():
    """A transient falls three or four orders of magnitude. On linear axes every
    gate after the first few is indistinguishable from zero."""
    before, after, _ = _ran("""
        kind: filters
        steps:
          - cull_negatives: {}
    """)
    fig = figures.figure("cull_negatives", before, after)
    assert fig.axes[0].get_xscale() == "log"
    assert fig.axes[0].get_yscale() == "log"


@needs_matplotlib
def test_a_step_that_changes_nothing_draws_nothing():
    before = make_dataset(n=6, n_lines=1, n_gates=4)
    assert figures.figure("rename_array", before, before.copy()) is None


@needs_matplotlib
def test_every_drawable_filter_can_actually_be_drawn():
    """Declaring a shape is not the same as the shape working for that filter.
    Drawn against an unchanged pair, which is enough to exercise the code."""
    before = make_dataset(n=12, n_lines=1, n_gates=6)
    after = before.copy()
    for step, shape in sorted(figures.SHAPES.items()):
        if shape == figures.NOTHING:
            continue
        assert figures.figure(step, before, after) is not None, step


# ── The reference is generated, and stays current ────────────────────────────

def test_the_committed_reference_matches_the_registry():
    """The check that keeps this from drifting. Documentation that *can* drift
    from the code will, and the only reliable fix is a failing test rather than
    a habit. Regenerate with:

        python -c "import pathlib; from ymerflow_processing import reference; \\
                   [pathlib.Path('docs/reference', n).write_text(t) \\
                    for n, t in reference.generate(None).items()]"
    """
    stale = []
    for name, expected in reference.generate(REFERENCE_DIR).items():
        path = REFERENCE_DIR / name
        if not path.is_file() or path.read_text() != expected:
            stale.append(name)
    assert not stale, (
        f"docs/reference is out of date with the registry: {', '.join(stale)}. "
        f"Regenerate it — see this test's docstring.")


def test_every_step_appears_in_the_reference():
    text = (REFERENCE_DIR / "filters.md").read_text()
    for step in load_steps(FILTER_GROUP):
        assert f"`{step}`" in text, step


def test_a_string_default_keeps_its_quotes():
    """Stripping quotes to unrender a stringified annotation also strips them
    from string defaults, so `output_column: str = 'qs_altim'` came out reading
    as a name rather than a string."""
    text = (REFERENCE_DIR / "qc.md").read_text()
    assert "output_column: str = 'qs_altim'" in text
    assert "output_column: str = qs_altim" not in text


def test_a_parameter_description_stops_at_its_own_entry():
    """Every step's last parameter was carrying 'Returns (dataset, StepReport)'
    glued to the end of its description."""
    for name in ("filters.md", "qc.md", "compare.md"):
        assert "Returns ``(dataset" not in (REFERENCE_DIR / name).read_text()


def test_the_agent_index_carries_what_a_signature_cannot():
    """Ordering and invariants are what a plausible-looking wrong pipeline gets
    wrong, and neither is inferable from a signature."""
    text = (REFERENCE_DIR / "llms.txt").read_text()
    assert "## Invariants" in text and "## Ordering" in text
    assert "Culling is subtractive" in text
    assert "before `assign_uncertainty`" in text
    # consumes is declared per check and matters for how a caller invokes it.
    assert "consumes: datasets" in text


# ── Both docstring styles, so the conversion can go one module at a time ─────

def _described(func):
    return {name: text for name, _, text in reference._parameters(func)}


def test_a_numpy_parameters_section_is_read():
    def step(dataset, max_uncertainty: float = 0.2, start_gate: int = 0):
        """Disable gates whose uncertainty is too large.

        Parameters
        ----------
        max_uncertainty : float
            Largest relative uncertainty to keep.
        start_gate : int
            Where evaluation begins, 0-based.
        """

    assert _described(step) == {
        "max_uncertainty": "Largest relative uncertainty to keep.",
        "start_gate": "Where evaluation begins, 0-based.",
    }


def test_a_numpy_description_may_run_over_several_lines():
    def step(dataset, window: int = 7):
        """Smooth along the line.

        Parameters
        ----------
        window : int
            Soundings smoothed over. Larger values
            smear the very signature being looked for.
        """

    assert _described(step)["window"] == (
        "Soundings smoothed over. Larger values smear the very signature "
        "being looked for.")


def test_a_numpy_parameters_section_stops_at_the_next_heading():
    """The trap the Sphinx parser already had: a description running on into
    whatever follows it, so every last parameter carried the return type."""
    def step(dataset, criterion: str = "profile"):
        """Cull coupled data.

        Parameters
        ----------
        criterion : str
            Which test to apply.

        Returns
        -------
        tuple
            ``(dataset, StepReport)``.
        """

    assert _described(step)["criterion"] == "Which test to apply."


def test_the_sphinx_style_is_still_read():
    """Accepted until the last module is converted; removing it earlier would
    blank every description in the generated reference."""
    def step(dataset, max_misfit: float = 2.0):
        """Cull poorly fit gates.

        :param max_misfit: Largest acceptable gatewise misfit.

        Returns ``(dataset, StepReport)``.
        """

    assert _described(step)["max_misfit"] == (
        "Largest acceptable gatewise misfit.")


def test_a_heading_is_not_swallowed_into_the_summary():
    def step(dataset, window: int = 7):
        """Smooth along the line.
        Parameters
        ----------
        window : int
            Soundings smoothed over.
        """

    assert reference._summary(step) == "Smooth along the line."
