"""Two implementations over one input — 02's acceptance criterion for the rebuild.

The design question these exercise is not "do the numbers match". It is whether a
**deliberate improvement** has somewhere to go. A report whose vocabulary is only
matches and does-not-match forces someone to widen a tolerance until the
improvement disappears into it, and then the comparison means nothing.
"""

import json

import numpy as np
import pytest

from conftest import make_dataset
from ymerflow_processing.core.compare.equivalence import (
    ABSENT, AGREES, CONSUMES, DATA, DIFFERS, EXPECTED, MASK, STEP, UNCERTAINTY,
    implementation_equivalence,
)
from ymerflow_processing.core.mask import CULLED
from ymerflow_processing.core.pipeline import (
    COMPARE_GROUP, ComparisonRunner, Pipeline, load_steps,
)

N = 10
GATES = 6


def pair(**kw):
    """A candidate and a baseline that start identical."""
    return make_dataset(n=N, n_lines=1, n_gates=GATES, **kw), \
        make_dataset(n=N, n_lines=1, n_gates=GATES, **kw)


def check(candidate, baseline, **kw):
    kw.setdefault("justification", "Float round-trip through a text format.")
    return implementation_equivalence([candidate], reference=baseline, **kw)


def outcome(report, quantity, channel="LM"):
    quantities = report.distributions["quantities"]
    key = next(k for k in quantities if channel in k or channel == "")
    return quantities[key][quantity]["outcome"]


# ── The bar is not bit-for-bit ───────────────────────────────────────────────

def test_identical_implementations_are_equivalent():
    candidate, baseline = pair()
    report = check(candidate, baseline)

    assert report.distributions["verdict"] == "equivalent"
    assert outcome(report, DATA) == AGREES


def test_an_undeclared_difference_is_the_finding():
    candidate, baseline = pair()
    candidate.layer_data[candidate.channels[0].data_key].iloc[:, 2] *= 1.5

    report = check(candidate, baseline)
    assert report.distributions["verdict"] == "differs"
    assert outcome(report, DATA) == DIFFERS


def test_a_declared_difference_is_a_pass():
    """Bit-for-bit agreement with a defective baseline means reproducing its
    defects, so 'these differ, deliberately, here is why' has to be an outcome
    the report can express — not only a failure."""
    candidate, baseline = pair()
    candidate.layer_data[candidate.channels[0].std_key].iloc[:, :] *= 0.5

    report = check(candidate, baseline, expected_differences=[
        {"quantity": UNCERTAINTY,
         "reason": "The baseline clipped the denominator and produced ~100% "
                   "uncertainty on normalized data."}])

    assert report.distributions["verdict"] == "equivalent_with_expected_differences"
    assert outcome(report, UNCERTAINTY) == EXPECTED
    assert outcome(report, DATA) == AGREES


def test_the_declaration_carries_its_reason_into_the_report():
    """The report is pasted into a migration record: a reader learns why."""
    candidate, baseline = pair()
    candidate.layer_data[candidate.channels[0].std_key].iloc[:, :] *= 0.5
    report = check(candidate, baseline, expected_differences=[
        {"quantity": UNCERTAINTY, "reason": "Denominator no longer clipped."}])

    assert "clipped" in json.dumps(report.distributions)


# ── A declared difference can still fail ─────────────────────────────────────

def test_a_declared_direction_that_is_violated_is_still_a_finding():
    """What makes 'better' testable rather than assertable. Declaring the new
    uncertainty should be smaller means a candidate that comes out larger is a
    regression, even though the difference was declared."""
    candidate, baseline = pair()
    candidate.layer_data[candidate.channels[0].std_key].iloc[:, :] *= 2.0

    report = check(candidate, baseline, expected_differences=[
        {"quantity": UNCERTAINTY, "direction": "smaller",
         "reason": "The baseline clipped the denominator."}])

    assert report.distributions["verdict"] == "differs"
    assert outcome(report, UNCERTAINTY) == DIFFERS
    assert any("wrong way" in note for note in report.notes)


def test_a_declared_direction_that_holds_passes():
    candidate, baseline = pair()
    candidate.layer_data[candidate.channels[0].std_key].iloc[:, :] *= 0.5

    report = check(candidate, baseline, expected_differences=[
        {"quantity": UNCERTAINTY, "direction": "smaller",
         "reason": "The baseline clipped the denominator."}])
    assert outcome(report, UNCERTAINTY) == EXPECTED


def test_a_declaration_nothing_exercised_is_reported():
    """Stale, or the comparison never reached what it describes. Either way it
    should not read as agreement."""
    candidate, baseline = pair()
    report = check(candidate, baseline, expected_differences=[
        {"quantity": UNCERTAINTY, "reason": "Denominator no longer clipped."}])

    assert report.distributions["verdict"] == "equivalent"
    assert any("never exercised" in note for note in report.notes)


def test_a_declaration_can_name_one_channel():
    candidate, baseline = pair()
    for channel in candidate.channels:
        candidate.layer_data[channel.std_key].iloc[:, :] *= 0.5

    report = check(candidate, baseline, expected_differences=[
        {"quantity": UNCERTAINTY, "channel": candidate.channels[0].key,
         "reason": "Only the first moment's error path changed."}])

    assert report.distributions["verdict"] == "differs"   # the other channel


def test_a_declaration_without_a_reason_is_refused():
    candidate, baseline = pair()
    with pytest.raises(ValueError, match="must carry a reason"):
        check(candidate, baseline,
              expected_differences=[{"quantity": UNCERTAINTY}])


def test_a_declaration_naming_a_column_rather_than_a_quantity_is_refused():
    """Quantities are semantic because the two sides may not share a column
    name — which is the whole reason alignment does not use one."""
    candidate, baseline = pair()
    with pytest.raises(ValueError, match="must name a quantity"):
        check(candidate, baseline, expected_differences=[
            {"quantity": "dbdt_std_ch1gt", "reason": "..."}])


# ── Alignment cannot assume a shared schema ──────────────────────────────────

def test_channels_match_on_gate_times_not_on_names():
    """A channel IS its gate times. The second moment is the second moment
    whether the file calls it ch2, HM, or nothing at all."""
    candidate, baseline = pair()
    for channel in baseline.channels:
        baseline.survey.xyz.model_info.setdefault("dummy", None)
    report = check(candidate, baseline)

    quantities = report.distributions["quantities"]
    assert report.distributions["alignment"]["channels_compared"] == 2
    assert all("baseline_channel" in entry for entry in quantities.values())


def _reverse(dataset):
    """The same survey, written out in the opposite order."""
    dataset.survey.xyz.flightlines = dataset.flightlines.iloc[::-1].copy()
    for key in list(dataset.layer_data):
        dataset.layer_data[key] = dataset.layer_data[key].iloc[::-1].copy()
    return dataset


def test_soundings_match_by_timestamp_when_reordered():
    """An implementation that sorted its output differently is still the same
    survey. Pairing by row would compare every sounding against a stranger."""
    candidate, baseline = pair()
    for dataset in (candidate, baseline):
        dataset.survey.xyz.flightlines["timestamp"] = np.arange(N) * 1e-5
    _reverse(baseline)

    report = check(candidate, baseline)
    assert report.distributions["alignment"]["soundings_matched_by"] == "timestamp"
    assert report.distributions["verdict"] == "equivalent"


def test_position_is_the_fallback_identity_when_there_is_no_timestamp():
    """Good to whatever tolerance the two sides round coordinates to."""
    candidate, baseline = pair()
    _reverse(baseline)

    report = check(candidate, baseline)
    assert report.distributions["alignment"]["soundings_matched_by"] == "position"
    assert report.distributions["verdict"] == "equivalent"


def test_a_reordered_side_would_be_compared_against_strangers_without_this():
    """The reason alignment is not positional: reversing one side and grading it
    row-against-row finds differences that are not there."""
    candidate, baseline = pair()
    values = candidate.layer_data[candidate.channels[0].data_key]
    values.iloc[:, :] = np.arange(N)[:, None] * np.ones(GATES)[None, :]
    baseline.layer_data[baseline.channels[0].data_key].iloc[:, :] = \
        values.to_numpy()
    _reverse(baseline)

    assert check(candidate, baseline).distributions["verdict"] == "equivalent"


def test_pairing_by_order_is_said_out_loud():
    """It is an assumption, not a match: nothing in the numbers would say so."""
    candidate, baseline = pair()
    for dataset in (candidate, baseline):
        frame = dataset.flightlines
        for column in (dataset.survey.xyz.get_column("timestamp"),
                       dataset.survey.xyz.x_column, dataset.survey.xyz.y_column):
            if column in frame.columns:
                frame.drop(columns=[column], inplace=False)
        dataset.survey.xyz.flightlines = frame.drop(columns=[
            c for c in (dataset.survey.xyz.get_column("timestamp"),
                        dataset.survey.xyz.x_column, dataset.survey.xyz.y_column)
            if c in frame.columns])

    report = check(candidate, baseline)
    assert report.distributions["alignment"]["soundings_matched_by"] == "order"
    assert any("BY ORDER" in note for note in report.notes)


def test_unmatchable_sides_of_different_length_are_refused():
    candidate, _ = pair()
    baseline = make_dataset(n=N + 3, n_lines=1, n_gates=GATES)
    for dataset in (candidate, baseline):
        xyz = dataset.survey.xyz
        xyz.flightlines = dataset.flightlines.drop(columns=[
            c for c in (xyz.get_column("timestamp"), xyz.x_column, xyz.y_column)
            if c in dataset.flightlines.columns])

    with pytest.raises(ValueError, match="no alignment to fall back to"):
        check(candidate, baseline)


def test_a_side_that_dropped_soundings_is_reported_not_misaligned():
    candidate, baseline = pair()
    keep = baseline.flightlines.index[:-2]
    baseline.survey.xyz.flightlines = baseline.flightlines.loc[keep].copy()
    for key in list(baseline.layer_data):
        baseline.layer_data[key] = baseline.layer_data[key].loc[keep].copy()

    report = check(candidate, baseline)
    assert report.distributions["alignment"]["soundings_compared"] == N - 2
    assert any("only in the candidate" in note for note in report.notes)


# ── Values only where both sides kept the datum ──────────────────────────────

def test_what_is_under_a_mask_is_not_compared():
    """An implementation may leave anything at all under its mask, so comparing
    there measures nothing."""
    candidate, baseline = pair()
    channel = candidate.channels[0]
    candidate.layer_data[channel.data_key].iloc[:, 4] = 1e9
    for dataset in (candidate, baseline):
        dataset.layer_data[dataset.channels[0].inuse_key].iloc[:, 4] = CULLED

    assert outcome(check(candidate, baseline), DATA) == AGREES


def test_the_mask_itself_is_compared():
    """Which is where 'this one culled more' belongs."""
    candidate, baseline = pair()
    candidate.layer_data[candidate.channels[0].inuse_key].iloc[:, 3] = CULLED

    report = check(candidate, baseline)
    assert outcome(report, MASK) == DIFFERS
    assert outcome(report, DATA) == AGREES


def test_mask_provenance_is_not_a_difference():
    """The magnitude records why a datum went; what is compared is whether it
    is in use."""
    candidate, baseline = pair()
    for dataset in (candidate, baseline):
        dataset.layer_data[dataset.channels[0].inuse_key].iloc[:, 3] = CULLED
    candidate.layer_data[candidate.channels[0].inuse_key].iloc[:, 1] = 7
    baseline.layer_data[baseline.channels[0].inuse_key].iloc[:, 1] = 1

    assert outcome(check(candidate, baseline), MASK) == AGREES


# ── Tolerance ────────────────────────────────────────────────────────────────

def test_a_difference_inside_the_tolerance_agrees():
    candidate, baseline = pair()
    candidate.layer_data[candidate.channels[0].data_key] *= (1 + 1e-9)
    assert outcome(check(candidate, baseline, relative=1e-6), DATA) == AGREES


def test_an_absolute_tolerance_is_available_because_a_fraction_of_zero_is_zero():
    candidate, baseline = pair()
    baseline.layer_data[baseline.channels[0].data_key].iloc[:, :] = 0.0
    candidate.layer_data[candidate.channels[0].data_key].iloc[:, :] = 1e-15

    assert outcome(check(candidate, baseline), DATA) == DIFFERS
    assert outcome(check(candidate, baseline, absolute=1e-12), DATA) == AGREES


def test_a_tolerance_with_no_justification_is_flagged():
    """A number somebody tuned until the comparison passed."""
    candidate, baseline = pair()
    report = implementation_equivalence([candidate], reference=baseline)
    assert any("without a reason" in note for note in report.notes)


def test_the_worst_disagreement_is_located():
    candidate, baseline = pair()
    candidate.layer_data[candidate.channels[0].data_key].iloc[3, 2] *= 100

    report = check(candidate, baseline)
    worst = report.distributions["quantities"]
    entry = next(v for v in worst.values() if v[DATA]["outcome"] == DIFFERS)
    assert entry[DATA]["worst_at"]["sounding"] == 3
    assert entry[DATA]["worst_at"]["gate"] == 2


# ── Awkward inputs ───────────────────────────────────────────────────────────

def test_a_single_sounding():
    candidate = make_dataset(n=1, n_lines=1, n_gates=GATES)
    baseline = make_dataset(n=1, n_lines=1, n_gates=GATES)
    assert check(candidate, baseline).distributions["verdict"] == "equivalent"


def test_an_all_nan_gate_agrees_with_an_all_nan_gate():
    """Two absences are agreement; an absence against a number is not."""
    candidate, baseline = pair()
    for dataset in (candidate, baseline):
        dataset.layer_data[dataset.channels[0].data_key].iloc[:, 2] = np.nan

    assert outcome(check(candidate, baseline), DATA) == AGREES


def test_a_nan_against_a_number_is_a_difference():
    candidate, baseline = pair()
    candidate.layer_data[candidate.channels[0].data_key].iloc[:, 2] = np.nan
    assert outcome(check(candidate, baseline), DATA) == DIFFERS


def test_an_empty_side_is_recorded_rather_than_passed():
    """An empty result agrees with everything, which is exactly why it must not
    read as agreement."""
    candidate, baseline = pair()
    candidate.survey.xyz.flightlines = candidate.flightlines.iloc[:0]

    report = check(candidate, baseline)
    assert report.distributions["verdict"] == ABSENT
    assert any("agrees with everything" in note for note in report.notes)


def test_a_delivery_already_culled_on_both_sides_compares_what_is_left():
    candidate, baseline = pair()
    for dataset in (candidate, baseline):
        for channel in dataset.channels:
            dataset.layer_data[channel.inuse_key].iloc[:, :3] = CULLED
    candidate.layer_data[candidate.channels[0].data_key].iloc[:, :3] = -999

    report = check(candidate, baseline)
    assert report.distributions["verdict"] == "equivalent"


def test_a_quantity_only_one_side_carries_is_absent_not_different():
    candidate, baseline = pair()
    del candidate.layer_data[candidate.channels[0].std_key]

    report = check(candidate, baseline)
    assert outcome(report, UNCERTAINTY) == ABSENT
    assert report.distributions["verdict"] != "differs"


def test_a_system_description_naming_gates_the_arrays_do_not_hold():
    """The method layer warns when a description disagrees with the data it
    describes, so the axis is a claim about the columns rather than a count of
    them, and a gate it names past the end cannot be compared."""
    candidate, baseline = pair()
    keep = list(range(GATES - 2))
    for channel in candidate.channels:
        for key in channel.array_keys:
            candidate.layer_data[key] = candidate.layer_data[key].iloc[:, keep]

    report = check(candidate, baseline)
    assert any("the arrays do not hold" in note for note in report.notes)
    assert report.distributions["verdict"] == "equivalent"


# ── Contract ─────────────────────────────────────────────────────────────────

def test_it_declares_that_it_needs_datasets():
    """02: no report carries per-datum arrays, and per-datum is the question."""
    assert CONSUMES == "datasets"
    assert ComparisonRunner(Pipeline.parse(f"""
        kind: compare
        steps:
          - {STEP}: {{}}
    """)).consumes(STEP) == "datasets"


def test_reports_are_the_default_for_a_check_that_does_not_declare():
    assert ComparisonRunner(Pipeline.parse("""
        kind: compare
        steps:
          - noise_floor_drift: {}
    """)).consumes("noise_floor_drift") == "reports"


def test_a_report_given_where_a_dataset_was_needed_says_so():
    candidate, _ = pair()
    with pytest.raises(TypeError, match="consumes: datasets"):
        implementation_equivalence([candidate], reference={"steps": []})


def test_it_needs_a_baseline():
    candidate, _ = pair()
    with pytest.raises(ValueError, match="nothing to be equivalent to"):
        implementation_equivalence([candidate])


def test_it_compares_one_pair_at_a_time():
    """The migration register is per entry point, so a run is per entry point."""
    candidate, baseline = pair()
    with pytest.raises(ValueError, match="one candidate against one baseline"):
        implementation_equivalence([candidate, candidate], reference=baseline)


def test_it_runs_from_a_pipeline_document():
    candidate, baseline = pair()
    report = ComparisonRunner(Pipeline.parse(f"""
        kind: compare
        steps:
          - {STEP}:
              relative: 1.0e-6
              justification: Float round-trip through a text format.
    """)).run([candidate], reference=baseline)

    assert report.steps[0].distributions["verdict"] == "equivalent"


def test_the_report_survives_serialization():
    """It is pasted into a migration record."""
    candidate, baseline = pair()
    candidate.layer_data[candidate.channels[0].std_key] *= 0.5
    report = check(candidate, baseline, expected_differences=[
        {"quantity": UNCERTAINTY, "direction": "smaller",
         "reason": "Denominator no longer clipped."}])

    restored = json.loads(json.dumps(report.to_dict(), default=str))
    assert restored["distributions"]["verdict"] == \
        "equivalent_with_expected_differences"
    assert restored["distributions"]["tolerance"]["justification"]


def test_registered():
    assert STEP in load_steps(COMPARE_GROUP)
