"""filter_20 — the vendor's own statement about its unusable early gates.

The first culling filter, and the simplest: no predicate over the data at all,
just a count read from the system description. It exercises the whole mask path
end to end, which is most of why it went first.
"""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.cull.initial_gates import (
    STEP_CODE, declared_initial_gates, disable_initial_gates,
)
from ymerflow_processing.core.mask import (
    CULLED, IN_USE, provenance_key, step_codes,
)
from ymerflow_processing.core.pipeline import FILTER_GROUP, load_steps

GATES = 8


def delivery(remove=(3, 5), n=6, **kw):
    """A dataset whose description declares initial gates to remove."""
    dataset = make_dataset(n=n, n_lines=1, n_gates=GATES, **kw)
    for index, count in enumerate(remove, start=1):
        block = dataset.survey.gex.gex_dict.get(f"Channel{index}")
        if block is not None and count is not None:
            block["RemoveInitialGates"] = float(count)
    return dataset


def mask(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].inuse_key].to_numpy()


# ── What it disables ─────────────────────────────────────────────────────────

def test_each_channel_loses_the_gates_its_own_description_declares():
    """Per channel, not one count for the instrument: LM and HM have different
    turn-off behaviour and declare different windows."""
    dataset, report = disable_initial_gates(delivery(remove=(3, 5)))

    assert (mask(dataset, 0)[:, :3] == CULLED).all()
    assert (mask(dataset, 0)[:, 3:] == IN_USE).all()
    assert (mask(dataset, 1)[:, :5] == CULLED).all()
    assert (mask(dataset, 1)[:, 5:] == IN_USE).all()
    assert report.distributions["removed"] == 6 * (3 + 5)


def test_it_masks_rather_than_trimming():
    """Everything is captured in the mask. Nothing in the pipeline may narrow a
    gate axis, and the time axis must stay aligned with the data."""
    dataset = delivery()
    channel = dataset.channels[0]
    before = dataset.layer_data[channel.data_key].copy()
    axis_before = channel.axis.values.copy()

    dataset, _ = disable_initial_gates(dataset)

    assert dataset.layer_data[channel.data_key].shape == (6, GATES)
    assert dataset.layer_data[channel.inuse_key].shape == (6, GATES)
    pd.testing.assert_frame_equal(dataset.layer_data[channel.data_key], before)
    # The gate-time table is untouched: RemoveInitialGates is a record of a
    # window, not an instruction to re-slice one.
    assert np.array_equal(dataset.channels[0].axis.values, axis_before)


def test_the_boundary_is_exclusive():
    """The sharp edge. Positional slicing is exclusive; the .loc equivalent is
    inclusive on integer labels and would take one good gate every time."""
    dataset, _ = disable_initial_gates(delivery(remove=(5, 5)))

    assert mask(dataset, 0)[0, 4] == CULLED      # the fifth gate, index 4
    assert mask(dataset, 0)[0, 5] == IN_USE      # the sixth survives


def test_zero_declared_removes_nothing():
    dataset, report = disable_initial_gates(delivery(remove=(0, 0)))

    assert (mask(dataset, 0) == IN_USE).all()
    assert report.distributions["removed"] == 0


def test_it_changes_no_measurement_uncertainty_or_attribute():
    dataset = delivery()
    arrays = {k: v.copy() for k, v in dataset.layer_data.items()
              if "inuse" not in k.lower()}
    flightlines = dataset.flightlines.copy()

    dataset, _ = disable_initial_gates(dataset)

    for key, array in arrays.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    pd.testing.assert_frame_equal(dataset.flightlines, flightlines)


# ── Idempotency and composition ──────────────────────────────────────────────

def test_running_it_twice_changes_nothing():
    """R4, and what makes it safe for the importer to apply it too."""
    dataset, first = disable_initial_gates(delivery())
    snapshot = {k: v.copy() for k, v in dataset.layer_data.items()}
    dataset, second = disable_initial_gates(dataset)

    for key, array in snapshot.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    assert second.distributions["removed"] == 0


def test_it_composes_with_a_delivery_that_arrived_already_trimmed():
    """A contractor may have applied it. Subtractive semantics mean running it
    again is a no-op rather than a double-count."""
    dataset = delivery(remove=(3, 3))
    for channel in dataset.channels:
        dataset.layer_data[channel.inuse_key].iloc[:, :3] = CULLED

    dataset, report = disable_initial_gates(dataset)

    assert report.distributions["removed"] == 0
    assert (mask(dataset, 0)[:, :3] == CULLED).all()
    assert (mask(dataset, 0)[:, 3:] == IN_USE).all()


def test_it_never_re_enables_a_gate_something_else_disabled():
    dataset = delivery(remove=(2, 2))
    dataset.layer_data[dataset.channels[0].inuse_key].iloc[:, 7] = CULLED

    dataset, _ = disable_initial_gates(dataset)
    assert mask(dataset, 0)[0, 7] == CULLED


# ── Provenance ───────────────────────────────────────────────────────────────

def test_the_gates_are_attributed_to_this_filter():
    dataset, _ = disable_initial_gates(delivery(remove=(3, 0)))
    why = dataset.layer_data[provenance_key(dataset.channels[0].inuse_key)]

    assert (why.to_numpy()[:, :3] == STEP_CODE).all()
    assert (why.to_numpy()[:, 3:] == 0).all()
    assert STEP_CODE == 20        # its number in the plan, so it self-documents


def test_nothing_here_is_collateral():
    """No policy is applied: the vendor named a window, it was not inferred from
    a neighbouring gate, so every disabled gate is a direct hit."""
    dataset, report = disable_initial_gates(delivery(remove=(3, 0)))
    outcome = report.distributions["per_channel"]

    first = outcome[str(dataset.channels[0])]
    assert first["removed_direct"] == 18
    assert first["removed_collateral"] == 0


def test_the_dataset_records_what_the_code_means():
    """An array of 20s is unreadable without it, and whoever opens the file will
    not have the run report."""
    dataset, _ = disable_initial_gates(delivery())
    assert step_codes(dataset)[STEP_CODE] == "disable_initial_gates"


# ── Reading the declaration ──────────────────────────────────────────────────

def test_a_channel_declaring_nothing_is_left_alone_and_said_so():
    """Silence is different from zero, and is reported differently."""
    dataset = delivery()
    del dataset.survey.gex.gex_dict["Channel1"]["RemoveInitialGates"]

    dataset, report = disable_initial_gates(dataset)

    assert (mask(dataset, 0) == IN_USE).all()
    assert any("declares no initial gates" in n for n in report.notes)


def test_a_declaration_wider_than_the_data_is_an_error():
    """The description and the delivery disagreeing is a fact worth knowing, not
    something to clamp silently."""
    with pytest.raises(ValueError, match="only 8 gates"):
        disable_initial_gates(delivery(remove=(20, 0)))


def test_the_declaration_can_be_overridden_per_channel():
    """For a delivery whose description is known to be wrong."""
    dataset, report = disable_initial_gates(delivery(remove=(3, 3)),
                                            gates={"ch1": 6})

    assert (mask(dataset, 0)[:, :6] == CULLED).all()
    assert (mask(dataset, 1)[:, :3] == CULLED).all()      # its own, untouched

    first = report.distributions["per_channel"][str(dataset.channels[0])]
    assert first["declared"] == 3 and first["applied"] == 6
    assert first["overridden"] is True


def test_the_declared_value_is_reported_even_when_overridden():
    """So the two can never be confused after the fact."""
    _, report = disable_initial_gates(delivery(remove=(3, 3)), gates=1)
    for outcome in report.distributions["per_channel"].values():
        assert outcome["declared"] == 3
        assert outcome["applied"] == 1


def test_declared_initial_gates_reads_the_description():
    dataset = delivery(remove=(4, 7))
    assert declared_initial_gates(dataset, dataset.channels[0]) == 4
    assert declared_initial_gates(dataset, dataset.channels[1]) == 7


def test_no_system_description_means_nothing_is_declared():
    dataset = delivery(with_gex=False)
    dataset, report = disable_initial_gates(dataset)

    assert (mask(dataset, 0) == IN_USE).all()
    assert any("declares no initial gates" in n for n in report.notes)


# ── Plumbing ─────────────────────────────────────────────────────────────────

def test_the_report_survives_serialization():
    _, report = disable_initial_gates(delivery())
    restored = json.loads(json.dumps(report.to_dict(), default=str))
    assert restored["step"] == "disable_initial_gates"
    assert restored["distributions"]["provenance_code"] == STEP_CODE


def test_registered():
    assert "disable_initial_gates" in load_steps(FILTER_GROUP)


# ── A mandate, not a step someone remembers ──────────────────────────────────

def test_a_qc_run_applies_it_without_being_asked():
    """The whole point of promoting it. Left optional it is the easiest thing in
    the library to forget, and forgetting it is silent — the inversion runs,
    converges, and is confidently wrong about the shallow section."""
    from ymerflow_processing.core.pipeline import Pipeline, QCRunner

    dataset = delivery(remove=(3, 5), n=20)
    dataset.flightlines["tx_alt"] = np.full(20, 30.0)
    dataset, _ = QCRunner(Pipeline.parse(
        "kind: qc\nsteps: [{height_above_ground: "
        "{good_max: 45, marginal_max: 65}}]")).run(dataset)

    assert (mask(dataset, 0)[:, :3] == CULLED).all()
    assert (mask(dataset, 1)[:, :5] == CULLED).all()


def test_a_filter_run_applies_it_before_any_step():
    from ymerflow_processing.core.pipeline import Pipeline, ProcessingRunner

    dataset = delivery(remove=(3, 3), n=20)
    dataset, _ = ProcessingRunner(Pipeline.parse(
        "kind: filters\nsteps: [{select_channels: "
        "{criterion: moment, keep: [LM]}}]")).run(dataset)

    assert (mask(dataset, 0)[:, :3] == CULLED).all()


def test_declaring_zero_leaves_a_delivery_untouched():
    """R6's byte-identical promise. Materializing a mask to say "nothing to
    disable" would write arrays into a delivery that had none."""
    from ymerflow_processing.core.pipeline import Pipeline, QCRunner

    dataset = delivery(remove=(0, 0), n=20)
    dataset.flightlines["tx_alt"] = np.full(20, 30.0)
    for channel in dataset.channels:
        del dataset.layer_data[channel.inuse_key]
    before = {k: v.copy() for k, v in dataset.layer_data.items()}

    dataset, _ = QCRunner(Pipeline.parse(
        "kind: qc\nsteps: [{height_above_ground: "
        "{good_max: 45, marginal_max: 65}}]")).run(dataset)

    assert set(dataset.layer_data) == set(before)
    for key, array in before.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


def test_the_mandate_form_is_a_no_op_the_second_time():
    from ymerflow_processing.core.cull.initial_gates import (
        honor_system_description,
    )
    dataset = delivery(remove=(3, 3))

    assert honor_system_description(dataset) == 6 * 3 * 2
    assert honor_system_description(dataset) == 0


def test_a_description_disagreeing_with_the_delivery_does_not_break_a_load():
    """It raises from the explicit step, where someone asked and is watching.
    Raising it from a load would make an unreadable file out of a readable one."""
    from ymerflow_processing.core.cull.initial_gates import (
        honor_system_description,
    )
    dataset = delivery(remove=(20, 0))

    assert honor_system_description(dataset) == 0        # warned, not raised
    with pytest.raises(ValueError, match="only 8 gates"):
        disable_initial_gates(dataset)
