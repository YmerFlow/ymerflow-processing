"""filter_33 — replaying a human's judgement through an automated pipeline.

The step that makes the chain reproducible in the presence of hand editing.
Most of what follows guards the addressing: a diff that carries its own sounding
indices is safe to replay, and one applied to the wrong state must say so rather
than edit the wrong soundings.
"""

import json

import numpy as np
import pandas as pd
import pytest
from libaarhusxyz.xyz import XYZ

from conftest import make_dataset
from ymerflow_processing.filters.cull_nominate_gates import nominate_gates
from ymerflow_processing.filters.cull_manual_edits import (
    APPLY_INDEX, STEP_CODE, apply_manual_edits,
)
from ymerflow_processing.mask import CULLED, IN_USE, provenance_key
from ymerflow_processing.pipeline import (
    FILTER_GROUP, Pipeline, ProcessingRunner, load_steps,
)

N = 6
GATES = 8


def delivery(**kw):
    return make_dataset(n=N, n_lines=1, n_gates=GATES, **kw)


def mask(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].inuse_key].to_numpy()


def edit_set(soundings, overrides, channel=1, marker=True):
    """A sparse diff: NaN everywhere except the entries stated.

    ``overrides`` is ``{(row_in_diff, gate): value}`` where value is 0 or 1.
    """
    rows = len(soundings)
    array = np.full((rows, GATES), np.nan)
    for (row, gate), value in overrides.items():
        array[row, gate] = value
    return XYZ({
        "flightlines": pd.DataFrame({APPLY_INDEX: list(soundings)}),
        "layer_data": {f"InUse_Ch{channel:02d}": pd.DataFrame(array)},
        "model_info": {"diff_dummy": True} if marker else {},
        "file_meta": {},
    })


# ── Applying an edit ─────────────────────────────────────────────────────────

def test_an_edit_set_addresses_a_named_moment_by_its_own_mask_name():
    """`LM_Z_G01_inuse` carries no channel number for the numbered patterns to
    capture. The editor writes its diff under the mask's own name, so that name
    is what the replay has to recognise."""
    from test_named_moments import LM_GATES, delivery as named_delivery
    dataset = named_delivery(keys=("LM_Z_G01", "HM_Z_G01"))
    array = np.full((1, LM_GATES), np.nan)
    array[0, 5] = 0
    diff = XYZ({
        "flightlines": pd.DataFrame({APPLY_INDEX: [2]}),
        "layer_data": {"LM_Z_G01_inuse": pd.DataFrame(array)},
        "model_info": {"diff_dummy": True}, "file_meta": {},
    })

    dataset, report = apply_manual_edits(dataset, diff)

    low = next(c for c in dataset.channels if c.data_key == "LM_Z_G01")
    assert dataset.layer_data[low.inuse_key].to_numpy()[2, 5] == CULLED
    assert report.distributions["removed"] == 1


def test_a_hand_disabled_gate_is_disabled():
    diff = edit_set([2], {(0, 4): 0})
    dataset, report = apply_manual_edits(delivery(), diff)

    assert mask(dataset)[2, 4] == CULLED
    assert mask(dataset)[2, 3] == IN_USE
    assert report.distributions["removed"] == 1


def test_a_hand_enabled_gate_is_restored():
    """A person overruling an automated cull is the clearest case there is for
    turning data back on."""
    dataset, _ = nominate_gates(delivery(), disable_gates={"ch1": "0-3"})
    diff = edit_set([1], {(0, 0): 1, (0, 1): 1})

    dataset, report = apply_manual_edits(dataset, diff)

    assert mask(dataset)[1, 0] == IN_USE
    assert mask(dataset)[1, 1] == IN_USE
    assert mask(dataset)[1, 2] == CULLED       # not named by the edit
    assert mask(dataset)[0, 0] == CULLED       # a different sounding
    assert report.distributions["restored"] == 2


def test_both_directions_in_one_edit():
    dataset, _ = nominate_gates(delivery(), disable_gates={"ch1": 0})
    diff = edit_set([3], {(0, 0): 1, (0, 5): 0})

    dataset, report = apply_manual_edits(dataset, diff)

    assert mask(dataset)[3, 0] == IN_USE
    assert mask(dataset)[3, 5] == CULLED
    assert report.distributions["restored"] == 1
    assert report.distributions["removed"] == 1


def test_nan_means_no_override():
    """Only non-NaN entries say anything; everything else is left as found."""
    dataset, _ = nominate_gates(delivery(), disable_gates={"ch1": "0-2"})
    diff = edit_set([0], {(0, 0): 1})           # the rest of the row is NaN

    dataset, _ = apply_manual_edits(dataset, diff)

    assert mask(dataset)[0, 0] == IN_USE
    assert mask(dataset)[0, 1] == CULLED
    assert mask(dataset)[0, 2] == CULLED


def test_no_nan_ever_reaches_the_mask():
    """The sentinel is the diff's, not the mask's. Downstream readers disagree
    about NaN, so it is consumed here and goes no further."""
    diff = edit_set([1], {(0, 2): 0})
    dataset, _ = apply_manual_edits(delivery(), diff)

    for channel in dataset.channels:
        values = dataset.layer_data[channel.inuse_key].to_numpy()
        assert np.isfinite(values).all()
        assert set(np.unique(values)) <= {IN_USE, CULLED}


def test_it_changes_no_measurement_or_uncertainty():
    """01: it carries only the mask. A hand-edited value would be
    unattributable; a hand-disabled datum is a judgement with a reason."""
    dataset = delivery()
    arrays = {k: v.copy() for k, v in dataset.layer_data.items()
              if "inuse" not in k.lower()}

    dataset, _ = apply_manual_edits(dataset, edit_set([1], {(0, 2): 0}))

    for key, array in arrays.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


def test_only_the_named_channel_is_touched():
    diff = edit_set([1], {(0, 2): 0}, channel=1)
    dataset, _ = apply_manual_edits(delivery(), diff)

    assert (mask(dataset, 1) == IN_USE).all()


# ── The addressing ───────────────────────────────────────────────────────────

def test_the_diff_carries_its_own_sounding_indices():
    """Which is what makes replay safe — it does not depend on row order
    surviving whatever ran before it."""
    diff = edit_set([0, 4], {(0, 1): 0, (1, 6): 0})
    dataset, report = apply_manual_edits(delivery(), diff)

    assert mask(dataset)[0, 1] == CULLED
    assert mask(dataset)[4, 6] == CULLED
    assert mask(dataset)[1, 1] == IN_USE
    assert report.distributions["soundings_edited"] == 2


def test_an_edit_authored_against_a_different_state_is_refused():
    """Applying it would edit the wrong soundings, silently."""
    with pytest.raises(IndexError, match="different state"):
        apply_manual_edits(delivery(), edit_set([99], {(0, 0): 0}))


def test_a_diff_with_no_addressing_is_refused():
    diff = XYZ({"flightlines": pd.DataFrame({"line_id": [1]}),
                "layer_data": {"InUse_Ch01": pd.DataFrame(np.full((1, GATES), 0.0))},
                "model_info": {"diff_dummy": True}, "file_meta": {}})

    with pytest.raises(KeyError, match="cannot be replayed safely"):
        apply_manual_edits(delivery(), diff)


def test_a_row_count_disagreement_is_refused():
    diff = edit_set([0, 1], {(0, 0): 0})
    diff.layer_data["InUse_Ch01"] = pd.DataFrame(np.full((3, GATES), np.nan))

    with pytest.raises(ValueError, match="must agree"):
        apply_manual_edits(delivery(), diff)


def test_a_gate_count_disagreement_is_noted_not_applied():
    diff = edit_set([0], {(0, 0): 0})
    diff.layer_data["InUse_Ch01"] = pd.DataFrame(np.full((1, 4), 0.0))

    dataset, report = apply_manual_edits(delivery(), diff)
    assert (mask(dataset) == IN_USE).all()
    assert any("do not match this dataset's gate count" in n
               for n in report.notes)


def test_a_file_with_no_marker_is_applied_with_a_note():
    diff = edit_set([1], {(0, 2): 0}, marker=False)
    dataset, report = apply_manual_edits(delivery(), diff)

    assert mask(dataset)[1, 2] == CULLED
    assert any("no 'diff_dummy' marker" in n for n in report.notes)


# ── Replay ───────────────────────────────────────────────────────────────────

def test_re_running_the_pipeline_does_not_undo_the_edit():
    """R6's whole point: the standard failure mode when automated and manual
    edits mix is the automated pass silently reverting the human's."""
    diff = edit_set([2], {(0, 6): 0})
    document = """
        kind: filters
        steps:
          - nominate_gates: {disable_gates: {ch1: "0-1"}}
    """
    dataset, _ = apply_manual_edits(delivery(), diff)
    dataset, _ = ProcessingRunner(Pipeline.parse(document)).run(dataset)

    assert mask(dataset)[2, 6] == CULLED       # the edit survived
    assert mask(dataset)[2, 0] == CULLED       # and the automated pass ran


def test_applying_the_same_edit_twice_changes_nothing():
    diff = edit_set([2], {(0, 4): 0})
    dataset, _ = apply_manual_edits(delivery(), diff)
    snapshot = {k: v.copy() for k, v in dataset.layer_data.items()}
    dataset, second = apply_manual_edits(dataset, diff)

    for key, array in snapshot.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    assert second.distributions["removed"] == 0


# ── The checkpoint, and provenance ───────────────────────────────────────────

def test_the_checkpoint_name_is_recorded():
    """A diff is meaningless without knowing which state it is a difference
    from, which is why 01 keeps the two as one step."""
    _, report = apply_manual_edits(delivery(), edit_set([1], {(0, 2): 0}),
                                   checkpoint="after-automated-culling")
    assert report.distributions["checkpoint"] == "after-automated-culling"


def test_an_unnamed_checkpoint_is_noted():
    _, report = apply_manual_edits(delivery(), edit_set([1], {(0, 2): 0}))
    assert any("No checkpoint name" in n for n in report.notes)


def test_the_edit_is_attributed_to_this_filter():
    dataset, _ = apply_manual_edits(delivery(), edit_set([1], {(0, 2): 0}))
    codes = dataset.layer_data[
        provenance_key(dataset.channels[0].inuse_key)].to_numpy()

    assert codes[1, 2] == STEP_CODE
    assert codes[0, 2] == 0
    assert STEP_CODE == 33


def test_a_restored_gate_reads_as_restored_not_culled():
    dataset, _ = nominate_gates(delivery(), disable_gates={"ch1": 0})
    dataset, _ = apply_manual_edits(dataset, edit_set([1], {(0, 0): 1}))

    codes = dataset.layer_data[
        provenance_key(dataset.channels[0].inuse_key)].to_numpy()
    assert codes[1, 0] == STEP_CODE and mask(dataset)[1, 0] == IN_USE
    assert codes[0, 0] == 21 and mask(dataset)[0, 0] == CULLED


def test_the_report_survives_serialization():
    _, report = apply_manual_edits(delivery(), edit_set([1], {(0, 2): 0}),
                                   checkpoint="mid-chain")
    restored = json.loads(json.dumps(report.to_dict(), default=str))
    assert restored["step"] == "apply_manual_edits"
    assert restored["distributions"]["checkpoint"] == "mid-chain"


def test_registered():
    assert "apply_manual_edits" in load_steps(FILTER_GROUP)


def test_it_reads_the_msgpack_the_platform_actually_writes(tmp_path):
    """The delivered form. The platform's plot editor saves an
    inuse_diff_{dataset}.msgpack and creates a process version with it as a
    parameter, so a path is what arrives — not an object we constructed."""
    path = tmp_path / "inuse_diff_delivery.msgpack"
    edit_set([2], {(0, 4): 0}).dump(str(path))

    dataset, report = apply_manual_edits(delivery(), path)

    assert mask(dataset)[2, 4] == CULLED
    assert report.distributions["removed"] == 1


def test_an_already_loaded_diff_is_accepted_too():
    """R8 asks for external inputs typed rather than as paths, and a loaded XYZ
    is the typed form. A path is accepted because a standalone run has nothing
    to materialise it — the same layering load() follows."""
    dataset, _ = apply_manual_edits(delivery(), edit_set([2], {(0, 4): 0}))
    assert mask(dataset)[2, 4] == CULLED
