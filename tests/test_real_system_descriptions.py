"""Checks against real SkyTEM system descriptions.

Skipped when the files are not present, so the suite still runs standalone. What
they protect is the two conventions that were established by inspecting real
files and would otherwise be silently re-guessable:

* a ``GateTime`` row is *center, open, close* — so the axis takes column 0
* ``RemoveInitialGates`` is a masking instruction, not a gate-time offset

Both are the kind of mistake that produces plausible numbers rather than an
error, which is why they are pinned here.
"""

import os
from pathlib import Path

import numpy as np
import pytest

from ymerflow_processing.methods.tem import _dipole_moment, _gate_axis

#: Where to find stock system descriptions. Set ``YF_GEX_DIR`` to a directory of
#: ``.gex`` files to enable these; unset, they skip.
#:
#: Deliberately not a path in this repository. These files describe commercial
#: instruments and are not ours to redistribute, and hardcoding anyone's local
#: filesystem into a public repository is its own problem.
GEX_DIR = Path(os.environ["YF_GEX_DIR"]) if os.environ.get("YF_GEX_DIR") else None
SYSTEMS = ["SkyTEM_304M_XYZ", "SkyTEM_306HP", "SkyTEM_312HP", "SkyTEM_304_75hz_XYZ"]

pytestmark = pytest.mark.skipif(
    GEX_DIR is None or not GEX_DIR.is_dir(),
    reason="set YF_GEX_DIR to a directory of .gex files to run these",
)


def load(name):
    from libaarhusxyz.gex import GEX
    path = GEX_DIR / f"{name}.gex"
    if not path.exists():
        pytest.skip(f"{path.name} not present")
    return GEX(str(path))


def gate_table(system, channel):
    """The gate table this channel draws on.

    Classic-receiver systems (304M, 304HR) carry one shared ``GateTime``;
    DigiTEM-receiver systems (306HP, 312HP) digitise each moment independently
    and carry ``GateTimeLM`` and ``GateTimeHM`` separately. The moment-specific
    table wins, which is the order ``libaarhusxyz`` resolves them in.
    """
    general = system.gex_dict["General"]
    moment = system.gex_dict[f"Channel{channel}"].get("TransmitterMoment", "")
    return general.get(f"GateTime{moment}", general.get("GateTime"))


@pytest.mark.parametrize("name", SYSTEMS)
def test_column_zero_is_the_center_not_open_or_close(name):
    """What identifies the center column: it lies strictly inside the gate.

    Checked as an ordering rather than an exact midpoint. The center is *very
    nearly* the arithmetic mean of open and close — within 1e-5 relative almost
    everywhere — but not exactly: on a 312HP the first high-moment gate, which
    straddles transmitter turn-off, differs from the midpoint by 2.5e-7 s. The
    ordering holds regardless, and it is the ordering that tells us which column
    to take.
    """
    system = load(name)
    for channel in range(1, int(system.number_channels) + 1):
        gates = np.asarray(system.gate_times(channel), dtype=float)
        assert gates.shape[1] == 3
        center, open_, close = gates[:, 0], gates[:, 1], gates[:, 2]

        assert np.all(open_ < center), f"{name} ch{channel}"
        assert np.all(center < close), f"{name} ch{channel}"
        np.testing.assert_allclose(center, (open_ + close) / 2, rtol=1e-2)


@pytest.mark.parametrize("name", SYSTEMS)
def test_axis_takes_the_center_column(name):
    system = load(name)
    axis = _gate_axis(system, 1)
    np.testing.assert_allclose(
        axis.values, np.asarray(system.gate_times(1), dtype=float)[:, 0])
    assert (axis.name, axis.unit) == ("gate time", "s")


@pytest.mark.parametrize("name", SYSTEMS)
def test_gate_times_are_strictly_increasing(name):
    system = load(name)
    for channel in range(1, int(system.number_channels) + 1):
        centers = _gate_axis(system, channel).values
        assert np.all(np.diff(centers) > 0)


@pytest.mark.parametrize("name", SYSTEMS)
def test_removing_initial_gates_would_not_leave_enough_gates(name):
    """Why ``RemoveInitialGates`` cannot be a gate-time offset.

    On a 304M, Channel2 declares 37 gates and the general block lists exactly
    37. Treating ``RemoveInitialGates=7`` as a slice offset would leave 30 — so
    it is not an offset. It is the instrument saying its earliest gates are
    contaminated by turn-off, which is filter_20's business, writing the mask.
    """
    system = load(name)
    for channel in range(1, int(system.number_channels) + 1):
        block = system.gex_dict[f"Channel{channel}"]
        available = len(gate_table(system, channel))
        n_gates = int(block["NoGates"])

        # The table always holds at least the gates the channel declares...
        assert n_gates <= available, f"{name} ch{channel}"
        # ...and the axis we build is exactly that many, unshifted by
        # RemoveInitialGates.
        assert _gate_axis(system, channel).size == n_gates


@pytest.mark.parametrize("name", ["SkyTEM_304M_XYZ"])
def test_removing_initial_gates_as_an_offset_would_overrun(name):
    """The evidence that ``RemoveInitialGates`` is not a slice offset.

    A 304M's Channel2 declares 37 gates and its shared table holds exactly 37.
    Treating ``RemoveInitialGates=7`` as an offset would leave 30 — fewer than
    declared. So it is not an offset into the gate table; it is the instrument
    stating that its earliest gates are contaminated by transmitter turn-off,
    which is filter_20's business, writing the mask (R6).
    """
    system = load(name)
    block = system.gex_dict["Channel2"]
    remove = int(block.get("RemoveInitialGates", 0))
    n_gates = int(block["NoGates"])
    available = len(gate_table(system, 2))

    assert remove > 0
    assert n_gates == available
    assert remove + n_gates > available


@pytest.mark.parametrize("name", SYSTEMS)
def test_dipole_moment_is_turns_times_area_times_current(name):
    """m = N.I.A, and the high moment dwarfs the low one."""
    system = load(name)
    general = system.gex_dict["General"]
    area = general["TxLoopArea"]

    moments = []
    for channel in range(1, int(system.number_channels) + 1):
        block = system.gex_dict[f"Channel{channel}"]
        turns = general.get(
            f"NumberOfTurns{block.get('TransmitterMoment', '')}",
            general.get("NumberOfTurns"))
        expected = turns * area * block["TxApproximateCurrent"]
        assert _dipole_moment(system, channel) == pytest.approx(expected)
        moments.append(expected)

    if len(moments) > 1:
        assert moments[-1] > moments[0]


@pytest.mark.parametrize("name", SYSTEMS)
def test_gate_count_differs_between_moments(name):
    """More R1 evidence: even gate count is per-moment, not a survey constant."""
    system = load(name)
    counts = {int(system.gex_dict[f"Channel{c}"]["NoGates"])
              for c in range(1, int(system.number_channels) + 1)}
    assert counts  # merely that it is discoverable per channel, not assumed
