"""A third naming family: moments named rather than numbered.

Found by pointing the library at a real SkyTEM delivery for the first time. It
ships `LM_Z_G01`, `HM_Z_G01`, `HM_X_G01` — a convention that identifies a channel
by **transmitter moment and receiver component**, with no number anywhere. The
library discovered zero channels in it, so every filter and check that needs one
was a no-op and the run reported success.

The thing worth holding onto: `G01` looks like a channel index and is not. It is
the same on every array in the file.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset
from ymerflow_processing.core.dataset import ChannelNaming, discover_by_naming
from ymerflow_processing.core.select_channels import (
    select_channels, unclaimed_arrays,
)
from ymerflow_processing.em.tem.method import (
    NAMED_MOMENT, NAMED_MOMENT_INUSE, TEM_NAMING, _by_moment_and_component,
)

N, LM_GATES, HM_GATES = 6, 28, 37


class FakeGEX:
    """A system description declaring LM on channel 1 and HM on channel 2, both Z.

    Which is what the real delivery's own description says — and it declares
    nothing at all about the horizontal component it ships alongside.
    """

    def __init__(self, channels=(("LM", "Z", LM_GATES), ("HM", "Z", HM_GATES))):
        self.gex_dict = {
            f"Channel{i}": {"ApproxDipoleMoment": 1000.0 * i}
            for i, _ in enumerate(channels, start=1)
        }
        self._channels = channels

    def transmitter_moment(self, index):
        return self._channels[index - 1][0]

    def rx_orientation(self, index):
        return self._channels[index - 1][1]

    def gate_times(self, index):
        n = self._channels[index - 1][2]
        centers = np.logspace(-5, -3, n)
        return np.column_stack([centers, centers * 0.9, centers * 1.1])


def delivery(keys=("LM_Z_G01", "HM_Z_G01", "HM_X_G01"), system=None):
    """A dataset in the delivered-product naming, with no std and no mask."""
    dataset = make_dataset(n=N, n_lines=1, n_gates=4)
    for key in list(dataset.layer_data):
        del dataset.layer_data[key]
    widths = {"LM_Z_G01": LM_GATES, "HM_Z_G01": HM_GATES, "HM_X_G01": HM_GATES}
    for key in keys:
        dataset.layer_data[key] = pd.DataFrame(
            np.full((N, widths.get(key, HM_GATES)), 1e-9))
    dataset.survey.gex = system if system is not None else FakeGEX()
    return dataset


# ── The resolver asks; it does not assume ────────────────────────────────────

def test_a_moment_name_resolves_through_the_system_description():
    """Nothing in an array called HM_Z_G01 says whether the high moment is
    channel 1 or channel 2. Only the description knows."""
    system = FakeGEX()
    assert _by_moment_and_component(("lm", "z"), system) == 1
    assert _by_moment_and_component(("hm", "z"), system) == 2


def test_a_component_the_description_does_not_declare_resolves_to_nothing():
    """SkyTEM's data files carry a horizontal component, but the description
    is built for the Z component only."""
    assert _by_moment_and_component(("hm", "x"), FakeGEX()) is None


def test_the_moments_can_be_declared_in_either_order():
    """Nothing says the low moment is channel 1 — the description does."""
    swapped = FakeGEX((("HM", "Z", HM_GATES), ("LM", "Z", LM_GATES)))
    assert _by_moment_and_component(("lm", "z"), swapped) == 2
    assert _by_moment_and_component(("hm", "z"), swapped) == 1


def test_without_a_description_nothing_resolves_and_order_is_not_a_fallback(caplog):
    """Sorted by name a high moment precedes a low one, so guessing would put
    the moments the wrong way round with nothing to give it away."""
    assert _by_moment_and_component(("hm", "z"), None) is None
    assert "Order is not a fallback" in caplog.text


def test_a_malformed_description_does_not_loop_forever():
    class Endless:
        gex_dict = type("D", (), {"get": lambda self, k: {"x": 1}})()

        def transmitter_moment(self, index):
            return "ZZ"

        def rx_orientation(self, index):
            return "q"

    assert _by_moment_and_component(("lm", "z"), Endless()) is None


# ── Discovery, end to end ────────────────────────────────────────────────────

def test_the_delivered_naming_now_resolves():
    dataset = delivery()
    found = {c.key: c for c in dataset.channels}

    assert sorted(found) == ["ch1", "ch2"]
    assert found["ch1"].name == "LM" and found["ch1"].axis.size == LM_GATES
    assert found["ch2"].name == "HM" and found["ch2"].axis.size == HM_GATES


def test_the_undeclared_component_is_unclaimed_rather_than_a_channel():
    """It must not reach an inversion: DualMomentTEMXYZSystem hardcodes
    rx_orientation='z' and never reads the declared polarization, so a
    three-component delivery presented as moments is read as a confident wrong
    model rather than refused."""
    dataset = delivery()
    assert "HM_X_G01" in unclaimed_arrays(dataset)
    assert "HM_X_G01" not in {c.data_key for c in dataset.channels}


def test_the_unclaimed_component_can_be_dropped_when_asked():
    dataset = delivery()
    dataset, report = select_channels(dataset, criterion="component",
                                      keep=["z"], drop_unclaimed=True)
    assert "HM_X_G01" not in dataset.layer_data


def test_it_is_kept_by_default():
    """Undeclared rather than empty, and at least one of them is real coherent
    signal the incumbent toolchain merely discards."""
    dataset = delivery()
    dataset, _ = select_channels(dataset, criterion="component", keep=["z"])
    assert "HM_X_G01" in dataset.layer_data


def test_a_single_moment_system_yields_one_channel():
    """R1: the count comes from the data, never from an assumption."""
    dataset = delivery(keys=("LM_Z_G01",),
                       system=FakeGEX((("LM", "Z", LM_GATES),)))
    assert len(dataset.channels) == 1


# ── The traps ────────────────────────────────────────────────────────────────

def test_the_mask_this_library_writes_is_not_read_back_as_data():
    """Unanchored, the data pattern also matches LM_Z_G01_inuse, so the mask
    would be discovered as a second data array for the same channel. The same
    trap the provenance prefix was moved to avoid, met from the other side."""
    import re
    assert re.match(NAMED_MOMENT, "LM_Z_G01_inuse", re.IGNORECASE) is None
    assert re.match(NAMED_MOMENT, "LM_Z_G01", re.IGNORECASE) is not None


def test_the_mask_is_claimed_by_its_own_channel():
    from ymerflow_processing.core.mask import ensure

    dataset = delivery()
    ensure(dataset, dataset.channels[0])
    channel = dataset.channels[0]

    assert channel.inuse_key == "LM_Z_G01_inuse"
    assert "LM_Z_G01_inuse" not in unclaimed_arrays(dataset)


def test_provenance_is_not_mistaken_for_a_channel_of_its_own():
    import re
    for pattern in (NAMED_MOMENT, NAMED_MOMENT_INUSE):
        assert re.match(pattern, "why_LM_Z_G01_inuse", re.IGNORECASE) is None


def test_g01_is_not_a_channel_index():
    """It is the same on every array in the file, and reading it as an index
    would fold all three arrays into one channel."""
    dataset = delivery()
    assert {c.index for c in dataset.channels} == {1, 2}


# ── The mechanism stays in core, the meaning stays in the method ─────────────

def test_a_numeric_pattern_still_needs_no_resolver():
    """Backwards compatible: a single all-digit group is the index itself."""
    naming = ChannelNaming(data=(r"^dbdt_ch(\d+)",))
    assert list(discover_by_naming(["dbdt_ch1gt", "dbdt_ch2gt"], naming)) == [1, 2]


def test_a_named_pattern_without_a_resolver_says_so_rather_than_guessing(caplog):
    naming = ChannelNaming(data=(NAMED_MOMENT,))
    assert discover_by_naming(["LM_Z_G01"], naming) == {}
    assert "not an index" in caplog.text


def test_the_resolver_receives_the_context_it_was_given():
    seen = []

    def resolve(groups, context):
        seen.append((groups, context))
        return 1

    naming = ChannelNaming(data=(NAMED_MOMENT,), resolve=resolve)
    discover_by_naming(["LM_Z_G01"], naming, "the system")
    assert seen == [(("LM", "Z"), "the system")]


def test_all_three_families_coexist():
    """Two were in circulation when this was written; a third turned up in the
    first real delivery, and there will be a fourth."""
    assert len(TEM_NAMING.data) == 3
    naming = TEM_NAMING
    for key in ("dbdt_ch1gt", "Gate_Ch01", "LM_Z_G01"):
        assert any(__import__("re").match(p, key, __import__("re").IGNORECASE)
                   for p in naming.data), key


def test_our_own_provenance_is_not_reported_as_an_undeclared_component():
    """The note explains that a vendor may ship a component it omits from the
    system description — a confusing thing to read about a column this library
    created two steps ago."""
    from ymerflow_processing.core.mask import cull, mark_step

    dataset = delivery()
    mark_step(dataset, 32, "cull_negatives")
    channel = dataset.channels[0]
    cull(dataset, channel,
         np.tile(np.arange(LM_GATES) >= 20, (N, 1)), step_code=32)

    unclaimed = unclaimed_arrays(dataset)
    assert "HM_X_G01" in unclaimed                     # the vendor's, reported
    assert not [k for k in unclaimed if k.startswith("why_")]   # ours, not


def test_the_inventory_and_select_channels_agree_on_what_is_unclaimed():
    """They had two implementations of the same idea, and the second had not
    learned what the first knew."""
    from ymerflow_processing.core.inventory import take
    from ymerflow_processing.core.mask import cull, mark_step

    dataset = delivery()
    mark_step(dataset, 32, "cull_negatives")
    cull(dataset, dataset.channels[0],
         np.tile(np.arange(LM_GATES) >= 20, (N, 1)), step_code=32)

    assert (set(take(dataset)["arrays"]["unresolved"])
            == set(unclaimed_arrays(dataset)))
