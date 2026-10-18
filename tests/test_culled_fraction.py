"""qc_12 — how much of each sounding survived processing.

The only check whose answer depends on a step having run earlier in the same
session, and the only one that reads the mask rather than an attribute. Most of
what follows pins the denominator, because measuring against the wrong thing
reports the contractor's culling as ours.
"""

import json

import numpy as np
import pandas as pd
import pytest

from conftest import make_dataset, tier_names
from ymerflow_processing.core.cull.nominate_gates import nominate_gates
from ymerflow_processing.core.cull.negatives import cull_negatives
from ymerflow_processing.core.mask import CULLED
from ymerflow_processing.core.pipeline import (
    QC_GROUP, Pipeline, QCRunner, load_steps,
)
from ymerflow_processing.core.qc.culled_fraction import culled_fraction
from ymerflow_processing.core.qc.thresholds import GOOD, MARGINAL, OUT_OF_SPEC

N = 4
GATES = 8


def delivery(**kw):
    return make_dataset(n=N, n_lines=1, n_gates=GATES, **kw)


def culled(dataset, gates="0-3", channel="ch1"):
    dataset, _ = nominate_gates(dataset, disable_gates={channel: gates})
    return dataset


# ── The measurement ──────────────────────────────────────────────────────────

def test_it_measures_what_this_pipeline_culled():
    """Four of eight gates in one of two channels: a quarter of the sounding."""
    dataset, report = culled_fraction(culled(delivery()))

    assert dataset.flightlines["cull_frac"].iloc[0] == pytest.approx(0.25)
    assert report.distributions["culled_here"] == N * 4


def test_the_fraction_is_across_channels_not_within_one():
    """The question is how much of this sounding survived, and a sounding is not
    one channel."""
    dataset = culled(culled(delivery()), gates="0-7", channel="ch2")
    dataset, _ = culled_fraction(dataset)

    # Four of ch1's eight and all eight of ch2's: twelve of sixteen.
    assert dataset.flightlines["cull_frac"].iloc[0] == pytest.approx(12 / 16)


def test_the_per_channel_split_is_reported():
    """"Half gone" reads very differently if it is half of each moment rather
    than all of one."""
    _, report = culled_fraction(culled(delivery()))
    per_channel = report.distributions["per_channel"]

    assert per_channel["ch1 (LM/z)"]["culled_here"] == N * 4
    assert per_channel["ch2 (HM/z)"]["culled_here"] == 0


def test_nothing_culled_measures_zero():
    dataset, report = culled_fraction(delivery())
    assert (dataset.flightlines["cull_frac"] == 0.0).all()
    assert report.distributions["culled_here"] == 0


def test_it_changes_no_measurement_uncertainty_or_mask():
    """R6 from the other side: this reads the mask culling wrote."""
    dataset = culled(delivery())
    arrays = {k: v.copy() for k, v in dataset.layer_data.items()}

    dataset, _ = culled_fraction(dataset)

    for key, array in arrays.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


# ── The denominator ──────────────────────────────────────────────────────────

def test_data_that_arrived_culled_is_not_reported_as_ours():
    """Measuring against the full array would report the contractor's culling
    as this pipeline's, which flatters or damns the wrong party."""
    dataset = delivery()
    dataset.layer_data[dataset.channels[0].inuse_key].iloc[:, :4] = CULLED

    dataset, report = culled_fraction(dataset)

    assert (dataset.flightlines["cull_frac"] == 0.0).all()
    assert report.distributions["arrived_culled"] == N * 4
    assert report.distributions["culled_here"] == 0
    assert any("arrived already disabled" in n for n in report.notes)


def test_the_two_are_reported_side_by_side():
    dataset = delivery()
    dataset.layer_data[dataset.channels[0].inuse_key].iloc[:, 7] = CULLED
    dataset = culled(dataset, gates="0-1")

    _, report = culled_fraction(dataset)
    assert report.distributions["culled_here"] == N * 2
    assert report.distributions["arrived_culled"] == N


def test_a_delivery_with_no_mask_at_all_is_skipped():
    """It reports on the culling group's work, so with none done it says so
    rather than reporting a confident zero."""
    dataset = delivery()
    for channel in dataset.channels:
        del dataset.layer_data[channel.inuse_key]

    dataset, report = culled_fraction(dataset)

    assert "cull_frac" not in dataset.flightlines.columns
    assert any("nothing has been culled" in n for n in report.notes)


# ── Verdicts ─────────────────────────────────────────────────────────────────

def test_without_thresholds_it_measures_and_says_nothing():
    dataset, report = culled_fraction(culled(delivery()))

    assert "cull_frac" in dataset.flightlines.columns
    assert "qs_culled" not in dataset.flightlines.columns
    assert report.verdict_column is None
    assert any("left unjudged" in n for n in report.notes)


def test_with_thresholds_it_classifies():
    dataset, report = culled_fraction(culled(delivery()),
                                      good_max=0.1, marginal_max=0.2)

    assert tier_names(dataset.flightlines["qs_culled"])[0] == OUT_OF_SPEC
    assert report.verdict_column == "qs_culled"
    assert report.finding == "processing"


def test_half_a_threshold_is_an_error():
    with pytest.raises(ValueError, match="both bounds"):
        culled_fraction(culled(delivery()), good_max=0.1)


def test_it_reports_as_neither_flight_nor_instrument():
    """Culling is this library's doing, not the contractor's and not the
    instrument's, so it names itself as a reason and feeds neither summary."""
    dataset, _ = QCRunner(Pipeline.parse("""
        kind: qc
        steps:
          - height_above_ground: {good_max: 45, marginal_max: 65}
          - culled_fraction: {good_max: 0.1, marginal_max: 0.2}
    """)).run(culled(delivery(height=np.full(N, 30.0))))

    row = dataset.flightlines.iloc[0]
    assert tier_names([row["qs_culled"]])[0] == OUT_OF_SPEC
    assert tier_names([row["qs_verdict"]])[0] == GOOD      # the flying was fine
    assert pd.isna(row["qs_sensor"])
    assert "culled_fraction" in row["qs_reasons"]


# ── Attribution ──────────────────────────────────────────────────────────────

def test_it_names_the_steps_that_did_the_culling():
    dataset = delivery()
    values = dataset.layer_data[dataset.channels[0].data_key]
    values.iloc[1, 5] = -1e-10
    dataset, _ = cull_negatives(dataset)
    dataset = culled(dataset, gates="0-1")

    _, report = culled_fraction(dataset)
    assert report.distributions["by_step"] == {21: "nominate_gates",
                                               32: "cull_negatives"}


def test_a_restored_gate_is_not_counted_as_culled():
    """It carries a provenance code but is in use — the mask disambiguates."""
    dataset = culled(delivery(), gates="0-3")
    dataset, _ = nominate_gates(dataset, enable_gates={"ch1": "0-3"})

    dataset, report = culled_fraction(dataset)
    assert (dataset.flightlines["cull_frac"] == 0.0).all()
    assert report.distributions["culled_here"] == 0


# ── Plumbing ─────────────────────────────────────────────────────────────────

def test_it_is_idempotent():
    dataset = culled(delivery())
    dataset, first = culled_fraction(dataset)
    before = dataset.flightlines["cull_frac"].copy()
    dataset, second = culled_fraction(dataset)

    pd.testing.assert_series_equal(dataset.flightlines["cull_frac"], before)
    assert second.distributions["culled_here"] == first.distributions["culled_here"]


def test_the_report_survives_serialization():
    _, report = culled_fraction(culled(delivery()), good_max=0.1,
                                marginal_max=0.2)
    restored = json.loads(json.dumps(report.to_dict(), default=str))
    assert restored["step"] == "culled_fraction"
    assert restored["distributions"]["culled_here"] == N * 4


def test_the_column_names_fit_a_shapefile_field():
    dataset, _ = culled_fraction(culled(delivery()), good_max=0.1,
                                 marginal_max=0.2)
    for name in ("qs_culled", "cull_frac"):
        assert name in dataset.flightlines.columns
        assert len(name) <= 10


def test_registered():
    assert "culled_fraction" in load_steps(QC_GROUP)
