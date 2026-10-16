"""The weighted grade — points per sounding, averaged up the grains.

A verdict answers "does this pass". A grade answers "how bad, relative to
everything else", which a verdict cannot: once two lines are both Out of Spec
they tie, and a reflight conversation needs them ordered.

These two must not be confused, and most of what follows is pinning the line
between them: the grade dilutes by construction, so it is never the gate.
"""

import numpy as np
import pytest

from conftest import make_dataset, tier_names
from ymerflow_processing.core.pipeline import Pipeline, QCRunner
from ymerflow_processing.core.qc.rollup import (
    DEFAULT_WEIGHTS, UNWEIGHTED, summarize_run,
)
from ymerflow_processing.core.qc.thresholds import (
    GOOD, MARGINAL, MAX_POINTS, OUT_OF_SPEC, TIER_POINTS, score,
)

HEIGHT = {"good_max": 45, "marginal_max": 65}
SPEED = {"good_min": 15, "good_max": 25, "marginal_min": 10, "marginal_max": 30}


def run(document, **kw):
    kw.setdefault("n_lines", 2)
    kw.setdefault("n_flights", 2)
    dataset = make_dataset(n=40, **kw)
    return QCRunner(Pipeline.parse(document)).run(dataset)


# ── Points ───────────────────────────────────────────────────────────────────

def test_the_scale_runs_worst_to_best():
    assert TIER_POINTS[OUT_OF_SPEC] == 0
    assert TIER_POINTS[MARGINAL] == 1
    assert TIER_POINTS[GOOD] == 2
    assert MAX_POINTS == 2


def test_an_unmeasured_sounding_has_no_grade_rather_than_a_zero():
    """Zero would make a line nobody checked look identical to one that failed
    outright — the more alarming reading, for the case with the least evidence."""
    points = score([GOOD, None, OUT_OF_SPEC])
    assert points[0] == 2.0
    assert np.isnan(points[1])
    assert points[2] == 0.0


# ── The grade is not the gate ────────────────────────────────────────────────

def test_a_light_check_failing_grades_well_and_still_fails():
    """The whole reason both exist. Speed is out of spec while height — eight
    times its weight — is clean, so the grade stays near the top. The verdict
    does not care about weight and reads Out of Spec regardless.

    This is dilution working as intended, and it is why the grade cannot be the
    gate: read the score alone and this sounding looks fine."""
    dataset, _ = run(f"""
        kind: qc
        steps:
          - height_above_ground: {HEIGHT}
          - ground_speed: {SPEED}
    """, height=np.full(40, 30.0), speed=np.full(40, 45.0))

    row = dataset.flightlines.iloc[0]
    assert tier_names([row["qs_verdict"]])[0] == OUT_OF_SPEC          # the gate is unmoved
    assert row["qs_score"] == pytest.approx(4.0 / 2.25)
    assert row["qs_score"] > 0.75 * MAX_POINTS       # and the grade barely dips


def test_a_heavy_check_failing_drags_the_grade_down_hard():
    """The other side of the same weighting. Height carries 2.0 of the 2.25 on
    offer, so failing it costs almost everything."""
    dataset, _ = run(f"""
        kind: qc
        steps:
          - height_above_ground: {HEIGHT}
          - ground_speed: {SPEED}
    """, height=np.full(40, 90.0), speed=np.full(40, 20.0))

    row = dataset.flightlines.iloc[0]
    assert tier_names([row["qs_verdict"]])[0] == OUT_OF_SPEC
    assert row["qs_score"] == pytest.approx(0.5 / 2.25)
    assert row["qs_score"] < 0.25 * MAX_POINTS


def test_a_clean_sounding_scores_the_maximum():
    dataset, _ = run(f"""
        kind: qc
        steps:
          - height_above_ground: {HEIGHT}
    """, height=np.full(40, 30.0))

    assert (dataset.flightlines["qs_score"] == MAX_POINTS).all()
    assert set(tier_names(dataset.flightlines["qs_verdict"])) == {GOOD}


# ── Weights ──────────────────────────────────────────────────────────────────

def test_the_defaults_are_the_stated_ones():
    """Pinned because they are a judgement call someone made deliberately, and a
    silent drift would re-rank every survey ever graded."""
    assert DEFAULT_WEIGHTS == {
        "height_above_ground": 2.0,
        "system_attitude": 2.0,
        "ground_speed": 0.25,
        "sounding_spacing": 0.25,
        "transmitter_stability": 0.25,
        "altimeter_agreement": 0.0,
        "powerline_noise": 0.0,
        "data_misfit": 0.0,
    }


def test_height_outweighs_speed_eight_to_one():
    """2.0 against 0.25. Failing height must cost far more than failing speed."""
    document = f"""
        kind: qc
        steps:
          - height_above_ground: {HEIGHT}
          - ground_speed: {SPEED}
    """
    bad_height, _ = run(document, height=np.full(40, 90.0),
                        speed=np.full(40, 20.0))
    bad_speed, _ = run(document, height=np.full(40, 30.0),
                       speed=np.full(40, 45.0))

    assert bad_height.flightlines["qs_score"][0] < bad_speed.flightlines["qs_score"][0]
    # 2.0*0 + 0.25*2 over 2.25, against 2.0*2 + 0.25*0 over 2.25.
    assert bad_height.flightlines["qs_score"][0] == pytest.approx(0.5 / 2.25)
    assert bad_speed.flightlines["qs_score"][0] == pytest.approx(4.0 / 2.25)


def test_a_pipeline_may_override_a_weight():
    document = """
        kind: qc
        rollup:
          weights: {{ground_speed: 8.0}}
        steps:
          - height_above_ground: {height}
          - ground_speed: {speed}
    """.format(height=HEIGHT, speed=SPEED)
    dataset, report = run(document, height=np.full(40, 30.0),
                          speed=np.full(40, 45.0))

    assert report.rollup["weights"]["ground_speed"] == 8.0
    assert report.rollup["weights"]["height_above_ground"] == 2.0   # untouched
    # Now speed dominates, so failing it drags the grade below halfway.
    assert dataset.flightlines["qs_score"][0] == pytest.approx(4.0 / 10.0)


def test_a_zero_weight_flags_without_grading():
    """altimeter_agreement's default. It must still say what it found."""
    dataset, report = run("""
        kind: qc
        steps:
          - height_above_ground: {good_max: 45, marginal_max: 65}
          - altimeter_agreement:
              columns: [tx_alt, alt_two]
              good_max: 1.0
              marginal_max: 2.0
    """, height=np.full(40, 30.0), second_altimeter=np.full(40, 55.0))

    assert report.rollup["weights"]["altimeter_agreement"] == 0.0
    # Graded on height alone, so a clean flight still scores full marks.
    assert (dataset.flightlines["qs_score"] == MAX_POINTS).all()
    # And the check is still present, still able to raise a sensor fault.
    assert "qs_altim" in dataset.flightlines.columns
    assert "qs_sensor" in dataset.flightlines.columns


def test_every_weight_that_applied_is_recorded():
    """Including the ones nobody set, so an unintended default is visible."""
    _, report = run(f"""
        kind: qc
        steps:
          - height_above_ground: {HEIGHT}
          - ground_speed: {SPEED}
    """, height=np.full(40, 30.0), speed=np.full(40, 20.0))

    assert report.rollup["weights"] == {"height_above_ground": 2.0,
                                        "ground_speed": 0.25}
    assert report.rollup["max_score"] == MAX_POINTS


def test_an_unlisted_check_gets_a_full_say_rather_than_none():
    """A new check must be weighted down on purpose, not vanish by omission."""
    assert UNWEIGHTED == 1.0
    assert "powerline_noise" in DEFAULT_WEIGHTS      # this one was on purpose


def test_weighting_a_step_that_never_ran_is_an_error():
    with pytest.raises(KeyError, match="produced no verdict"):
        run(f"""
            kind: qc
            rollup:
              weights: {{ground_speeed: 1.0}}
            steps:
              - height_above_ground: {HEIGHT}
        """, height=np.full(40, 30.0))


def test_a_negative_weight_is_an_error():
    with pytest.raises(ValueError, match="must not be negative"):
        run(f"""
            kind: qc
            rollup:
              weights: {{height_above_ground: -1.0}}
            steps:
              - height_above_ground: {HEIGHT}
        """, height=np.full(40, 30.0))


# ── Averaging up the grains ──────────────────────────────────────────────────

def test_each_grain_averages_its_soundings():
    """Half a line failing height grades halfway between the two."""
    heights = np.concatenate([np.full(10, 30.0), np.full(10, 90.0),
                              np.full(20, 30.0)])
    dataset, _ = run(f"""
        kind: qc
        steps:
          - height_above_ground: {HEIGHT}
    """, height=heights)

    frame = dataset.flightlines
    assert frame["ql_score"][0] == pytest.approx(1.0)      # line 100: half failed
    assert frame["ql_score"][20] == pytest.approx(2.0)     # line 101: clean
    assert frame["qd_score"][0] == pytest.approx(1.5)      # the delivery


def test_the_delivery_average_is_over_soundings_not_over_lines():
    """Otherwise a two-sounding line weighs as much as a thousand-sounding one."""
    heights = np.concatenate([np.full(30, 90.0), np.full(10, 30.0)])
    dataset, _ = run(f"""
        kind: qc
        steps:
          - height_above_ground: {HEIGHT}
    """, height=heights, n_lines=2)

    # 30 soundings at 0 and 10 at 2 is 0.5, not the 1.0 a per-line mean gives.
    assert dataset.flightlines["qd_score"][0] == pytest.approx(0.5)


def test_the_grade_sits_alongside_the_fractions_rather_than_replacing_them():
    """A mean aliases: 1.0 is either all-Marginal or half-Good-half-OOS. The
    tier fractions are what tell those apart, so both are written."""
    dataset, _ = run(f"""
        kind: qc
        steps:
          - height_above_ground: {HEIGHT}
    """, height=np.concatenate([np.full(20, 30.0), np.full(20, 90.0)]))

    frame = dataset.flightlines
    assert frame["qd_score"][0] == pytest.approx(1.0)
    assert frame["qd_good"][0] == pytest.approx(0.5)
    assert frame["qd_marg"][0] == pytest.approx(0.0)
    assert frame["qd_oos"][0] == pytest.approx(0.5)


def test_no_check_reaching_a_verdict_leaves_no_grade():
    dataset = make_dataset(n=10, n_lines=1)
    assert summarize_run(dataset, [], None) == {"columns": {}, "checks": {}}
    assert "qs_score" not in dataset.flightlines.columns
