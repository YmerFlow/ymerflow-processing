"""Per-line aggregation: both rules measured, neither applied."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ymerflow_processing.core.qc.aggregate import (
    apply_rule, find_flight_column, grain_measures, longest_run, per_flight,
    per_line,
)
from ymerflow_processing.core.qc.thresholds import GOOD, MARGINAL, OUT_OF_SPEC


# ── Contiguous runs ──────────────────────────────────────────────────────────

def test_longest_run_of_nothing_is_zero():
    assert longest_run([False, False, False]) == 0
    assert longest_run([]) == 0


def test_longest_run_finds_the_longest_not_the_first():
    assert longest_run([True, False, True, True, True, False, True]) == 3


def test_longest_run_handles_runs_at_both_ends():
    assert longest_run([True, True, False, True]) == 2
    assert longest_run([True, False, True, True]) == 2
    assert longest_run([True] * 5) == 5


def test_a_scattered_total_and_one_stretch_are_distinguished():
    """The reason both rules are reported rather than just a fraction."""
    scattered = np.array([True, False] * 10)
    contiguous = np.array([True] * 10 + [False] * 10)

    assert scattered.sum() == contiguous.sum() == 10
    assert longest_run(scattered) == 1
    assert longest_run(contiguous) == 10


# ── Per-line summary ─────────────────────────────────────────────────────────

def _verdicts(sequence):
    return pd.Categorical(sequence, categories=[GOOD, MARGINAL, OUT_OF_SPEC],
                          ordered=True)


def test_per_line_counts_and_fractions():
    verdicts = _verdicts([GOOD, GOOD, OUT_OF_SPEC, OUT_OF_SPEC,
                          GOOD, MARGINAL, GOOD, GOOD])
    lines = [100] * 4 + [200] * 4

    summary = per_line(verdicts, lines)

    assert summary["100"]["counts"] == {GOOD: 2, MARGINAL: 0, OUT_OF_SPEC: 2}
    assert summary["100"]["fraction_out_of_spec"] == 0.5
    assert summary["100"]["longest_out_of_spec_run"] == 2

    assert summary["200"]["fraction_out_of_spec"] == 0.0
    # Marginal is not Good, and is counted whole: no weight has been supplied.
    assert summary["200"]["fraction_not_good"] == 0.25


def test_fractions_are_of_classified_soundings_only():
    """A line half unmeasured must not read as half in spec."""
    verdicts = _verdicts([GOOD, OUT_OF_SPEC, None, None])
    summary = per_line(verdicts, [100] * 4)

    assert summary["100"]["n_soundings"] == 4
    assert summary["100"]["n_classified"] == 2
    assert summary["100"]["fraction_out_of_spec"] == 0.5


def test_a_line_with_no_classified_soundings_reports_none_not_zero():
    summary = per_line(_verdicts([None, None]), [100, 100])
    assert summary["100"]["fraction_out_of_spec"] is None


def test_run_distance_uses_position_when_present():
    """10 m spacing, four contiguous bad soundings -> 30 m of flying."""
    verdicts = _verdicts([GOOD, OUT_OF_SPEC, OUT_OF_SPEC, OUT_OF_SPEC,
                          OUT_OF_SPEC, GOOD])
    summary = per_line(verdicts, [100] * 6,
                       easting=np.arange(6) * 10.0, northing=np.zeros(6))

    assert summary["100"]["longest_out_of_spec_run"] == 4
    assert summary["100"]["longest_out_of_spec_run_distance"] == 30.0


def test_run_distance_is_absent_without_position():
    summary = per_line(_verdicts([OUT_OF_SPEC]), [100])
    assert summary["100"]["longest_out_of_spec_run_distance"] is None


def test_a_single_bad_sounding_has_no_length():
    summary = per_line(_verdicts([GOOD, OUT_OF_SPEC, GOOD]), [100] * 3,
                       easting=np.arange(3) * 10.0, northing=np.zeros(3))
    assert summary["100"]["longest_out_of_spec_run"] == 1
    assert summary["100"]["longest_out_of_spec_run_distance"] == 0.0


# ── Measures broadcast to every sounding ─────────────────────────────────────

def test_each_group_measure_is_repeated_on_its_soundings():
    """A vector export has one row per sounding, so the line's figure has to sit
    on each of them for a map to be colored by it."""
    verdicts = _verdicts([OUT_OF_SPEC, OUT_OF_SPEC, GOOD, GOOD, GOOD, GOOD])
    lines = [100] * 3 + [200] * 3
    x, y = np.arange(6) * 10.0, np.zeros(6)

    measures = grain_measures(verdicts, lines, easting=x, northing=y)

    assert list(measures["oos"][:3]) == [pytest.approx(2 / 3)] * 3
    assert list(measures["oos"][3:]) == [0.0] * 3
    assert measures["run"][0] == 10.0
    assert list(measures["n"]) == [3.0] * 6


def test_one_bad_sounding_does_not_read_as_a_failed_line():
    """What replaced the worst-verdict column: one in six is a fraction of
    0.17, not a line-wide Out of Spec."""
    verdicts = _verdicts([OUT_OF_SPEC] + [GOOD] * 5)
    measures = grain_measures(verdicts, [100] * 6)
    assert measures["oos"][0] == pytest.approx(1 / 6)


def test_every_tier_gets_a_fraction_not_just_the_failing_one():
    """0.02 out of spec does not say whether the rest was Good or Marginal, and
    those are different deliveries."""
    verdicts = _verdicts([OUT_OF_SPEC] + [MARGINAL] * 3 + [GOOD] * 4)
    measures = grain_measures(verdicts, [100] * 8)

    assert measures["good"][0] == pytest.approx(0.5)
    assert measures["marg"][0] == pytest.approx(3 / 8)
    assert measures["oos"][0] == pytest.approx(1 / 8)


def test_the_three_fractions_sum_to_one_over_what_was_classified():
    """And 'n' is the denominator, so a half-measured line cannot pass for
    a fully measured one."""
    verdicts = _verdicts([OUT_OF_SPEC, MARGINAL, GOOD, GOOD, None, None])
    measures = grain_measures(verdicts, [100] * 6)

    total = measures["good"][0] + measures["marg"][0] + measures["oos"][0]
    assert total == pytest.approx(1.0)
    assert measures["n"][0] == 4.0          # not 6


def test_a_group_with_nothing_classified_measures_nothing():
    measures = grain_measures(_verdicts([None, None]), [100] * 2)
    assert np.isnan(measures["good"][0])
    assert np.isnan(measures["oos"][0])
    assert measures["n"][0] == 0.0


# ── Turning the measures into a verdict ──────────────────────────────────────

def test_no_thresholds_means_no_verdict():
    """Measure and do not decide, unless a contract says how."""
    assert apply_rule([0.9], [5000.0]) is None


def test_either_measure_can_trip_the_rule():
    """They catch different problems, so requiring both would miss whichever
    case the contract was written for."""
    scattered = apply_rule([0.9], [20.0], max_fraction=0.05, max_run=1000)
    one_stretch = apply_rule([0.02], [5000.0], max_fraction=0.05, max_run=1000)

    assert scattered[0] == OUT_OF_SPEC        # broadly bad, never long
    assert one_stretch[0] == OUT_OF_SPEC      # briefly bad, but continuously


def test_a_line_inside_both_thresholds_passes():
    verdict = apply_rule([0.02], [20.0], max_fraction=0.05, max_run=1000)
    assert verdict[0] == GOOD


def test_one_threshold_alone_is_enough_to_have_a_rule():
    assert apply_rule([0.9], [0.0], max_fraction=0.05)[0] == OUT_OF_SPEC
    assert apply_rule([0.0], [5000.0], max_run=1000)[0] == OUT_OF_SPEC


def test_an_unmeasurable_group_does_not_fail():
    """A line nothing could be judged on has no fraction; that is not a breach."""
    assert apply_rule([np.nan], [np.nan], max_fraction=0.05)[0] == GOOD


# ── Per flight ───────────────────────────────────────────────────────────────
# A delivery commonly holds three or four flights, so one QC run has to break
# down by flight as well as by line.

def test_per_flight_groups_several_lines():
    verdicts = _verdicts([GOOD] * 4 + [OUT_OF_SPEC] * 4)
    lines = [100, 100, 101, 101, 102, 102, 103, 103]
    flights = [6, 6, 6, 6, 7, 7, 7, 7]

    summary = per_flight(verdicts, flights, lines)

    assert set(summary) == {"6", "7"}
    assert summary["6"]["fraction_out_of_spec"] == 0.0
    assert summary["7"]["fraction_out_of_spec"] == 1.0
    assert summary["7"]["n_soundings"] == 4


def test_a_run_does_not_span_the_turn_between_two_lines():
    """The aircraft turns between lines, so two bad stretches either side of a
    turn are two problems, not one long one."""
    verdicts = _verdicts([GOOD, OUT_OF_SPEC, OUT_OF_SPEC,
                          OUT_OF_SPEC, OUT_OF_SPEC, GOOD])
    lines = [100, 100, 100, 101, 101, 101]   # boundary falls mid-run
    flights = [6] * 6

    segmented = per_flight(verdicts, flights, lines)
    unsegmented = per_flight(verdicts, flights)

    assert segmented["6"]["longest_out_of_spec_run"] == 2       # 2 on each line
    assert unsegmented["6"]["longest_out_of_spec_run"] == 4     # overstated


def test_per_flight_takes_the_worst_line_not_the_sum():
    verdicts = _verdicts([OUT_OF_SPEC, OUT_OF_SPEC, GOOD,
                          OUT_OF_SPEC, OUT_OF_SPEC, OUT_OF_SPEC])
    lines = [100, 100, 100, 101, 101, 101]

    summary = per_flight(verdicts, [6] * 6, lines)
    assert summary["6"]["longest_out_of_spec_run"] == 3         # line 101's run


def test_per_flight_reports_run_distance():
    verdicts = _verdicts([GOOD, OUT_OF_SPEC, OUT_OF_SPEC, OUT_OF_SPEC])
    summary = per_flight(verdicts, [6] * 4, [100] * 4,
                         easting=np.arange(4) * 10.0, northing=np.zeros(4))
    assert summary["6"]["longest_out_of_spec_run_distance"] == 20.0


def test_per_flight_and_per_line_agree_on_totals():
    """Different grains over the same soundings must count the same things."""
    verdicts = _verdicts([GOOD, OUT_OF_SPEC] * 6)
    lines = np.repeat([100, 101, 102], 4)
    flights = np.repeat([6, 7], 6)

    by_line = per_line(verdicts, lines)
    by_flight = per_flight(verdicts, flights, lines)

    total = lambda s: sum(v["counts"][OUT_OF_SPEC] for v in s.values())
    assert total(by_line) == total(by_flight) == 6


# ── Finding the flight column ────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["flight", "Flight", "flight_no", "FlightID"])
def test_flight_column_is_discovered_under_its_usual_spellings(name):
    frame = pd.DataFrame({name: [1, 1, 2], "other": [0, 0, 0]})
    assert find_flight_column(frame) == name


def test_no_flight_column_is_not_an_error():
    assert find_flight_column(pd.DataFrame({"line_id": [1]})) is None


# ── Gap tolerance ────────────────────────────────────────────────────────────
# A contract saying "out of spec for 1 km or more" describes a stretch of
# flying. One sounding that dips back into spec does not end it — but without
# closing, it halves the measured run.

def _line(flags, spacing=10.0):
    """A straight line at `spacing` apart, verdicts from a boolean mask."""
    verdicts = _verdicts([OUT_OF_SPEC if f else GOOD for f in flags])
    x = np.arange(len(flags)) * spacing
    return verdicts, x, np.zeros(len(flags))


def test_one_recovered_sounding_halves_the_run_without_closing():
    flags = np.zeros(160, bool); flags[20:140] = True; flags[80] = False
    verdicts, x, y = _line(flags)

    summary = per_line(verdicts, [100] * 160, x, y)["100"]
    assert summary["longest_out_of_spec_run_distance"] == pytest.approx(590)
    # and the closed figure equals the raw one when no tolerance was given
    assert summary["longest_out_of_spec_run_closed_distance"] == pytest.approx(590)


def test_closing_bridges_the_recovered_sounding():
    flags = np.zeros(160, bool); flags[20:140] = True; flags[80] = False
    verdicts, x, y = _line(flags)

    summary = per_line(verdicts, [100] * 160, x, y, gap_tolerance=100)["100"]
    assert summary["longest_out_of_spec_run_distance"] == pytest.approx(590)
    assert summary["longest_out_of_spec_run_closed_distance"] == pytest.approx(1190)
    assert summary["gap_tolerance"] == 100


def test_the_raw_measurement_is_always_kept():
    """Closing must never hide what was actually flown."""
    flags = np.zeros(60, bool); flags[10:50] = True; flags[30] = False
    verdicts, x, y = _line(flags)

    summary = per_line(verdicts, [100] * 60, x, y, gap_tolerance=500)["100"]
    assert summary["longest_out_of_spec_run_distance"] < (
        summary["longest_out_of_spec_run_closed_distance"])


def test_a_gap_longer_than_the_tolerance_is_not_bridged():
    """Genuinely returning to spec for a while does end the run."""
    flags = np.zeros(100, bool); flags[10:40] = True; flags[60:90] = True
    verdicts, x, y = _line(flags)          # 200 m of in-spec flying between

    summary = per_line(verdicts, [100] * 100, x, y, gap_tolerance=100)["100"]
    assert summary["longest_out_of_spec_run_closed"] == 30


def test_closing_does_not_reach_past_the_ends():
    """Leading and trailing in-spec flying has no run on the far side to join."""
    flags = np.zeros(20, bool); flags[10:] = True
    verdicts, x, y = _line(flags)

    summary = per_line(verdicts, [100] * 20, x, y, gap_tolerance=1000)["100"]
    assert summary["longest_out_of_spec_run_closed"] == 10


def test_closing_does_not_change_the_fraction():
    """Runs are where the gap matters; closing the fraction would overstate."""
    flags = np.zeros(160, bool); flags[20:140] = True; flags[80] = False
    verdicts, x, y = _line(flags)

    raw = per_line(verdicts, [100] * 160, x, y)["100"]
    closed = per_line(verdicts, [100] * 160, x, y, gap_tolerance=100)["100"]
    assert raw["fraction_out_of_spec"] == closed["fraction_out_of_spec"]


def test_closing_still_does_not_cross_the_turn_between_lines():
    """A tolerance large enough to swallow the turn must not join two lines."""
    verdicts = _verdicts([OUT_OF_SPEC] * 6)
    lines = [100] * 3 + [101] * 3
    x = np.array([0.0, 10.0, 20.0, 0.0, 10.0, 20.0])
    y = np.array([0.0] * 3 + [1000.0] * 3)

    summary = per_flight(verdicts, [6] * 6, lines, x, y, gap_tolerance=5000)["6"]
    assert summary["longest_out_of_spec_run_closed"] == 3
