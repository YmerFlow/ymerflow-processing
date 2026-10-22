"""qc_09 — noise floor drift, the first comparison check.

The shape 02 adds for things only visible cumulatively. One delivery's floor is
a number; twenty deliveries' floors are a diagnosis, and no per-run check can
see the second.

Most of what follows is about the *ordering*, because a series trended in the
wrong order produces a plausible slope and nothing gives it away.
"""

import json

import numpy as np
import pytest

from ymerflow_processing.compare.drift import (
    _flight_key, noise_floor_drift,
)
from ymerflow_processing.pipeline import (
    COMPARE_GROUP, ComparisonRunner, Pipeline, load_steps,
)


def stored(level, identity="d", flights=None, when=None, supplied=(),
           step="set_noise_floor"):
    """A stored run report of the shape set_noise_floor produces."""
    return {
        "identity": identity,
        "flights": flights,
        "time_range": when,
        "steps": [{"step": step,
                   "distributions": {"model": {"level_1ms": level,
                                               "exponent": -0.5,
                                               "supplied": list(supplied)}}}],
    }


def series(report):
    return [entry["delivery"] for entry in report.distributions["series"]]


# ── The series ───────────────────────────────────────────────────────────────

def test_it_trends_a_rising_floor():
    reports = [stored(1e-9 * 1.2 ** n, identity=f"d{n}", flights=[n])
               for n in range(6)]
    report = noise_floor_drift(reports)
    trend = report.distributions["trend"]

    assert trend["fractional_change_per_delivery"] == pytest.approx(0.2, abs=0.01)
    assert trend["total_change"] == pytest.approx(1.2 ** 5, rel=1e-6)


def test_a_steady_survey_trends_flat():
    report = noise_floor_drift([stored(1e-9, flights=[n]) for n in range(6)])
    assert report.distributions["trend"][
        "fractional_change_per_delivery"] == pytest.approx(0.0, abs=1e-9)


def test_the_trend_is_fitted_in_log_space():
    """A floor is a scale: 5e-10 to 1e-9 is the same event as 1e-9 to 2e-9, and
    a linear fit would call the second twenty times larger."""
    low = noise_floor_drift([stored(5e-10 * 2 ** n, flights=[n])
                             for n in range(4)])
    high = noise_floor_drift([stored(1e-8 * 2 ** n, flights=[n])
                              for n in range(4)])

    assert (low.distributions["trend"]["fractional_change_per_delivery"]
            == pytest.approx(
                high.distributions["trend"]["fractional_change_per_delivery"]))


def test_the_largest_single_step_is_reported_separately():
    """A trend can be small while one step is enormous, and averaging hides
    exactly the delivery someone needs to look at."""
    reports = [stored(1e-9, identity="a", flights=[1]),
               stored(1.01e-9, identity="b", flights=[2]),
               stored(8e-9, identity="c", flights=[3]),
               stored(8.05e-9, identity="d", flights=[4])]
    report = noise_floor_drift(reports)
    step = report.distributions["largest_step"]

    assert step["between"] == ["b", "c"]
    assert step["largest_ratio"] == pytest.approx(8e-9 / 1.01e-9, rel=1e-6)


# ── Ordering, which is the part that goes silently wrong ─────────────────────

def test_a_time_range_orders_the_series():
    reports = [stored(3e-9, identity="third", when=["2026-05-13"]),
               stored(1e-9, identity="first", when=["2026-05-11"]),
               stored(2e-9, identity="second", when=["2026-05-12"])]
    report = noise_floor_drift(reports)

    assert series(report) == ["first", "second", "third"]
    assert report.distributions["ordered_by"] == "time_range"


def test_flight_numbers_order_numerically_not_as_text():
    """Sorted as text "10" comes before "9", so a sequentially numbered survey
    would trend its tenth flight before its ninth and shift every delivery
    after it. Nothing in the numbers would give that away."""
    reports = [stored(1e-9, identity="ninth", flights=[9]),
               stored(2e-9, identity="tenth", flights=[10]),
               stored(3e-9, identity="hundredth", flights=[100])]
    report = noise_floor_drift(reports)

    assert series(report) == ["ninth", "tenth", "hundredth"]
    assert report.distributions["ordered_by"] == "flight number"


def test_a_series_mixing_ordering_keys_is_labelled_by_the_weakest():
    """One dated delivery among flight-numbered ones does not make the series
    chronological. The label came from the first entry, which is the one with
    the strongest key, so a mixed series read as ordered by time range."""
    reports = [stored(1e-9, identity="dated", when=["2026-05-11"], flights=[3]),
               stored(2e-9, identity="f1", flights=[1]),
               stored(3e-9, identity="f2", flights=[2])]
    report = noise_floor_drift(reports)

    assert report.distributions["ordered_by"] == "flight number"
    assert any("mixes" in n for n in report.notes)


def test_a_date_like_flight_number_also_orders_correctly():
    """SkyTEM writes 26051201 — a date with an index. Numeric ordering handles
    it without knowing that is what it is."""
    reports = [stored(2e-9, identity="later", flights=[26051201]),
               stored(1e-9, identity="earlier", flights=[26051105])]
    assert series(noise_floor_drift(reports)) == ["earlier", "later"]


def test_a_delivery_is_placed_by_its_earliest_flight():
    """One delivered file commonly holds three or four flights."""
    reports = [stored(2e-9, identity="second", flights=[20, 21, 22]),
               stored(1e-9, identity="first", flights=[17, 18, 19])]
    assert series(noise_floor_drift(reports)) == ["first", "second"]


def test_a_non_numeric_scheme_still_orders_rather_than_raising():
    assert _flight_key(["A9", "A10"]) == (1, "A10")


def test_a_time_range_beats_a_flight_number():
    """A time is chronological by construction; a flight number is
    chronological by convention. Where both exist they agree anyway."""
    reports = [stored(1e-9, identity="a", flights=[99], when=["2026-05-11"]),
               stored(2e-9, identity="b", flights=[1], when=["2026-05-12"])]

    report = noise_floor_drift(reports)
    assert series(report) == ["a", "b"]
    assert report.distributions["ordered_by"] == "time_range"


def test_ordering_by_flight_number_says_it_is_a_convention():
    report = noise_floor_drift([stored(1e-9, flights=[n]) for n in range(3)])
    assert any("convention rather than a measurement" in n
               for n in report.notes)


def test_with_neither_the_order_given_is_used_and_said():
    report = noise_floor_drift([stored(1e-9), stored(2e-9)])

    assert report.distributions["ordered_by"] == "the order given"
    assert any("nothing more" in n for n in report.notes)


def test_the_identity_never_orders_anything():
    """Delivery names routinely embed an export date, so ordering by one would
    enter a single delivery into the series twice after a re-export."""
    reports = [stored(1e-9, identity="zzz_20260511", flights=[1]),
               stored(2e-9, identity="aaa_20260512", flights=[2])]
    assert series(noise_floor_drift(reports)) == ["zzz_20260511",
                                                  "aaa_20260512"]


# ── The baseline ─────────────────────────────────────────────────────────────

def test_the_baseline_is_the_median_without_a_reference():
    """Robust to the outlier being looked for."""
    reports = [stored(1e-9, flights=[n]) for n in range(4)]
    reports.append(stored(1e-6, identity="bad", flights=[9]))

    report = noise_floor_drift(reports)
    assert report.distributions["baseline"]["level_1ms"] == pytest.approx(1e-9)
    assert report.distributions["baseline"]["from"] == "series median"


def test_a_reference_becomes_the_baseline():
    report = noise_floor_drift([stored(2e-9, flights=[1])],
                               reference=stored(1e-9))

    assert report.distributions["baseline"]["from"] == "reference"
    assert report.distributions["series"][0][
        "ratio_to_baseline"] == pytest.approx(2.0)


def test_a_reference_with_no_model_falls_back_and_says_so():
    report = noise_floor_drift([stored(1e-9, flights=[1])],
                               reference={"steps": []})
    assert any("baseline is the series median" in n for n in report.notes)


# ── Judging, only when asked ─────────────────────────────────────────────────

def test_without_a_threshold_it_measures_and_says_nothing():
    report = noise_floor_drift([stored(1e-9, flights=[n]) for n in range(3)])

    assert report.distributions["exceeded"] == []
    assert any("left unjudged" in n for n in report.notes)


def test_an_outlier_is_named_when_a_threshold_is_given():
    reports = [stored(1e-9, identity=f"d{n}", flights=[n]) for n in range(4)]
    reports.append(stored(1e-7, identity="noisy", flights=[9]))

    report = noise_floor_drift(reports, max_ratio=3.0)
    assert report.distributions["exceeded"] == ["noisy"]
    assert any("cannot tell which" in n for n in report.notes)


def test_a_floor_far_below_the_baseline_is_flagged_too():
    """Either direction: a floor that collapses is as much a change as one that
    rises."""
    reports = [stored(1e-9, identity=f"d{n}", flights=[n]) for n in range(4)]
    reports.append(stored(1e-12, identity="quiet", flights=[9]))

    report = noise_floor_drift(reports, max_ratio=3.0)
    assert report.distributions["exceeded"] == ["quiet"]


def test_a_ratio_of_one_or_less_is_refused():
    with pytest.raises(ValueError, match="must exceed 1"):
        noise_floor_drift([stored(1e-9)], max_ratio=1.0)


# ── What it will not silently trend ──────────────────────────────────────────

def test_a_supplied_floor_is_named_rather_than_trended_quietly():
    """An estimate and a number someone typed are different evidence, and a
    series mixing them trends the typing as much as the data."""
    reports = [stored(1e-9, identity="measured", flights=[1]),
               stored(5e-9, identity="typed", flights=[2],
                      supplied=["level_1ms"])]

    report = noise_floor_drift(reports)
    assert any("supplied rather than estimated" in n for n in report.notes)
    assert any("typed" in n for n in report.notes)


def test_a_report_with_no_model_is_excluded_and_named():
    reports = [stored(1e-9, identity="has", flights=[1]),
               {"identity": "none", "steps": []}]

    report = noise_floor_drift(reports)
    assert series(report) == ["has"]
    assert any("none" in n and "carry no" in n for n in report.notes)


def test_no_models_at_all_is_a_note_not_a_crash():
    report = noise_floor_drift([{"identity": "a", "steps": []}])
    assert report.distributions == {}
    assert any("no series" in n for n in report.notes)


def test_no_reports_at_all_is_a_note():
    report = noise_floor_drift([])
    assert any("nothing to trend" in n for n in report.notes)


# ── Plumbing ─────────────────────────────────────────────────────────────────

def test_it_runs_from_the_comparison_runner():
    reports = [stored(1e-9 * 1.5 ** n, identity=f"d{n}", flights=[n])
               for n in range(4)]
    result = ComparisonRunner(Pipeline.parse("""
        kind: compare
        steps:
          - noise_floor_drift: {max_ratio: 4.0}
    """)).run(reports)

    assert [s.step for s in result.steps] == ["noise_floor_drift"]
    assert result.steps[0].params == {"max_ratio": 4.0}


def test_a_pair_is_simply_a_two_item_sequence():
    """02: items subsumes pairwise comparison without a separate shape."""
    report = noise_floor_drift([stored(1e-9, identity="before", flights=[1]),
                               stored(4e-9, identity="after", flights=[2])])
    assert report.distributions["trend"]["total_change"] == pytest.approx(4.0)


def test_the_report_survives_serialization():
    report = noise_floor_drift([stored(1e-9, flights=[n]) for n in range(3)],
                               max_ratio=2.0)
    restored = json.loads(json.dumps(report.to_dict(), default=str))
    assert restored["step"] == "noise_floor_drift"
    assert len(restored["distributions"]["series"]) == 3


def test_registered():
    assert "noise_floor_drift" in load_steps(COMPARE_GROUP)


# ── The assumption is checked, not asserted ──────────────────────────────────

def test_a_duplicated_flight_number_is_called_out():
    """A re-export entered twice, or a contractor restarting their numbering.
    Either way the series trends one flight more than once."""
    reports = [stored(1e-9, identity="first", flights=[7]),
               stored(2e-9, identity="again", flights=[7])]

    report = noise_floor_drift(reports)
    assert any("more than one delivery" in n for n in report.notes)
    assert any("first" in n and "again" in n for n in report.notes)


def test_numbering_that_disagrees_with_the_given_order_is_called_out():
    """items is ordered by contract, so if the caller passed deliveries
    chronologically and the flight numbers disagree, the numbering is not
    chronological. That is the case this exists to catch."""
    reports = [stored(1e-9, identity="flown first", flights=[50]),
               stored(2e-9, identity="flown second", flights=[3])]

    report = noise_floor_drift(reports)
    assert any("disagrees with the order the reports were given" in n
               for n in report.notes)
    # And it still orders by the numbers — it reports the conflict rather than
    # silently picking one.
    assert series(report) == ["flown second", "flown first"]


def test_agreeing_numbering_says_nothing():
    """A warning that fires on correct use is worse than none."""
    reports = [stored(1e-9, identity="a", flights=[1]),
               stored(2e-9, identity="b", flights=[2])]

    report = noise_floor_drift(reports)
    assert not any("disagrees" in n for n in report.notes)
    assert not any("more than one delivery" in n for n in report.notes)


def test_a_non_numeric_scheme_is_called_out():
    reports = [stored(1e-9, identity="a", flights=["A9"]),
               stored(2e-9, identity="b", flights=["A10"])]

    report = noise_floor_drift(reports)
    assert any("ordered as text" in n for n in report.notes)


def test_the_checks_do_not_run_when_time_ordered():
    """There is nothing to check: the ordering did not rest on the assumption."""
    reports = [stored(1e-9, identity="a", flights=[50], when=["2026-05-11"]),
               stored(2e-9, identity="b", flights=[3], when=["2026-05-12"])]

    report = noise_floor_drift(reports)
    assert not any("disagrees" in n for n in report.notes)
