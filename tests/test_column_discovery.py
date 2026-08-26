"""Finding the columns a real delivery actually uses.

Every case here comes from one real SkyTEM delivery, the first this library was
pointed at. Three of nine QC checks found nothing to grade, and a fourth graded
the wrong quantity — and none of it failed. The checks reported politely that
columns were absent, the run exited 0, and a reader would have concluded the
survey was flown badly.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import make_dataset
from ymerflow_processing.core.attitude import (
    DEFAULT_PITCH, DEFAULT_ROLL, resolve_attitude_columns,
)
from ymerflow_processing.core.dataset import (
    HEIGHT_COLUMNS, PITCH_COLUMNS, ROLL_COLUMNS, find_column,
)
from ymerflow_processing.core.qc.qc_01_height import height_above_ground
from ymerflow_processing.core.qc.qc_02_attitude import system_attitude
from ymerflow_processing.core.qc.qc_04_current import find_current_columns

SPEC = {"good_max": 45, "marginal_max": 65}


def renamed(**mapping):
    """A delivery whose columns are spelled the way the real one spells them."""
    dataset = make_dataset(n=12, n_lines=1, n_gates=5,
                           height=np.full(12, 35.0))
    dataset.survey.xyz.flightlines = dataset.flightlines.rename(columns=mapping)
    return dataset


# ── Case, and why variants are not enough ────────────────────────────────────

def test_an_interior_capital_is_found():
    """The specific gap. libaarhusxyz reaches these by generating lower, upper
    and title variants of 'anglex' — none of which is 'AngleX', which is what
    the delivery spells it. So attitude silently vanished from the run."""
    dataset = renamed(tilt_x="AngleX", tilt_y="AngleY")
    frame = dataset.flightlines

    assert find_column(frame, PITCH_COLUMNS) == "AngleX"
    assert find_column(frame, ROLL_COLUMNS) == "AngleY"


def test_the_attitude_check_now_grades_a_real_spelling():
    dataset = renamed(tilt_x="AngleX", tilt_y="AngleY")
    dataset, report = system_attitude(dataset, good=5, marginal=7.5)

    assert report.verdict_column == "qs_tilt"
    assert not any("No attitude columns" in n for n in report.notes)


def test_tilt_correction_finds_them_too():
    """Not only the check. correct_tilt defaulted to the literal name 'tilt_x',
    so on this delivery it would have raised rather than corrected."""
    dataset = renamed(tilt_x="AngleX", tilt_y="AngleY")
    assert resolve_attitude_columns(dataset, None, None) == ("AngleX", "AngleY")


def test_absent_attitude_still_falls_back_to_the_canonical_name():
    """zero_attitude CREATES these columns where a delivery carries none, so it
    needs a name to create them under."""
    dataset = make_dataset(n=4, n_lines=1, n_gates=4)
    dataset.survey.xyz.flightlines = dataset.flightlines.drop(
        columns=["tilt_x", "tilt_y"])

    assert resolve_attitude_columns(dataset, None, None) == (DEFAULT_PITCH,
                                                             DEFAULT_ROLL)


def test_a_named_column_still_wins():
    dataset = renamed(tilt_x="AngleX", tilt_y="AngleY")
    assert resolve_attitude_columns(dataset, "Lat", "Lon") == ("Lat", "Lon")


# ── Height is not elevation ──────────────────────────────────────────────────

def test_the_specific_spelling_beats_the_generic_one():
    """The delivery carries BOTH: 'Height' at tens of metres above ground and
    'Alt' at hundreds above the geoid. Preferring 'alt' graded every sounding
    against elevation."""
    dataset = make_dataset(n=6, n_lines=1, n_gates=4, height=np.full(6, 40.0))
    # The real delivery carries no tx_alt at all — only these two.
    frame = dataset.flightlines.drop(columns=["tx_alt"])
    frame["Alt"] = 450.0
    frame["Height"] = 40.0

    assert find_column(frame, HEIGHT_COLUMNS) == "Height"


def test_grading_elevation_is_called_out_rather_than_reported_as_a_bad_survey():
    """8995 soundings out of spec is what this looked like, and nothing said
    the input was the wrong quantity."""
    dataset = make_dataset(n=20, n_lines=1, n_gates=4, height=np.full(20, 40.0))
    frame = dataset.flightlines
    frame["topo"] = 400.0
    frame["tx_alt"] = 440.0          # elevation: ground plus a flying height

    dataset, report = height_above_ground(dataset, column="tx_alt", **SPEC)
    assert any("looks like ELEVATION" in n for n in report.notes)


def test_it_stays_quiet_on_a_genuine_height():
    """A warning that fires on correct use is worse than none."""
    dataset = make_dataset(n=20, n_lines=1, n_gates=4, height=np.full(20, 40.0))
    dataset.flightlines["topo"] = 400.0

    dataset, report = height_above_ground(dataset, **SPEC)
    assert not any("ELEVATION" in n for n in report.notes)


def test_it_stays_quiet_when_there_is_nothing_to_compare_against():
    dataset = make_dataset(n=8, n_lines=1, n_gates=4, height=np.full(8, 40.0))
    dataset.survey.xyz.flightlines = dataset.flightlines.drop(columns=["topo"])

    dataset, report = height_above_ground(dataset, **SPEC)
    assert not any("ELEVATION" in n for n in report.notes)


def test_all_nan_heights_do_not_warn_about_a_nan_slice():
    """A delivery with no heights is a legitimate input, not a mistake."""
    dataset = make_dataset(n=8, n_lines=1, n_gates=4)
    dataset.flightlines["tx_alt"] = np.nan
    dataset.flightlines["topo"] = 400.0

    with np.errstate(all="raise"):
        dataset, report = height_above_ground(dataset, **SPEC)
    assert not any("ELEVATION" in n for n in report.notes)


def test_the_column_it_read_is_in_the_report():
    """R3: the parameters actually used, including the ones nobody supplied.
    Not recording it is why reading the wrong one took three steps to find."""
    dataset = make_dataset(n=6, n_lines=1, n_gates=4, height=np.full(6, 40.0))
    _, report = height_above_ground(dataset, **SPEC)

    assert report.distributions["height"]["column"] == "tx_alt"


# ── Transmitter current ──────────────────────────────────────────────────────

def test_an_abbreviated_per_moment_current_is_found():
    """The delivery spells them Curr_LM and Curr_HM."""
    dataset = make_dataset(n=6, n_lines=1, n_gates=4)
    frame = dataset.flightlines
    frame["Curr_LM"] = 8.7
    frame["Curr_HM"] = 111.8

    found = find_current_columns(frame, dataset.channels)
    assert found == {"ch1": "Curr_LM", "ch2": "Curr_HM"}


def test_a_generic_current_is_still_refused_when_two_moments_could_claim_it():
    """With two moments, 'current' is ambiguous and guessing would be worse
    than reporting nothing."""
    dataset = make_dataset(n=6, n_lines=1, n_gates=4)
    dataset.flightlines["current"] = 10.0

    assert find_current_columns(dataset.flightlines, dataset.channels) == {}
