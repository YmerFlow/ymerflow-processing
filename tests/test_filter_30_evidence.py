"""filter_30 — data too close to a cultural conductor.

The only culling filter whose evidence is not in the data. A powerline crossing
is a clean, coherent, believable response — of a powerline — so every shape test
passes it and an inversion turns it into ground. What identifies it is the map.
"""

import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("geopandas", reason="geopandas is declared in setup.py")

from conftest import make_dataset, make_infrastructure
from ymerflow_processing.core.cull.filter_30_evidence import (
    STEP_CODES, cull_by_external_evidence, distance_to_features,
    gate_distances,
)
from ymerflow_processing.core.misfit import gatewise_misfit
from ymerflow_processing.core.mask import CULLED, IN_USE, provenance_key
from ymerflow_processing.core.pipeline import FILTER_GROUP, load_steps

N = 10
GATES = 8
UTM = "EPSG:32612"


def delivery(**kw):
    """Soundings along y = 0 at x = 0, 10, 20 ... so distance to a line at
    y = 500 is 500 for every one of them."""
    dataset = make_dataset(n=N, n_lines=1, n_gates=GATES, **kw)
    dataset.crs = UTM
    return dataset


def mask(dataset, which=0):
    return dataset.layer_data[dataset.channels[which].inuse_key].to_numpy()


# ── Distance ─────────────────────────────────────────────────────────────────

def test_distance_is_to_the_nearest_feature():
    features = make_infrastructure(lines=[[(0, 100), (1000, 100)],
                                          [(0, 900), (1000, 900)]])
    distance = distance_to_features([0.0, 0.0], [0.0, 800.0], features)

    assert distance[0] == pytest.approx(100.0)
    assert distance[1] == pytest.approx(100.0)     # the far line is nearer here


def test_a_sounding_with_no_position_has_no_distance():
    features = make_infrastructure()
    distance = distance_to_features([0.0, np.nan], [0.0, 0.0], features)

    assert np.isfinite(distance[0])
    assert np.isnan(distance[1])


def test_an_empty_feature_set_leaves_every_distance_unknown():
    import geopandas as gpd
    empty = gpd.GeoDataFrame({"kind": []}, geometry=[], crs=UTM)
    assert np.isnan(distance_to_features([0.0], [0.0], empty)).all()


# ── The exclusion widens with gate time ──────────────────────────────────────

def test_late_gates_are_excluded_from_further_away():
    """01: late-time data senses a larger volume and is affected from further
    away."""
    times = np.logspace(-6, -3, GATES)
    exclusion = gate_distances(100.0, 400.0, times)

    assert exclusion[0] == pytest.approx(100.0)
    assert exclusion[-1] == pytest.approx(400.0)
    assert np.all(np.diff(exclusion) > 0)


def test_one_distance_gives_a_constant_exclusion():
    times = np.logspace(-6, -3, 4)
    assert np.allclose(gate_distances(200.0, 200.0, times), 200.0)


def test_a_sounding_loses_its_late_gates_and_keeps_its_early_ones():
    """The right shape: early gates sense a small volume directly beneath the
    loop and are not reaching the conductor."""
    features = make_infrastructure(lines=[(0, 300), (1000, 300)])
    dataset, _ = cull_by_external_evidence(delivery(), 'proximity', features,
                                          distance_first=100.0,
                                          distance_last=500.0)

    assert (mask(dataset)[:, 0] == IN_USE).all()
    assert (mask(dataset)[:, -1] == CULLED).all()


def test_a_survey_well_clear_of_everything_is_untouched():
    features = make_infrastructure(lines=[(0, 5000), (1000, 5000)])
    dataset, report = cull_by_external_evidence(delivery(), 'proximity', features,
                                               distance_first=100.0,
                                               distance_last=400.0)

    assert (mask(dataset) == IN_USE).all()
    assert report.distributions["removed"] == 0


def test_a_survey_on_top_of_a_feature_loses_everything():
    features = make_infrastructure(lines=[(0, 0), (1000, 0)])
    dataset, _ = cull_by_external_evidence(delivery(), 'proximity', features,
                                          distance_first=50.0)

    assert (mask(dataset) == CULLED).all()


# ── Per moment ───────────────────────────────────────────────────────────────

def test_each_moment_can_have_its_own_exclusion():
    """A high moment at late time senses far more ground than a low moment at
    early time, so one distance for the instrument is wrong for both."""
    features = make_infrastructure(lines=[(0, 300), (1000, 300)])
    dataset, report = cull_by_external_evidence(
        delivery(), 'proximity', features,
        distance_first={"ch1": 50.0, "ch2": 400.0})

    assert (mask(dataset, 0) == IN_USE).all()          # 300 > 50
    assert (mask(dataset, 1) == CULLED).all()          # 300 < 400
    outcome = report.distributions["per_channel"]
    assert outcome["ch1 (LM/z)"]["exclusion"]["first"] == 50.0
    assert outcome["ch2 (HM/z)"]["exclusion"]["first"] == 400.0


def test_a_moment_may_be_named_rather_than_keyed():
    features = make_infrastructure(lines=[(0, 300), (1000, 300)])
    dataset, _ = cull_by_external_evidence(delivery(), 'proximity', features,
                                          distance_first={"HM": 400.0})

    assert (mask(dataset, 0) == IN_USE).all()
    assert (mask(dataset, 1) == CULLED).all()


# ── A type, not a format ─────────────────────────────────────────────────────

def test_it_takes_a_geodataframe():
    """R8's typed form. This filter is the example R8 uses for the problem —
    its old default parameter was literally '/path/to/my/shape/file.shp'."""
    features = make_infrastructure(lines=[(0, 0), (1000, 0)])
    dataset, _ = cull_by_external_evidence(delivery(), 'proximity', features,
                                          distance_first=50.0)
    assert (mask(dataset) == CULLED).all()


@pytest.mark.parametrize("suffix", [".gpkg", ".shp", ".geojson"])
def test_the_format_on_disk_is_not_this_filter_s_business(suffix, tmp_path):
    """It reads through a standard loader and does not care which format it
    was."""
    path = make_infrastructure(tmp_path / f"lines{suffix}",
                               lines=[(0, 0), (1000, 0)])
    dataset, _ = cull_by_external_evidence(delivery(), 'proximity', path,
                                          distance_first=50.0)
    assert (mask(dataset) == CULLED).all()


def test_features_are_reprojected_to_the_survey():
    """A survey in one projection and an infrastructure layer in another is
    routine; using them as-is would place every feature somewhere wrong."""
    features = make_infrastructure(lines=[(0, 0), (1000, 0)], crs=UTM)
    in_degrees = features.to_crs("EPSG:4326")

    dataset, _ = cull_by_external_evidence(delivery(), 'proximity', in_degrees,
                                          distance_first=50.0)
    assert (mask(dataset) == CULLED).all()


def test_geometry_with_no_crs_is_refused():
    """Guessing would put every feature somewhere plausible and wrong."""
    features = make_infrastructure(lines=[(0, 0), (1000, 0)])
    features.crs = None

    with pytest.raises(ValueError, match="declares no CRS"):
        cull_by_external_evidence(delivery(), 'proximity', features, distance_first=50.0)


def test_a_geographic_survey_crs_is_refused():
    """An exclusion of 200 in a geographic CRS is 200 degrees, and nothing in
    the arithmetic would object."""
    dataset = delivery()
    dataset.crs = "EPSG:4326"

    with pytest.raises(ValueError, match="is geographic"):
        cull_by_external_evidence(dataset, 'proximity', make_infrastructure(),
                                 distance_first=200.0)


def test_no_crs_at_all_is_a_note_not_a_refusal():
    dataset = delivery()
    dataset.crs = None
    features = make_infrastructure(lines=[(0, 0), (1000, 0)])

    dataset, report = cull_by_external_evidence(dataset, 'proximity', features,
                                               distance_first=50.0)
    assert any("declares no CRS" in n for n in report.notes)
    assert (mask(dataset) == CULLED).all()


# ── Shared behaviour ─────────────────────────────────────────────────────────

def test_it_changes_no_measurement_or_uncertainty():
    features = make_infrastructure(lines=[(0, 0), (1000, 0)])
    dataset = delivery()
    arrays = {k: v.copy() for k, v in dataset.layer_data.items()
              if "inuse" not in k.lower()}

    dataset, _ = cull_by_external_evidence(dataset, 'proximity', features,
                                          distance_first=50.0)
    for key, array in arrays.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


def test_it_is_idempotent():
    features = make_infrastructure(lines=[(0, 0), (1000, 0)])
    dataset, _ = cull_by_external_evidence(delivery(), 'proximity', features,
                                          distance_first=50.0)
    snapshot = {k: v.copy() for k, v in dataset.layer_data.items()}
    dataset, second = cull_by_external_evidence(dataset, 'proximity', features,
                                               distance_first=50.0)

    for key, array in snapshot.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)
    assert second.distributions["removed"] == 0


def test_a_sounding_with_no_position_is_not_culled():
    """Where it is has not been established, which is a different problem from
    being near something."""
    features = make_infrastructure(lines=[(0, 0), (1000, 0)])
    dataset = delivery()
    dataset.flightlines.loc[3, "x"] = np.nan

    dataset, report = cull_by_external_evidence(dataset, 'proximity', features,
                                               distance_first=50.0)
    assert (mask(dataset)[3] == IN_USE).all()
    assert any("no position" in n for n in report.notes)


def test_the_cull_is_attributed_to_this_filter():
    features = make_infrastructure(lines=[(0, 0), (1000, 0)])
    dataset, _ = cull_by_external_evidence(delivery(), 'proximity', features,
                                          distance_first=50.0)
    why = dataset.layer_data[provenance_key(dataset.channels[0].inuse_key)]

    assert (why.to_numpy() == STEP_CODES['proximity']).all()
    assert STEP_CODES['proximity'] == 30


def test_the_distance_distribution_is_reported():
    """The number someone needs to choose an exclusion."""
    features = make_infrastructure(lines=[(0, 300), (1000, 300)])
    _, report = cull_by_external_evidence(delivery(), 'proximity', features,
                                         distance_first=50.0)

    assert report.distributions["distance_to_nearest"]["mean"] == pytest.approx(300.0)
    assert report.distributions["features"] == 1


def test_the_report_survives_serialization():
    features = make_infrastructure(lines=[(0, 0), (1000, 0)])
    _, report = cull_by_external_evidence(delivery(), 'proximity', features,
                                         distance_first=50.0)
    restored = json.loads(json.dumps(report.to_dict(), default=str))
    assert restored["distributions"]["provenance_code"] == 30


def test_registered():
    assert "cull_by_external_evidence" in load_steps(FILTER_GROUP)


# ── Misfit, and the conjunction (filter_31) ──────────────────────────────────

def forward(dataset, scale=1.0, gates=None):
    """A forward synthetic: the measured data, optionally missed at some gates."""
    from libaarhusxyz.xyz import XYZ

    layers = {}
    for channel in dataset.channels:
        predicted = dataset.layer_data[channel.data_key].to_numpy().copy()
        if gates is not None:
            predicted[:, gates] *= scale
        layers[channel.data_key] = pd.DataFrame(predicted)
    return XYZ({"flightlines": dataset.flightlines.copy(),
                "layer_data": layers, "model_info": {}, "file_meta": {}})


def test_the_misfit_is_measured_minus_synthetic_over_uncertainty():
    """The quantity is data - synthetic. The threshold is on it divided by
    uncertainty, which is what makes one number work across a transient
    spanning orders of magnitude."""
    dataset = delivery()
    dataset.layer_data[dataset.channels[0].std_key].iloc[:, :] = 0.10
    synthetic = forward(dataset, scale=0.8, gates=[5])   # 20% low at gate 5

    misfit = gatewise_misfit(dataset, synthetic, dataset.channels[0])

    # A 20% shortfall against a 10% sigma is two standard deviations.
    assert misfit[0, 5] == pytest.approx(2.0, rel=1e-6)
    assert misfit[0, 0] == pytest.approx(0.0)


def test_a_stated_residual_is_used_where_the_result_carries_one():
    """An inversion writes (dobs - dpred) * W.diagonal() itself."""
    from libaarhusxyz.xyz import XYZ
    dataset = delivery()
    stated = np.full((N, GATES), 7.0)
    result = XYZ({"flightlines": dataset.flightlines.copy(),
                  "layer_data": {"dbdt_err_ch1gt": pd.DataFrame(stated)},
                  "model_info": {}, "file_meta": {}})

    assert np.allclose(gatewise_misfit(dataset, result, dataset.channels[0]),
                       7.0)


def test_predictions_with_no_uncertainty_to_normalize_by_are_refused():
    """A raw difference cannot be thresholded across a transient spanning
    orders of magnitude."""
    dataset = delivery()
    synthetic = forward(dataset)
    del dataset.layer_data[dataset.channels[0].std_key]

    with pytest.raises(ValueError, match="raw difference cannot be"):
        gatewise_misfit(dataset, synthetic, dataset.channels[0])


def test_misfit_alone_culls_the_badly_fit_gates():
    dataset = delivery()
    for channel in dataset.channels:
        dataset.layer_data[channel.std_key].iloc[:, :] = 0.10
    synthetic = forward(dataset, scale=0.5, gates=[6, 7])

    dataset, report = cull_by_external_evidence(
        dataset, "misfit", result=synthetic, max_misfit=3.0)

    assert (mask(dataset)[:, 6:] == CULLED).all()
    assert (mask(dataset)[:, 0] == IN_USE).all()
    assert report.distributions["provenance_code"] == 31


def test_misfit_alone_says_what_01_says_about_it():
    """A poor fit may be genuine geology a 1D model cannot represent, and
    removing it removes the evidence that the model was wrong."""
    dataset = delivery()
    synthetic = forward(dataset)
    _, report = cull_by_external_evidence(dataset, "misfit", result=synthetic,
                                          max_misfit=3.0)

    assert any("over-culls" in n for n in report.notes)


def test_both_is_more_selective_than_either():
    """01: proximity alone over-culls, misfit alone over-culls, the conjunction
    is more selective than either."""
    def prepared():
        dataset = delivery()
        for channel in dataset.channels:
            dataset.layer_data[channel.std_key].iloc[:, :] = 0.10
        return dataset

    far = make_infrastructure(lines=[(0, 5000), (1000, 5000)])
    near = make_infrastructure(lines=[(0, 0), (1000, 0)])

    # Badly fit everywhere, but the line is nowhere near a feature.
    dataset = prepared()
    dataset, both = cull_by_external_evidence(
        dataset, "both", infrastructure=far, distance_first=100.0,
        result=forward(dataset, scale=0.5, gates=[6, 7]), max_misfit=3.0)
    assert both.distributions["removed"] == 0

    # The same misfit, now beside a feature.
    dataset = prepared()
    dataset, conjunction = cull_by_external_evidence(
        dataset, "both", infrastructure=near, distance_first=100.0,
        result=forward(dataset, scale=0.5, gates=[6, 7]), max_misfit=3.0)
    assert conjunction.distributions["removed"] > 0

    # And near the feature but fitting well, nothing goes either.
    dataset = prepared()
    dataset, fits = cull_by_external_evidence(
        dataset, "both", infrastructure=near, distance_first=100.0,
        result=forward(dataset), max_misfit=3.0)
    assert fits.distributions["removed"] == 0


def test_each_criterion_carries_only_its_own_parameters():
    """R7: nothing in the signature is dead depending on a flag."""
    with pytest.raises(ValueError, match="needs infrastructure and"):
        cull_by_external_evidence(delivery(), "proximity")
    with pytest.raises(ValueError, match="needs max_misfit"):
        cull_by_external_evidence(delivery(), "misfit")
    with pytest.raises(ValueError, match="needs max_misfit"):
        cull_by_external_evidence(delivery(), "both",
                                  infrastructure=make_infrastructure(),
                                  distance_first=100.0)


def test_an_unknown_criterion_names_the_available_ones():
    with pytest.raises(ValueError, match="proximity, misfit, both"):
        cull_by_external_evidence(delivery(), "nearby")


def test_a_non_positive_misfit_threshold_is_refused():
    with pytest.raises(ValueError, match="must be positive"):
        cull_by_external_evidence(delivery(), "misfit",
                                  result=forward(delivery()), max_misfit=0)


def test_proximity_and_the_conjunction_are_attributed_separately():
    """The report records the criterion, but the provenance array is what
    someone reads six months later."""
    assert STEP_CODES["proximity"] == 30
    assert STEP_CODES["misfit"] == STEP_CODES["both"] == 31


# ── The misfit column, and creating one where it can ─────────────────────────

def test_it_thresholds_a_column_when_one_exists():
    """01's resolution: filter_31 stops being special and becomes an ordinary
    culling filter thresholding a column."""
    from ymerflow_processing.core.misfit import compute_misfit

    dataset = delivery()
    for channel in dataset.channels:
        dataset.layer_data[channel.std_key].iloc[:, :] = 0.10
    dataset, _ = compute_misfit(dataset, forward(dataset, scale=0.5,
                                                 gates=[6, 7]))

    # No result passed — the column is all it needs now.
    dataset, report = cull_by_external_evidence(dataset, "misfit",
                                                max_misfit=3.0)
    assert (mask(dataset)[:, 6:] == CULLED).all()
    assert report.distributions["removed"] > 0


def test_it_creates_the_column_when_there_is_none_and_it_can():
    """A caller with the result to hand should not have to run a separate step
    to make one appear."""
    from ymerflow_processing.core.misfit import misfit_key

    dataset = delivery()
    for channel in dataset.channels:
        dataset.layer_data[channel.std_key].iloc[:, :] = 0.10
    key = misfit_key(dataset.channels[0])
    assert key not in dataset.layer_data

    dataset, _ = cull_by_external_evidence(
        dataset, "misfit", result=forward(dataset, scale=0.5, gates=[6, 7]),
        max_misfit=3.0)

    assert key in dataset.layer_data          # written, not just used


def test_a_created_column_is_not_recomputed_by_the_next_consumer():
    dataset = delivery()
    for channel in dataset.channels:
        dataset.layer_data[channel.std_key].iloc[:, :] = 0.10
    synthetic = forward(dataset, scale=0.5, gates=[6, 7])

    dataset, _ = cull_by_external_evidence(dataset, "misfit", result=synthetic,
                                           max_misfit=3.0)
    # Second pass with no result at all: the column carries it.
    dataset, second = cull_by_external_evidence(dataset, "misfit",
                                                max_misfit=3.0)
    assert second.distributions["removed"] == 0     # idempotent, and it ran


def test_with_neither_a_column_nor_a_result_it_says_both_ways_out():
    dataset, report = cull_by_external_evidence(delivery(), "misfit",
                                                max_misfit=3.0)

    assert report.distributions["removed"] == 0
    assert any("pass result=" in n and "compute_misfit" in n
               for n in report.notes)
