"""filter_05 — altitude expressed against a proper terrain model.

Altitude is distance above the ground; elevation is distance above the geoid.
Nearly every assertion here is about keeping those two apart, because the whole
filter is one subtraction and every way of getting it wrong is silent.
"""

import json

import numpy as np
import pandas as pd
import pytest

# Declared in setup.py, so its absence is an incomplete environment rather than
# an optional feature. Skipped rather than failed so the reason is legible in
# one line instead of nineteen identical tracebacks.
pytest.importorskip("rasterio",
                    reason="rasterio is required by filter_05 and is declared "
                           "in setup.py; this environment does not have it")

from conftest import make_dataset, make_terrain
from ymerflow_processing.core.pipeline import (
    FILTER_GROUP, Pipeline, ProcessingRunner, load_steps,
)
from ymerflow_processing.core.terrain import (
    STASHED_ALTITUDE, STASHED_ELEVATION, apply_terrain_model, sample_terrain,
)

N = 8
UTM = "EPSG:32612"


def delivery(altitude=40.0, elevation=250.0, **kw):
    """Soundings at x = 0, 10, 20 ... y = 0, so they sit inside the fixture."""
    dataset = make_dataset(n=N, n_lines=1, n_gates=4, height=np.full(N, altitude),
                           **kw)
    dataset.flightlines["topo"] = float(elevation)
    return dataset


# ── The definition ───────────────────────────────────────────────────────────

def test_the_instrument_stays_where_it_was_and_altitude_moves(tmp_path):
    """A better ground surface does not move the aircraft. It changes how far
    above the ground the aircraft was."""
    terrain = make_terrain(tmp_path / "dtm.tif", elevation=200.0)
    dataset, _ = apply_terrain_model(delivery(altitude=40.0, elevation=250.0),
                                     terrain, crs=UTM)
    frame = dataset.flightlines

    # Instrument elevation was 250 + 40 = 290. Ground is really 200.
    assert (frame["topo"] == 200.0).all()
    assert (frame["tx_alt"] == 90.0).all()
    assert (frame["topo"] + frame["tx_alt"] == 290.0).all()


def test_a_lower_ground_surface_means_a_greater_altitude(tmp_path):
    high = make_terrain(tmp_path / "high.tif", elevation=240.0)
    low = make_terrain(tmp_path / "low.tif", elevation=180.0)

    over_high, _ = apply_terrain_model(delivery(), high, crs=UTM)
    over_low, _ = apply_terrain_model(delivery(), low, crs=UTM)

    assert over_high.flightlines["tx_alt"].iloc[0] == 50.0
    assert over_low.flightlines["tx_alt"].iloc[0] == 110.0


def test_elevation_may_be_below_the_geoid(tmp_path):
    """Elevation is distance above *or below*; nothing here assumes positive."""
    terrain = make_terrain(tmp_path / "dtm.tif", elevation=-15.0)
    dataset, _ = apply_terrain_model(delivery(altitude=40.0, elevation=0.0),
                                     terrain, crs=UTM)

    assert (dataset.flightlines["topo"] == -15.0).all()
    assert (dataset.flightlines["tx_alt"] == 55.0).all()


def test_preserving_altitude_moves_the_instrument_instead(tmp_path):
    """The alternative, available and deliberately not the default: in a SkyTEM
    delivery both columns come from the same lidar, so holding altitude fixed
    preserves the very error this filter exists to remove."""
    terrain = make_terrain(tmp_path / "dtm.tif", elevation=200.0)
    dataset, report = apply_terrain_model(delivery(altitude=40.0, elevation=250.0),
                                          terrain, crs=UTM, preserve="altitude")
    frame = dataset.flightlines

    assert (frame["tx_alt"] == 40.0).all()
    assert (frame["topo"] == 200.0).all()
    assert (frame["topo"] + frame["tx_alt"] == 240.0).all()   # the bird moved
    assert report.distributions["preserve"] == "altitude"


def test_an_unknown_preserve_is_refused():
    with pytest.raises(ValueError, match="Unknown preserve"):
        apply_terrain_model(delivery(), None, preserve="whatever")


# ── The originals ────────────────────────────────────────────────────────────

def test_the_delivered_values_are_stashed(tmp_path):
    terrain = make_terrain(tmp_path / "dtm.tif", elevation=200.0)
    dataset, _ = apply_terrain_model(delivery(altitude=40.0, elevation=250.0),
                                     terrain, crs=UTM)
    frame = dataset.flightlines

    assert (frame[STASHED_ALTITUDE] == 40.0).all()
    assert (frame[STASHED_ELEVATION] == 250.0).all()


def test_the_stash_names_fit_a_shapefile_field():
    for name in (STASHED_ALTITUDE, STASHED_ELEVATION):
        assert len(name) <= 10


def test_stashing_can_be_declined(tmp_path):
    terrain = make_terrain(tmp_path / "dtm.tif")
    dataset, _ = apply_terrain_model(delivery(), terrain, crs=UTM, stash=False)
    assert STASHED_ALTITUDE not in dataset.flightlines.columns


def test_it_changes_no_measurement_or_uncertainty(tmp_path):
    terrain = make_terrain(tmp_path / "dtm.tif")
    dataset = delivery()
    arrays = {k: v.copy() for k, v in dataset.layer_data.items()}

    dataset, _ = apply_terrain_model(dataset, terrain, crs=UTM)

    for key, array in arrays.items():
        pd.testing.assert_frame_equal(dataset.layer_data[key], array)


# ── Where the model does not reach ───────────────────────────────────────────

def test_a_sounding_outside_the_model_is_unset_not_left_alone(tmp_path):
    """Keeping delivered values would leave one dataset carrying two
    definitions of ground with nothing marking which is which."""
    terrain = make_terrain(tmp_path / "dtm.tif", bounds=(-100, -100, 40, 40))
    dataset, report = apply_terrain_model(delivery(), terrain, crs=UTM)
    frame = dataset.flightlines

    assert np.isnan(frame["tx_alt"].iloc[-1])
    assert np.isnan(frame["topo"].iloc[-1])
    assert frame["tx_alt"].iloc[0] == 90.0
    assert report.distributions["coverage"]["outside_or_nodata"] > 0
    assert any("outside the terrain model" in n for n in report.notes)


def test_the_originals_survive_a_coverage_gap(tmp_path):
    """Nothing is lost — the delivered values are still there to look at."""
    terrain = make_terrain(tmp_path / "dtm.tif", bounds=(-100, -100, 40, 40))
    dataset, _ = apply_terrain_model(delivery(), terrain, crs=UTM)

    assert (dataset.flightlines[STASHED_ALTITUDE] == 40.0).all()


def test_a_nodata_cell_is_treated_as_no_coverage(tmp_path):
    terrain = make_terrain(tmp_path / "dtm.tif", elevation=200.0,
                           bounds=(-100, -100, 1000, 1000),
                           # Origin (-100, 1000), 10 m cells, so soundings at
                           # y=0 land on row 100 and x=0,10,20 on cols 10,11,12.
                           holes=[(100, c) for c in range(10, 13)])
    dataset, report = apply_terrain_model(delivery(), terrain, crs=UTM)

    assert report.distributions["coverage"]["outside_or_nodata"] > 0
    assert dataset.flightlines["topo"].isna().any()


# ── The datum check ──────────────────────────────────────────────────────────

def test_a_datum_mismatch_shows_as_a_large_median_and_small_spread(tmp_path):
    """The whole survey shifted by nearly the same amount — GPS reports height
    above the ellipsoid and terrain models are usually above the geoid."""
    terrain = make_terrain(tmp_path / "dtm.tif", elevation=200.0)
    dataset = delivery(elevation=232.0)                   # a 32 m separation
    dataset.flightlines["topo"] += np.linspace(-0.2, 0.2, N)   # slight noise

    _, report = apply_terrain_model(dataset, terrain, crs=UTM)
    offset = report.distributions["elevation_offset"]

    assert offset["median"] == pytest.approx(32.0, abs=0.2)
    assert offset["robust_spread"] < 1.0


def test_terrain_error_shows_as_a_small_median_and_large_spread(tmp_path):
    """The opposite signature, and the case this filter is actually for."""
    terrain = make_terrain(tmp_path / "dtm.tif", elevation=200.0)
    dataset = delivery(elevation=200.0)
    dataset.flightlines["topo"] += np.linspace(-40.0, 40.0, N)

    _, report = apply_terrain_model(dataset, terrain, crs=UTM)
    offset = report.distributions["elevation_offset"]

    assert abs(offset["median"]) < 1.0
    assert offset["robust_spread"] > 10.0


def test_neither_signature_is_judged(tmp_path):
    """Where the boundary lies depends on the terrain and the model. A house
    number would cry wolf on real relief or wave through a datum error."""
    terrain = make_terrain(tmp_path / "dtm.tif", elevation=200.0)
    _, report = apply_terrain_model(delivery(elevation=280.0), terrain, crs=UTM)

    assert "reading" in report.distributions["elevation_offset"]
    assert not any("datum" in n.lower() for n in report.notes)


# ── CRS ──────────────────────────────────────────────────────────────────────

def test_positions_are_reprojected_when_the_crs_differ(tmp_path):
    """A survey in one projection and a model in another is routine; sampling
    as-is would read from the wrong place entirely."""
    terrain = make_terrain(tmp_path / "dtm.tif", elevation=200.0, crs="EPSG:4326",
                           bounds=(-113.0, 38.0, -112.0, 39.0), cell=0.001)
    easting = np.full(3, 400000.0)
    northing = np.full(3, 4250000.0)

    sampled, raster_crs = sample_terrain(terrain, easting, northing,
                                         source_crs=UTM)
    assert str(raster_crs) == "EPSG:4326"
    assert np.isfinite(sampled).all()
    assert (sampled == 200.0).all()


def test_a_dataset_with_no_crs_is_sampled_as_is_and_told(tmp_path):
    terrain = make_terrain(tmp_path / "dtm.tif")
    _, report = apply_terrain_model(delivery(), terrain)

    assert any("declares no CRS" in n for n in report.notes)
    assert report.distributions["crs"]["survey"] is None


# ── Plumbing ─────────────────────────────────────────────────────────────────

def test_it_runs_from_a_pipeline(tmp_path):
    terrain = make_terrain(tmp_path / "dtm.tif", elevation=200.0)
    dataset, _ = ProcessingRunner(Pipeline.parse(f"""
        kind: filters
        steps:
          - apply_terrain_model:
              terrain: {terrain}
              crs: {UTM}
    """)).run(delivery())

    assert (dataset.flightlines["tx_alt"] == 90.0).all()


def test_an_open_raster_is_accepted_too(tmp_path):
    """R8: external inputs typed rather than as paths, and an opened dataset is
    the typed form."""
    import rasterio
    path = make_terrain(tmp_path / "dtm.tif", elevation=200.0)
    with rasterio.open(path) as handle:
        dataset, _ = apply_terrain_model(delivery(), handle, crs=UTM)
    assert (dataset.flightlines["tx_alt"] == 90.0).all()


def test_a_missing_column_is_an_error(tmp_path):
    terrain = make_terrain(tmp_path / "dtm.tif")
    dataset = delivery()
    dataset.survey.xyz.flightlines = dataset.flightlines.drop(columns=["topo"])

    with pytest.raises(KeyError, match="elevation column"):
        apply_terrain_model(dataset, terrain, crs=UTM)


def test_the_report_survives_serialization(tmp_path):
    terrain = make_terrain(tmp_path / "dtm.tif")
    _, report = apply_terrain_model(delivery(), terrain, crs=UTM)
    restored = json.loads(json.dumps(report.to_dict(), default=str))
    assert restored["step"] == "apply_terrain_model"
    assert restored["distributions"]["preserve"] == "elevation"


def test_registered():
    assert "apply_terrain_model" in load_steps(FILTER_GROUP)
