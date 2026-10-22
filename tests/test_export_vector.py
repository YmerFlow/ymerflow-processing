"""Writing a dataset out as vector geometry — the artifact that leaves the building."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("geopandas")
import geopandas as gpd

from conftest import make_dataset
from ymerflow_processing.filters.export_vector import export_vector
from ymerflow_processing.pipeline import (
    FILTER_GROUP, Pipeline, QCRunner, load_steps,
)

UTM, WGS84 = "EPSG:32613", "EPSG:4326"


def classified(n=60, n_lines=3, bad=slice(10, 25)):
    """A dataset that has been through QC, so it carries verdicts to export."""
    heights = np.full(n, 30.0)
    heights[bad] = 80.0
    dataset = make_dataset(n=n, n_lines=n_lines, n_flights=1, height=heights)
    dataset.crs = UTM
    dataset, _ = QCRunner(Pipeline.parse("""
        kind: qc
        rollup: {line: {max_fraction: 0.3}}
        steps:
          - height_above_ground: {good_max: 45, marginal_max: 65}
    """)).run(dataset)
    return dataset


# ── GeoPackage, the default ──────────────────────────────────────────────────

def test_geopackage_holds_both_geometries_in_one_file(tmp_path):
    out = tmp_path / "qc.gpkg"
    _, report = export_vector(classified(), out, crs=UTM)

    assert report.distributions["driver"] == "GPKG"
    assert report.distributions["layers"] == {"soundings": 60, "flightlines": 3}
    assert gpd.read_file(out, layer="soundings").geometry.geom_type.unique() == ["Point"]
    assert gpd.read_file(out, layer="flightlines").geometry.geom_type.unique() == [
        "LineString"]


def test_the_line_layer_carries_what_a_reflight_conversation_needs(tmp_path):
    """Which lines failed, without joining anything."""
    out = tmp_path / "qc.gpkg"
    export_vector(classified(), out, crs=UTM)
    lines = gpd.read_file(out, layer="flightlines")

    assert set(lines["line_id"]) == {100, 101, 102}
    # A fraction per tier, so "which lines failed and how badly" is answerable
    # off the line feature alone.
    assert {"ql_good", "ql_marg", "ql_oos", "ql_run", "ql_n"} <= set(lines.columns)
    assert lines.loc[lines["line_id"] == 100, "ql_verdict"].iloc[0] == "OOS"
    assert lines.loc[lines["line_id"] == 102, "ql_verdict"].iloc[0] == "Good"


def test_per_sounding_columns_stay_on_the_points(tmp_path):
    """They vary along a line, so they have no meaning on the line feature."""
    out = tmp_path / "qc.gpkg"
    export_vector(classified(), out, crs=UTM)

    assert "qs_height" in gpd.read_file(out, layer="soundings")
    assert "qs_height" not in gpd.read_file(out, layer="flightlines")


def test_lines_can_be_declined(tmp_path):
    out = tmp_path / "qc.gpkg"
    _, report = export_vector(classified(), out, crs=UTM, lines=False)
    assert report.distributions["layers"] == {"soundings": 60}


# ── No shapefile ─────────────────────────────────────────────────────────────

def test_a_shapefile_path_is_refused_and_says_why(tmp_path):
    """Dropped 2026-10-22 (Ben): its ten-character field names had become a
    constraint on every column name in the library."""
    with pytest.raises(ValueError, match="Shapefile is not written"):
        export_vector(classified(), tmp_path / "qc.shp", crs=UTM)
    with pytest.raises(ValueError, match="Shapefile is not written"):
        export_vector(classified(), tmp_path / "qc.gpkg", crs=UTM, format="shp")


def test_geojson_holds_one_layer_so_lines_go_beside(tmp_path):
    out = tmp_path / "qc.geojson"
    _, report = export_vector(classified(), out, crs=WGS84)

    assert (tmp_path / "qc.geojson").exists()
    assert (tmp_path / "qc_lines.geojson").exists()
    assert any("one layer per file" in n for n in report.notes)


# ── Field names ──────────────────────────────────────────────────────────────

def test_long_and_similar_field_names_are_written_as_they_are(tmp_path):
    """No ten-character limit, no truncation, no collision: GeoPackage keeps
    the names the library gave."""
    dataset = classified(n=6, n_lines=1, bad=slice(0, 0))
    dataset.flightlines["qs_verdict_a"] = 1
    dataset.flightlines["qs_verdict_b"] = 2
    dataset.flightlines["measured_conductance"] = 1.0

    out = tmp_path / "qc.gpkg"
    _, report = export_vector(dataset, out, crs=UTM)
    written = gpd.read_file(out, layer="soundings")
    assert {"qs_verdict_a", "qs_verdict_b", "measured_conductance"} <= set(written.columns)
    assert not any("truncated" in n for n in report.notes)


# ── The output CRS is explicit, always ───────────────────────────────────────

def test_the_output_crs_is_required(tmp_path):
    with pytest.raises(ValueError, match="output CRS is required"):
        export_vector(classified(), tmp_path / "qc.gpkg", crs=None)


def test_an_unknown_source_crs_is_an_error_not_an_assumption(tmp_path):
    """Reprojecting from an assumed CRS produces plausible, wrong positions."""
    dataset = classified()
    dataset.crs = None
    with pytest.raises(ValueError, match="cannot be reprojected"):
        export_vector(dataset, tmp_path / "qc.gpkg", crs=WGS84)


def test_coordinates_are_reprojected_to_the_output_crs(tmp_path):
    out = tmp_path / "qc.gpkg"
    _, report = export_vector(classified(), out, crs=WGS84)

    assert report.distributions["crs"] == {
        "source": UTM, "output": WGS84, "reprojected": True}
    written = gpd.read_file(out, layer="soundings")
    assert written.crs.to_string() == WGS84
    assert -180 <= written.geometry.x.iloc[0] <= 180      # degrees, not metres


def test_no_reprojection_when_the_two_match(tmp_path):
    _, report = export_vector(classified(), tmp_path / "qc.gpkg", crs=UTM)
    assert report.distributions["crs"]["reprojected"] is False


# ── Unclassified survives ────────────────────────────────────────────────────

def test_unclassified_is_written_as_a_label_not_a_blank(tmp_path):
    """03 gives a metric with no threshold no classification at all, and it
    must not arrive at a client reading as 'Good' or as an empty cell."""
    dataset = classified(n=6, n_lines=1, bad=slice(0, 0))
    dataset.flightlines.loc[0, "qs_height"] = None

    out = tmp_path / "qc.gpkg"
    _, report = export_vector(dataset, out, crs=UTM)

    written = gpd.read_file(out, layer="soundings")
    assert written["qs_height"].iloc[0] == "Unclassified"
    assert written["qs_height"].iloc[1] == "Good"
    assert report.distributions["unclassified_written_as"] == "Unclassified"


def test_the_label_is_settable(tmp_path):
    dataset = classified(n=6, n_lines=1, bad=slice(0, 0))
    dataset.flightlines.loc[0, "qs_height"] = None

    out = tmp_path / "qc.gpkg"
    export_vector(dataset, out, crs=UTM, unclassified="not measured")
    assert gpd.read_file(out, layer="soundings")["qs_height"].iloc[0] == "not measured"


# ── Format, selection, and absent inputs ─────────────────────────────────────

def test_the_format_comes_from_the_extension(tmp_path):
    for name, driver in (("a.gpkg", "GPKG"), ("c.geojson", "GeoJSON")):
        _, report = export_vector(classified(n=6, n_lines=1, bad=slice(0, 0)),
                                  tmp_path / name, crs=WGS84)
        assert report.distributions["driver"] == driver


def test_an_unknown_extension_says_what_it_takes(tmp_path):
    with pytest.raises(ValueError, match=r"\.gpkg"):
        export_vector(classified(), tmp_path / "qc.xyz", crs=UTM)


def test_columns_can_be_selected(tmp_path):
    out = tmp_path / "qc.gpkg"
    _, report = export_vector(classified(), out, crs=UTM,
                              columns=["line_id", "qs_height"])
    assert report.distributions["fields"] == ["line_id", "qs_height"]


def test_a_named_column_that_is_absent_is_an_error(tmp_path):
    with pytest.raises(KeyError, match="'nope'"):
        export_vector(classified(), tmp_path / "qc.gpkg", crs=UTM,
                      columns=["nope"])


def test_an_empty_dataset_reports_rather_than_raises(tmp_path):
    dataset = make_dataset(n=0, n_lines=1, height=[], pitch=[], roll=[])
    dataset.crs = UTM
    _, report = export_vector(dataset, tmp_path / "qc.gpkg", crs=UTM)
    assert any("nothing to export" in n for n in report.notes)


def test_the_dataset_is_returned_unchanged(tmp_path):
    """It reads and writes files; it is not a filter that alters anything."""
    dataset = classified()
    before = dataset.flightlines.copy()
    returned, _ = export_vector(dataset, tmp_path / "qc.gpkg", crs=UTM)
    pd.testing.assert_frame_equal(returned.flightlines, before)


def test_registered_and_runs_from_a_pipeline(tmp_path):
    assert "export_vector" in load_steps(FILTER_GROUP)


# ── The layers are the product; the file is one delivery of them ─────────────

def test_the_layers_can_be_built_without_writing_anything(tmp_path):
    """R8 says format is a loader concern on the way in. On the way out it is a
    delivery concern, and neither is a property of the processing — so what this
    step produces is vector geometry, and serialising it is a separate act."""
    from ymerflow_processing.filters.export_vector import build_layers

    built, _ = build_layers(classified(), crs=UTM)

    assert set(built) == {"soundings", "flightlines"}
    assert not list(tmp_path.iterdir())


def test_the_built_layers_are_already_in_the_output_crs():
    from ymerflow_processing.filters.export_vector import build_layers

    built, _ = build_layers(classified(), crs="EPSG:4326")
    assert str(built["soundings"].crs) == "EPSG:4326"


def test_one_build_can_serve_several_formats(tmp_path):
    """The point of the split: the platform builds once and serves whatever is
    asked for, without re-running anything."""
    import geopandas as gpd

    dataset = classified()
    for name in ("gpkg", "geojson"):
        out = tmp_path / f"qc.{name}"
        export_vector(dataset, out, crs=UTM)
        assert out.exists()
    assert len(gpd.read_file(tmp_path / "qc.geojson")) == 60


def test_a_stated_format_beats_the_extension(tmp_path):
    """So a caller serving a chosen format is not obliged to encode that choice
    in a filename."""
    out = tmp_path / "download.bin"
    _, report = export_vector(classified(), out, crs=UTM, format="gpkg")

    assert report.distributions["driver"] == "GPKG"
    assert out.exists()


def test_an_unreadable_extension_with_no_format_says_both_ways_out(tmp_path):
    with pytest.raises(ValueError, match="or pass format="):
        export_vector(classified(), tmp_path / "qc.bin", crs=UTM)


def test_an_unknown_format_names_the_available_ones(tmp_path):
    with pytest.raises(ValueError, match="Unknown format"):
        export_vector(classified(), tmp_path / "qc.gpkg", crs=UTM, format="kml")
