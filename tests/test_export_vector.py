"""Writing a dataset out as vector geometry — the artifact that leaves the building."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("geopandas")
import geopandas as gpd

from conftest import make_dataset
from ymerflow_processing.core.export_vector import (
    export_vector, truncation_collisions,
)
from ymerflow_processing.core.pipeline import (
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


# ── Shapefile, only when a contract names it ─────────────────────────────────

def test_a_shapefile_holds_one_geometry_type_so_lines_go_beside(tmp_path):
    out = tmp_path / "qc.shp"
    _, report = export_vector(classified(), out, crs=UTM)

    assert (tmp_path / "qc.shp").exists()
    assert (tmp_path / "qc_lines.shp").exists()
    assert any("one layer per file" in n for n in report.notes)


def test_a_shapefile_declares_its_encoding(tmp_path):
    """DBF's encoding is ambiguous and readers guess differently."""
    out = tmp_path / "qc.shp"
    export_vector(classified(), out, crs=UTM)

    assert (tmp_path / "qc.cpg").read_text() == "UTF-8"
    assert (tmp_path / "qc_lines.cpg").read_text() == "UTF-8"


def test_a_shapefile_carries_a_projection_file(tmp_path):
    out = tmp_path / "qc.shp"
    export_vector(classified(), out, crs=UTM)
    assert "UTM_Zone_13N" in (tmp_path / "qc.prj").read_text()


# ── Field names ──────────────────────────────────────────────────────────────

def test_colliding_field_names_are_refused_rather_than_silently_merged(tmp_path):
    """A shapefile truncates to ten and one column simply replaces the other."""
    dataset = classified(n=6, n_lines=1, bad=slice(0, 0))
    dataset.flightlines["qs_verdict_a"] = 1
    dataset.flightlines["qs_verdict_b"] = 2

    with pytest.raises(ValueError, match="would collide"):
        export_vector(dataset, tmp_path / "qc.shp", crs=UTM)


def test_the_collision_check_directly():
    assert truncation_collisions(["qs_verdict_a", "qs_verdict_b", "qs_height"]) == {
        "qs_verdict": ["qs_verdict_a", "qs_verdict_b"]}
    assert truncation_collisions(["qs_height", "qs_speed"]) == {}


def test_names_that_merely_truncate_are_noted_not_refused(tmp_path):
    """A long name colliding with nothing loses only its tail — worth saying,
    not worth refusing."""
    dataset = classified(n=6, n_lines=1, bad=slice(0, 0))
    dataset.flightlines["measured_conductance"] = 1.0

    _, report = export_vector(dataset, tmp_path / "qc.shp", crs=UTM)

    assert any("measured_conductance" in n and "truncated" in n
               for n in report.notes)


def test_the_fixture_needs_no_truncation_at_all(tmp_path):
    """The ten-character discipline, checked end to end: a full QC run exports
    to a shapefile with nothing shortened and nothing colliding."""
    _, report = export_vector(classified(), tmp_path / "qc.shp", crs=UTM)
    assert not any("truncated" in n for n in report.notes)
    assert all(len(f) <= 10 for f in report.distributions["fields"])


def test_geopackage_has_no_such_limit(tmp_path):
    dataset = classified(n=6, n_lines=1, bad=slice(0, 0))
    dataset.flightlines["qs_verdict_a"] = 1
    dataset.flightlines["qs_verdict_b"] = 2

    out = tmp_path / "qc.gpkg"
    export_vector(dataset, out, crs=UTM)          # no complaint
    written = gpd.read_file(out, layer="soundings")
    assert {"qs_verdict_a", "qs_verdict_b"} <= set(written.columns)


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
    """03 gives a metric with no threshold no classification at all. A
    shapefile cannot express that, so it must not arrive as 'Good'."""
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
    for name, driver in (("a.gpkg", "GPKG"), ("b.shp", "ESRI Shapefile"),
                         ("c.geojson", "GeoJSON")):
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
