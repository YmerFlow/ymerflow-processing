"""filter_30 — disable data too close to a cultural conductor.

**What it does to the dataset:** disables gates whose sounding sits within an
exclusion distance of a mapped feature, plus whatever the culling policy widens
that to. It changes no measurement, no uncertainty and no array shape (R6).

``01``: *"Cultural conductors couple into the receiver and produce anomalies that
are real measurements of the wrong thing."*

That phrase is the justification and also the reason no shape test can find these.
A powerline crossing produces a clean, coherent, entirely believable response —
of a powerline. filter_25 to filter_28 pass it, filter_32 passes it, and an
inversion turns it into a conductor in the ground. What identifies it is not the
data but the map.

The exclusion widens with gate time
------------------------------------
``01``: *"late-time data senses a larger volume and is affected from further
away."* So the distance is not one number: it is given at the first gate and at
the last, and interpolated in ``log t`` between them, the same basis filter_10
uses for its window widths and for the same reason — gates are logarithmic.

The consequence is that near a feature a sounding loses its late gates while
keeping its early ones, which is the right shape: the early gates sense a small
volume directly beneath the loop and are not reaching the conductor.

Per moment, because the moments do not sense the same volume
--------------------------------------------------------------
``01`` asks for *"per-moment distances"*. A high moment at late time senses far
more ground than a low moment at early time, so one exclusion for the instrument
would be too wide for one channel and too narrow for the other.

A type, not a format
--------------------
R8 names this filter as the example of the problem — its old default parameter
was literally the string ``'/path/to/my/shape/file.shp'``. So it takes **vector
geometry**: a ``GeoDataFrame`` is the typed form, and a path is accepted because
a standalone run has nothing to materialise one. Whether that path is a
shapefile, a GeoPackage or a KML is geopandas' business, not this filter's.

Distance is measured in the survey's own CRS
---------------------------------------------
Which means the CRS must be projected. An exclusion of 200 in a geographic CRS is
200 *degrees*, which is most of the planet, and nothing in the arithmetic would
object. That is refused rather than warned about.

What this cannot do, and what complements it
---------------------------------------------
It needs a map, and it culls everything near a feature whether or not that datum
was actually affected. qc_05 is the complement: it finds crossings in the
powerline monitor with no map at all, but cannot see a line flown *parallel* to a
conductor, because affecting the whole window makes the elevated level the
background. One needs a map and over-culls; the other needs no map and misses the
parallel case. Neither replaces the other.

filter_31 is the intended refinement — proximity *and* poor fit together, which
is more selective than either — and ``01`` defers it because it needs an
inversion to have run.
"""

from __future__ import annotations

import logging

import numpy as np

from ..distribution import summarize
from ..mask import apply_policy, cull, in_use, mark_step
from ..pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "cull_near_infrastructure"

#: This filter's number in ``building_plans/01``, used as its provenance code.
STEP_CODE = 30


def _open(infrastructure, crs):
    """The features as a GeoDataFrame in ``crs``, from an object or a path."""
    import geopandas as gpd

    frame = (infrastructure if hasattr(infrastructure, "geometry")
             else gpd.read_file(str(infrastructure)))
    if crs is None:
        return frame
    if frame.crs is None:
        raise ValueError(
            "The infrastructure geometry declares no CRS, so it cannot be "
            "placed against the survey. Give it one before using it — guessing "
            "would put every feature somewhere plausible and wrong."
        )
    return frame.to_crs(crs) if str(frame.crs) != str(crs) else frame


def distance_to_features(easting, northing, features) -> np.ndarray:
    """Distance from each sounding to the nearest feature, in CRS units.

    ``NaN`` where the sounding has no position. An R-tree nearest query rather
    than a distance to every feature: a national infrastructure layer has tens
    of thousands of geometries and a survey has tens of thousands of soundings.
    """
    from shapely import STRtree, points

    x = np.asarray(easting, dtype=float)
    y = np.asarray(northing, dtype=float)
    out = np.full(x.shape, np.nan)

    placed = np.isfinite(x) & np.isfinite(y)
    geometries = [g for g in features.geometry if g is not None and not g.is_empty]
    if not placed.any() or not geometries:
        return out

    tree = STRtree(geometries)
    _, distance = tree.query_nearest(points(x[placed], y[placed]),
                                     return_distance=True, all_matches=False)
    out[placed] = distance
    return out


def gate_distances(first: float, last: float, times) -> np.ndarray:
    """Per-gate exclusion distance, interpolated linearly in log time."""
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        return np.full(max(times.size, 1), float(first))

    usable = times > 0
    span = np.zeros(times.size)
    if usable.sum() >= 2:
        logs = np.log(np.where(usable, times, np.nan))
        low, high = np.nanmin(logs), np.nanmax(logs)
        if high > low:
            span = np.where(usable, (logs - low) / (high - low), 0.0)
    return float(first) + span * (float(last) - float(first))


def cull_near_infrastructure(dataset, infrastructure, distance_first,
                             distance_last=None, crs=None, start_gate: int = 0,
                             spread_earlier: int = 0,
                             disable_tail: bool = False):
    """Disable data within an exclusion distance of mapped features.

    :param infrastructure: Vector geometry — a ``GeoDataFrame``, or a path to
        anything geopandas can read.
    :param distance_first: Exclusion distance at the first gate, in the survey's
        CRS units. Per channel key or moment name, or one value for all.
    :param distance_last: The same at the last gate. Equal to ``distance_first``
        if unset, giving a constant exclusion.
    :param crs: The survey's CRS, if the dataset does not carry one. Must be
        projected: an exclusion of 200 in a geographic CRS is 200 degrees.
    :param start_gate: Where evaluation begins, 0-based.
    :param spread_earlier: Also disable this many gates before each hit.
    :param disable_tail: Also disable everything after the earliest hit. Usually
        redundant here, since the exclusion already widens with time.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if frame is None or len(frame) == 0 or not dataset.channels:
        report.notes.append("No soundings or no channels; nothing to cull.")
        return dataset, report

    survey_crs = crs if crs is not None else getattr(dataset, "crs", None)
    if survey_crs is not None:
        from pyproj import CRS as ProjCRS
        if ProjCRS.from_user_input(survey_crs).is_geographic:
            raise ValueError(
                f"The survey CRS {survey_crs!r} is geographic, so a distance in "
                f"it is a distance in degrees — an exclusion of "
                f"{distance_first} would be most of the planet. Reproject to a "
                f"projected CRS before culling by distance."
            )
    else:
        report.notes.append(
            "The dataset declares no CRS, so the features were used as they are "
            "and both are assumed to share a projected CRS with the same units. "
            "Pass crs= to have them reprojected and checked."
        )

    features = _open(infrastructure, survey_crs)
    xyz = dataset.survey.xyz
    if xyz.x_column not in frame.columns or xyz.y_column not in frame.columns:
        raise KeyError("No position columns; distance cannot be measured.")

    distance = distance_to_features(frame[xyz.x_column].to_numpy(),
                                    frame[xyz.y_column].to_numpy(), features)
    unplaced = int((~np.isfinite(distance)).sum())

    firsts = dataset.resolve_per_channel(distance_first)
    lasts = (dataset.resolve_per_channel(distance_last)
             if distance_last is not None else dict(firsts))

    per_channel, culled_any = {}, False
    for channel in dataset.channels:
        if channel.key not in firsts:
            continue
        times = np.asarray(channel.axis.values, dtype=float)
        gates = dataset.layer_data[channel.data_key].shape[1]
        if times.size != gates:
            continue

        exclusion = gate_distances(firsts[channel.key],
                                   lasts.get(channel.key, firsts[channel.key]),
                                   times)
        # A sounding with no position is not culled: NaN fails the comparison,
        # which is the wanted answer. Where it is has not been established, and
        # that is a different problem from being near something.
        near = distance[:, None] <= exclusion[None, :]
        near &= np.isfinite(distance)[:, None] & in_use(dataset, channel)

        widened = apply_policy(near, start_gate=start_gate,
                               spread_earlier=spread_earlier,
                               disable_tail=disable_tail)
        direct = near.copy()
        direct[:, :start_gate] = False

        outcome = {
            "exclusion": {"first": float(exclusion[0]),
                          "last": float(exclusion[-1])},
            "within": int(direct.sum()),
            "soundings_affected": int(direct.any(axis=1).sum()),
        }
        if widened.any():
            culled_any = True
            outcome.update(cull(dataset, channel, widened,
                                step_code=STEP_CODE, direct=direct))
        else:
            outcome["removed"] = 0
        per_channel[str(channel)] = outcome

    if culled_any:
        mark_step(dataset, STEP_CODE, STEP)
    if unplaced:
        report.notes.append(
            f"{unplaced} sounding(s) have no position and were not culled. "
            f"Where they are has not been established, which is a different "
            f"problem from being near something."
        )

    removed = sum(c["removed"] for c in per_channel.values())
    report.soundings_affected = max(
        (c["soundings_affected"] for c in per_channel.values()), default=0)
    report.distributions = {
        "per_channel": per_channel,
        "removed": removed,
        "features": int(len(features)),
        "crs": str(survey_crs) if survey_crs else None,
        "distance_to_nearest": summarize(distance),
        "policy": {"start_gate": start_gate, "spread_earlier": spread_earlier,
                   "disable_tail": disable_tail},
        "provenance_code": STEP_CODE,
    }
    log.info("%s: disabled %d gate-sample(s) near %d feature(s)",
             STEP, removed, len(features))
    return dataset, report
