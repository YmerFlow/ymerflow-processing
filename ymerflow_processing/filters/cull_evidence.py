"""filter_30 and filter_31 — data that outside evidence says is suspect.

Two steps over one engine: ``cull_near_infrastructure`` (filter_30) disables
gates whose sounding sits within an exclusion distance of a mapped feature, and
``cull_by_misfit`` (filter_31) disables gates a previous inversion fit poorly —
optionally only ``within`` that same distance of a feature, which is the
conjunction ``01`` describes as more selective than either. Both add whatever the
culling policy widens a hit to. Neither changes a measurement, an uncertainty or
an array shape (R6).

They were one step with a three-valued ``criterion`` until 2026-10-21. Split
because a person choosing a step in a form should see "cull near infrastructure"
and "cull by misfit", not a mode switch inside "external evidence" — and because
the two have almost no parameters in common.

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

import logging

import numpy as np

from geopandas import GeoDataFrame
from libaarhusxyz.xyz import XYZ

from ..dataset import Dataset
from ..distribution import summarize
from ..mask import apply_policy, cull, in_use, mark_step
from .misfit import ensure_misfit
from ..pipeline import StepReport

log = logging.getLogger(__name__)

CATEGORY = "cull gates"

INFRASTRUCTURE = "cull_near_infrastructure"
MISFIT = "cull_by_misfit"

#: Provenance codes, one per step, so a per-datum attribution still says
#: *which* evidence removed a point — the report records it, but the array is
#: what someone reads six months later. A misfit cull restricted to ``within``
#: a feature is still a misfit cull.
STEP_CODES = {INFRASTRUCTURE: 30, MISFIT: 31}

#: How an inversion result names its gatewise residual, already divided by
#: uncertainty — ``(dobs - dpred) * W.diagonal()`` — so a threshold reads in
#: standard deviations rather than in the response's own units.
MISFIT_PATTERNS = (r"^dbdt_err_ch(\d+)", r"^err_ch(\d+)")


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

    An R-tree nearest query rather than a distance to every feature: a national
    infrastructure layer has tens of thousands of geometries and a survey has
    tens of thousands of soundings.

    Parameters
    ----------
    easting, northing : array_like
        Sounding positions, in the same CRS as ``features``.
    features : geopandas.GeoSeries | geopandas.GeoDataFrame
        The mapped geometry.

    Returns
    -------
    numpy.ndarray
        One distance per sounding; ``NaN`` where the sounding has no position.
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
    """Per-gate exclusion distance, interpolated linearly in log time.

    Parameters
    ----------
    first, last : float
        The distance at the first and last gate.
    times : array_like
        Gate centre times; only their log spacing matters.

    Returns
    -------
    numpy.ndarray
        One distance per gate.
    """
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


def cull_near_infrastructure(dataset, infrastructure: str | GeoDataFrame,
                             distance_first: float | dict[str, float],
                             distance_last: float | dict[str, float] | None = None,
                             crs: str | None = None, start_gate: int = 0,
                             spread_earlier: int = 0,
                             disable_tail: bool = False):
    """Disable gates whose sounding lies within an exclusion distance of a mapped feature.

    Parameters
    ----------
    dataset : Dataset
        The dataset to cull; its in-use masks are written.
    infrastructure : geopandas.GeoDataFrame | geopandas.GeoSeries | str
        Vector geometry of the features to stay clear of — powerlines, fences,
        pipelines — or a path geopandas can read.
    distance_first : float | dict
        Exclusion distance at the first gate, in the survey's CRS units. Per
        channel key or moment name, or one value for all.
    distance_last : float | dict | None
        The same at the last gate; interpolated in log time between the two.
        Constant if unset.
    crs : str | None
        The survey's CRS, if the dataset does not carry one.
    start_gate : int
        Where evaluation begins, 0-based.
    spread_earlier : int
        Also disable this many gates before each hit.
    disable_tail : bool
        Also disable everything after the earliest hit.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    if infrastructure is None or distance_first is None:
        raise ValueError(
            f"{INFRASTRUCTURE} needs infrastructure and distance_first.")
    return _cull(dataset, INFRASTRUCTURE, infrastructure, distance_first,
                 distance_last, None, None, crs, start_gate, spread_earlier,
                 disable_tail)


def cull_by_misfit(dataset, max_misfit: float,
                   result: str | XYZ | Dataset | None = None,
                   within: str | GeoDataFrame | None = None,
                   distance_first: float | dict[str, float] | None = None,
                   distance_last: float | dict[str, float] | None = None,
                   crs: str | None = None, start_gate: int = 0,
                   spread_earlier: int = 0, disable_tail: bool = False):
    """Disable gates a previous inversion fit poorly, optionally only near mapped features.

    Parameters
    ----------
    dataset : Dataset
        The dataset to cull; its in-use masks are written, and a misfit column
        may be added.
    max_misfit : float
        Largest acceptable gatewise misfit, **in standard deviations**. Data
        missed by more than this is suspect.
    result : XYZ | Dataset | str | Path | None
        A previous inversion's or forward model's output. **Only needed when
        the dataset has no misfit column yet** — this reads the column first
        and computes one from the result only if there is none, writing it so
        nothing recomputes it. Running ``compute_misfit`` beforehand makes this
        an ordinary filter thresholding a column, which is what ``01`` asks for.
    within : geopandas.GeoDataFrame | geopandas.GeoSeries | str | None
        Restrict the cull to soundings within ``distance_first`` /
        ``distance_last`` of these features. Poorly fit **and** near a feature
        is more selective than either alone (``01``): a poor fit away from
        anything mapped may be geology a 1D model cannot represent, and
        removing it removes the evidence that the model was wrong.
    distance_first : float | dict | None
        Exclusion distance at the first gate, in the survey's CRS units, when
        ``within`` is given. Per channel key or moment name, or one value.
    distance_last : float | dict | None
        The same at the last gate. Constant if unset.
    crs : str | None
        The survey's CRS, if the dataset does not carry one.
    start_gate : int
        Where evaluation begins, 0-based.
    spread_earlier : int
        Also disable this many gates before each hit.
    disable_tail : bool
        Also disable everything after the earliest hit.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    if max_misfit is None:
        raise ValueError(
            f"{MISFIT} needs max_misfit. The misfit is in standard deviations, "
            f"so max_misfit is how many the model may miss by.")
    if max_misfit <= 0:
        raise ValueError(
            f"max_misfit counts standard deviations and must be positive; got "
            f"{max_misfit:g}.")
    if within is not None and distance_first is None:
        raise ValueError(
            f"{MISFIT} with within= needs distance_first: how close to a "
            f"feature a poorly fit sounding must be to count.")
    return _cull(dataset, MISFIT, within, distance_first, distance_last,
                 result, max_misfit, crs, start_gate, spread_earlier,
                 disable_tail)


def _cull(dataset, step, infrastructure, distance_first, distance_last, result,
          max_misfit, crs, start_gate, spread_earlier, disable_tail):
    """The shared engine: proximity and misfit tests, intersected, then culled."""
    report = StepReport(step=step)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    wants_geometry = infrastructure is not None
    wants_misfit = max_misfit is not None

    if frame is None or len(frame) == 0 or not dataset.channels:
        report.notes.append("No soundings or no channels; nothing to cull.")
        return dataset, report

    step_code = STEP_CODES[step]
    distance = None
    features = None
    survey_crs = None

    if wants_geometry:
        survey_crs = crs if crs is not None else getattr(dataset, "crs", None)
        if survey_crs is not None:
            from pyproj import CRS as ProjCRS
            if ProjCRS.from_user_input(survey_crs).is_geographic:
                raise ValueError(
                    f"The survey CRS {survey_crs!r} is geographic, so a "
                    f"distance in it is a distance in degrees — an exclusion of "
                    f"{distance_first} would be most of the planet. Reproject "
                    f"to a projected CRS before culling by distance."
                )
        else:
            report.notes.append(
                "The dataset declares no CRS, so the features were used as they "
                "are and both are assumed to share a projected CRS with the "
                "same units. Pass crs= to have them reprojected and checked."
            )
        features = _open(infrastructure, survey_crs)
        xyz = dataset.survey.xyz
        if xyz.x_column not in frame.columns or xyz.y_column not in frame.columns:
            raise KeyError("No position columns; distance cannot be measured.")
        distance = distance_to_features(frame[xyz.x_column].to_numpy(),
                                        frame[xyz.y_column].to_numpy(), features)
        unplaced = int((~np.isfinite(distance)).sum())
        if unplaced:
            report.notes.append(
                f"{unplaced} sounding(s) have no position and were not culled "
                f"on proximity. Where they are has not been established, which "
                f"is a different problem from being near something."
            )

    firsts = (dataset.resolve_per_channel(distance_first)
              if wants_geometry else {})
    lasts = (dataset.resolve_per_channel(distance_last)
             if wants_geometry and distance_last is not None else dict(firsts))

    per_channel, culled_any, without = {}, False, []
    for channel in dataset.channels:
        gates = dataset.layer_data[channel.data_key].shape[1]
        times = np.asarray(channel.axis.values, dtype=float)
        if times.size != gates:
            continue
        live = in_use(dataset, channel)
        outcome: dict = {}

        near = np.ones((len(frame), gates), dtype=bool)
        if wants_geometry:
            if channel.key not in firsts:
                continue
            exclusion = gate_distances(firsts[channel.key],
                                       lasts.get(channel.key,
                                                 firsts[channel.key]), times)
            near = (distance[:, None] <= exclusion[None, :])
            near &= np.isfinite(distance)[:, None]
            outcome["exclusion"] = {"first": float(exclusion[0]),
                                    "last": float(exclusion[-1])}
            outcome["within"] = int((near & live).sum())

        poor = np.ones((len(frame), gates), dtype=bool)
        if wants_misfit:
            # The column is the canonical form, so it is looked for first.
            # Where it is absent and a result was given, it is computed and
            # written — so the next consumer finds it rather than recomputing.
            misfit = ensure_misfit(dataset, channel, result)
            if misfit is None:
                without.append(str(channel))
                continue
            poor = np.abs(misfit) > max_misfit
            poor &= np.isfinite(misfit)
            outcome["misfit"] = summarize(misfit[np.isfinite(misfit)])
            outcome["poorly_fit"] = int((poor & live).sum())

        suspect = near & poor & live
        widened = apply_policy(suspect, start_gate=start_gate,
                               spread_earlier=spread_earlier,
                               disable_tail=disable_tail)
        direct = suspect.copy()
        direct[:, :start_gate] = False

        outcome["suspect"] = int(direct.sum())
        if widened.any():
            culled_any = True
            outcome.update(cull(dataset, channel, widened,
                                step_code=step_code, direct=direct))
        else:
            outcome["removed"] = 0
        per_channel[str(channel)] = outcome

    if culled_any:
        mark_step(dataset, step_code, step)
    if without:
        report.notes.append(
            f"No misfit column for {', '.join(without)}, and none could be "
            f"computed — pass result= with an inversion or forward model, or "
            f"run compute_misfit first. Those channels were not judged."
        )

    removed = sum(c["removed"] for c in per_channel.values())
    report.soundings_affected = removed
    report.distributions = {
        "within": wants_geometry if step == MISFIT else None,
        "per_channel": per_channel,
        "removed": removed,
        "features": None if features is None else int(len(features)),
        "crs": str(survey_crs) if survey_crs else None,
        "distance_to_nearest": (None if distance is None
                                else summarize(distance)),
        "max_misfit": max_misfit,
        "policy": {"start_gate": start_gate, "spread_earlier": spread_earlier,
                   "disable_tail": disable_tail},
        "provenance_code": step_code,
    }
    if wants_misfit and not wants_geometry:
        report.notes.append(
            "Culled on misfit alone. 01 warns that this over-culls: a poor fit "
            "may be genuine geology a 1D model cannot represent, and removing "
            "it removes the evidence that the model was wrong. Restricting it "
            "with within= to soundings near mapped features is more selective."
        )
    log.info("%s: culled %d gate-sample(s)", step, removed)
    return dataset, report
