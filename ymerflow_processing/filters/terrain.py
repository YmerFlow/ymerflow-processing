"""filter_05 — express altitude against a proper terrain model.

**What it does to the dataset:** replaces the ground elevation with one sampled
from a digital terrain model, and recomputes altitude to match. It changes no
measurement, no uncertainty and no mask.

Two words, kept apart
---------------------
**Altitude** is distance above the ground. **Elevation** is distance above (or
below) the geoid. The library's columns are ``tx_alt`` and ``topo``
respectively, and ``libaarhusxyz`` states the relation between them by plotting
``alt_column + z_column`` as the instrument's position:

    instrument elevation = ground elevation + altitude

Everything here follows from reading that as a definition. Elevation is the
datum-referenced quantity; altitude is derived from it.

What gets corrected, and what is held fixed
-------------------------------------------
**The instrument's elevation is held fixed and altitude is recomputed.** The
aircraft was where it was — that comes from GPS, which is an absolute
measurement against a datum. What a better terrain model changes is not where
the aircraft flew but *how far above the ground* it flew.

So:

    instrument elevation  =  delivered ground elevation + delivered altitude
    new ground elevation  =  terrain model, sampled at the sounding
    new altitude          =  instrument elevation - new ground elevation

The alternative — trusting the altimeter and moving the aircraft — is available
as ``preserve="altitude"``, but it is the wrong default. In a SkyTEM delivery the
normalizer maps ``dem_lidar`` to ``topo`` and ``height_lidar`` to ``tx_altitude``:
**both come from the same lidar**, so the delivered ground elevation already
carries whatever the altimeter got wrong. Holding altitude fixed would preserve
exactly the error this filter exists to remove.

Why it matters that this is a separate step
-------------------------------------------
qc_01 grades height above ground and deliberately takes no terrain model, because
producing a correct one is this filter's job and doing it in two places gives two
chances to disagree (R9). The consequence is that until this runs, qc_01 grades
whatever altitude the delivery arrived with — and the heaviest-weighted check in
the whole QC scheme rests on a column nothing has validated.

A vertical datum mismatch looks like terrain error and is not
------------------------------------------------------------
GPS reports height above the **ellipsoid**; terrain models are usually **above
the geoid**. The separation between them is tens of metres in places, varies
slowly, and is not something this library can resolve — it has no geoid model.

But it can be *seen*. A datum mismatch shows up as a large offset between the
delivered ground elevation and the sampled one, with a **small spread**: the
whole survey shifted by nearly the same amount. Genuine terrain error is the
opposite — small offset, large spread. So the offset's median and its robust
spread are both reported, and reading them together is what tells the two apart.

No threshold is applied to that. Where the boundary lies depends on the terrain
and the model, and a house number here would either cry wolf on real relief or
wave through a datum error that puts every sounding tens of metres wrong.

Where the model does not reach
------------------------------
A sounding outside the model's coverage, or over a nodata cell, gets ``NaN`` for
both altitude and elevation rather than keeping its delivered values. Keeping
them would leave one dataset carrying two different definitions of ground with
nothing marking which is which — and a later check grading that column would be
comparing against a datum it cannot know. NaN is unclassified, which every check
already handles, and the count is reported.

The originals are stashed either way, so nothing is lost.

Sources
-------
The altitude/elevation distinction and the instrument-position relation are
``libaarhusxyz``'s own; see ``XYZ.plot_line``. Sampling is nearest-neighbour by
default because a terrain model's cell *is* the ground surface it asserts, and
interpolating between cells invents relief the model does not claim.
"""

import logging

import numpy as np

from ..pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "apply_terrain_model"

#: Where the delivered values are kept. Ten characters and eight, so both
#: survive a shapefile field name intact.
STASHED_ALTITUDE = "tx_alt_raw"
STASHED_ELEVATION = "topo_raw"

#: Scales a median absolute deviation so its spread reads in units comparable to
#: a standard deviation for normally distributed values.
MAD_TO_SIGMA = 1.4826

PRESERVE_ELEVATION = "elevation"
PRESERVE_ALTITUDE = "altitude"
PRESERVE = (PRESERVE_ELEVATION, PRESERVE_ALTITUDE)


def sample_terrain(raster, easting, northing, source_crs=None, band: int = 1):
    """Ground elevation at each position, ``NaN`` off the model or on nodata.

    Parameters
    ----------
    raster : rasterio.DatasetReader | str | Path
        An open ``rasterio`` dataset or a path to one. R8 asks for external
        inputs typed rather than as paths, and an opened dataset is the typed
        form; a path is accepted because a standalone run has nothing to
        materialise it, which is the same layering :func:`core.dataset.load`
        follows.
    easting, northing : array_like
        The positions to sample at.
    source_crs : str | None
        What the positions are in, if it differs from the raster's CRS.
    band : int
        Which raster band holds elevation.

    Returns
    -------
    numpy.ndarray
        One elevation per position.
    """
    import rasterio
    from rasterio.warp import transform as warp_transform

    opened = raster if hasattr(raster, "read") else rasterio.open(str(raster))
    try:
        x = np.asarray(easting, dtype=float)
        y = np.asarray(northing, dtype=float)

        if source_crs is not None and opened.crs is not None:
            if rasterio.crs.CRS.from_user_input(source_crs) != opened.crs:
                x, y = warp_transform(source_crs, opened.crs,
                                      x.tolist(), y.tolist())
                x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

        elevation = np.full(x.shape, np.nan)
        finite = np.isfinite(x) & np.isfinite(y)
        if not finite.any():
            return elevation, opened.crs

        # Nearest-neighbour: a terrain model's cell *is* the ground surface it
        # asserts, and interpolating between cells invents relief it does not
        # claim.
        sampled = np.array(
            [value[0] for value in opened.sample(
                zip(x[finite].tolist(), y[finite].tolist()), indexes=band)],
            dtype=float)
        if opened.nodata is not None:
            sampled = np.where(sampled == opened.nodata, np.nan, sampled)
        elevation[finite] = sampled
        return elevation, opened.crs
    finally:
        if opened is not raster:
            opened.close()


def apply_terrain_model(dataset, terrain, crs=None, band: int = 1,
                        preserve: str = PRESERVE_ELEVATION,
                        altitude_column: str | None = None,
                        elevation_column: str | None = None,
                        stash: bool = True):
    """Re-express altitude against a terrain model.

    Parameters
    ----------
    dataset : Dataset
        The dataset whose altitude or elevation column is rewritten.
    terrain : rasterio.DatasetReader | str | Path
        The terrain model — an open ``rasterio`` dataset or a path. Its values
        are ground elevation in the same vertical units as the delivery.
    crs : str | None
        The survey's horizontal CRS, needed only if it differs from the
        raster's. Taken from the dataset when it carries one.
    band : int
        Which raster band holds elevation.
    preserve : str
        ``elevation`` — hold the instrument's elevation and recompute altitude,
        which is the default and almost always right; or ``altitude`` — trust
        the altimeter and move the instrument instead.
    altitude_column : str | None
        Which column holds altitude. Discovered if unset.
    elevation_column : str | None
        Which holds ground elevation. Discovered if unset.
    stash : bool
        Keep the delivered values under ``tx_alt_raw`` and ``topo_raw``.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if preserve not in PRESERVE:
        raise ValueError(
            f"Unknown preserve {preserve!r}. Available: {', '.join(PRESERVE)}"
        )
    if frame is None or len(frame) == 0:
        report.notes.append("No soundings; nothing to correct.")
        return dataset, report

    xyz = dataset.survey.xyz
    altitude_column = altitude_column or xyz.alt_column
    elevation_column = elevation_column or xyz.z_column
    for role, name in (("altitude", altitude_column),
                       ("elevation", elevation_column)):
        if name is None or name not in frame.columns:
            raise KeyError(
                f"No {role} column {name!r} in flightlines. "
                f"Present: {', '.join(map(str, frame.columns))}"
            )
    if xyz.x_column not in frame.columns or xyz.y_column not in frame.columns:
        raise KeyError("No position columns; the terrain cannot be sampled.")

    altitude = frame[altitude_column].to_numpy(dtype=float)
    elevation = frame[elevation_column].to_numpy(dtype=float)
    # The definition everything here follows from.
    instrument = elevation + altitude

    source_crs = crs if crs is not None else getattr(dataset, "crs", None)
    ground, raster_crs = sample_terrain(
        terrain, frame[xyz.x_column].to_numpy(), frame[xyz.y_column].to_numpy(),
        source_crs=source_crs, band=band)

    if source_crs is None and raster_crs is not None:
        report.notes.append(
            "The dataset declares no CRS, so positions were sampled against the "
            f"terrain model as-is, assuming both are in {raster_crs}. If they "
            "are not, every sample is from the wrong place — pass crs= to have "
            "them reprojected."
        )

    missing = ~np.isfinite(ground)
    if stash:
        frame[STASHED_ALTITUDE] = altitude
        frame[STASHED_ELEVATION] = elevation

    if preserve == PRESERVE_ELEVATION:
        new_elevation = ground
        new_altitude = instrument - ground
    else:
        new_elevation = ground
        new_altitude = altitude

    # Where the model does not reach, both columns go missing rather than
    # keeping delivered values — one dataset carrying two definitions of ground
    # with nothing marking which is which is worse than an admitted gap.
    frame[altitude_column] = np.where(missing, np.nan, new_altitude)
    frame[elevation_column] = np.where(missing, np.nan, new_elevation)

    # A vertical datum mismatch and genuine terrain error look the same in the
    # median and opposite in the spread, so both are reported and neither is
    # judged. See the module docstring.
    offset = elevation - ground
    usable = offset[np.isfinite(offset)]
    if usable.size:
        median = float(np.median(usable))
        spread = float(np.median(np.abs(usable - median)) * MAD_TO_SIGMA)
    else:
        median = spread = None

    report.soundings_affected = int((~missing).sum())
    report.distributions = {
        "preserve": preserve,
        "columns": {"altitude": altitude_column, "elevation": elevation_column},
        "crs": {"survey": str(source_crs) if source_crs else None,
                "terrain": str(raster_crs) if raster_crs else None},
        "coverage": {"sampled": int((~missing).sum()),
                     "outside_or_nodata": int(missing.sum())},
        "elevation_offset": {
            "median": median,
            "robust_spread": spread,
            "reading": (
                "A large median with a small spread is a vertical datum "
                "mismatch — GPS reports height above the ellipsoid and terrain "
                "models are usually above the geoid. A small median with a "
                "large spread is genuine terrain error, which is what this "
                "filter is for."
            ),
        },
        "stashed": {"altitude": STASHED_ALTITUDE,
                    "elevation": STASHED_ELEVATION} if stash else None,
    }

    if missing.any():
        report.notes.append(
            f"{int(missing.sum())} sounding(s) fall outside the terrain model or "
            f"on nodata; their altitude and elevation are unset rather than "
            f"left at delivered values. The originals are in "
            f"{STASHED_ALTITUDE!r} and {STASHED_ELEVATION!r}."
            if stash else
            f"{int(missing.sum())} sounding(s) fall outside the terrain model "
            f"or on nodata; their altitude and elevation are unset."
        )
    log.info("%s: sampled %d of %d sounding(s); median elevation offset %s",
             STEP, int((~missing).sum()), len(frame),
             "n/a" if median is None else f"{median:.3g}")
    return dataset, report
