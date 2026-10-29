"""The reconstruction criterion: Asif et al. (2022), Algorithm 1, over any basis.

A transient the basis cannot reproduce is not describing a 1D earth the way the
rest of the survey does. The arithmetic, with the paper's symbols:

1. Average the data and its standard deviation along the line over ``window``
   (the paper's ``W``, there 7 raw stacks), to smooth short-term fluctuation.
   Internal to the evaluation: the decision applies to every sounding, and the
   data are not changed.
2. From the first gate whose averaged relative uncertainty exceeds
   ``max_uncertainty`` (``Δ_max``, 20%) onward, mask the transient: that is
   noise, not coupling, and not this criterion's business.
3. Reconstruct the averaged transient from the basis and take the error in
   sigmas at each evaluated gate, ``β = |d_recon − d| / Δd``.
4. The confidence ``C`` is the fraction of evaluated gates with
   ``β < max_reconstruction_error`` (``α``, 2.5).
5. Average ``C`` along the line over twice ``window`` (``2W``), so no single
   sounding decides.
6. Decide per sounding (the paper's fuzzy band):

   - ``C_avg > min_confidence`` (``β_max``): keep every gate;
   - ``C_avg`` below ``cull_below_confidence`` (``0.80 × β_max`` in the paper):
     every gate of the sounding is coupled;
   - between the two: only gates with ``β ≤ 1`` survive. Gates past the noise
     cutoff are left to the steps whose business they are.

Windows are distances along the line, in the survey's horizontal units, never
counts of soundings: sampling rates differ between lines of one survey, and a
vendor filter can correlate neighbors at the full rate.
"""

import numpy as np

from ..mask import CULLED

#: Asif et al. (2022), Algorithm 1: the lower edge of the fuzzy band, as a
#: fraction of ``min_confidence``.
FUZZY_FLOOR = 0.80


def positions(dataset):
    """Sounding positions and line identifiers, or ``None`` if there are none.

    Parameters
    ----------
    dataset : Dataset
        Whose ``flightlines`` are read.

    Returns
    -------
    tuple | None
        ``(x, y, lines)``: easting and northing in the survey's horizontal
        units, and a line identifier per sounding (all one line if the survey
        names none). ``None`` when there is no easting or northing.
    """
    frame = dataset.flightlines
    xyz = dataset.survey.xyz
    if frame is None or xyz.x_column not in frame.columns or xyz.y_column not in frame.columns:
        return None
    x = frame[xyz.x_column].to_numpy(dtype=float)
    y = frame[xyz.y_column].to_numpy(dtype=float)
    line = xyz.line_id_column
    lines = frame[line].to_numpy() if line in frame.columns else np.zeros(len(frame))
    return x, y, lines


def along_line_mean(values, x, y, lines, length: float) -> np.ndarray:
    """Mean over the soundings of the same line within ``length / 2`` along it.

    Distance is accumulated along each line in file order, so a turn or a gap
    in the line is counted as the distance flown, and a window never crosses
    from one line to another.

    Parameters
    ----------
    values : numpy.ndarray
        ``(soundings,)`` or ``(soundings, gates)``; ``NaN`` is ignored.
    x, y : numpy.ndarray
        Sounding positions, in the survey's horizontal units.
    lines : numpy.ndarray
        A line identifier per sounding.
    length : float
        The full window length, in the same units as ``x`` and ``y``. Zero
        returns ``values`` unchanged.

    Returns
    -------
    numpy.ndarray
        The same shape as ``values``; ``NaN`` where a window holds no value.
    """
    values = np.asarray(values, dtype=float)
    flat = values.ndim == 1
    values = values[:, None] if flat else values
    out = np.full(values.shape, np.nan)
    half = max(float(length), 0.0) / 2.0
    for line in np.unique(lines):
        rows = np.flatnonzero(lines == line)
        steps = np.hypot(np.diff(x[rows]), np.diff(y[rows]))
        along = np.concatenate([[0.0], np.cumsum(np.nan_to_num(steps))])
        start = np.searchsorted(along, along - half, side="left")
        stop = np.searchsorted(along, along + half, side="right")
        known = np.isfinite(values[rows])
        sums = np.vstack([np.zeros(values.shape[1]),
                          np.cumsum(np.where(known, values[rows], 0.0), axis=0)])
        counts = np.vstack([np.zeros(values.shape[1]), np.cumsum(known, axis=0)])
        total = sums[stop] - sums[start]
        n = counts[stop] - counts[start]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[rows] = np.where(n > 0, total / n, np.nan)
    return out[:, 0] if flat else out


def _live(dataset, channel) -> np.ndarray:
    """Where the channel is in use, without creating a mask if it has none."""
    shape = dataset.layer_data[channel.data_key].shape
    key = channel.inuse_key
    if key is None or key not in dataset.layer_data:
        return np.ones(shape, dtype=bool)
    return dataset.layer_data[key].to_numpy() != CULLED


def detect(dataset, channel, reconstructor, *, max_reconstruction_error: float,
           min_confidence: float, cull_below_confidence: float, window: float,
           max_uncertainty: float):
    """Which gates of one channel the basis cannot reproduce, per Algorithm 1.

    Parameters
    ----------
    dataset : Dataset
        The survey; needs sounding positions and the channel's uncertainty.
    channel : Channel
        The channel judged.
    reconstructor : Reconstructor
        The basis; fitted here, on this channel of this survey.
    max_reconstruction_error : float
        ``α``: a gate reconstructs when its error is below this, in standard
        deviations of the assigned uncertainty.
    min_confidence : float
        ``β_max``: a sounding whose smoothed confidence exceeds this keeps every
        gate. A fraction, 0..1.
    cull_below_confidence : float
        Below this smoothed confidence every gate of the sounding is coupled.
        A fraction, at most ``min_confidence``.
    window : float
        ``W``: the along-line length the data are averaged over before judging,
        in the survey's horizontal units. The confidence is averaged over
        twice this.
    max_uncertainty : float
        ``Δ_max``: from the first gate whose averaged relative uncertainty
        exceeds this, the transient is not judged. A fraction.

    Returns
    -------
    tuple
        ``(direct, confidence, error)``: the gates the criterion marks coupled,
        ``(soundings, gates)`` bool; the smoothed confidence per sounding, ``NaN``
        where nothing could be judged; and the error in sigmas per gate, ``NaN``
        where not evaluated.
    """
    x, y, lines = positions(dataset)
    data = dataset.layer_data[channel.data_key].to_numpy(dtype=float)
    relative = dataset.layer_data[channel.std_key].to_numpy(dtype=float)
    usable = _live(dataset, channel) & np.isfinite(data) & np.isfinite(relative) & (data > 0)

    averaged = along_line_mean(np.where(usable, data, np.nan), x, y, lines, window)
    sigma = along_line_mean(np.where(usable, relative * np.abs(data), np.nan),
                            x, y, lines, window)
    with np.errstate(invalid="ignore", divide="ignore"):
        noisy = (sigma / averaged) > max_uncertainty
    gate = np.arange(data.shape[1])[None, :]
    first = np.where(noisy.any(axis=1), noisy.argmax(axis=1), data.shape[1])[:, None]
    evaluated = np.isfinite(averaged) & np.isfinite(sigma) & (averaged > 0) & (gate < first)

    error = np.full(data.shape, np.nan)
    gates = evaluated.any(axis=0)
    if gates.any():
        transients = np.where(evaluated, averaged, np.nan)[:, gates]
        reconstructor.fit(transients)
        rebuilt = reconstructor.reconstruct(transients)
        with np.errstate(invalid="ignore", divide="ignore"):
            error[:, gates] = np.abs(rebuilt - averaged[:, gates]) / sigma[:, gates]
    evaluated &= np.isfinite(error)

    judged = evaluated.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = np.where(judged > 0,
                       (evaluated & (error < max_reconstruction_error)).sum(axis=1) / judged,
                       np.nan)
    confidence = along_line_mean(raw, x, y, lines, 2.0 * window)

    known = np.isfinite(confidence)
    low = known & (confidence < cull_below_confidence)
    moderate = known & ~low & (confidence <= min_confidence)
    direct = np.zeros(data.shape, dtype=bool)
    direct[low] = True
    direct |= moderate[:, None] & evaluated & (error > 1.0)
    return direct, confidence, error
