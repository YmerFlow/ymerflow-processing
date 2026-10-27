"""filter_10 — average each sounding with its along-line neighbours.

**What it does to the dataset:** replaces values and uncertainties with a
weighted average over neighbouring soundings, culls where too little valid data
fell in the window, and optionally decimates the result. It writes values,
uncertainty and the mask — more than any other step here, which is why the
originals are stashed.

``01`` calls this *"the most consequential filter in the library and the one most
worth rebuilding rather than porting"* and lists four defects to design out. Each
is addressed below, and each has a test named for it.

Defect 1 — a width of one must be a no-op
------------------------------------------
The old implementation silently produced all-invalid output for a width of one.
A window of one sounding is the sounding itself; averaging it with nothing is the
identity, and that is what happens here. Not an error, not a special case
threaded through the arithmetic — it falls out of a kernel of length one being
``[1.0]``.

Defect 2 — validity is effective sample size, not a count
----------------------------------------------------------
``01``: *"the validity gate counts samples rather than effective sample size,
which is wrong for any non-flat weighting."*

Under a taper, a window can be nominally full while almost all of its weight sits
on one sounding. Counting samples calls that well-determined; it is not. So the
gate is Kish's effective sample size,

    n_eff = (Σw)² / Σw²

which equals the count exactly for flat weights and falls toward one as the
weight concentrates. ``min_valid_fraction`` is compared against ``n_eff / width``,
so it means the same thing whatever kernel is chosen.

Defect 3 — the kernel is tapered, not flat
-------------------------------------------
``01``: *"a neighbour at the edge of the window counts as much as the sounding
itself; a distance taper preserves narrow targets far better."*

A flat kernel is a box filter: it has sidelobes, it rings, and a conductor
narrower than the window is smeared across the whole of it with equal authority.
A Hann taper is the default — smooth, zero at the edges, so a sounding entering
the window contributes nothing at first and grows in. ``boxcar`` remains
available, because reproducing what a previous processing run did is a real need.

Defect 4 — one convolution per gate, not a loop per datum
----------------------------------------------------------
``01``: *"a nested per-gate, per-sounding loop over row slices, which is orders
of magnitude slower."*

The weighted mean is a ratio of two convolutions. With per-datum weights ``u`` and
a kernel ``k``:

    numerator   = (u · x) * k
    denominator = u * k

so each gate of each line costs two convolutions rather than a loop over
soundings. Gate count is tens and line count is tens; sounding count is thousands,
and it is the sounding axis that is vectorised.

Widths vary with gate time
--------------------------
``01``: *"narrow at early gates, wider at late gates — so that late-time data,
which is noisier, is smoothed more."* Given a width at the first gate and at the
last, the rest are interpolated linearly in **log time**, which is how gates are
spaced. For log-spaced gates that coincides with interpolating on gate index; for
irregularly spaced ones it is the one that follows the physics.

Widths are counts of soundings rather than distances. That is the literal reading
of "widths at the first and last gate", and it is what the decimation parameter —
which *is* a distance — complements.

Uncertainty informs the weighting and is propagated
---------------------------------------------------
``01``: *"Where per-datum uncertainties exist they inform the weighting; where
they do not, the spread within the window is used to estimate one. Output always
carries uncertainty as a fraction of the value."*

Weights are the taper times inverse variance, so a noisy neighbour pulls less. The
propagated variance of a weighted mean is ``Σ(w²σ²) / (Σw)²``. Where no
uncertainty exists, the weighted spread *within* the window estimates one — which
is a different quantity and is reported as such, since it measures disagreement
among neighbours rather than any instrument's stated error.

Output is relative, as the format requires everywhere else.

Nothing crosses the turn between lines
--------------------------------------
A window that spanned two lines would average a sounding with one flown minutes
earlier and hundreds of metres away. Each line is convolved separately.
"""

import logging

import numpy as np
import pandas as pd

from ..distribution import summarize
from ..dataset import STASH_PREFIX, stash_key
from ..mask import CULLED, apply_policy, cull, in_use, mark_step
from ..pipeline import StepReport

log = logging.getLogger(__name__)

CATEGORY = "averaging"

STEP = "average_along_line"

#: This filter's number in ``building_plans/01``, used as its provenance code.
STEP_CODE = 10

HANN, GAUSSIAN, BOXCAR = "hann", "gaussian", "boxcar"
METHODS = (HANN, GAUSSIAN, BOXCAR)


def kernel(width: int, method: str = HANN) -> np.ndarray:
    """The averaging taper, normalized to sum to one.

    A width of one is ``[1.0]`` for every method, which is what makes the no-op
    fall out rather than being special-cased.

    Parameters
    ----------
    width : int
        Window width in soundings; made odd so the window is centred.
    method : str
        ``hann``, ``gaussian`` or ``boxcar``.

    Returns
    -------
    numpy.ndarray
        The taper, ``width`` long (or one more), summing to one.
    """
    width = int(width)
    if width < 1:
        raise ValueError(f"A window width is a count of soundings and must be "
                         f"at least 1; got {width}.")
    if width == 1:
        return np.ones(1)

    if method == BOXCAR:
        shape = np.ones(width)
    elif method == HANN:
        # Zero at both ends, so a sounding entering the window contributes
        # nothing at first and grows in. np.hanning's endpoints are exactly 0,
        # which would waste two samples, so the interior is used.
        shape = np.hanning(width + 2)[1:-1]
    elif method == GAUSSIAN:
        # Truncated at two sigma, which puts the edge weight at ~13% of centre.
        half = (width - 1) / 2.0
        offsets = np.arange(width) - half
        shape = np.exp(-0.5 * (offsets / (half / 2.0 or 1.0)) ** 2)
    else:
        raise ValueError(
            f"Unknown method {method!r}. Available: {', '.join(METHODS)}")
    return shape / shape.sum()


def gate_widths(first: int, last: int, times) -> np.ndarray:
    """Per-gate window widths, interpolated linearly in log time.

    Parameters
    ----------
    first, last : int
        The width at the first and last gate.
    times : array_like
        Gate centre times; only their log spacing matters.

    Returns
    -------
    numpy.ndarray
        One integer width per gate.
    """
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        return np.full(max(times.size, 1), int(first))

    usable = times > 0
    span = np.zeros(times.size)
    if usable.sum() >= 2:
        logs = np.log(np.where(usable, times, np.nan))
        low, high = np.nanmin(logs), np.nanmax(logs)
        span = np.where(usable, (logs - low) / (high - low) if high > low else 0.0, 0.0)
    widths = np.rint(first + span * (last - first)).astype(int)
    return np.maximum(widths, 1)


def _convolve(values, weights, taper):
    """Weighted sum and the sums its mean and variance need, along one line.

    Returns four arrays, one value per sounding: ``Σkwx``, ``Σkw``, ``Σk²w²``
    (Kish's denominator, for the effective sample size) and ``Σk²w`` (the
    variance numerator), where ``k`` is the taper and ``w`` the weights. The
    window is centred and edges are handled by shortening rather than by
    wrapping or zero-padding the data — a sounding at the start of a line has
    fewer neighbours, and that is a fact about it rather than a reason to
    invent some.
    """
    def full(series):
        return np.convolve(np.asarray(series, dtype=float), taper, mode="same")

    weights = np.asarray(weights, dtype=float)
    squared_taper = np.asarray(taper, dtype=float) ** 2

    weighted = full(np.where(np.isfinite(values), values, 0.0) * weights)
    total = full(weights)
    # Kish's denominator, and the variance numerator. They are different sums
    # and were conflated in a first draft, which made the propagated
    # uncertainty of a tapered average larger than the data it came from.
    squared = np.convolve(weights ** 2, squared_taper, mode="same")
    for_variance = np.convolve(weights, squared_taper, mode="same")
    return weighted, total, squared, for_variance


def average_along_line(dataset, width_first: int, width_last: int | None = None,
                       method: str = HANN, min_valid_fraction: float = 0.5,
                       decimate_to: float | None = None,
                       line_column: str | None = None, stash: bool = True):
    """Average each sounding with its along-line neighbours.

    Parameters
    ----------
    dataset : Dataset
        The dataset to average; data and uncertainty arrays are replaced, and
        with ``decimate_to`` soundings are dropped.
    width_first : int
        Window width in soundings at the first gate.
    width_last : int | None
        Width at the last gate. Equal to ``width_first`` if unset, giving a
        constant window.
    method : str
        ``hann`` (default), ``gaussian`` or ``boxcar``. The last is a flat
        kernel and is kept for reproducing a previous run, not because it is a
        good choice.
    min_valid_fraction : float
        The **effective** sample size a window must reach, as a fraction of its
        width, before an output is produced. Effective, not counted — see the
        module docstring.
    decimate_to : float | None
        Keep only soundings this far apart along the line, in CRS units. Drops
        soundings, changing the dataset's extent.
    line_column : str | None
        Which column identifies the line. Discovered if unset.
    stash : bool
        Keep the unaveraged values and uncertainties alongside.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if method not in METHODS:
        raise ValueError(
            f"Unknown method {method!r}. Available: {', '.join(METHODS)}")
    if not 0 < min_valid_fraction <= 1:
        raise ValueError(
            f"min_valid_fraction is a fraction of the window and must be in "
            f"(0, 1]; got {min_valid_fraction:g}.")
    width_last = width_first if width_last is None else width_last
    for name, value in (("width_first", width_first), ("width_last", width_last)):
        if int(value) < 1:
            raise ValueError(f"{name} is a count of soundings and must be at "
                             f"least 1; got {value}.")

    if frame is None or len(frame) == 0 or not dataset.channels:
        report.notes.append("No soundings or no channels; nothing to average.")
        return dataset, report

    xyz = dataset.survey.xyz
    line_column = line_column or xyz.line_id_column
    lines = (frame[line_column].to_numpy() if line_column in frame.columns
             else np.zeros(len(frame)))
    if line_column not in frame.columns:
        report.notes.append(
            "No line identifier; the whole delivery was treated as one line, so "
            "a window may span the turn between two.")

    per_channel, culled_any = {}, False
    for channel in dataset.channels:
        values = dataset.layer_data[channel.data_key].to_numpy(dtype=float)
        times = np.asarray(channel.axis.values, dtype=float)
        if times.size != values.shape[1]:
            continue

        has_std = bool(channel.std_key and channel.std_key in dataset.layer_data)
        relative = (dataset.layer_data[channel.std_key].to_numpy(dtype=float)
                    if has_std else None)
        live = in_use(dataset, channel)

        if stash:
            # This writes values, uncertainty and the mask, so what was
            # delivered is kept under the orig_ prefix.
            dataset.layer_data[stash_key(channel.data_key)] = \
                dataset.layer_data[channel.data_key].copy()
            if has_std:
                dataset.layer_data[stash_key(channel.std_key)] = \
                    dataset.layer_data[channel.std_key].copy()

        widths = gate_widths(int(width_first), int(width_last), times)
        averaged = np.full(values.shape, np.nan)
        propagated = np.full(values.shape, np.nan)
        thin = np.zeros(values.shape, dtype=bool)
        spread_used = False

        for gate in range(values.shape[1]):
            taper = kernel(int(widths[gate]), method)
            column = values[:, gate]
            usable = np.isfinite(column) & live[:, gate]

            # Inverse variance where an uncertainty exists, so a noisy
            # neighbour pulls less. Absolute, since the stored figure is
            # relative to the value.
            if has_std:
                sigma = np.abs(relative[:, gate]) * np.abs(column)
                stated = usable & (sigma > 0) & np.isfinite(sigma)
                weight = np.where(stated, 1.0 / np.where(stated, sigma, 1.0) ** 2,
                                  0.0)
                # A usable datum whose sigma is zero or absent gets the typical
                # weight rather than none. Zero sigma is not certainty — it is
                # what a *relative* uncertainty gives for a value of zero — and
                # excluding those data drops every genuine zero from the
                # average. Infinite weight is the opposite failure and would
                # make one datum the whole answer.
                typical = np.median(weight[stated]) if stated.any() else 1.0
                weight = np.where(usable & ~stated, typical, weight)
            else:
                weight = usable.astype(float)

            for key in np.unique(lines):
                rows = lines == key
                if rows.sum() == 0:
                    continue
                total_wx, total_w, total_w2, for_variance = _convolve(
                    column[rows], weight[rows], taper)

                with np.errstate(divide="ignore", invalid="ignore"):
                    mean = np.where(total_w > 0, total_wx / total_w, np.nan)
                    # Kish: equals the count for flat weights and falls toward
                    # one as the weight concentrates.
                    effective = np.where(total_w2 > 0,
                                         total_w ** 2 / total_w2, 0.0)

                # A hair of tolerance: a genuinely full window computes an
                # effective size of 4.999999999999998, so an exact comparison
                # rejects every window at min_valid_fraction of 1.
                needed = min_valid_fraction * widths[gate]
                enough = effective >= needed * (1 - 1e-9)
                averaged[rows, gate] = np.where(enough, mean, np.nan)
                thin[rows, gate] = ~enough

                if has_std:
                    # Var of a weighted mean is sum(W^2 s^2) / (sum W)^2, and
                    # with W = k/s^2 that is sum(k^2 w) / (sum k w)^2. It is
                    # *not* 1/sum(w): that holds only for a flat kernel, and
                    # under a Hann taper it overstates the result by 15%.
                    with np.errstate(divide="ignore", invalid="ignore"):
                        variance = np.where(total_w > 0,
                                            for_variance / total_w ** 2, np.nan)
                else:
                    # No stated uncertainty, so the disagreement among
                    # neighbours estimates one. A different quantity, reported
                    # as such.
                    spread_used = True
                    residual = np.where(np.isfinite(column[rows]), column[rows], 0.0)
                    sq_wx, sq_w, _, _ = _convolve(residual ** 2, weight[rows], taper)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        second = np.where(sq_w > 0, sq_wx / sq_w, np.nan)
                        variance = np.maximum(second - mean ** 2, 0.0)
                with np.errstate(divide="ignore", invalid="ignore"):
                    propagated[rows, gate] = np.sqrt(variance) / np.abs(mean)

        index = dataset.layer_data[channel.data_key].index
        columns = dataset.layer_data[channel.data_key].columns
        dataset.layer_data[channel.data_key] = pd.DataFrame(
            averaged, index=index, columns=columns)
        if has_std:
            dataset.layer_data[channel.std_key] = pd.DataFrame(
                np.where(np.isfinite(propagated), propagated, np.nan),
                index=index, columns=columns)

        outcome = {
            "widths": {"first": int(widths[0]), "last": int(widths[-1])},
            "method": method,
            "weighted_by_uncertainty": has_std,
            "spread_estimated_uncertainty": spread_used,
            "too_thin": int(thin.sum()),
            "values": summarize(averaged[np.isfinite(averaged)]),
        }
        if thin.any():
            culled_any = True
            outcome.update(cull(dataset, channel, thin & live,
                                step_code=STEP_CODE))
        per_channel[str(channel)] = outcome

    if culled_any:
        mark_step(dataset, STEP_CODE, STEP)

    decimated = None
    if decimate_to:
        decimated = _decimate(dataset, lines, decimate_to, report)

    report.distributions = {
        "per_channel": per_channel,
        "method": method,
        "min_valid_fraction": float(min_valid_fraction),
        "decimated": decimated,
        "stashed": STASH_PREFIX if stash else None,
        "provenance_code": STEP_CODE,
    }
    log.info("%s: averaged %d channel(s) with a %s kernel",
             STEP, len(per_channel), method)
    return dataset, report


def _decimate(dataset, lines, spacing: float, report) -> dict:
    """Keep soundings no closer than ``spacing`` along each line.

    Applied after averaging, never before: decimating first would throw away the
    neighbours the average is made from, which is the difference between
    smoothing and simply taking every nth sounding.
    """
    frame = dataset.flightlines
    xyz = dataset.survey.xyz
    if xyz.x_column not in frame.columns or xyz.y_column not in frame.columns:
        report.notes.append(
            "No position columns, so the result could not be decimated.")
        return None

    x = frame[xyz.x_column].to_numpy(dtype=float)
    y = frame[xyz.y_column].to_numpy(dtype=float)
    keep = np.zeros(len(frame), dtype=bool)

    for key in np.unique(lines):
        rows = np.flatnonzero(lines == key)
        if rows.size == 0:
            continue
        anchor = rows[0]
        keep[anchor] = True
        for row in rows[1:]:
            if np.hypot(x[row] - x[anchor], y[row] - y[anchor]) >= spacing:
                keep[row] = True
                anchor = row

    before = len(frame)
    dataset.survey.xyz.flightlines = frame[keep].reset_index(drop=True)
    for name, array in list(dataset.layer_data.items()):
        if len(array) == before:
            dataset.layer_data[name] = array[keep].reset_index(drop=True)
    return {"target_spacing": float(spacing), "before": before,
            "after": int(keep.sum())}
