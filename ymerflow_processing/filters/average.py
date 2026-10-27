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

The hybrid method — emerald-processing's, ported
------------------------------------------------
``method="hybrid"`` is Ben's ``rolling_hybrid_mean_df`` from emerald-processing,
vectorized. Per gate, per window of soundings:

1. Culled and non-finite neighbours drop out; a window needs
   ``ceil(min_valid_fraction × width)`` of them (at least two) — a **count**,
   as emerald counts, not the effective sample size the tapered methods use.
2. An alpha trim drops the highest and lowest ``alpha`` of the values
   (``floor(n × alpha)`` each end, so nothing is trimmed below 1 / alpha).
3. Inverse-variance weights, with each error raised to the larger of the
   window's median error and the line's median error for that gate, then
   capped at ``max_weight_factor`` × the median weight, so no quiet neighbour
   takes over.
4. The value is the weighted mean.
5. The error is the sum of squares total — the weighted mean of the stated
   variances plus the weighted scatter of the values about the mean:

       σ² = Σw·σᵢ² / Σw  +  Σw·(xᵢ − x̄)² / Σw

   It counts what the errors claim and how much the neighbours actually
   disagree, and is **not** divided by the number of neighbours.

That last point is a choice about the data, not an omission. Dividing by N
assumes the neighbours are independent, and full-rate deliveries that the
contractor has already filtered along the line are not: on the Estancia VTEM
finals, averaging ten neighbours lowers the measured scatter by about 1.16×,
not √10. ``independent_fraction`` says how independent they are: the error
variance is divided by ``max(1, independent_fraction × n)``. 0 (the default)
is emerald's behavior; 1 is fully independent neighbours, the standard error of
the mean.

The hybrid method's widths are always odd — the sounding plus the same number
of neighbours each side — rounded as emerald rounds them.

Widths ramp over the gates in use
---------------------------------
For every method, the ramp from ``width_first`` to ``width_last`` runs from the
first gate in use anywhere in the channel to the last, so ``width_first`` lands
on the first gate that is averaged rather than on a padded or disabled one.

Nothing crosses the turn between lines
--------------------------------------
A window that spanned two lines would average a sounding with one flown minutes
earlier and hundreds of metres away. Each line is convolved separately.
"""

import logging
import warnings

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

HANN, GAUSSIAN, BOXCAR, HYBRID = "hann", "gaussian", "boxcar", "hybrid"
METHODS = (HANN, GAUSSIAN, BOXCAR, HYBRID)


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


def round_to_odd(width) -> np.ndarray:
    """Round widths [soundings] to the nearest odd number, as emerald-processing does.

    ``ceil(w / 2) · 2 − 1``: 7 stays 7, 8 becomes 7, 8.1 becomes 9.
    """
    return (np.ceil(np.asarray(width, dtype=float) / 2) * 2 - 1).astype(int)


def gate_widths(first: int, last: int, times, active=None,
                odd: bool = False) -> np.ndarray:
    """Per-gate window widths [soundings], interpolated linearly in log time.

    Parameters
    ----------
    first, last : int
        The width at the first and last gate, in soundings.
    times : array_like
        Gate centre times [s]; only their log spacing matters.
    active : array_like of bool, optional
        Which gates are in use. The ramp runs from the first active gate to the
        last, so ``first`` lands on the first gate that is averaged. Gates
        before it get ``first`` and gates after it ``last``. All gates with a
        positive time if unset.
    odd : bool
        Round each width to the nearest odd number, so the window is the
        sounding plus the same number of neighbours on each side.

    Returns
    -------
    numpy.ndarray
        One integer width per gate, at least 1.
    """
    times = np.asarray(times, dtype=float)
    if times.size < 2:
        widths = np.full(max(times.size, 1), float(first))
        return np.maximum(round_to_odd(widths) if odd else widths.astype(int), 1)

    usable = times > 0
    if active is not None:
        usable &= np.asarray(active, dtype=bool)
    span = np.zeros(times.size)
    if usable.sum() >= 2:
        logs = np.log(np.where(times > 0, times, np.nan))
        low = np.nanmin(np.where(usable, logs, np.nan))
        high = np.nanmax(np.where(usable, logs, np.nan))
        if high > low:
            span = np.clip(np.where(times > 0, (logs - low) / (high - low), 0.0), 0.0, 1.0)
    exact = first + span * (last - first)
    widths = round_to_odd(exact) if odd else np.rint(exact).astype(int)
    return np.maximum(widths, 1)


def _windows(series, width):
    """Centered windows of ``width`` soundings along one line, NaN past its ends."""
    half = width // 2
    padded = np.concatenate([np.full(half, np.nan), np.asarray(series, dtype=float),
                             np.full(width - 1 - half, np.nan)])
    return np.lib.stride_tricks.sliding_window_view(padded, width)


def hybrid_mean(values, errors, width: int, min_valid_fraction: float = 0.35,
                alpha: float = 0.1, max_weight_factor: float = 10.0,
                independent_fraction: float = 0.0):
    """Ben's hybrid average along one line for one gate, vectorized.

    A port of emerald-processing's ``rolling_hybrid_mean_df``: alpha trim, then
    capped inverse-variance weights, then the sum-of-squares-total error. See
    the module docstring.

    Parameters
    ----------
    values : array_like
        The gate's values along the line, NaN where culled [data units].
    errors : array_like
        Absolute 1-sigma errors, same units, NaN where unknown. Zeros mean "no
        error information" for the weights, as emerald treats them.
    width : int
        Window width in soundings; odd widths are centered.
    min_valid_fraction : float
        Neighbours needed, as a fraction of ``width``: ``max(2, ceil(f × width))``.
    alpha : float
        Fraction trimmed from each end of the sorted values.
    max_weight_factor : float
        Cap on any weight, as a multiple of the window's median weight.
    independent_fraction : float
        How independent the neighbours are, 0 to 1. The error variance is
        divided by ``max(1, independent_fraction × n_kept)``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(mean, error, enough)``: the averaged value [data units], its
        absolute 1-sigma error [data units], and where the window had enough
        valid neighbours to produce a value.
    """
    values = np.asarray(values, dtype=float)
    errors = np.abs(np.asarray(errors, dtype=float))
    valid_one = np.isfinite(values) & np.isfinite(errors)
    if width <= 1:
        # The identity: the sounding with its own error, as emerald does.
        return (np.where(valid_one, values, np.nan),
                np.where(valid_one, errors, np.nan), valid_one)

    x, e = _windows(values, width), _windows(errors, width)
    valid = np.isfinite(x) & np.isfinite(e)
    n_valid = valid.sum(axis=1)
    min_periods = max(2, int(np.ceil(width * min_valid_fraction)))

    # Alpha trim: floor(n·alpha) from each end of the sorted valid values, and
    # none when fewer than three are valid or the trim would take them all.
    n_trim = np.floor(n_valid * alpha).astype(int)
    n_trim = np.where((n_valid < 3) | (2 * n_trim >= n_valid), 0, n_trim)
    rank = np.argsort(np.argsort(np.where(valid, x, np.inf), axis=1, kind="stable"),
                      axis=1, kind="stable")
    kept = valid & (rank >= n_trim[:, None]) & (rank < (n_valid - n_trim)[:, None])
    n_kept = kept.sum(axis=1)

    # The gate-level error floor: the line's median non-zero error at this gate.
    line_errors = errors[valid_one & (errors > 0)]
    gate_floor = float(np.median(line_errors)) if line_errors.size else 0.0

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN windows
        e_kept = np.where(kept, e, np.nan)
        window_floor = np.nanmedian(np.where(e_kept > 0, e_kept, np.nan), axis=1)
        floor = np.fmax(np.nan_to_num(window_floor, nan=0.0), gate_floor)
        has_floor = floor > 0
        weights = 1.0 / np.fmax(e_kept, floor[:, None]) ** 2
        cap = np.nanmedian(weights, axis=1) * max_weight_factor
        weights = np.fmin(weights, cap[:, None])
        weights = np.where(has_floor[:, None], weights, 1.0)      # no errors: equal
        weights = np.where(kept, weights, 0.0)

        x_kept = np.where(kept, x, 0.0)
        total = weights.sum(axis=1)
        mean = (weights * x_kept).sum(axis=1) / total
        # Floating-point cancellation: a mean twelve orders below the data it
        # came from is zero, not a ghost value.
        mean_abs = np.abs(x_kept).sum(axis=1) / n_kept
        mean = np.where((mean_abs > 0) & (np.abs(mean) < 1e-12 * mean_abs), 0.0, mean)

        stated = (weights * np.where(kept, e, 0.0) ** 2).sum(axis=1) / total
        scatter = (weights * np.where(kept, x - mean[:, None], 0.0) ** 2).sum(axis=1) / total
        variance = (stated + scatter) / np.maximum(1.0, independent_fraction * n_kept)

    enough = (n_valid >= min_periods) & (n_kept >= 2)
    return (np.where(enough, mean, np.nan), np.where(enough, np.sqrt(variance), np.nan),
            enough)


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
                       line_column: str | None = None, stash: bool = True,
                       alpha: float = 0.1, max_weight_factor: float = 10.0,
                       independent_fraction: float = 0.0):
    """Average each sounding with its along-line neighbours.

    Parameters
    ----------
    dataset : Dataset
        The dataset to average; data and uncertainty arrays are replaced, and
        with ``decimate_to`` soundings are dropped.
    width_first : int
        Window width in soundings at the first gate in use.
    width_last : int | None
        Width at the last gate in use. Equal to ``width_first`` if unset,
        giving a constant window.
    method : str
        ``hann`` (default), ``gaussian``, ``boxcar`` or ``hybrid``. ``hybrid``
        is emerald-processing's alpha-trimmed, capped inverse-variance mean
        with the sum-of-squares-total error, and always uses odd widths; see
        the module docstring. ``boxcar`` is a flat kernel kept for reproducing
        a previous run.
    min_valid_fraction : float
        For the tapered methods, the **effective** sample size a window must
        reach, as a fraction of its width. For ``hybrid``, the **count** of
        valid neighbours, ``max(2, ceil(fraction × width))``, as emerald counts
        it.
    decimate_to : float | None
        Keep only soundings this far apart along the line, in CRS units. Drops
        soundings, changing the dataset's extent.
    line_column : str | None
        Which column identifies the line. Discovered if unset.
    stash : bool
        Keep the unaveraged values and uncertainties alongside.
    alpha : float
        ``hybrid`` only: the fraction trimmed from each end of the sorted
        values in a window.
    max_weight_factor : float
        ``hybrid`` only: the cap on any weight, as a multiple of the window's
        median weight.
    independent_fraction : float
        ``hybrid`` only, 0 to 1: how independent neighbouring soundings are.
        The error variance is divided by ``max(1, fraction × n)``. 0 keeps
        emerald's error, the window's total spread; 1 is the standard error of
        the mean for fully independent soundings.

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
    if not 0 <= alpha < 0.5:
        raise ValueError(f"alpha is the fraction trimmed from EACH end and must be "
                         f"in [0, 0.5); got {alpha:g}.")
    if not max_weight_factor > 0:
        raise ValueError(f"max_weight_factor must be positive; got {max_weight_factor:g}.")
    if not 0 <= independent_fraction <= 1:
        raise ValueError(f"independent_fraction must be in [0, 1]; got "
                         f"{independent_fraction:g}.")

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

        widths = gate_widths(int(width_first), int(width_last), times,
                             active=live.any(axis=0), odd=method == HYBRID)
        averaged = np.full(values.shape, np.nan)
        propagated = np.full(values.shape, np.nan)
        thin = np.zeros(values.shape, dtype=bool)
        spread_used = False

        for gate in range(values.shape[1]):
            column = values[:, gate]
            usable = np.isfinite(column) & live[:, gate]

            if method == HYBRID:
                # Absolute errors; without a stated uncertainty, zero, which the
                # hybrid reads as "no error information": equal weights, and an
                # error from the scatter alone.
                sigma = (np.abs(relative[:, gate]) * np.abs(column) if has_std
                         else np.zeros(column.size))
                spread_used = spread_used or not has_std
                for key in np.unique(lines):
                    rows = lines == key
                    mean, error, enough = hybrid_mean(
                        np.where(usable[rows], column[rows], np.nan),
                        np.where(usable[rows], sigma[rows], np.nan),
                        int(widths[gate]), min_valid_fraction=min_valid_fraction,
                        alpha=alpha, max_weight_factor=max_weight_factor,
                        independent_fraction=independent_fraction)
                    averaged[rows, gate] = mean
                    thin[rows, gate] = ~enough
                    with np.errstate(divide="ignore", invalid="ignore"):
                        propagated[rows, gate] = error / np.abs(mean)
                continue

            taper = kernel(int(widths[gate]), method)

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
            "widths": {"first": int(widths[0]), "last": int(widths[-1]),
                       "per_gate": [int(w) for w in widths]},
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
        "hybrid": ({"alpha": float(alpha), "max_weight_factor": float(max_weight_factor),
                    "independent_fraction": float(independent_fraction)}
                   if method == HYBRID else None),
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
