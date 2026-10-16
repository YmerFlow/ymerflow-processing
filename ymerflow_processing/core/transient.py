"""The shape of a transient, measured over the gates that are still in use.

The machinery filter_25 to filter_28 are thin wrappers around. It exists as its
own module because getting it wrong is the failure mode those four share, and
because writing it four times would give four chances to get it wrong
differently.

Surviving gates, at their real times
------------------------------------
A decay rate is a difference between neighbours, and **a culled gate is not a
neighbour**. If gate 6 has been disabled, the slope at gate 7 is measured against
gate 5 — not against a hole, and not against gate 6's value as though it were
still there.

That makes the offset between the points variable, so the times cannot be
treated as evenly spaced. They are not evenly spaced anyway: gates are
logarithmic, so even an untouched transient has a different Δt at every step.
Computing a "second difference" as ``y[i-1] - 2y[i] + y[i+1]`` assumes uniform
spacing and is wrong on both counts — silently, because the result is a plausible
number.

So every quantity here is a divided difference over the actual ``log t`` of the
nearest surviving neighbours on each side.

Log-log, because that is where a transient is a line
-----------------------------------------------------
Over a homogeneous half-space the late-time response is a power law in time, so
in ``log t`` against ``log y`` it is a straight line and its **slope** is the
decay exponent. Ambient noise has a different characteristic slope, and real data
lies between them — which is what makes an out-of-range slope a physical
statement rather than a threshold on a number.

**Curvature** is the second derivative in that same space: how fast the decay
rate is itself changing. Where slope catches a transient decaying at an
impossible rate, curvature catches one *changing* its rate impossibly fast.

Where it is undefined, and why that is not a finding
-----------------------------------------------------
``NaN``, not zero, and not an objection:

- **The first and last surviving gate.** A central difference needs a neighbour
  on each side. ``01`` says this of curvature explicitly and it is equally true
  of slope.
- **A non-positive value.** ``log`` of it does not exist. A negative measurement
  is filter_32's finding to make; treating it as a shape problem here would
  double-count it and attribute it to the wrong cause.
- **Fewer than two, or three, surviving gates in the sounding.** There is nothing
  to difference.

None of these are "the shape is wrong". They are "the shape is unmeasurable
here", which is a different statement and must not become a cull.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


def neighbours(live) -> tuple[np.ndarray, np.ndarray]:
    """Index of the nearest surviving gate on each side, per datum.

    ``-1`` where there is none. Vectorised over soundings: the running maximum
    of "index if live else -1" is the previous survivor, and the same trick
    reversed gives the next one.
    """
    live = np.asarray(live, dtype=bool)
    rows, gates = live.shape
    index = np.broadcast_to(np.arange(gates), live.shape)

    forward = np.where(live, index, -1)
    previous = np.full(live.shape, -1)
    previous[:, 1:] = np.maximum.accumulate(forward, axis=1)[:, :-1]

    backward = np.where(live, index, gates)
    following = np.full(live.shape, -1)
    following[:, :-1] = np.minimum.accumulate(backward[:, ::-1], axis=1)[:, ::-1][:, 1:]
    following = np.where(following >= gates, -1, following)
    return previous, following


def _gather(values, index):
    """``values[row, index[row, gate]]``, with ``NaN`` where the index is -1."""
    safe = np.where(index < 0, 0, index)
    taken = np.take_along_axis(values, safe, axis=1)
    return np.where(index < 0, np.nan, taken)


def log_slope(values, times, live=None) -> np.ndarray:
    """``d(log y) / d(log t)`` at each datum, over surviving neighbours.

    Negative for a decaying transient, and typically between about -1 and -3 for
    airborne TEM over an earth. A central divided difference, so the uneven
    spacing of log-spaced gates is handled rather than assumed away.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    live = (np.ones(values.shape, dtype=bool) if live is None
            else np.asarray(live, dtype=bool))

    # log of a non-positive value does not exist. A negative measurement is
    # filter_32's finding; treating it as a shape problem would double-count it.
    usable = live & np.isfinite(values) & (values > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_y = np.where(usable, np.log(np.where(usable, values, 1.0)), np.nan)
    log_t = np.log(np.where(times > 0, times, np.nan))
    log_t = np.broadcast_to(log_t, values.shape)

    previous, following = neighbours(usable)
    y_before, y_after = _gather(log_y, previous), _gather(log_y, following)
    t_before, t_after = _gather(log_t, previous), _gather(log_t, following)

    with np.errstate(divide="ignore", invalid="ignore"):
        slope = (y_after - y_before) / (t_after - t_before)
    return np.where(usable, slope, np.nan)


def log_curvature(values, times, live=None) -> np.ndarray:
    """The second derivative of ``log y`` in ``log t``, over surviving neighbours.

    The non-uniform three-point form,

        2 · [ (y₂-y₁)/(x₂-x₁) − (y₁-y₀)/(x₁-x₀) ] / (x₂-x₀)

    which reduces to the familiar ``y₀ − 2y₁ + y₂`` only when the spacing is
    even. It never is here — gates are logarithmic and culling makes the gaps
    variable on top of that.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    live = (np.ones(values.shape, dtype=bool) if live is None
            else np.asarray(live, dtype=bool))

    usable = live & np.isfinite(values) & (values > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_y = np.where(usable, np.log(np.where(usable, values, 1.0)), np.nan)
    log_t = np.broadcast_to(np.log(np.where(times > 0, times, np.nan)),
                            values.shape)

    previous, following = neighbours(usable)
    y0, y2 = _gather(log_y, previous), _gather(log_y, following)
    x0, x2 = _gather(log_t, previous), _gather(log_t, following)
    y1 = np.where(usable, log_y, np.nan)
    x1 = np.where(usable, log_t, np.nan)

    with np.errstate(divide="ignore", invalid="ignore"):
        rising = (y2 - y1) / (x2 - x1)
        falling = (y1 - y0) / (x1 - x0)
        curvature = 2.0 * (rising - falling) / (x2 - x0)
    return np.where(usable, curvature, np.nan)
