"""Lining two runs up so their data can be compared, without a shared schema.

Shared by every comparison that reaches into datasets rather than reports —
``implementation_equivalence``, ``version_diff``, ``parameter_sweep`` — because
all three face the same problem and none of them may solve it by column name.

Two runs of the same pipeline agree on their schema. Two *implementations* need
not, and even one pipeline run twice can drop a sounding, cull a gate or reorder
its output. So identity is taken from what the instrument recorded rather than
from what a file happens to call things:

**Soundings** by timestamp where there is one, else by position on the ground,
else — only when both sides have the same count — by order, which is said out
loud because it is an assumption rather than a match.

Timestamp first is not a preference. A timestamp is an identity: the instrument
recorded it, and it survives re-export, renaming and reordering. Position is a
*near*-identity with one specific way of being wrong — a rigid offset between two
runs by some multiple of the sounding spacing re-pairs each sounding with a
neighbor rather than failing to pair, leaving the pairing monotone and the
numbers plausible. Only the overlap shrinks, which is why a large unmatched
fraction is reported rather than passed over.

**Channels** by their gate times, which is what a channel physically *is*. An
instrument's second moment is its second moment whether the file calls it
``ch2``, ``HM`` or nothing at all.

**Gates** by center time within a tolerance, so a run that dropped three gates is
compared over the ones it kept rather than shifted against the other side and
declared wrong everywhere.

**Quantities** semantically — ``data``, ``uncertainty``, ``mask`` — since the two
sides may not share a column name for any of them.
"""

import numpy as np

#: The quantities compared, named semantically rather than by column.
DATA = "data"
UNCERTAINTY = "uncertainty"
MASK = "mask"
QUANTITIES = (DATA, UNCERTAINTY, MASK)

#: How closely gate center times must agree to be the same gate.
GATE_RTOL = 1e-3

#: How close two soundings must be on the ground to be the same sounding, in CRS
#: units, when there is no timestamp to match on.
POSITION_TOLERANCE = 0.5

#: The unmatched fraction above which position matching says so. Not an error —
#: two runs may genuinely have kept different soundings — but it is also what a
#: rigid coordinate offset looks like, and that one is worth naming out loud.
UNMATCHED_SUSPICIOUS = 0.10


# ── Aligning the two sides ───────────────────────────────────────────────────

def sounding_keys(dataset, position_tolerance):
    """What identifies each sounding, most trustworthy form first.

    A timestamp is an identity: the instrument recorded it, and it survives
    re-export, renaming and reordering. Position is a near-identity, good to
    whatever tolerance the two sides round coordinates to. Order is neither, so
    it is used only as a last resort and always said out loud.
    """
    frame = dataset.flightlines
    xyz = dataset.survey.xyz

    stamp = xyz.get_column("timestamp")
    if stamp and stamp in frame.columns:
        values = frame[stamp].to_numpy(dtype=float)
        if np.isfinite(values).all():
            # Microseconds, from days. Two implementations that both round trip
            # through a text format should agree to well within that.
            return [round(float(v) * 86_400e6) for v in values], "timestamp"

    if xyz.x_column in frame.columns and xyz.y_column in frame.columns:
        x = frame[xyz.x_column].to_numpy(dtype=float)
        y = frame[xyz.y_column].to_numpy(dtype=float)
        if np.isfinite(x).all() and np.isfinite(y).all():
            grid = float(position_tolerance)
            return [(round(a / grid), round(b / grid)) for a, b in zip(x, y)], \
                "position"

    return None, None


def align_soundings(candidate, baseline, position_tolerance, report):
    """Row indices into each side that refer to the same soundings."""
    keys_c, how = sounding_keys(candidate, position_tolerance)
    keys_b, how_b = sounding_keys(baseline, position_tolerance)

    if keys_c is None or keys_b is None or how != how_b:
        n_c, n_b = len(candidate.flightlines), len(baseline.flightlines)
        if n_c != n_b:
            raise ValueError(
                f"The two sides share no timestamp or position to match "
                f"soundings on, and hold different numbers of them "
                f"({n_c} and {n_b}). There is no alignment to fall back to."
            )
        report.notes.append(
            "Neither side carries a timestamp or coordinates both could be "
            "matched on, so soundings are paired BY ORDER. That is an "
            "assumption, not a match: if either implementation reordered or "
            "dropped a sounding, every comparison below is against the wrong "
            "row and nothing in the numbers would say so."
        )
        rows = np.arange(n_c)
        return rows, rows, "order"

    index_b = {}
    for row, key in enumerate(keys_b):
        index_b.setdefault(key, []).append(row)

    pairs, unmatched_c = [], 0
    for row, key in enumerate(keys_c):
        rows = index_b.get(key)
        if rows:
            pairs.append((row, rows.pop(0)))
        else:
            unmatched_c += 1
    unmatched_b = sum(len(rows) for rows in index_b.values())

    if not pairs:
        raise ValueError(
            f"No sounding in the candidate matched one in the baseline by "
            f"{how}. The two sides are not the same survey, or the tolerance "
            f"is too tight."
        )
    if unmatched_c or unmatched_b:
        report.notes.append(
            f"{unmatched_c} sounding(s) exist only in the candidate and "
            f"{unmatched_b} only in the baseline, matched by {how}. Those are "
            f"not compared — a sounding one side dropped is a difference in "
            f"what was kept, which the mask comparison measures, not a "
            f"difference in a value."
        )

    # Position is a near-identity, not an identity, and it has one specific way
    # of being wrong: a rigid offset between the two runs by some multiple of
    # the sounding spacing re-pairs each sounding with a *neighbor* rather than
    # failing to pair. The pairing stays monotone and the values stay plausible,
    # so nothing downstream looks wrong — only the overlap shrinks, by roughly
    # the offset in soundings. That cannot be detected from here, but it can be
    # named where it would show, which is a large unmatched fraction.
    if how == "position" and pairs:
        worst = max(unmatched_c, unmatched_b) / (len(pairs) + max(unmatched_c,
                                                                 unmatched_b))
        if worst > UNMATCHED_SUSPICIOUS:
            report.notes.append(
                f"{worst:.0%} of soundings matched by position had no partner. "
                f"A genuine difference in what each run kept looks like this — "
                f"and so does a rigid offset between the two, which would pair "
                f"every sounding with a neighbor instead of itself and leave "
                f"the numbers looking reasonable. Check that the two runs share "
                f"a coordinate reference before reading anything below."
            )

    rows_c = np.array([p[0] for p in pairs])
    rows_b = np.array([p[1] for p in pairs])
    return rows_c, rows_b, how


def align_channels(candidate, baseline, report):
    """Channel pairs, matched on gate times rather than on names.

    A channel *is* its gate times. Two implementations may call the second
    moment ``ch2``, ``HM`` or nothing at all, and the times settle it.
    """
    remaining = list(baseline.channels)
    pairs, unmatched = [], []

    for channel in candidate.channels:
        times = axis_of(channel)
        best, score = None, 0
        for other in remaining:
            overlap = gate_overlap(times, axis_of(other))
            if overlap is not None and len(overlap[0]) > score:
                best, score = other, len(overlap[0])
        if best is None:
            # No axis on either side, or nothing in common. Fall back to the
            # index the instrument numbers them by.
            best = next((o for o in remaining if o.index == channel.index), None)
            if best is not None and times is None and axis_of(best) is None:
                report.notes.append(
                    f"{channel} was matched to {best} by channel index, since "
                    f"neither side carries gate times to match on.")
        if best is None:
            unmatched.append(str(channel))
            continue
        remaining.remove(best)
        pairs.append((channel, best))

    if unmatched:
        report.notes.append(
            f"The baseline has no channel matching {', '.join(unmatched)}, so "
            f"they are not compared.")
    if remaining:
        report.notes.append(
            f"The candidate has no channel matching "
            f"{', '.join(str(c) for c in remaining)} in the baseline.")
    return pairs


def axis_of(channel):
    values = getattr(channel.axis, "values", None)
    if values is None:
        return None
    array = np.asarray(values, dtype=float).ravel()
    return array if array.size else None


def gate_overlap(times_c, times_b, rtol=GATE_RTOL):
    """Column indices into each side naming the same gates."""
    if times_c is None or times_b is None:
        return None
    cols_c, cols_b = [], []
    used = set()
    for i, t in enumerate(times_c):
        if not np.isfinite(t):
            continue
        close = np.nonzero(np.isclose(times_b, t, rtol=rtol, atol=0.0))[0]
        for j in close:
            if int(j) not in used:
                cols_c.append(i)
                cols_b.append(int(j))
                used.add(int(j))
                break
    if not cols_c:
        return None
    return np.array(cols_c), np.array(cols_b)


def key_for(channel, quantity):
    return {DATA: channel.data_key, UNCERTAINTY: channel.std_key,
            MASK: channel.inuse_key}[quantity]


def block(dataset, channel, quantity, rows, cols):
    key = key_for(channel, quantity)
    if not key or key not in dataset.layer_data:
        return None
    array = dataset.layer_data[key].to_numpy(dtype=float)
    return array[np.ix_(rows, cols)]
