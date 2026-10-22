"""filter_23 — disable soundings left with too few usable gates to invert.

**What it does to the dataset:** turns off the remaining gates of any sounding
whose usable count has fallen below a minimum. It changes no measurement, no
uncertainty, and no array shape (R6).

``01``: "A sounding reduced to a handful of points cannot constrain a model, and
leaving it in produces confident nonsense." That last phrase is the reason this
exists rather than being left to the inversion. Too few gates does not make an
inversion fail — it makes it succeed, quickly, on a model the data never
constrained. Nothing downstream reports a problem, so nothing prompts anyone to
look at the result sceptically.

It runs late, and that is not a preference
------------------------------------------
``01`` says it "runs late, since it depends on what earlier culling removed", and
this is the first step in the library whose *answer* depends on pipeline order
rather than merely its efficiency. Run first it does nothing, because nothing has
been removed yet; run last it sees the mask every other predicate left behind.

That makes it the natural closing step of a culling pipeline, and it is worth
saying plainly in a report rather than assuming: the count is taken against the
mask **as it stands when this runs**, so the same document with the steps
reordered legitimately produces a different answer.

Usable means what the inversion will actually use
-------------------------------------------------
A gate is counted when it is **in use and has a finite value and a finite
uncertainty** — the same three conditions the forward code applies when it builds
its data vector. Counting the mask alone would overstate what survives: a gate
left in use but carrying no measurement contributes nothing to constraining a
model, and a sounding of ten such gates is as empty as one with none.

Per channel, not per sounding
-----------------------------
The minimum is applied to each channel's own count. A dual-moment sounding whose
low moment has been reduced to three gates while its high moment still has thirty
is not a useless sounding — it is a sounding with one usable moment, and
disabling the other would throw away good data to punish its neighbour.

Whether an inversion can *use* a sounding with only one moment is a question
about how that inversion is configured, and it is not answerable here.

**There is no default minimum.** How many gates constrain a model depends on how
many layers the inversion solves for, which this library does not know.
"""

import logging

import numpy as np

from ..mask import cull, in_use, mark_step
from ..pipeline import StepReport

log = logging.getLogger(__name__)

CATEGORY = "cull soundings"

STEP = "cull_sparse_soundings"

#: This filter's number in ``building_plans/01``, used as its provenance code.
STEP_CODE = 23


def usable(dataset, channel) -> np.ndarray:
    """Which gates the inversion would actually use, per sounding per gate.

    In use, with a finite value and a finite uncertainty — the three conditions
    the forward code applies together. Counting the mask alone would overstate
    what survives.

    Parameters
    ----------
    dataset : Dataset
        The dataset read.
    channel : Channel
        The channel whose data, uncertainty and mask are combined.

    Returns
    -------
    numpy.ndarray
        Boolean, soundings by gates; ``True`` where all three hold.
    """
    live = in_use(dataset, channel)
    values = dataset.layer_data[channel.data_key].to_numpy(dtype=float)
    good = live & np.isfinite(values)

    if channel.std_key and channel.std_key in dataset.layer_data:
        std = dataset.layer_data[channel.std_key].to_numpy(dtype=float)
        good &= np.isfinite(std)
    return good


def cull_sparse_soundings(dataset, minimum_gates: int):
    """Disable soundings whose usable gate count has fallen below ``minimum``.

    Parameters
    ----------
    dataset : Dataset
        The dataset to cull; its in-use masks are written.
    minimum_gates : int
        Fewest usable gates a sounding may have and still be kept, per channel.
        Inclusive — a sounding with exactly this many is kept, since "at least
        N gates" is how the requirement is stated. **No default:** how many
        gates constrain a model depends on how many layers the inversion solves
        for.

    Returns
    -------
    tuple
        ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    minimum_gates = int(minimum_gates)
    if minimum_gates < 1:
        raise ValueError(
            f"minimum_gates counts gates and must be at least 1; got "
            f"{minimum_gates}. Zero would disable nothing and is better said by "
            f"leaving the step out."
        )

    channels = dataset.channels
    if not channels:
        report.notes.append("No channels; nothing to count.")
        return dataset, report

    per_channel, culled_any = {}, False
    # Which soundings lost a channel, in any channel. Summing the per-channel
    # counts would report more "soundings" than the delivery has, since one
    # sounding can be thinned in every channel independently.
    touched = np.zeros(len(frame), dtype=bool)
    for channel in channels:
        good = usable(dataset, channel)
        counts = good.sum(axis=1)

        # Already empty is not this step's doing, and counting it as such would
        # credit filter_23 with whatever actually emptied the sounding.
        already_empty = counts == 0
        sparse = (counts < minimum_gates) & ~already_empty

        outcome = {
            "minimum_gates": minimum_gates,
            "soundings_sparse": int(sparse.sum()),
            "soundings_already_empty": int(already_empty.sum()),
            "median_usable_gates": float(np.median(counts)) if counts.size else None,
        }

        if sparse.any():
            culled_any = True
            touched |= sparse
            # The whole row: a sounding below the minimum keeps nothing, since
            # the gates that remain are exactly the ones too few to be useful.
            across = np.repeat(sparse[:, None], good.shape[1], axis=1)
            outcome.update(cull(dataset, channel, across, step_code=STEP_CODE))
        else:
            outcome["removed"] = 0
        per_channel[str(channel)] = outcome

    if culled_any:
        mark_step(dataset, STEP_CODE, STEP)

    report.soundings_affected = int(touched.sum())
    report.distributions = {
        "per_channel": per_channel,
        # Distinct soundings that lost at least one channel. The per-channel
        # figures above say which, and add to more than this wherever a
        # sounding was thinned in more than one.
        "soundings_culled": int(touched.sum()),
        "channel_soundings_culled": sum(c["soundings_sparse"]
                                        for c in per_channel.values()),
        "provenance_code": STEP_CODE,
    }
    report.notes.append(
        "Counted against the mask as it stands. This step depends on what "
        "earlier culling removed, so it belongs last in a pipeline — run first "
        "it has nothing to find."
    )
    log.info("%s: disabled %d sounding(s) below %d usable gate(s)",
             STEP, int(touched.sum()), minimum_gates)
    return dataset, report
