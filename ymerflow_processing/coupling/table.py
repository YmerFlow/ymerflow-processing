"""The per-gate coupling table: what the detector found, whether or not it culled.

One table per channel, the same shape as the channel's mask, written on every
run. ``action: flag`` stops there; ``action: cull`` then applies it to the mask.
One code path, so a flagging run and a culling run cannot drift apart, and the
table survives both, which is what a second, manual pass needs to reverse
anything.

The values speak the provenance array's language (``mask.provenance``):

    ``0``   clean
    ``+1``  detected directly by the criterion
    ``-1``  swept up by the culling policy (``spread_earlier``, ``disable_tail``)

It holds the **post-policy** decision, what a cull would take, because the
point of flagging is to show what culling would do.

A prefix, never a suffix
------------------------
Channel naming patterns are anchored at the start of a key, so a suffixed
``dbdt_inuse_ch1gt_coupled`` would still match the in-use pattern and discovery
would find two masks for one channel. ``mask.PROVENANCE_PREFIX`` and
``dataset.STASH_PREFIX`` exist for the same reason.

Not an undeclared array
-----------------------
The table is gate-shaped and belongs to no channel, so anything detecting
undeclared data by shape would report it as something the delivery carried.
:func:`is_table` is the one predicate the inventory and ``select_moments`` use
to tell it apart.
"""

import numpy as np
import pandas as pd

from ..dataset import role_key
from ..mask import cull

#: Prefixed onto a channel's mask key to name its coupling table.
PREFIX = "coupled_"

CLEAN = 0
DIRECT = 1
COLLATERAL = -1


def is_table(key: str) -> bool:
    """Whether this ``layer_data`` key is a coupling table this library wrote.

    Parameters
    ----------
    key : str
        A ``layer_data`` key.

    Returns
    -------
    bool
        ``True`` for a coupling table.
    """
    return str(key).lower().startswith(PREFIX)


def key_for(inuse_key: str) -> str:
    """Where the coupling table for a channel with mask key ``inuse_key`` lives.

    Parameters
    ----------
    inuse_key : str
        The channel's in-use mask key.

    Returns
    -------
    str
        The ``layer_data`` key of its coupling table.
    """
    return f"{PREFIX}{inuse_key}"


def channel_key(channel) -> str:
    """The table key for ``channel``, whether or not its mask exists yet.

    Parameters
    ----------
    channel : Channel
        Whose table.

    Returns
    -------
    str
        The ``layer_data`` key of its coupling table.
    """
    return key_for(channel.inuse_key or role_key(channel, "inuse"))


def ensure(dataset, channel) -> pd.DataFrame:
    """This channel's coupling table, created all clean if absent.

    Shaped from the channel's data, not its mask, so a flagging run creates no
    mask where the delivery carried none.

    Parameters
    ----------
    dataset : Dataset
        Where the table lives, or is created.
    channel : Channel
        Whose table.

    Returns
    -------
    pandas.DataFrame
        ``int8``, soundings by gates; ``0`` clean, ``+1`` direct, ``-1``
        collateral.
    """
    key = channel_key(channel)
    if key not in dataset.layer_data:
        reference = dataset.layer_data[channel.data_key]
        dataset.layer_data[key] = pd.DataFrame(
            np.full(reference.shape, CLEAN, dtype=np.int8),
            index=reference.index, columns=reference.columns)
    return dataset.layer_data[key]


def write(dataset, channel, bad, direct) -> pd.DataFrame:
    """Replace this channel's table with a decision.

    Replaced, not accumulated, so re-running a flagging pass on the same data
    writes the same table.

    Parameters
    ----------
    dataset : Dataset
        Where the table is written.
    channel : Channel
        Whose table.
    bad : numpy.ndarray
        ``(soundings, gates)``, ``True`` for everything a cull would take: the
        culling policy's output.
    direct : numpy.ndarray
        The same shape, ``True`` where the criterion itself objected.

    Returns
    -------
    pandas.DataFrame
        The table as written.
    """
    reference = ensure(dataset, channel)
    bad = np.asarray(bad, dtype=bool)
    direct = np.asarray(direct, dtype=bool) & bad
    values = np.where(direct, DIRECT, np.where(bad, COLLATERAL, CLEAN)).astype(np.int8)
    if values.shape != reference.shape:
        raise ValueError(
            f"A coupling decision of shape {values.shape} for channel "
            f"{channel.key}, whose data is {reference.shape}.")
    table = pd.DataFrame(values, index=reference.index, columns=reference.columns)
    dataset.layer_data[channel_key(channel)] = table
    return table


def apply(dataset, channel, step_code: int) -> dict:
    """Cull what this channel's table marks, recording direct against collateral.

    Parameters
    ----------
    dataset : Dataset
        Whose mask and provenance are written.
    channel : Channel
        Whose table is applied.
    step_code : int
        The step's provenance code, 1..127.

    Returns
    -------
    dict
        What changed, as :func:`~ymerflow_processing.mask.cull` reports it.
    """
    values = ensure(dataset, channel).to_numpy()
    return cull(dataset, channel, bad=values != CLEAN, step_code=step_code,
                direct=values == DIRECT)
