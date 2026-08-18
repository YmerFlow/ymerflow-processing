"""The in-use mask, and the policy every culling filter shares.

R6 is the governing rule:

    **QC writes attributes. Culling writes the mask. Manual edits override the
    mask, and survive re-runs because they replay as a diff.**

This is the mask half. Nothing here deletes anything — a culled datum is marked
not-in-use, which is reversible and auditable, and the arrays keep their shape.

What the mask actually is
-------------------------
**Per channel, per gate, per sounding.** One array per channel, shaped
``(soundings, gates)``, ones for keep and zeros for dropped. It reads as a single
vector until you look at the shape, and the shape is what makes a whole-sounding
cull and a whole-gate cull two different regions of the same array: a sounding
cull writes a row across every channel, a gate cull writes a column.

There is no per-sounding flag. One source of truth, so nothing downstream can
consult one and miss the other — which is the failure mode that a second, coarser
flag guarantees eventually.

The mask is materialized lazily
-------------------------------
``libaarhusxyz`` does not create in-use arrays, and this library must run
standalone, so a dataset can arrive without them. They are created — all in use —
the first time something writes, and **not at load**, because R6 promises that
running every QC check leaves the data byte-identical. Creating arrays at load
would break that promise for every pipeline that only ever measures.

Under the platform the importer has already created them, all ones, for any
channel with gate data that arrived without one. Both paths land in the same
place; neither may be assumed.

**Do not assume the mask starts all ones.** Delivered data can arrive already
culled by the contractor's own processing, and that is preserved as-is. Every
step therefore reports what *it* removed against the mask **as found**, never
against the total, or a re-run double-counts.

Culling is subtractive
----------------------
A culling step only ever turns data off. That makes order-independence and R4
idempotency nearly free — ANDing a mask with the same predicate twice changes
nothing the second time — and it means no automated step can quietly undo a
human's judgement.

Exactly two things may turn data back on, and both say so explicitly: filter_21,
which enables or disables nominated gates outright, and filter_33, which replays
a manual edit. Someone doing full primary-field modeling legitimately wants the
early gates back.

Never NaN
---------
``NaN`` is never written into a mask. Downstream readers are inconsistent about
it — one path in simpeg tests ``inuse == 0``, where ``NaN`` is not zero and the
datum silently survives; another does ``& inuse``, which raises. Neither outcome
is wanted. ``NaN`` belongs only to the filter_33 diff, where it means "no
override" and is consumed before anything reaches an inversion.

Why the policy lives here
-------------------------
``building_plans/01`` observes that most culling filters share three behaviors
and asks for them once rather than thirteen times:

    - **a starting gate** — evaluation begins at a nominated gate and considers
      everything after it
    - **spread-earlier** — when a problem is found, optionally also disable some
      number of gates before it, since a bad gate usually means its neighbours
      are suspect
    - **tail disabling** — when a problem is found, optionally disable every
      later gate in that sounding, on the reasoning that a transient does not
      recover once it has gone bad

    In the rebuild these should be a shared concern — a policy applied around a
    predicate — not copy-pasted parameters. **Each culling filter is then just a
    predicate over the data.**

So a predicate returns "these points look wrong" and touches neither the mask nor
the provenance array. Applying the policy, ANDing the mask, recording who did it
and counting what changed all happen once, here.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: In use, and culled. Integers rather than booleans because that is what both
#: naming conventions carry on disk and what the importer writes.
IN_USE = 1
CULLED = 0

#: Prefixed onto a mask key to name its provenance array.
#:
#: A **prefix**, not a suffix, and that is not cosmetic. Channel naming patterns
#: are anchored at the start of a key — ``^dbdt_inuse_ch(\d+)`` — so a suffixed
#: ``dbdt_inuse_ch1gt_why`` still matches the in-use pattern and discovery finds
#: two candidates for one channel's mask. It warned and picked correctly, but a
#: name that can be mistaken for the thing it describes is a trap waiting for a
#: pattern change. Prefixing puts it outside every anchored pattern.
PROVENANCE_PREFIX = "why_"

#: No step has claimed this datum.
UNTOUCHED = 0


#: Where the provenance codes are explained, so an array of 20s and -24s can be
#: read back later. The same lesson the verdict legend taught: a column of
#: numbers with no key is worse than no column, and the file has to carry its
#: own legend because whoever opens it will not have the run report.
STEP_REGISTRY = "cull_step_codes"


def provenance_key(inuse_key: str) -> str:
    """Where the provenance for ``inuse_key`` lives."""
    return f"{PROVENANCE_PREFIX}{inuse_key}"


def mark_step(dataset, code: int, step: str) -> None:
    """Record which step a provenance code stands for."""
    known = step_codes(dataset)
    if known.get(code) not in (None, step):
        raise ValueError(
            f"Provenance code {code} is already recorded as "
            f"{known[code]!r}; {step!r} cannot reuse it. Each culling step "
            f"needs its own code."
        )
    known[code] = step
    dataset.survey.xyz.model_info[STEP_REGISTRY] = " ".join(
        f"{c}:{n}" for c, n in sorted(known.items()))


def step_codes(dataset) -> dict[int, str]:
    """Provenance code to step name, as recorded on the dataset."""
    raw = dataset.survey.xyz.model_info.get(STEP_REGISTRY) or ""
    if not isinstance(raw, str):
        raw = " ".join(map(str, raw))
    out: dict[int, str] = {}
    for entry in raw.split():
        code, _, name = entry.partition(":")
        if name:
            out[int(code)] = name
    return out


def resolve(dataset, channel) -> str:
    """This channel's mask key, creating an all-in-use array if there is none.

    Returns the key rather than mutating the channel, because :class:`Channel`
    is frozen and ``Dataset.channels`` rediscovers on every access — so writing
    the array under a discoverable name is what makes the next lookup find it,
    and there is nothing to keep in sync.

    Resolved through the channel's own discovered key rather than by asking
    ``libaarhusxyz``. ``XYZ.layer_data_inuse_name`` looks like it returns a key
    and does not — it returns the stub it searched with, which is not a key for
    the ``dbdt_inuse_ch1gt`` convention (YmerFlow/libaarhusxyz#5). Channel
    discovery keeps the key it actually matched.
    """
    key = channel.inuse_key or _inuse_key_for(channel)
    if key in dataset.layer_data:
        return key

    reference = dataset.layer_data[channel.data_key]
    dataset.layer_data[key] = pd.DataFrame(
        np.full(reference.shape, IN_USE, dtype=np.int8),
        index=reference.index, columns=reference.columns)
    log.info("created an all-in-use mask for %s (%s)", channel.key, key)
    return key


def ensure(dataset, channel) -> pd.DataFrame:
    """This channel's mask, created all-in-use if the delivery carried none.

    Created on demand rather than at load, so a pipeline that only measures
    leaves the data exactly as it found it — which is what R6 promises.
    """
    return dataset.layer_data[resolve(dataset, channel)]


def _inuse_key_for(channel) -> str:
    """A mask key matching the channel's own data key.

    Both conventions in circulation put the role in the same position, so the
    mask key is the data key with the role substituted rather than a name
    assembled from a channel number — which is what makes this work for a
    convention nobody has enumerated yet.
    """
    data = channel.data_key
    if data.lower().startswith("gate_"):
        return "InUse_" + data[len("Gate_"):]
    if "dbdt_" in data.lower():
        return data.replace("dbdt_", "dbdt_inuse_", 1)
    return f"{data}_inuse"


def provenance(dataset, channel) -> pd.DataFrame:
    """This channel's provenance array, created empty if absent.

    ``int8`` and the same shape as the mask it explains. A code identifies the
    step; the **sign** says how the datum died — positive where it failed the
    predicate itself, negative where it was collateral from that step's spread
    or tail. So ``-4`` reads as "went because step 4 found a problem later in
    this transient".

    That distinction is what makes over-culling measurable rather than a
    suspicion: a step whose collateral vastly outnumbers its direct hits has
    ``spread_earlier`` or ``disable_tail`` set too aggressively, and without it
    the only way to find out is to re-run with the knobs at zero.

    Sign is safe to overload here in a way it would not be in the mask itself.
    Nothing downstream branches on this array — it is ours, and diagnostic.
    """
    mask = ensure(dataset, channel)
    key = provenance_key(resolve(dataset, channel))
    if key not in dataset.layer_data:
        dataset.layer_data[key] = pd.DataFrame(
            np.full(mask.shape, UNTOUCHED, dtype=np.int8),
            index=mask.index, columns=mask.columns)
    return dataset.layer_data[key]


def in_use(dataset, channel) -> np.ndarray:
    """Boolean view of the mask: ``True`` where a datum is still usable.

    What a predicate reads when its answer depends on which neighbours survive —
    slope and curvature both do, since they are computed between gates and a
    culled gate is not a gate to compute across.
    """
    return ensure(dataset, channel).to_numpy() != CULLED


def apply_policy(bad, start_gate: int = 0, spread_earlier: int = 0,
                 disable_tail: bool = False) -> np.ndarray:
    """Widen a predicate's answer according to the shared culling policy.

    ``bad`` is ``(soundings, gates)`` and ``True`` where the predicate objected.
    Returns the same shape, ``True`` for everything that should end up culled.

    Applied per sounding along the gate axis, and per channel by the caller —
    never across channels, since each channel has its own transient.

    ``start_gate``
        Where the predicate is meaningful. Curvature is undefined at the first
        gate; early gates are contaminated by transmitter turn-off. Objections
        before it are discarded.
    ``spread_earlier``
        How many gates before a hit to take with it. **The physical claim is
        that contamination precedes its visible onset** — the gate where a
        transient turns negative is where the effect becomes large enough to
        flip the sign, not where it started.
    ``disable_tail``
        Everything after the earliest hit in that sounding goes, on the
        reasoning that a transient does not recover once it has gone bad. With
        this on, the whole policy collapses to a usable window per sounding
        rather than a scatter of holes.

    ``spread_earlier`` deliberately reaches *below* ``start_gate``, clamped at
    gate 0. ``start_gate`` says where the predicate is valid, and the claim that
    an effect precedes its onset does not stop respecting physics at an
    evaluation boundary. Culling below it is cheap anyway — filter_20 has
    usually disabled those already.
    """
    bad = np.asarray(bad, dtype=bool)
    if bad.ndim != 2:
        raise ValueError(f"Expected a (soundings, gates) array, got {bad.shape}")
    if start_gate < 0 or spread_earlier < 0:
        raise ValueError(
            f"start_gate and spread_earlier count gates and cannot be negative; "
            f"got start_gate={start_gate}, spread_earlier={spread_earlier}"
        )

    # Objections outside the evaluated window are discarded before anything
    # spreads from them, so start_gate bounds what the predicate may claim
    # rather than merely trimming the result afterwards.
    hits = bad.copy()
    hits[:, :start_gate] = False

    out = hits.copy()
    gates = out.shape[1]
    index = np.arange(gates)[None, :]
    struck = hits.any(axis=1)[:, None]

    if disable_tail:
        # From the earliest hit onward, and only in soundings that had one.
        first = hits.argmax(axis=1)[:, None]
        edge = np.maximum(first - spread_earlier, 0)
        out |= struck & (index >= edge)
    elif spread_earlier:
        # Each hit takes its own neighbours, since the gates between hits are
        # still being judged individually.
        for step in range(1, spread_earlier + 1):
            out[:, :-step] |= hits[:, step:]
    return out


def enable(dataset, channel, good, step_code: int) -> dict:
    """Turn ``good`` back on in the channel's mask, and record who did it.

    **The exception to subtractive culling, and deliberately hard to reach.**
    Only two steps may call this: filter_21, which nominates gates outright, and
    filter_33, which replays a manual edit. Everything else only ever removes,
    which is what keeps order-independence and R4 idempotency nearly free and
    what stops an automated step quietly undoing a human's judgement.

    The legitimate case is not hypothetical: the early gates filter_20 disables
    on the vendor's word are exactly what someone doing full primary-field
    modeling needs back. Writing a mask rather than removing data is what makes
    that possible at all.

    Provenance records the restoring step's code, and the mask is what
    disambiguates it: a non-zero code against a **culled** entry says why the
    datum is off, and the same code against an **in-use** entry says it was
    deliberately restored. The two arrays are read together, so no third
    encoding is needed to tell "never culled" from "culled and put back".
    """
    key = resolve(dataset, channel)
    mask = dataset.layer_data[key]
    why = provenance(dataset, channel)

    good = np.asarray(good, dtype=bool)
    if good.shape != mask.shape:
        raise ValueError(
            f"Asked to enable {good.shape} for channel {channel.key}, whose "
            f"mask is {mask.shape}."
        )
    if step_code <= 0 or step_code > 127:
        raise ValueError(
            f"A step code must be 1..127 so its negation fits an int8; "
            f"got {step_code}"
        )

    dead = mask.to_numpy() == CULLED
    restoring = good & dead                    # only what was actually off

    values = mask.to_numpy(copy=True)
    values[restoring] = IN_USE
    dataset.layer_data[key] = pd.DataFrame(
        values, index=mask.index, columns=mask.columns)

    codes = why.to_numpy(copy=True)
    codes[restoring] = step_code
    dataset.layer_data[provenance_key(key)] = pd.DataFrame(
        codes, index=why.index, columns=why.columns)

    return {
        "channel": str(channel),
        "culled_before": int(dead.sum()),
        "restored": int(restoring.sum()),
        "in_use_after": int((values != CULLED).sum()),
    }


def cull(dataset, channel, bad, step_code: int, direct=None) -> dict:
    """Turn ``bad`` off in the channel's mask, and record who did it.

    ``bad`` is the policy's output — everything to cull. ``direct`` is the
    predicate's own answer before the policy widened it; where they differ, the
    difference is collateral and is recorded with a negative code.

    **Subtractive.** Already-culled data stays culled and is not re-attributed,
    so the provenance names the step that *first* removed each datum, which
    under these semantics is the causal one. A later step that would also have
    culled it is not recorded — a bitfield would record all of them and answers
    a question nobody has asked.

    Returns what changed, measured against the mask **as found**: a delivery may
    arrive already culled by the contractor, and counting against the total
    would claim credit for someone else's work.
    """
    key = resolve(dataset, channel)
    mask = dataset.layer_data[key]
    why = provenance(dataset, channel)

    bad = np.asarray(bad, dtype=bool)
    if bad.shape != mask.shape:
        raise ValueError(
            f"Predicate answered {bad.shape} for channel {channel.key}, whose "
            f"mask is {mask.shape}. A predicate must answer per sounding per "
            f"gate."
        )
    if step_code <= 0 or step_code > 127:
        raise ValueError(
            f"A step code must be 1..127 so its negation fits an int8; "
            f"got {step_code}"
        )

    live = mask.to_numpy() != CULLED
    removing = bad & live                      # only what this step actually takes
    direct = live & (np.asarray(direct, dtype=bool) if direct is not None else bad)

    values = mask.to_numpy(copy=True)
    values[removing] = CULLED
    dataset.layer_data[key] = pd.DataFrame(
        values, index=mask.index, columns=mask.columns)

    codes = why.to_numpy(copy=True)
    codes[removing & direct] = step_code
    codes[removing & ~direct] = -step_code
    dataset.layer_data[provenance_key(key)] = pd.DataFrame(
        codes, index=why.index, columns=why.columns)

    return {
        "channel": str(channel),
        "in_use_before": int(live.sum()),
        "removed": int(removing.sum()),
        "removed_direct": int((removing & direct).sum()),
        "removed_collateral": int((removing & ~direct).sum()),
        "in_use_after": int((values != CULLED).sum()),
        # Soundings left with nothing usable at all, which is what filter_23
        # exists to notice and what a reader wants to see happening early.
        "soundings_emptied": int(((values != CULLED).sum(axis=1) == 0).sum()
                                 - (live.sum(axis=1) == 0).sum()),
    }


def summarize(dataset) -> dict:
    """The mask as it stands, per channel. What qc_12 reports on."""
    out = {}
    for channel in dataset.channels:
        key = channel.inuse_key
        if key is None or key not in dataset.layer_data:
            continue
        values = dataset.layer_data[key].to_numpy()
        total = int(values.size)
        used = int((values != CULLED).sum())
        out[str(channel)] = {
            "total": total,
            "in_use": used,
            "culled": total - used,
            "culled_fraction": (total - used) / total if total else None,
            "soundings_empty": int(((values != CULLED).sum(axis=1) == 0).sum()),
        }
    return out
