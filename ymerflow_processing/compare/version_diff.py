"""What changed between two processing versions.

``02``'s first use for the comparison shape: *"the versioning story the product
already claims, currently demonstrated by eye."*

Demonstrated by eye is the problem. Two versions of a delivery differ in ways
nobody can see by flipping between two images — a hundredth of the soundings
regraded, four hundred more gates culled, an uncertainty floor moved — and the
question a reviewer actually has is not *do these differ* but **what did the
parameter change do**.

So this is not equivalence with the verdict removed. The two runs are *supposed*
to differ; a difference is the answer rather than the finding, and the report
carries no pass or fail at all.

It joins the two halves of the change
--------------------------------------
Both halves are already recorded. ``dataset.history`` carries each applied step
and its parameters, and the provenance array records which step culled each
datum. So the report can say **which parameters differ** and **which steps did
the culling that differs**, side by side, rather than leaving a reader to infer
the connection from two numbers.

That is the thing eyes cannot do. Spotting that a version culled more is easy;
attributing it to the step whose threshold moved is what makes the diff
actionable, and the dataset already knows.

What it measures
----------------
**The pipeline**, as steps added, removed, and reparameterized.

**The mask**, as data gained and lost, attributed per step. Gained matters as
much as lost: a version that culls *less* has loosened something, and that is
the change nobody looks for.

**The values**, over data both versions kept — since underneath a mask a run may
leave anything at all — as how much moved and by how much.

**The grades**, where both versions carry QC verdicts, as a migration between
tiers. A reviewer asking "did this get better" is asking this and nothing else,
and moving *both* ways at once is the interesting answer: a version can improve
the median while pushing a handful of soundings out of spec.
"""

import logging

import numpy as np

from ..dataset import Dataset
from ..mask import provenance, provenance_key, step_codes
from ..pipeline import StepReport
from ..qc.thresholds import TIER_POINTS, verdict_columns
from .align import (
    DATA, MASK, POSITION_TOLERANCE, QUANTITIES, UNCERTAINTY, align_channels,
    align_soundings, axis_of, block, gate_overlap,
)

log = logging.getLogger(__name__)

STEP = "version_diff"

#: Per ``02``'s table. No report carries per-datum arrays, and per-datum is
#: where a version difference actually lives.
CONSUMES = "datasets"


# ── The pipeline half ────────────────────────────────────────────────────────

def _pipeline_diff(new, old):
    """Steps added, removed and reparameterized between two runs.

    Positional rather than by name: a pipeline may run the same step twice with
    different parameters, and keying on the name alone would silently collapse
    the two.
    """
    steps_new = list(getattr(new, "history", []) or [])
    steps_old = list(getattr(old, "history", []) or [])

    added, removed, changed = [], [], []
    for position in range(max(len(steps_new), len(steps_old))):
        entry_new = steps_new[position] if position < len(steps_new) else None
        entry_old = steps_old[position] if position < len(steps_old) else None

        if entry_old is None:
            added.append({"at": position, "step": entry_new["step"],
                          "params": entry_new.get("params")})
        elif entry_new is None:
            removed.append({"at": position, "step": entry_old["step"],
                            "params": entry_old.get("params")})
        elif entry_new["step"] != entry_old["step"]:
            added.append({"at": position, "step": entry_new["step"],
                          "params": entry_new.get("params")})
            removed.append({"at": position, "step": entry_old["step"],
                            "params": entry_old.get("params")})
        else:
            params_new = entry_new.get("params") or {}
            params_old = entry_old.get("params") or {}
            moved = {
                key: {"old": params_old.get(key), "new": params_new.get(key)}
                for key in set(params_new) | set(params_old)
                if params_new.get(key) != params_old.get(key)
            }
            if moved:
                changed.append({"at": position, "step": entry_new["step"],
                                "parameters": moved})

    return {"added": added, "removed": removed, "reparameterized": changed,
            "steps_new": len(steps_new), "steps_old": len(steps_old)}


# ── The mask half ────────────────────────────────────────────────────────────

def _mask_diff(new, old, channel_new, channel_old, rows_n, rows_o,
               cols_n, cols_o):
    """What each version kept, and which step is responsible for the change."""
    kept_new = block(new, channel_new, MASK, rows_n, cols_n)
    kept_old = block(old, channel_old, MASK, rows_o, cols_o)
    if kept_new is None or kept_old is None:
        return None

    live_new = np.abs(kept_new) > 0
    live_old = np.abs(kept_old) > 0
    lost = live_old & ~live_new
    gained = live_new & ~live_old

    result = {
        "in_use_new": int(live_new.sum()),
        "in_use_old": int(live_old.sum()),
        "culled_by_the_new_version": int(lost.sum()),
        # A version that culls LESS has loosened something, and that is the
        # change nobody looks for.
        "restored_by_the_new_version": int(gained.sum()),
        "n": int(live_new.size),
    }

    # Both halves of the change, joined: which step did the culling that differs.
    codes = {**step_codes(old), **step_codes(new)}
    for side, dataset, channel, rows, cols, mask, field in (
            ("new", new, channel_new, rows_n, cols_n, lost, "culled_by_step"),
            ("old", old, channel_old, rows_o, cols_o, gained,
             "was_culled_by_step")):
        if not mask.any():
            continue
        key = provenance_key(channel.inuse_key or "")
        if key not in dataset.layer_data:
            continue
        why = provenance(dataset, channel).to_numpy()[np.ix_(rows, cols)]
        # The sign says how a datum died — the predicate itself, or collateral
        # from that step's spread. Attribution is by step, so magnitude.
        found = np.abs(why[mask]).astype(int)
        result[field] = {
            codes.get(code, f"code {code}"): int((found == code).sum())
            for code in np.unique(found) if code
        }
    return result


# ── The grade half ───────────────────────────────────────────────────────────

def _grade_diff(new, old, rows_n, rows_o):
    """How the QC verdicts moved, per verdict column both versions carry."""
    shared = sorted(set(verdict_columns(new)) & set(verdict_columns(old)))
    if not shared:
        return None

    moved = {}
    for column in shared:
        if (column not in new.flightlines.columns
                or column not in old.flightlines.columns):
            continue
        after = new.flightlines[column].to_numpy(dtype=float)[rows_n]
        before = old.flightlines[column].to_numpy(dtype=float)[rows_o]

        migration = {}
        for from_tier, from_code in TIER_POINTS.items():
            for to_tier, to_code in TIER_POINTS.items():
                if from_tier == to_tier:
                    continue
                count = int(((before == from_code) & (after == to_code)).sum())
                if count:
                    migration[f"{from_tier} -> {to_tier}"] = count
        if migration:
            moved[column] = migration
    return moved or None


# ── The check ────────────────────────────────────────────────────────────────

def version_diff(items: Dataset | list[Dataset], reference: Dataset | None = None,
                 quantities: list[str] = QUANTITIES,
                 relative: float = 1e-9, absolute: float = 0.0,
                 position_tolerance: float = POSITION_TOLERANCE):
    """What changed between two processing versions of the same data.

    Parameters
    ----------
    items : Dataset | list of Dataset
        The newer version — one dataset, or a one-item sequence.
    reference : Dataset
        The older version. Required: which of two versions is the newer one
        cannot be inferred, and getting it backwards reverses the sign of every
        finding while reporting the same magnitudes.
    quantities : list of str
        Which of ``data``, ``uncertainty``, ``mask`` to diff.
    relative : float
        Below this fractional change a value counts as unmoved. Tight by
        default — unlike an equivalence tolerance this is not a judgement about
        acceptability, only about float noise.
    absolute : float
        Absolute companion to ``relative``, for values that pass through zero.
    position_tolerance : float
        How close two soundings must be, in CRS units, to be the same sounding
        when there is no timestamp.

    Returns
    -------
    StepReport
        The comparison's report.

    Notes
    -----
    No pass or fail: the two versions are meant to differ, so a difference
    is the answer rather than the finding.
    """
    report = StepReport(step=STEP)

    if reference is None:
        raise ValueError(
            "version_diff needs a reference: the older version to diff "
            "against. Which of two versions is newer cannot be inferred, and "
            "getting it backwards reverses the sign of every finding while "
            "reporting the same magnitudes.")

    versions = list(items) if isinstance(items, (list, tuple)) else [items]
    if len(versions) != 1:
        raise ValueError(
            f"version_diff compares one version against one other; got "
            f"{len(versions)}. For a series varying one parameter, use "
            f"parameter_sweep.")
    new, old = versions[0], reference

    for side, dataset in (("newer", new), ("older", old)):
        if not hasattr(dataset, "layer_data"):
            raise TypeError(
                f"The {side} version is a {type(dataset).__name__}, not a "
                f"dataset. This check declares consumes: datasets — a version "
                f"difference lives in the arrays, which no report carries.")

    pipeline = _pipeline_diff(new, old)
    if not any((pipeline["added"], pipeline["removed"],
                pipeline["reparameterized"])):
        report.notes.append(
            "The two versions record the same steps with the same parameters. "
            "Any difference below came from something outside the pipeline — a "
            "different input, a different library version, or a step whose "
            "parameters were not recorded.")

    if not len(new.flightlines) or not len(old.flightlines):
        report.notes.append("One version has no soundings; only the pipeline "
                            "difference is reported.")
        report.distributions = {"pipeline": pipeline, "channels": {}}
        return report

    rows_n, rows_o, matched_by = align_soundings(
        new, old, position_tolerance, report)
    pairs = align_channels(new, old, report)

    channels, moved_total = {}, 0
    for channel_n, channel_o in pairs:
        width_n = new.layer_data[channel_n.data_key].shape[1]
        width_o = old.layer_data[channel_o.data_key].shape[1]
        overlap = gate_overlap(axis_of(channel_n), axis_of(channel_o))
        if overlap is None:
            cols_n = cols_o = np.arange(min(width_n, width_o))
        else:
            cols_n, cols_o = overlap
            fits = (cols_n < width_n) & (cols_o < width_o)
            cols_n, cols_o = cols_n[fits], cols_o[fits]
            if not cols_n.size:
                cols_n = cols_o = np.arange(min(width_n, width_o))

        entry = {"older_channel": str(channel_o),
                 "gates_compared": int(len(cols_n))}

        mask_change = _mask_diff(new, old, channel_n, channel_o,
                                 rows_n, rows_o, cols_n, cols_o)
        if MASK in quantities and mask_change is not None:
            entry[MASK] = mask_change

        # Values only where both versions kept the datum: underneath a mask a
        # run may leave anything at all, so a difference there measures nothing.
        live = None
        for dataset, channel, rows, cols in (
                (new, channel_n, rows_n, cols_n),
                (old, channel_o, rows_o, cols_o)):
            kept = block(dataset, channel, MASK, rows, cols)
            if kept is not None:
                here = np.abs(kept) > 0
                live = here if live is None else (live & here)

        for quantity in (q for q in quantities if q != MASK):
            values_n = block(new, channel_n, quantity, rows_n, cols_n)
            values_o = block(old, channel_o, quantity, rows_o, cols_o)
            if values_n is None or values_o is None:
                entry[quantity] = {
                    "note": f"Only the "
                            f"{'older' if values_n is None else 'newer'} "
                            f"version carries this quantity."}
                continue
            if live is not None:
                values_n = np.where(live, values_n, np.nan)
                values_o = np.where(live, values_o, np.nan)
            entry[quantity] = _value_diff(values_n, values_o, relative,
                                          absolute)
            moved_total += entry[quantity]["n_moved"]

        channels[str(channel_n)] = entry

    grades = _grade_diff(new, old, rows_n, rows_o)
    if grades:
        report.notes.append(
            "QC verdicts moved in both directions where that is shown below. A "
            "version can improve the median while pushing a handful of "
            "soundings out of spec, and the total alone would hide it.")

    report.soundings_total = len(new.flightlines)
    report.soundings_affected = moved_total
    report.distributions = {
        "pipeline": pipeline,
        "alignment": {"soundings_matched_by": matched_by,
                      "soundings_compared": int(len(rows_n)),
                      "channels_compared": len(pairs)},
        "channels": channels,
        "grades": grades,
        "tolerance": {"relative": relative, "absolute": absolute},
    }
    report.notes.append(
        "This is a diff, not a judgement: the two versions are meant to differ, "
        "so nothing here passes or fails.")

    log.info("%s: %d step change(s), %d datum(s) moved across %d channel(s)",
             STEP, len(pipeline["added"]) + len(pipeline["removed"])
             + len(pipeline["reparameterized"]), moved_total, len(pairs))
    return report


def _value_diff(new, old, relative, absolute):
    """How much moved, and by how much."""
    with np.errstate(invalid="ignore", divide="ignore"):
        difference = new - old
        magnitude = np.abs(difference)
        fraction = np.where(np.abs(old) > 0, magnitude / np.abs(old), np.nan)
        moved = magnitude > (absolute + relative * np.abs(old))

    comparable = np.isfinite(magnitude)
    moved = moved & comparable
    appeared = np.isfinite(new) & ~np.isfinite(old)
    vanished = ~np.isfinite(new) & np.isfinite(old)

    result = {
        "n_compared": int(comparable.sum()),
        "n_moved": int(moved.sum()),
        "fraction_moved": (float(moved.sum() / comparable.sum())
                           if comparable.any() else 0.0),
        "median_change": (float(np.nanmedian(difference[moved]))
                          if moved.any() else 0.0),
        "max_absolute_change": (float(np.nanmax(magnitude[moved]))
                                if moved.any() else 0.0),
        "max_relative_change": (float(np.nanmax(fraction[moved]))
                                if moved.any() and np.isfinite(
                                    fraction[moved]).any() else None),
    }
    # A value the newer version produced where the older had none is not a
    # change of magnitude, and averaging it in with ones that are would be
    # meaningless. Counted on its own.
    if appeared.any():
        result["n_appeared"] = int(appeared.sum())
    if vanished.any():
        result["n_vanished"] = int(vanished.sum())
    return result
