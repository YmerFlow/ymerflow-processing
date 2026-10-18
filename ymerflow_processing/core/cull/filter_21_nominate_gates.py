"""filter_21 — enable or disable nominated gates outright.

**What it does to the dataset:** sets named gates in the in-use mask, either
direction, for whole channels at a time. It changes no measurement, no
uncertainty, and no array shape (R6).

``01`` calls this "a manual override rather than a criterion", which is exactly
right and worth keeping in view: every other culling filter derives its answer
from the data or the system description, and this one derives it from a person
having decided. That is a legitimate thing to want and a dangerous thing to
leave implicit, so what it did is recorded per datum like everything else.

**It is one of only two steps that may turn data back on**, the other being
filter_33's manual edit. Everything else only ever removes.

Why enabling matters
--------------------
filter_20 disables the early gates on the vendor's word, because they sit in the
transmitter turn-off and cannot be inverted without modeling the system primary
field. Someone doing exactly that modeling needs them back. Writing a mask rather
than removing data is what makes that possible at all, and this is the step that
does it.

The same applies in reverse for a delivery whose description is wrong in the
other direction — gates the vendor did not flag but which are visibly unusable.

Naming gates
------------
Per channel, and a **set or a range** rather than one gate — ``01`` says "one
nominated gate", which is a limit nobody wants and which R1 argues against
anyway. Accepted forms, mixable in a list:

``3``
    One gate.
``[0, 1, 5]``
    Those gates.
``"0-4"``
    Gates 0 through 4 **inclusive** — five gates.

**A range is inclusive and a count is not**, and the two live one filter apart,
so it is worth stating: ``RemoveInitialGates: 5`` means five gates, 0 through 4,
and ``"0-4"`` names the same five. They agree on the outcome and differ in what
they are — a quantity against a span — which is why one is exclusive at the top
and the other is not.

Indices are positions in the delivered window, 0-based
------------------------------------------------------
The same origin as the gate-time axis: index 0 is the first delivered gate column
and the first row of the gate-time table, which are 1:1.

They are **not** the vendor's own gate numbers. ``libaarhusxyz`` normalizes the
origin away at parse time, so the vendor's numbering is not recoverable from a
loaded dataset and cannot be what this addresses.
"""

import logging

import numpy as np

from ..mask import cull, enable, mark_step
from ..pipeline import StepReport

log = logging.getLogger(__name__)

STEP = "nominate_gates"

#: This filter's number in ``building_plans/01``, used as its provenance code.
STEP_CODE = 21


def parse_gates(nominated, width: int) -> list[int]:
    """Gate indices from an int, a range string, or a list mixing both.

    Raises on anything outside the delivered window rather than clamping: a
    nomination naming a gate that does not exist is a mistake about the data,
    and silently doing four fifths of what was asked is how that goes unnoticed.
    """
    if isinstance(nominated, (int, np.integer)) or isinstance(nominated, str):
        nominated = [nominated]

    out: list[int] = []
    for entry in nominated:
        if isinstance(entry, str) and "-" in entry.strip().lstrip("-"):
            first, _, last = entry.strip().partition("-")
            try:
                low, high = int(first), int(last)
            except ValueError:
                raise ValueError(
                    f"Cannot read {entry!r} as a gate range. Use 'first-last', "
                    f"inclusive, e.g. '0-4' for the first five gates."
                ) from None
            if high < low:
                raise ValueError(
                    f"Gate range {entry!r} runs backwards; write it as "
                    f"'{high}-{low}' if that is what was meant."
                )
            out.extend(range(low, high + 1))       # inclusive: a span, not a count
        else:
            try:
                out.append(int(entry))
            except (TypeError, ValueError):
                raise ValueError(
                    f"Cannot read {entry!r} as a gate index or range."
                ) from None

    outside = sorted({g for g in out if g < 0 or g >= width})
    if outside:
        raise ValueError(
            f"Gate(s) {', '.join(map(str, outside))} are outside the delivered "
            f"window of {width} gates (0-{width - 1}). Indices are positions in "
            f"the delivered data, 0-based, not the vendor's own gate numbers."
        )
    return sorted(set(out))


def nominate_gates(dataset, enable_gates: int | str | list[int | str] | dict | None = None,
                   disable_gates: int | str | list[int | str] | dict | None = None):
    """Set named gates in the mask, either direction, per channel.

    Parameters
    ----------
    enable_gates
        Gates to turn back on, per channel key or moment name (``{"ch1":
        "0-5"}``), or one nomination for every channel.
    disable_gates
        Gates to turn off, in the same forms.

    Two parameters rather than one plus a direction flag. R7's argument against
    booleans applies: ``enable: false`` reads as "do not enable" rather than
    "disable", and a reader should not have to work out which.

    Returns ``(dataset, StepReport)``.
    """
    report = StepReport(step=STEP)
    frame = dataset.flightlines
    report.soundings_total = 0 if frame is None else len(frame)

    if enable_gates is None and disable_gates is None:
        raise ValueError(
            "nominate_gates needs enable_gates or disable_gates. With neither "
            "it would do nothing, which is better said by leaving the step out."
        )

    channels = dataset.channels
    if not channels:
        report.notes.append("No channels; nothing to nominate.")
        return dataset, report

    wanted = {
        "enabled": dataset.resolve_per_channel(enable_gates) if enable_gates is not None else {},
        "disabled": dataset.resolve_per_channel(disable_gates) if disable_gates is not None else {},
    }

    per_channel: dict = {}
    for channel in channels:
        width = dataset.layer_data[channel.data_key].shape[1]
        outcome: dict = {}

        for action, nominations in wanted.items():
            if channel.key not in nominations:
                continue
            gates = parse_gates(nominations[channel.key], width)
            if not gates:
                continue

            selected = np.zeros((len(frame), width), dtype=bool)
            selected[:, gates] = True
            # Whole columns: a gate is nominated for the channel, not for
            # particular soundings. Per-sounding judgement is filter_33's.
            if action == "enabled":
                result = enable(dataset, channel, selected, step_code=STEP_CODE)
            else:
                result = cull(dataset, channel, selected, step_code=STEP_CODE)
            outcome[action] = {"gates": gates, **result}

        if outcome:
            per_channel[str(channel)] = outcome

    if not per_channel:
        report.notes.append(
            "No nominated channel matched this dataset; nothing was changed."
        )
        return dataset, report

    mark_step(dataset, STEP_CODE, STEP)
    restored = sum(o.get("enabled", {}).get("restored", 0)
                   for o in per_channel.values())
    removed = sum(o.get("disabled", {}).get("removed", 0)
                  for o in per_channel.values())

    report.distributions = {
        "per_channel": per_channel,
        "restored": restored,
        "removed": removed,
        "provenance_code": STEP_CODE,
    }
    if restored:
        report.notes.append(
            f"{restored} gate-sample(s) were turned back on. This is one of "
            f"only two steps that may do that, and the provenance array records "
            f"it — a code against an in-use entry means restored, not culled."
        )
    log.info("%s: restored %d and removed %d gate-sample(s)",
             STEP, restored, removed)
    return dataset, report
