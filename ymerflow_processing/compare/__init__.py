"""Comparison checks: N reports (or N datasets) to one report.

``02`` makes this a fourth contract because it consumes many things and produces
one, which neither of the others expresses, and because *"stretching the filter
contract to cover it would make every filter's signature worse to serve one
case."*

    check(items, reference=None, **params) -> StepReport

``items`` is an **ordered** sequence — a pair is simply a two-item sequence, so
pairwise comparison needs no separate shape. ``reference`` is what everything is
compared *against* where the comparison is asymmetric, and it earns its place:
without it the baseline is ``items[0]`` by positional convention, which produces
subtly wrong answers when arguments arrive in the wrong order and nothing
complains.

Reports first, and reports by default
--------------------------------------
``02``: *"A dataset can always produce a report; a report can never reconstruct a
dataset."* So the report path is primary, and a check that does not declare
``consumes`` is a check over reports.

That keeps the common case a pure function over small records — no storage
backend, no credentials, trivially testable, runs anywhere. Cross-flight drift
reads twenty small stored records; loading twenty full datasets to recompute
numbers already computed would take minutes for something that should take
milliseconds.

It also dissolves the cumulative-QC problem without inventing persistent state.
R3 already requires every step to return a structured record, and those records
are already stored — so drift is not "QC state carried between runs", it is a
**query over data that already exists**.

Order by what was read from the data
-------------------------------------
``RunReport`` separates ``identity`` — a human label, normally the delivered
file's name — from ``flights`` and ``time_range``, which are read from the
soundings. Drift orders by the latter, because delivery names routinely embed an
export date and an identity taken from a filename is not stable across
re-exports. Ordering by it would enter one delivery into a series twice.
"""
