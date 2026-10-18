"""Acquisition QC — per-flight checks that answer "should this line be reflown"
while the aircraft is still on site.

Distinct from culling, and the distinction is the governing rule of the library:

    QC writes attributes. Culling writes the mask.

A check writes its Good / Marginal / Out of Spec verdict as a column on
``flightlines``, which changes nothing about what the inversion sees. Culling
writes the in-use mask, which does. QC records a judgment; culling acts on one.

Specification: ``building_plans/03_acquisition_qc.md``.

Checks register under the ``ymerflow.processing.qc`` entry-point group and are
run by :class:`~ymerflow_processing.pipeline.QCRunner`. What they share
lives here: the Good / Marginal / Out of Spec model in :mod:`.thresholds`, and
the per-line aggregation in :mod:`.aggregate`.
"""

from .aggregate import apply_rule, grain_measures, per_flight, per_line
from .thresholds import (
    GOOD, MARGINAL, OUT_OF_SPEC, TIERS,
    classify, describe, tier_counts, validate_spec,
)

__all__ = [
    "GOOD", "MARGINAL", "OUT_OF_SPEC", "TIERS",
    "classify", "describe", "tier_counts", "validate_spec",
    "per_line", "per_flight", "grain_measures", "apply_rule",
]
