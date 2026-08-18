"""Culling filters: steps that write the in-use mask.

R6's division of labour — QC writes attributes, culling writes the mask. Every
step here marks data not-in-use and changes no measurement, no uncertainty and
no array shape.

Each is a **predicate over the data**, wrapped by the shared policy in
:mod:`ymerflow_processing.core.mask`: a predicate says which points look wrong,
and the policy decides how far that objection spreads. Nothing here touches the
mask or the provenance array directly.

Provenance codes are the filter's own number from ``building_plans/01`` —
filter_20 is 20, filter_24 is 24. Unique by construction, stable across
releases, self-documenting in a diagnostic array, and comfortably inside the
1..127 an int8 can hold both ways.
"""
