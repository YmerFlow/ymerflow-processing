"""Processing filters: dataset in, dataset out.

Every step registered under ``ymerflow.processing.filters`` lives here, flat.
The ``cull_`` prefix marks the family that writes the in-use mask — R6's
division of labour is *QC writes attributes, culling writes the mask*, and the
prefix keeps that visible in a directory listing without a sub-package.

A culling filter marks data not-in-use and changes no measurement, no
uncertainty and no array shape. Each is a **predicate over the data**, wrapped
by the shared policy in :mod:`ymerflow_processing.mask`: a predicate says which
points look wrong, and the policy decides how far that objection spreads.
Nothing here touches the mask or the provenance array directly.

Provenance codes are the filter's own number from ``building_plans/01`` —
filter_20 is 20, filter_24 is 24. Unique by construction, stable across
releases, self-documenting in a diagnostic array, and comfortably inside the
1..127 an int8 can hold both ways. The number lives in each module's docstring
and ``STEP_CODE``, not in its filename, because in a filename it reads as the
order the steps are meant to run in.
"""
