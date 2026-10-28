"""Airborne geophysical data processing, QC and comparison.

Layout follows the roles the runners see — the three entry-point groups of
``building_plans/02`` — with the data model and its machinery at the top:

    dataset.py pipeline.py mask.py ...   the data model, runners, in-use mask
    filters/     dataset in, dataset out; ``cull_*`` write the mask
    qc/          dataset in, dataset plus verdicts out
    compare/     N reports or datasets in, one report out
    methods/     one adapter per measurement method (TEM today)

See ``building_plans/`` for the specification this is built from, and
``docs/plans/package-layout.md`` for why the layout is by role.
"""

__version__ = "0.4.1"
