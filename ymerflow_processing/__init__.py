"""Airborne geophysical data processing, QC and comparison.

Layout follows the specialization relationship: general airborne processing,
then EM, then the time- and frequency-domain methods.

    core/   any airborne survey — no method assumptions
    em/     shared by TEM and FEM
    em/tem/ gates, moments, transient decay

See ``building_plans/`` for the specification this is built from.
"""

__version__ = "0.1.0"
