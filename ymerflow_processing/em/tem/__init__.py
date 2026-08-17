"""Time-domain electromagnetics.

What belongs here is anything that needs gates, moments, or transient decay: the
noise model with its power-law floor, gate-indexed operations, slope and
curvature tests on the transient, and negative-data detection.

A filter that looks method-specific only because of how it is *configured*
belongs in ``core/`` with its parameterisation supplied from here — the
along-line averaging engine is generic; only its gate-time-varying window widths
are time-domain.
"""
