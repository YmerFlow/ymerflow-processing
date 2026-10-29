"""The parts of ``cull_by_coupling``, each testable on its own.

``table`` is the per-gate coupling table the step always writes; ``reconstruct``
holds the reconstruction basis, behind an interface so another basis can replace
the PCA one without the step changing; ``reconstruction`` is the criterion that
turns a reconstruction into a per-gate decision. The step itself, which parses
parameters, orchestrates and reports, is ``filters/cull_coupling.py``.

The design is ``docs/plans/coupling-detection.md``. Its profile and trend
criteria and its corroboration are later stages and are not here.
"""
