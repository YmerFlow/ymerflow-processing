"""Method adapters: what a measurement method tells the data model about itself.

One module per method, each registering a :class:`~ymerflow_processing.dataset.MethodAdapter`
under ``ymerflow.processing.methods``. An adapter has two responsibilities and
only these two: say how the method names its arrays, and say what their second
axis means — gate times for TEM, frequencies for FEM, energies for radiometrics.
Channel discovery itself is the data model's job.

Only time-domain EM has an adapter today. The split between a method-agnostic
data model and method-specific code was drawn before a second method existed,
and ``docs/plans/package-layout.md`` records the decision to redraw it only when
one does.
"""
