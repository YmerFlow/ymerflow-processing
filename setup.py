from setuptools import setup, find_packages

setup(
    name="ymerflow-processing",
    version="0.3.0",
    description="Airborne geophysical data processing, QC and comparison for YmerFlow",
    license="GPL-3.0-or-later",
    packages=find_packages(),
    # 02's first lever: "load a standard chain and adjust it" only works if
    # the chains install with the package.
    package_data={"ymerflow_processing": ["pipelines/*.yaml"]},
    python_requires=">=3.10",
    install_requires=[
        # MIT — the survey data model.
        #
        # Pinned to the YmerFlow fork rather than PyPI. Upstream is
        # unmaintained — last push April 2026 — and
        # carries a GEX attribute-protocol bug that breaks copy/deepcopy/pickle
        # on any Survey, reported upstream in 2024 and closed as a downstream
        # workaround. Fixed in our fork; see YmerFlow/libaarhusxyz#3.
        #
        # A direct reference cannot be uploaded to PyPI, so publishing this
        # package would first require publishing the fork under its own name.
        #
        # Spelled exactly as aem-processes spells it. pip compares direct
        # references as text, so ".git" against "@master" for the same commit
        # reads as two sources and the two packages cannot be installed
        # together. No ref means the default branch, master.
        "libaarhusxyz @ git+https://github.com/YmerFlow/libaarhusxyz.git",
        # Deliberately unpinned, and deliberately compatible with both majors.
        # numpy 2 is what this is developed against, but the inversion stack it
        # feeds still runs on 1.x, so a floor here would strand the environment
        # this library actually has to work in. That is a constraint on the code
        # rather than on the dependency: no numpy-2-only API, and the suite is
        # run against both before anything is committed.
        "numpy",
        "pandas>=2.2",    # R5: built against pandas 3.x semantics
        "geopandas",   # vector export, and R8 external inputs by type
        "rasterio",    # filter_05 samples a terrain model
        "pyyaml",
        "click",
        # The step reference is generated from the docstrings, read with
        # numpydoc's parser -- the same one swaggerspect builds the platform's
        # forms with, so the two cannot read a docstring differently.
        "numpydoc",
    ],
    extras_require={
        "test": ["pytest"],
        # Figures are a documentation concern, not a processing one. A runner
        # that only processes data should not pull a plotting stack to do it,
        # so matplotlib stays out of install_requires and the figure module
        # imports it lazily.
        "figures": ["matplotlib"],
    },
    entry_points={
        # Three runners, three groups — see building_plans/02.
        # dataset -> dataset
        "ymerflow.processing.filters": [
            "select_channels = ymerflow_processing.filters.select_channels:select_channels",
            "export_vector = ymerflow_processing.filters.export_vector:export_vector",
            "disable_initial_gates = ymerflow_processing.filters.cull_initial_gates:disable_initial_gates",
            "cull_by_geometry = ymerflow_processing.filters.cull_geometry:cull_by_geometry",
            "cull_by_uncertainty = ymerflow_processing.filters.cull_uncertainty:cull_by_uncertainty",
            "cull_by_external_evidence = ymerflow_processing.filters.cull_evidence:cull_by_external_evidence",
            "cull_by_decay_rate = ymerflow_processing.filters.cull_decay_rate:cull_by_decay_rate",
            "cull_by_curvature = ymerflow_processing.filters.cull_curvature:cull_by_curvature",
            "cull_negatives = ymerflow_processing.filters.cull_negatives:cull_negatives",
            "apply_manual_edits = ymerflow_processing.filters.cull_manual_edits:apply_manual_edits",
            "apply_terrain_model = ymerflow_processing.filters.terrain:apply_terrain_model",
            # R9: one noise model on the dataset; filter_12 and filter_29 read it.
            "estimate_noise_model = ymerflow_processing.filters.noise:estimate_noise_model",
            # filter_06-09: geometry, conventions and constants.
            "correct_tilt = ymerflow_processing.filters.attitude:correct_tilt",
            "zero_attitude = ymerflow_processing.filters.attitude:zero_attitude",
            "apply_nominal_current = ymerflow_processing.filters.normalize:apply_nominal_current",
            "compute_misfit = ymerflow_processing.filters.misfit:compute_misfit",
            "average_along_line = ymerflow_processing.filters.average:average_along_line",
            "inflate_uncertainty = ymerflow_processing.filters.inflate:inflate_uncertainty",
            "perturb_from_uncertainty = ymerflow_processing.filters.perturb:perturb_from_uncertainty",
            "assign_uncertainty = ymerflow_processing.filters.uncertainty:assign_uncertainty",
            "cull_below_noise_floor = ymerflow_processing.filters.cull_noise_floor:cull_below_noise_floor",
            # filter_01-04: line selection, collapsed to two by 01's decision.
            "classify_lines = ymerflow_processing.filters.lines:classify_lines",
            "select_lines = ymerflow_processing.filters.lines:select_lines",
            # filter_14-19: data management, one entry point each.
            "copy_attribute = ymerflow_processing.filters.manage:copy_attribute",
            "copy_array = ymerflow_processing.filters.manage:copy_array",
            "rename_attribute = ymerflow_processing.filters.manage:rename_attribute",
            "rename_array = ymerflow_processing.filters.manage:rename_array",
            "remove_attribute = ymerflow_processing.filters.manage:remove_attribute",
            "remove_array = ymerflow_processing.filters.manage:remove_array",
            "nominate_gates = ymerflow_processing.filters.cull_nominate_gates:nominate_gates",
            "cull_sparse_soundings = ymerflow_processing.filters.cull_sparse_soundings:cull_sparse_soundings",
        ],
        # N reports (or N datasets) -> one report
        "ymerflow.processing.compare": [
            "noise_floor_drift = ymerflow_processing.compare.drift:noise_floor_drift",
            "implementation_equivalence = ymerflow_processing.compare.equivalence:implementation_equivalence",
            "version_diff = ymerflow_processing.compare.version_diff:version_diff",
            "parameter_sweep = ymerflow_processing.compare.sweep:parameter_sweep",
        ],
        # dataset -> dataset + report
        "ymerflow.processing.qc": [
            "height_above_ground = ymerflow_processing.qc.height:height_above_ground",
            "system_attitude = ymerflow_processing.qc.attitude:system_attitude",
            "ground_speed = ymerflow_processing.qc.speed:ground_speed",
            "altimeter_agreement = ymerflow_processing.qc.altimeter:altimeter_agreement",
            "sounding_spacing = ymerflow_processing.qc.spacing:sounding_spacing",
            "transmitter_stability = ymerflow_processing.qc.current:transmitter_stability",
            "powerline_noise = ymerflow_processing.qc.powerline:powerline_noise",
            "culled_fraction = ymerflow_processing.qc.culled_fraction:culled_fraction",
            "data_misfit = ymerflow_processing.qc.data_misfit:data_misfit",
        ],
        # N reports (or N datasets) -> one report
        # How each method names its arrays and what their second axis means.
        # core/ owns the discovery mechanism; a method package owns only this.
        # Adding FEM, mag, rad or grav later is one more line here.
        "ymerflow.processing.methods": [
            "tem = ymerflow_processing.methods.tem:TEMMethod",
        ],
        "console_scripts": [
            "yf-process = ymerflow_processing.cli:main",
        ],
    },
)
