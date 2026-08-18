from setuptools import setup, find_packages

setup(
    name="ymerflow-processing",
    version="0.1.0",
    description="Airborne geophysical data processing, QC and comparison for YmerFlow",
    packages=find_packages(),
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
        "libaarhusxyz @ git+https://github.com/YmerFlow/libaarhusxyz@master",
        # Deliberately unpinned, and deliberately compatible with both majors.
        # numpy 2 is what this is developed against, but the inversion stack it
        # feeds still runs on 1.x, so a floor here would strand the environment
        # this library actually has to work in. That is a constraint on the code
        # rather than on the dependency: no numpy-2-only API, and the suite is
        # run against both before anything is committed.
        "numpy",
        "pandas>=2.2",    # R5: built against pandas 3.x semantics
        "geopandas",   # vector export, and R8 external inputs by type
        "pyyaml",
        "click",
    ],
    extras_require={
        "test": ["pytest"],
    },
    entry_points={
        # Three runners, three groups — see building_plans/02.
        # dataset -> dataset
        "ymerflow.processing.filters": [
            "select_channels = "
            "ymerflow_processing.core.select_channels:select_channels",
            "export_vector = "
            "ymerflow_processing.core.export_vector:export_vector",
            "disable_initial_gates = ymerflow_processing.core.cull."
            "filter_20_initial_gates:disable_initial_gates",
            "cull_by_geometry = ymerflow_processing.core.cull."
            "filter_22_geometry:cull_by_geometry",
            "cull_sparse_soundings = ymerflow_processing.core.cull."
            "filter_23_sparse_soundings:cull_sparse_soundings",
        ],
        # dataset -> dataset + report
        "ymerflow.processing.qc": [
            "height_above_ground = "
            "ymerflow_processing.core.qc.qc_01_height:height_above_ground",
            "system_attitude = "
            "ymerflow_processing.core.qc.qc_02_attitude:system_attitude",
            "ground_speed = "
            "ymerflow_processing.core.qc.qc_03_speed:ground_speed",
            "altimeter_agreement = "
            "ymerflow_processing.core.qc.qc_08_altimeter:altimeter_agreement",
            "sounding_spacing = "
            "ymerflow_processing.core.qc.qc_10_spacing:sounding_spacing",
            "transmitter_stability = "
            "ymerflow_processing.core.qc.qc_04_current:transmitter_stability",
            "powerline_noise = "
            "ymerflow_processing.core.qc.qc_05_powerline:powerline_noise",
        ],
        # N reports (or N datasets) -> one report
        "ymerflow.processing.compare": [],
        # How each method names its arrays and what their second axis means.
        # core/ owns the discovery mechanism; a method package owns only this.
        # Adding FEM, mag, rad or grav later is one more line here.
        "ymerflow.processing.methods": [
            "tem = ymerflow_processing.em.tem.method:TEMMethod",
        ],
        "console_scripts": [
            "yf-process = ymerflow_processing.core.cli:main",
        ],
    },
)
