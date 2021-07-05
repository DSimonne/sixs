import setuptools

with open("phdutils/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="phdutils",
    version="0.0.1",
    description="Python package for BCDI and SXRD",
    author="David Simonne",
    author_email="david.simonne@synchrotron-soleil.fr",
    url="",
    # scripts=[
    #     'phdutils/bcdi/terminal_scripts/compute_q.py',
    #     'phdutils/bcdi/terminal_scripts/correct_angles_detector.py',
    #     'phdutils/bcdi/terminal_scripts/facet_strain.py',
    #     'phdutils/bcdi/terminal_scripts/make_support.py',
    #     'phdutils/bcdi/terminal_scripts/movetodir.py',
    #     'phdutils/bcdi/terminal_scripts/preprocess_bcdi_merlin_ortho.py',
    #     'phdutils/bcdi/terminal_scripts/print_pos.py',
    #     'phdutils/bcdi/terminal_scripts/quick_phase_retrieval.sh',
    #     'phdutils/bcdi/terminal_scripts/quick_process.sh',
    #     'phdutils/bcdi/terminal_scripts/rotate.py',
    #     'phdutils/bcdi/terminal_scripts/slice_cxi.py',
    #     'phdutils/bcdi/terminal_scripts/strain-SIXS_merlin.py',
    #     'phdutils/bcdi/terminal_scripts/Sync_from_id01.sh',
    #     'phdutils/bcdi/terminal_scripts/Sync_to_id01.sh',
    #     ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],

    keywords = "BCDI SXRD",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
	include_package_data=True,
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "ipywidgets",
        "ipython",
        "scipy",
        "lmfit",
        "xrayutilities",
        "tables"
        ]
)