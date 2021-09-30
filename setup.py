import setuptools

with open("phdutils/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="phdutils",
    version="0.0.1",
    description="Python package for BCDI and SXRD",
    author="David Simonne",
    author_email="david.simonne@synchrotron-soleil.fr",
    data_files=[('', ["phdutils/bcdi/data_files/pynx_run.txt",
                      "phdutils/bcdi/data_files/CompareFacetsEvolution.ipynb", 
                      "phdutils/bcdi/data_files/PhasingNotebook.ipynb", 
                      "licence.txt",
                      "phdutils/sixs/alias_dict_2021.txt"
                     ])],
    url="https://github.com/DSimonne/PhDUtils/tree/master",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    scripts=[
        "phdutils/bcdi/scripts/compute_q.py",
        "phdutils/bcdi/scripts/run_correct_angles_detector.py",
        "phdutils/bcdi/scripts/run_movetodir.py",
        "phdutils/bcdi/scripts/run_preprocess_bcdi.py",
        "phdutils/bcdi/scripts/run_rotate.py",
        "phdutils/bcdi/scripts/run_slice_cxi.py",
        "phdutils/bcdi/scripts/run_std_filter.py",
        "phdutils/bcdi/scripts/run_strain.py",
        "phdutils/bcdi/scripts/job.slurm",
        "phdutils/bcdi/scripts/run_slurm_job.sh",
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
        "xrayutilities",
        "tables",
        "PyQt5"
        ]
)