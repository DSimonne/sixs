import setuptools

with open("phdutils/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="phdutils",
    version="0.0.1",
    description="Python package for SXRD",
    author="David Simonne",
    author_email="david.simonne@synchrotron-soleil.fr",
    data_files=[('', ["licence.txt",
                      "phdutils/sixs/alias_dict_2021.txt"
                     ])],
    url="https://github.com/DSimonne/PhDUtils/tree/master",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    keywords = "SXRD",
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