import setuptools

with open("sixs/README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sixs",
    version="0.0.1",
    description="Python package for data reduction at SixS",
    author="David Simonne, Andrea Resta",
    author_email="david.simonne@synchrotron-soleil.fr",
    data_files=[('', ["licence.txt",
                      "sixs/experiments/ammonia.yml",
                      ])],
    scripts=[
        "sixs/scripts/print_pos.py"
    ],
    url="https://github.com/DSimonne/sixs/tree/main",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    keywords=["SXRD", "XCAT", "RGA"],
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
