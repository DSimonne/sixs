# Welcome 

## This github is focused on data reduction tools at SixS
* Surface X-Ray Diffraction (SXRD) data analysis (CTR, reflectivity)
* Mass spectrometer data analysis (Residual Gas Analyser - RGA)

Contact : david.simonne@synchrotron-soleil.fr

You can install sixs via the `setup.py` script, so that you can use via a package after in python, see below

# Installing different packages

First, I advise you to create a Packages directory to keep these.
Secondly, I advise you to create a virtual environment to help with debogging, and so that once everything works, you don't update a package by mistake. To do so please follow the following steps:

## Create a virtual environment

* `mkdir py-env`
* `cd py-env/`
* `python3.9 -m venv .`
* `source bin/activate` # To activate the environment, then you can create an alias to do that

## Install sixs (with env activated)
* `git clone https://github.com/DSimonne/sixs.git`
* `cd sixs`
* `pip install .`

## An example notebook with data is available on the sixs3 computer

`/ruche/com-sixs/David/TestDataSXRD/TestDataAnalysis.ipynb`