# Welcome 

## This github is focused on data reduction tools at SixS
* Surface X-Ray Diffraction (SXRD) data analysis (CTR, reflectivity)
* Mass spectrometer data analysis (Residual Gas Analyser - RGA)

Contact : david.simonne@synchrotron-soleil.fr

You can install sixs via the setup.py script, so that you can use via a package after in python, see below

# Installing different packages

First, I advise you to create a Packages directory to keep these.
Secondly, I advise you to create a virtual environment to help with debogging, and so that once everything works, you don't update a package by mistake. To do so please follow the following steps:

## Create a virtual environment

* `mkdir py38-env`
* `cd py38-env/`
* `python3.8 -m venv .`
* `source bin/activate` # To activate the environment

## Install sixs
* `cd /Packages`
* `git clone https://github.com/DSimonne/sixs.git`
* `cd sixs`
* `source_py38`
* `pip install .`

# Data analysis at SixS

## First connect to sixs3
`ssh -X sixs3`

`df -h` (pour voir les disques accessibles)

`cd /nfs/ruche-sixs/sixs-soleil/com-sixs/David`

### IPython sur ligne

`xcat`

### Macros

`/nfs/ruche-sixs/sixs-soleil/com-sixs/2021/Shutdown5/test/`

You can create macros and run them with `do.run("<filename>")`
They must respect a pythonic synthax and are read lign by lign