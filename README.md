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


## How to kill detector (here xpad140)
* atk scan
* abort
* shclose()
* open jive
* see on which server ds_limadetector/xpad.s140 is
* open astor
* go to server
* ds_limadetector/xpad.s140
* kill server (does not work lol)
* compact pcicrate (level1)
* right click
* test device
* kill process by PID
* open jive
* go to device
* find detector
* I14-C-COO/DT/XPAD.S140
* served pid to type in argin value
* execute
* goes to red in astor
* open confluence/detectors/xpad140/calibration -> ssh
* ps aux|grep Server
* kill -9 xxxx
* make sure its gone
* open RebixXDAQ
* connect to ethernet server
* restart device in astor
* shopen

# Alignment procedure for BCDI

Once we have selected a clean part of the beam with the primary slits, the first step is to make sure that the beam is aligned with the FZP, OPS and CS out, especially within the rod carrying the coherence optical elements. We also have to make sure that there is enough space around the setup to allow the rotation of the goniometer on the Bragg peak in reciprocal space and that the cables from the RGA do not cut the laser that automatically disables the goniometer.

Afterwards, we must align the focal plane of the FZP on the COR on the goniometer and on the focal plane of the microscope (NAVITAR camera). To do so, we use a silicon rod as a sample. We use a Basler camera for the alignment procedure to directly look at the beam.

* Adjust the de-tuning of the mono (pitch).
* Find the beam reflected from the silicon tip on the camera to make sure that the tip is on the path of the beam, (scan along z);
* Use the silicon tip to put the beam in the focal plane of the microscope, by moving only the microscope, first move the microscope, then adjust with maximum zoom;
* Make sure that the silicon tip is in the focal plane of the FZP by taking numerous images at different positions of the FZP to find whether the silicon rod is before or after the focal point, then realign the OSA and CS.
* Find the part of the beam that is the most suitable for use, with the coherence slits opened to $500\,\mu m$, move to the center of the beam and close the slits to be sure that it is clean (with the Basler camera, in the direct beam). Remember to move base_x and base_z in the opposite direction (slits are mounted at 180°) to compensate;
* Lower the slits a $100\,\mu m$ horizontally;
* Close the slits to 50\*50 and make sure that, in the direct beam, we are on a good part of the detector;
* Align the Central Stop, the Fresnel Zone Plate and the OSA on that position, you can play with the slits' openings to better see the beam;
* Reverse the translation by $100\,\mu m$ so that the best part of the beam hits the FZP; make sure that the axes of the central stop are not in your future position;
* Make sure that the beam looks homogeneous, otherwise repeat the selection of the beam.
* Rattraper en basex ???

## Move the beam on the COR with sample
We are now certain that the elements are well aligned, that the silicon rod is in the focal plane of the lenses and of the microscope as well as in the COR of the goniometer; it is possible to replace the silicon rob by the sample. The beam quality should also be optimal.

Align the COR of µ by translating the sample in z, the direction in which the spot given by fluorescence of the sample is moving gives us an indication about the position of the COR of µ. Move the goniometer towards this position. Be careful about the angle between the sample orientation and the goniometer's axis.

We can move the sample so that the nanoparticles are in the COR of the goniometer, to do so move base_z to move the beam up or down on the microscope's image and move z to shift the beam left or right on the microscope's image.

We used a NAVITAR microscope to navigate on the sample surface and to select the right size of nanoparticle, which differs depending on the region on the sample The fluorescence of the sample also proved useful to find where the beam's footprint was located on the surface via the microscope.

## Sample introduction

Replace the Siemens star with the sample once the beam is characterized

Find omega to have the sample aligned horizontally, then scan along the vertical axis with slits opened to find the center of a square by taking the middle, then do again for a square 10 squares away, finally scan between to find a particle to analyze.
