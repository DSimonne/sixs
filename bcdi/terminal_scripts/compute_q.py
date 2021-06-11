#!/usr/bin/python3

"""Compute distance from gamma and delta"""

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import filedialog
import sys
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.experiment.experiment_utils as exp

# Print help
try:
    print ('Delta:',  sys.argv[1])
    print ('Gamma:',  sys.argv[2])
    print ('Wavelength:',  sys.argv[2])

except IndexError:
    print("""
        Arg 1: Delta
        Arg 2: Gamma
        Arg 3: Wavelength
        """)
    exit()

# give bragg x and bragg y in entry (pixel x and y where we want to have q)

###################################
# define setup related parameters #
###################################
beam_direction = (1, 0, 0)  # beam along z
sample_offsets = None  # tuple of offsets in degrees of the sample around (downstream, vertical up, outboard)
# convention: the sample offsets will be subtracted to the motor values
directbeam_x = 271  # x horizontal,  cch2 in xrayutilities
directbeam_y = 213   # y vertical,  cch1 in xrayutilities
direct_inplane = 0.0  # outer angle in xrayutilities
direct_outofplane = 0.0
sdd = 1.18  # sample to detector distance in m
energy = 8500  # in eV, offset of 6eV at ID01


######################################
# define beamline related parameters #
######################################
beamline = 'SIXS_2019'  # name of the beamline, used for data loading and normalization by monitor
# supported beamlines: 'ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10'
actuators = None  # {'rocking_angle': 'actuator_1_3'}
is_series = True  # specific to series measurement at P10
custom_scan = False  # True for a stack of images acquired without scan, e.g. with ct in a macro (no info in spec file)
custom_images = None  # list of image numbers for the custom_scan
custom_monitor = None  # monitor values for normalization for the custom_scan
custom_motors = None
# {"eta": np.linspace(16.989, 18.989, num=100, endpoint=False), "phi": 0, "nu": -0.75, "delta": 36.65}
# SIXS: beta, mu, gamma, delta
rocking_angle = "inplane"  # "outofplane" or "inplane"
specfile_name = "/home/david/.local/lib/python3.9/site-packages/phdutils/sixs/alias_dict_2021.txt"
# template for SIXS_2018: full path of the alias dictionnary 'alias_dict.txt', typically: root_folder + 'alias_dict.txt'

#############################################################
# define detector related parameters and region of interest #
#############################################################
detector = "Merlin"    # "Eiger2M" or "Maxipix" or "Eiger4M"
x_bragg = None  # horizontal pixel number of the Bragg peak, can be used for the definition of the ROI
y_bragg = None   # vertical pixel number of the Bragg peak, can be used for the definition of the ROI
roi_detector = None
# leave it as None to use the full detector. Use with center_fft='do_nothing' if you want this exact size.
high_threshold = 1000000  # everything above will be considered as hotpixel
hotpixels_file = "/home/david/Documents/PhD_local/PhDScripts/SIXS_January_2021/analysis/mask_merlin.npy"  # root_folder + 'hotpixels_HS4670.npz'  # non empty file path or None
flatfield_file = None  # root_folder + "flatfield_maxipix_8kev.npz"  # non empty file path or None
template_imagefile ="Pt_Al2O3_ascan_mu_%05d_R.nxs"


#######################
# Initialize detector #
#######################
detector = exp.Detector(name=detector, template_imagefile=template_imagefile, roi=roi_detector,
                        is_series=is_series)

####################
# Initialize setup #
####################
setup = exp.Setup(beamline=beamline, energy=energy, rocking_angle=rocking_angle, distance=sdd,
                  beam_direction=beam_direction, custom_scan=custom_scan, custom_images=custom_images,
                  custom_monitor=custom_monitor, custom_motors=custom_motors, pixel_x=detector.pixelsize_x,
                  pixel_y=detector.pixelsize_y, sample_offsets=sample_offsets, actuators=actuators)


x_direct_0 = directbeam_x + setup.inplane_coeff *\
             (direct_inplane*np.pi/180*sdd/detector.pixelsize_x)  # inplane_coeff is +1 or -1
y_direct_0 = directbeam_y - setup.outofplane_coeff *\
             direct_outofplane*np.pi/180*sdd/detector.pixelsize_y   # outofplane_coeff is +1 or -1

bragg_inplane = setup.inplane_angle + setup.inplane_coeff *\
                (detector.pixelsize_x*(bragg_x-x_direct_0)/sdd*180/np.pi)  # inplane_coeff is +1 or -1
bragg_outofplane = setup.outofplane_angle - setup.outofplane_coeff *\
                   detector.pixelsize_y*(bragg_y-y_direct_0)/sdd*180/np.pi   # outofplane_coeff is +1 or -1


# gamma is anti-clockwise
kout = 2 * np.pi / wavelength * np.array(
    [np.cos(np.pi * inplane_angle / 180) * np.cos(np.pi * outofplane_angle / 180),  # z
     np.sin(np.pi * outofplane_angle / 180),  # y
     np.sin(np.pi * inplane_angle / 180) * np.cos(np.pi * outofplane_angle / 180)])  # x

kin = 2*np.pi/wavelength * beam_direction

q = (kout - kin) * 1e-10

print(q)