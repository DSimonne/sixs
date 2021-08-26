import phdutils

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import glob
import errno
import os
import shutil
import math
from ast import literal_eval

# import lmfit
# from lmfit import minimize, Parameters, Parameter
# from lmfit.models import LinearModel, ConstantModel, QuadraticModel, PolynomialModel, StepModel
# from lmfit.models import GaussianModel, LorentzianModel, SplitLorentzianModel, VoigtModel, PseudoVoigtModel
# from lmfit.models import MoffatModel, Pearson7Model, StudentsTModel, BreitWignerModel, LognormalModel, ExponentialGaussianModel, SkewedGaussianModel, SkewedVoigtModel, DonaichModel
# import corner
# import numdifftools
# from scipy.stats import chisquare

import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive, fixed
from IPython.display import display, Markdown, Latex, clear_output

# from scipy import interpolate
# from scipy import optimize, signal
# from scipy import sparse

from datetime import datetime
import pickle
import inspect
import warnings

import tables as tb

# Import preprocess_bcdi modified for gui and usable as a function
from phdutils.bcdi.gui.preprocess_gui import *
from phdutils.bcdi.gui.correct_angles_gui import *
from phdutils.sixs import ReadNxs4 as rd

class Interface(object):
    """This  class is a Graphical User Interface (gui) that is meant to be used to process important amount of XAS datasets that focus on the same energy range and absoption edge.
        There are two ways of initializing the procedure in a jupyter notebook:
            _ gui = thorondor.gui.Interface(); One will have to write the name of the data folder in which all his datasets are saved.
            _ gui = thorondor.gui.Interface.get_class_list(data_folder = "<yourdata_folder>") if one has already worked on a Dataset and wishes to retrieve his work
        This class makes extensive use of the ipywidgets and is thus meant to be used with a jupyter notebook.
        Additional informations are provided in the "ReadMe" tab of the gui.
        The necessary Python packages are : numpy, pandas, matplotlib, glob, errno, os, shutil, ipywidgets, IPython, scipy, datetime, importlib, pickle, lmfit
        lmfit, xlrd, corner and inspect.
    """

    def __init__(self, class_list = False):
        """All the widgets for the gui are defined here. 
        """
        super(Interface, self).__init__()

        # Work in current direcoty ?
        self.work_dir = os.getcwd()
        self.path_package = inspect.getfile(phdutils).split("__")[0]

        # Widgets for the gui will need to separate later
        # Widgets for initialization 
        self._list_widgets_init = interactive(self.initialize_directories,

            ### Define scan related parameters
            label_scan = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Define working directory and scan number", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            sample_name = widgets.Text(
                value = "S",
                placeholder = "",
                description = 'Sample Name',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='45%'),
                style = {'description_width': 'initial'}),

            scans = widgets.BoundedIntText(
                value = "01415",
                description = 'Scan nb:',
                min = 0,
                max = 9999999,
                disabled = False,
                continuous_update = False,
                layout = Layout(width='45%'),
                style = {'description_width': 'initial'}),

            data_directory = widgets.Text(
                value = os.getcwd() + "/data_dir/",
                placeholder = "Path to data directory",
                description = 'Data directory',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            final_directory = widgets.Text(
                value = os.getcwd() + "/TestGui/",
                placeholder = "Path to target directory (parent to all scan directories)",
                description = 'Target directory',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            user_comment = widgets.Text(
                value = "",
                description = 'Comment',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                placeholder = "Comment regarding dataset...",
                style = {'description_width': 'initial'}),

            debug = widgets.ToggleButton(
                value = False,
                description = 'Debug',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'True to interact with plots, False to close it automatically',
                icon = 'check',
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'}),

            run_dir_init = widgets.ToggleButton(
                value = False,
                description = 'Initialize directories ...',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                icon = 'check',
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'}),
            )
        self._list_widgets_init.children[7].observe(self.init_handler, names = "value")

        # Widgets for preprocessing
        self._list_widgets_preprocessing = interactive(self.initialize_parameters,

            ### Define beamline related parameters
            label_beamline = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters specific to the beamline", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            beamline = widgets.Dropdown(
                options = ['ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10', 'NANOMAX', '34ID'],
                value = "SIXS_2019",
                description = 'Beamline',
                disabled = False,
                tooltip = "Name of the beamline, used for data loading and normalization by monitor",
                style = {'description_width': 'initial'}),

            rocking_angle = widgets.Dropdown(
                options = ['inplane', 'outofplane', 'energy'],
                value = "inplane",
                description = 'Rocking angle',
                disabled = False,
                tooltip = "Name of the beamline, used for data loading and normalization by monitor",
                style = {'description_width': 'initial'}),

            specfile_name = widgets.Text(
                placeholder = "alias_dict_2019.txt",
                value = "",
                description = 'Specfile name',
                disabled = False,
                continuous_update = False,
                tooltip = """For ID01: name of the spec file without, for SIXS_2018: full path of the alias dictionnary, typically root_folder + 'alias_dict_2019.txt',
                .fio for P10, not used for CRISTAL and SIXS_2019""",
                style = {'description_width': 'initial'}),

            follow_bragg = widgets.ToggleButton(
                value = False,
                description = 'Follow bragg',
                disabled = True,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'Only for energy scans, set to True if the detector was also scanned to follow the Bragg peak',
                icon = 'check'),

            actuators = widgets.Text(
                value = "{}",
                placeholder = "{}",
                description = 'Actuators',
                tooltip = "Optional dictionary that can be used to define the entries corresponding to actuators in data files (useful at CRISTAL where the location of data keeps changing)",
                readout = True,
                style = {'description_width': 'initial'},
                disabled = True),

            is_series = widgets.ToggleButton(
                value = False,
                description = 'Is series (P10)',
                disabled = True,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'specific to series measurement at P10',
                icon = 'check'),

            custom_scan = widgets.ToggleButton(
                value = False,
                description = 'Custom scan',
                disabled = True,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'set it to True for a stack of images acquired without scan, e.g. with ct in a macro, or when there is no spec/log file available',
                icon = 'check'),

            custom_images = widgets.IntText(
                value = 3, # np.arange(11353, 11453, 1)  # list of image numbers for the custom_scan
                description='Custom images',
                disabled = True,
                style = {'description_width': 'initial'}),

            custom_monitor = widgets.IntText(
                value = 51, # np.ones(51),  # monitor values for normalization for the custom_scan
                description='Custom monitor',
                disabled = True,
                style = {'description_width': 'initial'}),

            ### Parameters used in masking
            label_masking = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters used in masking", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            flag_interact = widgets.ToggleButton(
                value = False,
                description = 'Manual masking',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'True to interact with plots and manually mask points',
                icon = 'check'),

            background_plot = widgets.FloatText(
                value = 0.5,
                step = 0.01,
                max = 1,
                min = 0,
                description = 'Background plot:',
                layout = Layout(width='30%'),
                tooltip = "In level of grey in [0,1], 0 being dark. For visual comfort during masking",
                readout = True,
                style = {'description_width': 'initial'},
                disabled = False),

            ### Parameters related to data cropping/padding/centering
            label_centering = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters related to data cropping/padding/centering</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            centering  = widgets.Dropdown(
                options = ["max", "com", "manual"],
                value = "max",
                description = 'Centering of Bragg peak method:',
                disabled = False,
                layout = Layout(width='45%'),
                tooltip = "Bragg peak determination: 'max' or 'com', 'max' is better usually. It will be overridden by 'fix_bragg' if not empty",
                style = {'description_width': 'initial'}),

            fix_bragg = widgets.Text(
                placeholder = "[z_bragg, y_bragg, x_bragg]",
                description = 'Bragg peak position', # fix the Bragg peak position [z_bragg, y_bragg, x_bragg] considering the full detector
                disabled = True, # It is useful if hotpixels or intense aliens. Leave it [] otherwise.
                continuous_update = False,
                layout = Layout(width='45%'),
                style = {'description_width': 'initial'}),

            fix_size = widgets.Text(
                placeholder = "[zstart, zstop, ystart, ystop, xstart, xstop]",
                description = 'Fix array size', # crop the array to predefined size considering the full detector, leave it to [] otherwise [zstart, zstop, ystart, ystop, xstart, xstop]. ROI will be defaulted to []
                disabled = False,
                continuous_update = False,
                layout = Layout(width='45%'),
                style = {'description_width': 'initial'}),  

            center_fft = widgets.Dropdown(
                options = ['crop_sym_ZYX','crop_asym_ZYX','pad_asym_Z_crop_sym_YX', 'pad_sym_Z_crop_asym_YX','pad_sym_Z', 'pad_asym_Z', 'pad_sym_ZYX','pad_asym_ZYX', 'skip'],
                value = "crop_asym_ZYX",
                description = 'Center FFT',
                disabled = False,
                style = {'description_width': 'initial'}),

            pad_size = widgets.Text(
                placeholder = "[256, 512, 512]",
                description = 'Array size after padding', # used in 'pad_sym_Z_crop_sym_YX', 'pad_sym_Z', 'pad_sym_ZYX'
                disabled = False,
                continuous_update = False,
                layout = Layout(width='50%'),
                style = {'description_width': 'initial'}), 

            ### Parameters used in intensity normalization
            normalize_flux = widgets.Dropdown(
                options = ["skip", "monitor"],
                value = "skip",
                description = 'Normalize flux',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'Monitor to normalize the intensity by the default monitor values, skip to do nothing',
                icon = 'check',
                style = {'description_width': 'initial'}),


            ### Parameters for data filtering
            label_filtering = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters for data filtering</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            mask_zero_event = widgets.ToggleButton(
                value = False,
                description = 'Mask zero event',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'Mask pixels where the sum along the rocking curve is zero - may be dead pixels',
                icon = 'check'),

            flag_medianfilter = widgets.Dropdown(
                options = ['skip','median','interp_isolated', 'mask_isolated'],
                value = "skip",
                description = 'Flag median filter',
                disabled = False,
                tooltip = "set to 'median' for applying med2filter [3,3], set to 'interp_isolated' to interpolate isolated empty pixels based on 'medfilt_order' parameter, set to 'mask_isolated' it will mask isolated empty pixels, set to 'skip' will skip filtering",
                style = {'description_width': 'initial'}),

            medfilt_order = widgets.IntText(
                value = 7,
                description='Med filter order:',
                disabled = False,
                tooltip = "for custom median filter, number of pixels with intensity surrounding the empty pixel",
                style = {'description_width': 'initial'}),

            ### Parameter for data reduction
            label_reduction = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters for data reduction</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            binning = widgets.Text(
                value = "(1, 1, 1)",
                placeholder = "(1, 1, 1)",
                description = 'Binning for phasing',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "binning that will be used for phasing (stacking dimension, detector vertical axis, detector horizontal axis)"),

            ### Parameters used when reloading processed data
            label_reload = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters used when reloading processed data</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            reload_previous = widgets.ToggleButton(
                value = False,
                description = 'Reload previous',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'True to resume a previous masking (load data and mask)',
                icon = 'check'),

            reload_orthogonal = widgets.ToggleButton(
                value = False,
                description = 'Reload orthogonal',
                disabled = True,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'True if the reloaded data is already intepolated in an orthonormal frame',
                icon = 'check'),

            preprocessing_binning = widgets.Text(
                value = "(1, 1, 1)",
                placeholder = "(1, 1, 1)", # binning factors in each dimension of the binned data to be reloaded
                description = 'Binning used in data to be reloaded',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='30%'),
                style = {'description_width': 'initial'},
                tooltip = "binning that will be used for phasing (stacking dimension, detector vertical axis, detector horizontal axis)"),

            ### Saving options
            label_saving = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters used when saving the data</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            save_rawdata = widgets.ToggleButton(
                value = False,
                description = 'Save raw data',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'Save also the raw data when use_rawdata is False',
                icon = 'check'),

            save_to_npz = widgets.ToggleButton(
                value = True,
                description = 'Save to npz',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'True to save the processed data in npz format',
                icon = 'check'),

            save_to_mat = widgets.ToggleButton(
                value = False,
                description = 'Save to mat',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'True to save also in .mat format',
                icon = 'check'),

            save_to_vti = widgets.ToggleButton(
                value = False,
                description = 'Save to vti',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'Save the orthogonalized diffraction pattern to VTK file',
                icon = 'check'),

            save_asint = widgets.ToggleButton(
                value = False,
                description = 'Save as integers',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'if True, the result will be saved as an array of integers (save space)',
                icon = 'check'),

            ### Detector related parameters
            label_detector = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters related to the detector used</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            detector = widgets.Dropdown(
                options = ["Eiger2M", "Maxipix", "Eiger4M", "Merlin", "Timepix"],
                value = "Merlin",
                description = 'Detector',
                disabled = False,
                style = {'description_width': 'initial'}),

            x_bragg = widgets.IntText(
                value = 160,
                description = 'X Bragg, used for roi defintion:',
                disabled = False,
                tooltip = "Horizontal pixel number of the Bragg peak, can be used for the definition of the ROI",
                style = {'description_width': 'initial'}),

            y_bragg = widgets.IntText(
                value = 325,
                description = 'Y Bragg, used for roi defintion:',
                disabled = False,
                tooltip = "Vertical pixel number of the Bragg peak, can be used for the definition of the ROI",
                style = {'description_width': 'initial'}),

            photon_threshold = widgets.IntText(
                value = 0,
                description = 'Photon Threshold:',
                disabled = False,
                tooltip = "data[data < photon_threshold] = 0",
                style = {'description_width': 'initial'}),

            photon_filter = widgets.Dropdown(
                options = ['loading','postprocessing'],
                value = "loading",
                description = 'Photon filter',
                disabled = False,
                tooltip = "When the photon threshold should be applied, if 'loading', it is applied before binning; if 'postprocessing', it is applied at the end of the script before saving",
                style = {'description_width': 'initial'}),

            background_file = widgets.Text(
                value = "",
                placeholder = "self.work_dir + 'background.npz'",
                description = 'Background file',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            flatfield_file = widgets.Text(
                value = "",
                placeholder = f"{self.work_dir}flatfield_maxipix_8kev.npz",
                description = 'Flatfield file',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            hotpixels_file = widgets.Text(
                value = "/home/david/Documents/PhDScripts/SIXS_June_2021/reconstructions/analysis/mask_merlin_better_flipped.npy",
                placeholder = "mask_merlin.npz",
                description = 'Hotpixels file',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            # template_imagefile = widgets.Text(
            #     value = 'Pt_ascan_mu_%05d.nxs',
            #     description = 'Template imagefile',
            #     disabled = False,
            #     tooltip = """Template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'; Template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs';
            #                 Template for SIXS_2019: 'spare_ascan_mu_%05d.nxs';
            #                 Template for Cristal: 'S%d.nxs';
            #                 Template for P10: '_master.h5'; 
            #                 Template for NANOMAX: '%06d.h5'; 
            #                 Template for 34ID: 'Sample%dC_ES_data_51_256_256.npz'""",
            #     layout = Layout(width='90%'),
            #     style = {'description_width': 'initial'}),

            nb_pixel_x = widgets.IntText(
                description = 'Nb pixel x',
                disabled = False,
                tooltip = "fix to declare a known detector but with less pixels",
                style = {'description_width': 'initial'}),

            nb_pixel_y = widgets.IntText(
                description = 'Nb pixel y',
                disabled = False,
                tooltip = "fix to declare a known detector but with less pixels",
                style = {'description_width': 'initial'}),


            ### Define parameters below if you want to orthogonalize the data before phasing
            label_ortho = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters to define the data orthogonalization</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            use_rawdata = widgets.ToggleButton(
                value = True,
                description = 'Use Raw Data',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'False for using data gridded in laboratory frame/ True for using data in detector frame',
                icon = 'check'),

            interp_method = widgets.Dropdown(
                options = ['linearization','xrayutilities'],
                value = "linearization",
                description = 'Interpolation method',
                disabled = True,
                # tooltip = "",
                style = {'description_width': 'initial'}),

            fill_value_mask = widgets.Dropdown(
                options = [0, 1],
                value = 0,
                description = 'Fill value mask',
                disabled = True,
                tooltip = "It will define how the pixels outside of the data range are processed during the interpolation. Because of the large number of masked pixels, phase retrieval converges better if the pixels are not masked (0 intensity imposed). The data is by default set to 0 outside of the defined range.",
                style = {'description_width': 'initial'}),

            beam_direction = widgets.Text(
                value = "(1, 0, 0)",
                placeholder = "(1, 0, 0)",
                description = 'Beam direction in lab. frame',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='50%'),
                style = {'description_width': 'initial'},
                tooltip = "Beam direction in the laboratory frame (downstream, vertical up, outboard), beam along z"),

            sample_offsets = widgets.Text(
                value = "(0, 0)",
                placeholder = "(0, 0, 90, 0)",
                description = 'Sample offsets',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='25%'),
                style = {'description_width': 'initial'},
                tooltip = """Tuple of offsets in degrees of the sample for each sample circle (outer first). 
                            Convention: the sample offsets will be subtracted to the motor values"""),

            sdd = widgets.FloatText(
                value = 1.18,
                description = 'Sample Detector Dist. (m):',
                disabled = True,
                tooltip = "sample to detector distance in m",
                style = {'description_width': 'initial'}),

            energy = widgets.IntText(
                value = 8500,
                description = 'X-ray energy in eV',
                disabled = True,
                style = {'description_width': 'initial'}),

            custom_motors = widgets.Text(
                value = "{}",
                placeholder = "{}",
                description = 'Custom motors',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'},
                tooltip = "Use this to declare motor positions"),
            # {"mu": 0, "phi": -15.98, "chi": 90, "theta": 0, "delta": -0.5685, "gamma": 33.3147}
            # use this to declare motor positions if there is not log file
            # example: {"eta": np.linspace(16.989, 18.989, num=100, endpoint=False), "phi": 0, "nu": -0.75, "delta": 36.65}
            # ID01: eta, phi, nu, delta
            # CRISTAL: mgomega, gamma, delta
            # SIXS: beta, mu, gamma, delta
            # P10: om, phi, chi, mu, gamma, delta
            # NANOMAX: theta, phi, gamma, delta, energy, radius
            # 34ID: mu, phi (incident angle), chi, theta (inplane), delta (inplane), gamma (outofplane)


            ### Parameters for xrayutilities to orthogonalize the data before phasing
            label_xru = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Parameters used in xrayutilities to orthogonalize the data before phasing</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            #xrayutilities uses the xyz crystal frame: for incident angle = 0, x is downstream, y outboard, and z vertical up
            align_q = widgets.ToggleButton(
                value = True,
                description = 'Align q',
                disabled = True,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = """used only when interp_method is 'linearization', if True it rotates the crystal to align q along one axis of the array""",
                icon = 'check'),

            ref_axis_q = widgets.Dropdown(
                options = ["x", "y", "z"],
                value = "y",
                description = 'Ref axis q',
                disabled = True,
                layout = Layout(width='20%'),
                tooltip = "q will be aligned along that axis",
                style = {'description_width': 'initial'}),

            outofplane_angle = widgets.FloatText(
                value = 0,
                description = 'Outofplane angle',
                disabled = True,
                layout = Layout(width='25%'),
                style = {'description_width': 'initial'}),

            inplane_angle = widgets.FloatText(
                value = 0,
                description = 'Inplane angle',
                disabled = True,
                layout = Layout(width='25%'),
                style = {'description_width': 'initial'}),

            sample_inplane = widgets.Text(
                value = "(1, 0, 0)",
                placeholder = "(1, 0, 0)",
                description = 'Sample inplane',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "Sample inplane reference direction along the beam at 0 angles"),

            sample_outofplane = widgets.Text(
                value = "(0, 0, 1)",
                placeholder = "(0, 0, 1)",
                description = 'Sample outofplane',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "Surface normal of the sample at 0 angles"),

            offset_inplane = widgets.FloatText(
                value = 0,
                step = 0.01,
                description = 'Offset inplane',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "Outer detector angle offset, not important if you use raw data"),

            cch1 = widgets.IntText(
                value = 271,
                description = 'cch1',
                disabled = True,
                layout = Layout(width='15%'),
                tooltip = "cch1 parameter from xrayutilities 2D detector calibration, vertical",
                style = {'description_width': 'initial'}),

            cch2 = widgets.IntText(
                value = 213,
                description = 'cch2',
                disabled = True,
                layout = Layout(width='15%'),
                tooltip = "cch2 parameter from xrayutilities 2D detector calibration, horizontal",
                style = {'description_width': 'initial'}),

            direct_inplane = widgets.FloatText(
                value = 0,
                step = 0.01,
                min = 0,
                max = 360,
                description = 'Direct inplane angle:',
                layout = Layout(width='30%'),
                tooltip = "In level of grey in [0,1], 0 being dark. For visual comfort during masking",
                readout = True,
                style = {'description_width': 'initial'},
                disabled = False),

            direct_outofplane = widgets.FloatText(
                value = 0,
                step = 0.01,
                min = 0,
                max = 360,
                description = 'Direct outofplane angle:',
                layout = Layout(width='30%'),
                tooltip = "In level of grey in [0,1], 0 being dark. For visual comfort during masking",
                readout = True,
                style = {'description_width': 'initial'},
                disabled = False),

            detrot = widgets.FloatText(
                value = 0,
                step = 0.01,
                description = 'Detector rotation',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "Detrot parameter from xrayutilities 2D detector calibration"),

            tiltazimuth = widgets.FloatText(
                value = 360,
                step = 0.01,
                description = 'Tilt azimuth',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='15%'),
                style = {'description_width': 'initial'},
                tooltip = "tiltazimuth parameter from xrayutilities 2D detector calibration"),

            tilt = widgets.FloatText(
                value = 0,
                step = 0.01,
                description = 'Tilt',
                disabled = True,
                continuous_update = False,
                layout = Layout(width='15%'),
                style = {'description_width': 'initial'},
                tooltip = "tilt parameter from xrayutilities 2D detector calibration"),

            # Run preprocess
            label_preprocess = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Click below to run the data processing before phasing</p>", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            run_preprocess = widgets.ToggleButton(
                value = False,
                description = 'Run data preprocessing ...',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'},
                icon = 'check')
            )
        self._list_widgets_preprocessing.children[1].observe(self.beamline_handler, names = "value")
        self._list_widgets_preprocessing.children[8].observe(self.energy_scan_handler, names = "value")
        self._list_widgets_preprocessing.children[14].observe(self.bragg_peak_centering_handler, names = "value")
        self._list_widgets_preprocessing.children[26].observe(self.reload_data_handler, names = "value")
        self._list_widgets_preprocessing.children[47].observe(self.interpolation_handler, names = "value")
        self._list_widgets_preprocessing.children[-2].observe(self.preprocess_handler, names = "value")

        # Widgets for angles correction 
        self._list_widgets_correct = interactive(self.correct_angles,
            label_correct = widgets.HTML(
                description="<p style='font-weight: bold;font-size:1.2em'>Find the real values for the Bragg peak angles, needs correct xru parameters.", # 
                style = {'description_width': 'initial'},
                layout = Layout(width='90%', height = "35px")),

            csv_file = widgets.Text(
                value = os.getcwd() + "/<filename>.csv",
                placeholder = "Path to csv file",
                description = 'Csv file',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            temp_bool = widgets.ToggleButton(
                value = False,
                description = 'Estimate the temperature (Pt only)',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'Click to estimate the mean temperature of the sample from the Bragg peak angles',
                icon = 'check',
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'}),

            reflection = widgets.Text(
                value = "[1, 1, 1]",
                placeholder = "[1, 1, 1]",
                description = 'Reflection',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='30%'),
                style = {'description_width': 'initial'},
                tooltip = "Sample inplane reference direction along the beam at 0 angles"),

            reference_spacing = widgets.FloatText(
                value = 2.269545,
                step = 0.000001,
                min = 0,
                max = 100,
                description = 'Reference spacing (A):',
                layout = Layout(width='30%'),
                readout = True,
                style = {'description_width': 'initial'},
                disabled = False),

            reference_temperature = widgets.FloatText(
                value = 293.15,
                step = 0.01,
                min = 0,
                max = 2000,
                description = 'Reference temperature:',
                layout = Layout(width='30%'),
                readout = True,
                style = {'description_width': 'initial'},
                disabled = False),

            angles_bool = widgets.ToggleButton(
                value = False,
                description = 'Correct angles',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'Click to correct the Bragg peak angles',
                icon = 'check',
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'}),
            )
        self._list_widgets_correct.children[2].observe(self.temp_handler, names = "value")
        self._list_widgets_correct.children[-2].observe(self.correct_angles_handler, names = "value")

        # Create the final window
        self.tab_init = widgets.VBox([
            self._list_widgets_init.children[0],
            widgets.HBox(self._list_widgets_init.children[1:3]),
            self._list_widgets_init.children[3],
            self._list_widgets_init.children[4],
            self._list_widgets_init.children[5],
            widgets.HBox(self._list_widgets_init.children[6:8]),
            self._list_widgets_init.children[-1],
            ])

        self.tab_beamline = widgets.VBox([
            self._list_widgets_preprocessing.children[0],
            self._list_widgets_preprocessing.children[1],
            widgets.HBox(self._list_widgets_preprocessing.children[2:4]),
            self._list_widgets_preprocessing.children[4],
            widgets.HBox(self._list_widgets_preprocessing.children[5:7]),
            self._list_widgets_preprocessing.children[7],
            widgets.HBox(self._list_widgets_preprocessing.children[8:10]),
            self._list_widgets_preprocessing.children[10],
            widgets.HBox(self._list_widgets_preprocessing.children[11:13]),
            ])

        self.tab_reduction = widgets.VBox([
            self._list_widgets_preprocessing.children[13],
            widgets.HBox(self._list_widgets_preprocessing.children[14:16]), 
            widgets.HBox(self._list_widgets_preprocessing.children[17:20]),
            self._list_widgets_preprocessing.children[20],
            widgets.HBox(self._list_widgets_preprocessing.children[21:24]),
            self._list_widgets_preprocessing.children[24],
            ])

        self.tab_save_load = widgets.VBox([
            self._list_widgets_preprocessing.children[25],
            widgets.HBox(self._list_widgets_preprocessing.children[26:29]),
            self._list_widgets_preprocessing.children[29],
            widgets.HBox(self._list_widgets_preprocessing.children[30:35]),
            ])
        
        self.tab_detector = widgets.VBox([
            self._list_widgets_preprocessing.children[35],
            self._list_widgets_preprocessing.children[36],
            widgets.HBox(self._list_widgets_preprocessing.children[37:39]),
            widgets.HBox(self._list_widgets_preprocessing.children[39:41]),
            self._list_widgets_preprocessing.children[41],
            self._list_widgets_preprocessing.children[42],
            self._list_widgets_preprocessing.children[43],
            widgets.HBox(self._list_widgets_preprocessing.children[44:46]),
            ])

        self.tab_ortho = widgets.VBox([
            self._list_widgets_preprocessing.children[46],
            self._list_widgets_preprocessing.children[47],
            widgets.HBox(self._list_widgets_preprocessing.children[48:50]),
            self._list_widgets_preprocessing.children[50],
            self._list_widgets_preprocessing.children[51],
            self._list_widgets_preprocessing.children[52],
            widgets.HBox(self._list_widgets_preprocessing.children[53:55]),
            self._list_widgets_preprocessing.children[55],
            widgets.HBox(self._list_widgets_preprocessing.children[56:58]),
            widgets.HBox(self._list_widgets_preprocessing.children[58:61]),
            widgets.HBox(self._list_widgets_preprocessing.children[61:63]),
            widgets.HBox(self._list_widgets_preprocessing.children[63:67]),
            widgets.HBox(self._list_widgets_preprocessing.children[67:70]),
            ])

        self.tab_run = widgets.VBox([
            self._list_widgets_preprocessing.children[-3],
            self._list_widgets_preprocessing.children[-2],
            self._list_widgets_preprocessing.children[-1]
            ])

        self.tab_correct = widgets.VBox([
            self._list_widgets_correct.children[0],
            self._list_widgets_correct.children[1],
            self._list_widgets_correct.children[2],
            widgets.HBox(self._list_widgets_correct.children[3:6]),
            self._list_widgets_correct.children[6],
            self._list_widgets_correct.children[-1],
            ])

        self.window = widgets.Tab(children=[self.tab_init, self.tab_beamline, self.tab_reduction, self.tab_save_load, self.tab_detector, self.tab_ortho, self.tab_run, self.tab_correct])
        self.window.set_title(0, 'Scan detail')
        self.window.set_title(1, 'Beamline')
        self.window.set_title(2, "Data reduction")
        self.window.set_title(3, "Load/Save")
        self.window.set_title(4, 'Detector')
        self.window.set_title(5, 'Orthogonalization')
        self.window.set_title(6, 'Preprocess')
        self.window.set_title(7, 'Correct')

        display(self.window)

    # Widgets interactivbeam_directione functions
    def initialize_directories(self,
        label_scan,
        sample_name,
        scans,
        data_directory,
        final_directory,
        user_comment,
        debug,
        run_dir_init,
        ):
        """
        Function to move file from datadir to folder where it will be used by preprocess.bcdi
        Arg 1: Original data directory 
        Arg 2: Path of EXISTING target directory (e.g. Pt_Al2O3/) (subdirectories S{scan}/data & S{scan}/pynx_raw will 
            be updated/created)
        Arg 3: Scan(s) number, list or single value

        Looks recursively for one mu or omega scan including the scan number (glob.glob).

        Also moves all the notebooks needed for data analysis, and a pynx_run.txt file with all the parameters for phase retrieval,
            initialized for this dataset
        """

        if run_dir_init:
            # Save as attributes for use in future widgets

            # Transform string of list into python list object if multiple scans
            # if scans.startswith("["): # Should not happen in the gui
            #     self.scans = ast.literal_eval(scans)
                
            # else:
            #     self.scans = [scans]
            self.scans = scans
            self.sample_name = sample_name
            self.data_directory = data_directory
            self.user_comment = user_comment
            self.debug = debug
            self.data_dirname = None

            self.root_folder = final_directory
            # self.root_folder = root_folder

            # Scan folder
            self.scan_folder = self.root_folder + f"S{scans}/"
            print("Scan folder:", self.scan_folder)

            self.save_dir = None # scan_folder +"pynxraw/"

            # Data folder
            self.data_folder = self.scan_folder + "data/" # folder of the experiment, where all scans are stored
            print("Data folder:", self.data_folder)

            # Filename
            try:
                self.path_to_data = glob.glob(f"{self.data_directory}*mu*{self.scans}*")[0]
                print("File path:", self.path_to_data)
            except IndexError:
                    self.path_to_data = glob.glob(f"{self.data_directory}*omega*{self.scans}*")[0]
                    print("Omega scan") 

            self.template_imagefile = self.path_to_data.split("/data/")[-1].split("%05d"%self.scans)[0]+"%05d.nxs" #  +"%05d_R.nxs" If rotated before
            print("File template:", self.template_imagefile)


            # Create final directory is not yet existing
            if not os.path.isdir(self.root_folder):
                print(self.root_folder)
                full_path = ""
                for d in self.root_folder.split("/"):
                    full_path += d + "/"
                    try:
                        os.mkdir(full_path)
                    except FileExistsError:
                        pass

            print(f"Updating directories ...")

            # Scan directory
            try:
                os.mkdir(f"{self.root_folder}S{self.scans}")
                print(f"Created {self.root_folder}S{self.scans}")
            except FileExistsError:
                print(f"{self.root_folder}S{self.scans} exists")
                pass

            # /data directory
            try:
                os.mkdir(f"{self.root_folder}S{self.scans}/data")
                print(f"Created {self.root_folder}S{self.scans}/data")
            except FileExistsError:
                print(f"{self.root_folder}S{self.scans}/data exists")
                pass

            # /pynxraw directory
            try:
                os.mkdir(f"{self.root_folder}S{self.scans}/pynxraw")
                print(f"Created {self.root_folder}S{self.scans}/pynxraw")
            except FileExistsError:
                print(f"{self.root_folder}S{self.scans}/pynxraw exists")
                pass

            # /postprocessing directory
            try:
                os.mkdir(f"{self.root_folder}S{self.scans}/postprocessing")
                print(f"Created {self.root_folder}S{self.scans}/postprocessing")
            except FileExistsError:
                print(f"{self.root_folder}S{self.scans}/postprocessing exists")
                pass

            print(f"Moving files ...")

            # move data file
            try:
                shutil.copy2(self.path_to_data, f"{self.root_folder}S{self.scans}/data")
                print(f"Copied {self.path_to_data} to {self.root_folder}S{self.scans}/data")
            except FileExistsError:
                print(f"{self.root_folder}S{self.scans}/data/{self.path_to_data} exists")
                pass

            # move pynx_run.txt file
            try:
                shutil.copy(f"{self.path_package}bcdi/pynx_run.txt", f"{self.root_folder}S{self.scans}/pynxraw")
                print(f"Copied pynx_run.txt to {self.root_folder}S{self.scans}/pynxraw")
            except FileExistsError:
                print(f"{self.root_folder}S{self.scans}/pynxraw/pynx_run.txt exists")
                pass

            # Move notebooks
            try:
                shutil.copy(f"{self.path_package}bcdi/PhasingNotebook.ipynb", f"{self.root_folder}S{self.scans}/pynxraw")
                print(f"Copied PhasingNotebook.ipynb to {self.root_folder}S{self.scans}/pynxraw")
            except FileExistsError:
                print(f"{self.root_folder}S{self.scans}/pynxraw/PhasingNotebook.ipynb exists")
                pass

            try:
                shutil.copy(f"{self.path_package}bcdi/CompareFacetsEvolution.ipynb", f"{self.root_folder}S{self.scans}/postprocessing")
                print(f"Copied CompareFacetsEvolution.ipynb to {self.root_folder}S{self.scans}/postprocessing")
            except FileExistsError:
                print(f"{self.root_folder}S{self.scans}/postprocessing/CompareFacetsEvolution.ipynb exists")
                pass


    def initialize_parameters(self,
        label_beamline, beamline, actuators, is_series, custom_scan, custom_images, custom_monitor, specfile_name, rocking_angle, follow_bragg,
        label_masking, flag_interact, background_plot,
        label_centering, centering, fix_bragg, fix_size, center_fft, pad_size,
        normalize_flux, 
        label_filtering, mask_zero_event, flag_medianfilter, medfilt_order, binning,
        label_reload, reload_previous, reload_orthogonal, preprocessing_binning,
        label_saving, save_rawdata, save_to_npz, save_to_mat, save_to_vti, save_asint,
        label_detector, detector, x_bragg, y_bragg, photon_threshold, photon_filter, background_file, hotpixels_file, flatfield_file,
        #  template_imagefile,
        nb_pixel_x, nb_pixel_y,
        label_ortho, use_rawdata, interp_method, fill_value_mask, beam_direction, sample_offsets, sdd, energy, custom_motors,
        label_xru, align_q, ref_axis_q, outofplane_angle, inplane_angle, 
        sample_inplane, sample_outofplane, offset_inplane, cch1, cch2, direct_inplane, direct_outofplane, detrot, tiltazimuth, tilt,
        label_preprocess, run_preprocess):

        if run_preprocess:
            # Save parameter values as attributes
            self.binning = binning
            self.flag_interact = flag_interact
            self.background_plot = str(background_plot)
            self.centering = centering
            self.fix_bragg = fix_bragg
            self.fix_size = fix_size
            self.center_fft = center_fft
            self.pad_size = pad_size
            self.normalize_flux = normalize_flux
            self.mask_zero_event = mask_zero_event
            self.flag_medianfilter = flag_medianfilter
            self.medfilt_order = medfilt_order
            self.reload_previous = reload_previous
            self.reload_orthogonal = reload_orthogonal
            self.preprocessing_binning = preprocessing_binning
            self.save_rawdata = save_rawdata
            self.save_to_npz = save_to_npz
            self.save_to_mat = save_to_mat
            self.save_to_vti = save_to_vti
            self.save_asint = save_asint
            self.beamline = beamline
            self.actuators = actuators
            self.is_series = is_series
            self.custom_scan = custom_scan
            self.custom_images = custom_images
            self.custom_monitor = custom_monitor
            self.rocking_angle = rocking_angle
            self.follow_bragg = follow_bragg
            self.specfile_name = specfile_name
            self.detector = detector
            self.x_bragg = x_bragg
            self.y_bragg = y_bragg
            self.photon_threshold = photon_threshold
            self.photon_filter = photon_filter
            self.background_file = background_file
            self.hotpixels_file = hotpixels_file
            self.flatfield_file = flatfield_file
            # self.template_imagefile = template_imagefile
            self.nb_pixel_x = nb_pixel_x
            self.nb_pixel_y = nb_pixel_y
            self.use_rawdata = use_rawdata
            self.interp_method = interp_method
            self.fill_value_mask = fill_value_mask
            self.beam_direction = beam_direction
            self.sample_offsets = sample_offsets
            self.sdd = sdd
            self.energy = energy
            self.custom_motors = custom_motors
            self.align_q = align_q
            self.ref_axis_q = ref_axis_q
            self.outofplane_angle = outofplane_angle
            self.inplane_angle = inplane_angle
            self.sample_inplane = sample_inplane
            self.sample_outofplane = sample_outofplane
            self.offset_inplane = offset_inplane
            self.cch1 = cch1
            self.cch2 = cch2
            self.direct_inplane = direct_inplane
            self.direct_outofplane = direct_outofplane
            self.detrot = detrot
            self.tiltazimuth = tiltazimuth
            self.tilt = tilt

            # Extract dict, list and tuple from strings
            self.list_parameters = ["fix_bragg", "fix_size", "pad_size"]

            self.tuple_parameters = ["binning", "preprocessing_binning", "beam_direction", "sample_offsets", "sample_inplane", "sample_outofplane"]

            self.dict_parameters = ["actuators", "custom_motors"]

            try:
                for p in self.list_parameters:
                    if getattr(self, p) == "":
                        setattr(self, p, [])
                    else:
                        setattr(self, p, literal_eval(getattr(self, p)))
                    # print(f"{p}:", getattr(self, p))
            except ValueError:
                print(f"Wrong list syntax for {p}")

            try:
                for p in self.tuple_parameters:
                    if getattr(self, p) == "":
                        setattr(self, p, ())
                    else:
                        setattr(self, p, literal_eval(getattr(self, p)))
                    # print(f"{p}:", getattr(self, p))
            except ValueError:
                print(f"Wrong tuple syntax for {p}")

            try:
                for p in self.dict_parameters:
                    if getattr(self, p) == "":
                        setattr(self, p, None) # or {}
                    else:
                        if literal_eval(getattr(self, p)) == {}:
                            setattr(self, p, None)
                        else:
                            setattr(self, p, literal_eval(getattr(self, p)))
                    # print(f"{p}:", getattr(self, p))
            except ValueError:
                print(f"Wrong dict syntax for {p}")

            # Empty parameters are set to None (bcdi syntax)
            if self.data_dirname == "":
                self.data_dirname = None

            if self.background_file == "":
                self.background_file = None

            if self.hotpixels_file == "":
                self.hotpixels_file = None

            if self.flatfield_file == "":
                self.flatfield_file = None

            if self.specfile_name == "":
                self.specfile_name = None
                
            if self.nb_pixel_x == 0:
                self.nb_pixel_x = None

            if self.nb_pixel_y == 0:
                self.nb_pixel_y = None


            self.roi_detector = [self.y_bragg - 160, self.y_bragg + 160, self.x_bragg - 160, self.x_bragg + 160]
            self.roi_detector = []
            # [Vstart, Vstop, Hstart, Hstop]
            # leave it as [] to use the full detector. Use with center_fft='skip' if you want this exact size.

            self.linearity_func = None


            # Check is SIXS data, in that case rotate
            if self.beamline == "SIXS_2019":
                self.rotate_sixs_data()

            # On lance BCDI
            preprocess_bcdi(
                scans = self.scans,
                sample_name = self.sample_name,
                root_folder = self.root_folder,
                save_dir = self.save_dir,
                data_dirname = self.data_dirname,
                user_comment = self.user_comment,
                debug = self.debug,
                binning = self.binning,
                flag_interact = self.flag_interact,
                background_plot = self.background_plot,
                centering = self.centering,
                fix_bragg = self.fix_bragg,
                fix_size = self.fix_size,
                center_fft = self.center_fft,
                pad_size = self.pad_size,
                normalize_flux = self.normalize_flux,
                mask_zero_event = self.mask_zero_event,
                flag_medianfilter = self.flag_medianfilter,
                medfilt_order = self.medfilt_order,
                reload_previous = self.reload_previous,
                reload_orthogonal = self.reload_orthogonal,
                preprocessing_binning = self.preprocessing_binning,
                save_rawdata = self.save_rawdata,
                save_to_npz = self.save_to_npz,
                save_to_mat = self.save_to_mat,
                save_to_vti = self.save_to_vti,
                save_asint = self.save_asint,
                beamline = self.beamline,
                actuators = self.actuators,
                is_series = self.is_series,
                custom_scan = self.custom_scan,
                custom_images = self.custom_images,
                custom_monitor = self.custom_monitor,
                rocking_angle = self.rocking_angle,
                follow_bragg = self.follow_bragg,
                specfile_name = self.specfile_name,
                detector = self.detector,
                linearity_func = self.linearity_func,
                x_bragg = self.x_bragg,
                y_bragg = self.y_bragg,
                roi_detector = self.roi_detector,
                photon_threshold = self.photon_threshold,
                photon_filter = self.photon_filter,
                background_file = self.background_file,
                hotpixels_file = self.hotpixels_file,
                flatfield_file = self.flatfield_file,
                template_imagefile = self.template_imagefile,
                nb_pixel_x = self.nb_pixel_x,
                nb_pixel_y = self.nb_pixel_y,
                use_rawdata = self.use_rawdata,
                interp_method = self.interp_method,
                fill_value_mask = self.fill_value_mask,
                beam_direction = self.beam_direction,
                sample_offsets = self.sample_offsets,
                sdd = self.sdd,
                energy = self.energy,
                custom_motors = self.custom_motors,
                align_q = self.align_q,
                ref_axis_q = self.ref_axis_q,
                outofplane_angle = self.outofplane_angle,
                inplane_angle = self.inplane_angle,
                sample_inplane = self.sample_inplane,
                sample_outofplane = self.sample_outofplane,
                offset_inplane = self.offset_inplane,
                cch1 = self.cch1,
                cch2 = self.cch2,
                detrot = self.detrot,
                tiltazimuth = self.tiltazimuth,
                tilt = self.tilt,
            )

        if not run_preprocess:
            clear_output(True)


    def correct_angles(self,
        label_correct,
        csv_file,
        temp_bool,
        reflection,
        reference_spacing,
        reference_temperature,
        angles_bool,
        ):

        if angles_bool:
            # Save parameter values as attributes
            self.label_correct = label_correct
            self.csv_file = csv_file
            self.angles_bool = angles_bool
            self.temp_bool = temp_bool
            self.reference_spacing = reference_spacing
            self.reference_temperature = reference_temperature

            try:
                self.reflection = np.array(literal_eval(reflection))
            except ValueError:
                print(f"Wrong list syntax for refelction")

            # On lance la correction
            self.metadata = correct_angles_detector(
                filename = self.path_to_data,
                direct_inplane = self.direct_inplane,
                direct_outofplane = self.direct_outofplane,
                get_temperature = self.temp_bool,
                reflection = self.reflection,
                reference_spacing = self.reference_spacing, 
                reference_temperature = self.reference_temperature,
                high_threshold = 1000000,  
                save_dir = f"{self.root_folder}S{self.scans}/postprocessing/",
                csv_file = self.csv_file,
                scan = self.scans,
                root_folder = self.root_folder,
                sample_name = self.sample_name,
                filtered_data = False,
                peak_method = self.centering,
                normalize_flux = self.normalize_flux,
                debug = self.debug,
                beamline = self.beamline,
                actuators = self.actuators,
                is_series = self.is_series,
                custom_scan = self.custom_scan,
                custom_images = self.custom_images,
                custom_monitor = self.custom_monitor,
                custom_motors = self.custom_motors,
                rocking_angle = self.rocking_angle,
                specfile_name = self.specfile_name,
                detector = self.detector,
                x_bragg = self.x_bragg,
                y_bragg = self.y_bragg,
                roi_detector = None,
                hotpixels_file = self.hotpixels_file, 
                flatfield_file = self.flatfield_file,
                template_imagefile = self.template_imagefile,
                beam_direction = self.beam_direction,
                sample_offsets = self.sample_offsets,
                directbeam_x = self.cch1,
                directbeam_y = self.cch2,
                sdd = self.sdd,
                energy = self.energy,
            )

            # Save metadata
            for keys, values in self.metadata.items():
                setattr(self, keys, values)

        if not angles_bool:
            clear_output(True)


    # Non widgets functions
    def rotate_sixs_data(self):
        """
        Python script to rotate the data for vertical configuration
                Arg 1: Path of target directory (before /S{scan} ... )
                Arg 2: Scan(s) number, list or single value
        """

        print("Rotating SIXS data ...")
        with tb.open_file(self.path_to_data, "a") as f:
            # Get data
            try:
                # if rocking_angle == "omega":
                data_og = f.root.com.scan_data.data_02[:]
                # elif rocking_angle == "mu":
                #     data_og = f.root.com.scan_data.merlin_image[:]
                print("Calling merlin the enchanter in SBS...")
                self.scan_type = "SBS"
            except:
                try:
                    data_og = f.root.com.scan_data.self_image[:]
                    print("Calling merlin the enchanter in FLY...")
                    self.scan_type = "FLY"
                except:
                    print("This data does not result from Merlin :/")

            # Just an index for plotting schemes
            half = int(data_og.shape[0]/2)

            # Rotate data
            data = np.transpose(data_og, axes=(0, 2, 1))
            for idx in range(data.shape[0]):
                tmp = data[idx, :, :]
                data[idx, :, :] = np.fliplr(tmp)
            print("Data well rotated by 90°.")  

            print("Saving example figures...", end="\n\n")
            plt.figure(figsize = (16, 9))
            plt.imshow(data_og[half, :, :], vmax = 10)
            plt.xlabel('Delta')
            plt.ylabel('Gamma')
            plt.tight_layout()
            plt.savefig(self.root_folder + self.sample_name + str(self.scans) + "/data/data_before_rotation.png")
            plt.close()

            plt.figure(figsize = (16, 9))        
            plt.imshow(data[half, :, :], vmax = 10)
            plt.xlabel('Gamma')
            plt.ylabel('Delta')
            plt.tight_layout()
            plt.savefig(self.root_folder + self.sample_name + str(self.scans) + "/data/data_after_rotation.png")
            plt.close()

            # Overwrite data in copied file
            try:
                if self.scan_type == "SBS":
                    f.root.com.scan_data.data_02[:] = data
                elif self.scan_type == "FLY":
                    f.root.com.scan_data.test_image[:] = data
            except:
                print("Could not overwrite data ><")


    def extract_metadata(self):
        # Save rocking curve data
        np.savez(save_dir + "correct_detector_data.npz",
            tilt_values = tilt_values,
            rocking_curve = rocking_curve,
            interp_tilt = interp_tilt,
            interp_curve = interp_curve,
            COM_rocking_curve = tilt_values[z0],
            detector_data_COM = abs(data[int(round(z0)), :, :]),
            interp_fwhm = interp_fwhm
            )

        print(f"Saved data in {save_dir}correct_detector_data.npz")

        # Use this opportunity to save a lot more data !
        print(f"Opening {filename}")
        data = rd.DataSet(filename)

        # Add new data
        DF = pd.DataFrame([[scan, particle, q, qnorm, dist_plane, bragg_inplane, bragg_outofplane, data.x[0], data.y[0], data.z[0], data.mu[0], data.delta[0], data.omega[0],
                            data.gamma[0], data.gamma[0] - data.mu[0], (data.mu[-1] - data.mu[-0]) / len(data.mu), data.integration_time[0], len(data.integration_time), 
                            interp_fwhm, bragg_x, bragg_y, tilt_values[z0],
                            data.ssl3hg[0], data.ssl3vg[0], 
                            data.ssl1hg[0], data.ssl1vg[0]
                            ]],
                            columns = [
                                "scan", "particle", "q", "q_norm", "plane", "inplane_angle", "out_of_plane_angle", "x", "y", "z", "mu", "delta", "omega","gamma", 'gamma-mu',
                                "step size", "integration time", "steps", "FWHM", "bragg_x", "bragg_y", "COM_rocking_curve",
                                "ssl3hg", "ssl3vg", 
                                "ssl1hg", "ssl1vg", 
                            ])

        # Load all the data
        try:
            df = pd.read_csv(csv_file)

            # Replace old data linked to this scan, no problem if this row does not exist yet
            indices = df[df['scan'] == scan].index
            df.drop(indices , inplace=True)

            result = pd.concat([df, DF])

        except FileNotFoundError:
            result = DF

        # Save 
        result.to_csv(csv_file,
                    index = False,
                    columns = [
                        "scan", "particle", "q", "q_norm", "plane", "inplane_angle", "out_of_plane_angle", "x", "y", "z", "mu", "delta", "omega","gamma", 'gamma-mu',
                        "step size", "integration time", "steps", "ssl3hg", "ssl3vg", "FWHM", "bragg_x", "bragg_y", "COM_rocking_curve"
                    ])
        print(f"Saved in {csv_file}")


    # Below are handlers
    def init_handler(self, change):
        """Handles changes on the widget used for the initialization"""

        if not change.new:
            for w in self._list_widgets_init.children[:7]:
                w.disabled = False

        if change.new:
            for w in self._list_widgets_init.children[:7]:
                w.disabled = True

    def beamline_handler(self, change):
        "Handles changes on the widget used for the initialization"

        if change.new in ["SIXS_2019", "ID01"]:
            for w in self._list_widgets_preprocessing.children[2:7]:
                w.disabled = True

        if change.new not in ["SIXS_2019", "ID01"]:
            for w in self._list_widgets_preprocessing.children[2:7]:
                w.disabled = False

    def preprocess_handler(self, change):
        "Handles changes on the widget used for the initialization"

        if not change.new:
            for w in self._list_widgets_preprocessing.children[:-2]:
                w.disabled = False

        if change.new:
            for w in self._list_widgets_preprocessing.children[:-2]:
                w.disabled = True

    def bragg_peak_centering_handler(self, change):
        "Handles changes related to the centering of the Bragg peak"

        if change.new == "manual":
            self._list_widgets_preprocessing.children[15].disabled = False

        if change.new != "manual":
            self._list_widgets_preprocessing.children[15].disabled = True

    def energy_scan_handler(self, change):
        "Handles changes related to energy scans"

        if change.new == "energy":
            self._list_widgets_preprocessing.children[9].disabled = False

        if change.new != "energy":
            self._list_widgets_preprocessing.children[9].disabled = True

    def reload_data_handler(self, change):
        "Handles changes related to data reloading"

        if change.new:
            for w in self._list_widgets_preprocessing.children[27:29]:
                w.disabled = False

        if not change.new:
            for w in self._list_widgets_preprocessing.children[27:29]:
                w.disabled = True

    def interpolation_handler(self, change):
        "Handles changes related to data interpolation"

        if change.new:
            for w in self._list_widgets_preprocessing.children[48:68]:
                w.disabled = True

        if not change.new:
            for w in self._list_widgets_preprocessing.children[48:68]:
                w.disabled = False

    def temp_handler(self, change):
        "Handles changes related to data interpolation"

        if change.new:
            for w in self._list_widgets_correct.children[3:6]:
                w.disabled = False

        if not change.new:
            for w in self._list_widgets_correct.children[3:6]:
                w.disabled = True

    def correct_angles_handler(self, change):
        "Handles changes related to data interpolation"

        if change.new:
            for w in self._list_widgets_correct.children[:-2]:
                w.disabled = True

        if not change.new:
            for w in self._list_widgets_correct.children[:-2]:
                w.disabled = False

