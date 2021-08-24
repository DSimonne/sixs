try :
    import numpy as np
    import pandas as pd
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import glob
    import errno
    import os
    import shutil
    import math
    from ast import literal_eval

    import lmfit
    from lmfit import minimize, Parameters, Parameter
    from lmfit.models import LinearModel, ConstantModel, QuadraticModel, PolynomialModel, StepModel
    from lmfit.models import GaussianModel, LorentzianModel, SplitLorentzianModel, VoigtModel, PseudoVoigtModel
    from lmfit.models import MoffatModel, Pearson7Model, StudentsTModel, BreitWignerModel, LognormalModel, ExponentialGaussianModel, SkewedGaussianModel, SkewedVoigtModel, DonaichModel
    import corner
    import numdifftools
    from scipy.stats import chisquare

    import ipywidgets as widgets
    from ipywidgets import interact, Button, Layout, interactive, fixed
    from IPython.display import display, Markdown, Latex, clear_output

    from scipy import interpolate
    from scipy import optimize, signal
    from scipy import sparse

    from datetime import datetime
    import pickle
    import inspect
    import warnings

    import tables as tb

except ModuleNotFoundError:
    raise ModuleNotFoundError("""The following packages must be installed: numpy, pandas, matplotlib, glob, errno, os, shutil, math, lmfit, corner, numdifftools, scipy, ipywidgets, importlib, pickle, inspect, warnings""")

# Import preprocess_bcdi modified for gui and usable as a function
from phdutils.bcdi.gui.preprocess_gui import *

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

        # Widgets for the gui will need to separate later
        self._list_widgets = interactive(self.initialize_parameters,
            scans = widgets.BoundedIntText(
                value = "01305",
                description = 'Scan nb:',
                min = 0,
                max = 2000,
                disabled = False,
                continuous_update = False,
                layout = Layout(width='45%'),
                style = {'description_width': 'initial'}),

            root_folder = widgets.Text(
                value = os.getcwd() + "/",
                placeholder = "path/to/data",
                description = 'Root folder',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            save_dir = widgets.Text(
                value = "",
                placeholder = "Images will be saved there",
                description = 'Save dir',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

            data_dirname = widgets.Text(
                value = "",
                placeholder = "scan_folder/pynx/ or scan_folder/pynxraw/",
                description = 'Data dir',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='90%'),
                tooltip = "(default to scan_folder/pynx/ or scan_folder/pynxraw/",
                style = {'description_width': 'initial'}),

            sample_name = widgets.Text(
                value = "S",
                placeholder = "",
                description = 'Sample Name',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='45%'),
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
                icon = 'check'),
            
            binning = widgets.Text(
                value = "(1, 1, 1)",
                placeholder = "(1, 1, 1)",
                description = 'Binning for phasing',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "binning that will be used for phasing (stacking dimension, detector vertical axis, detector horizontal axis)"),

            ### Parameters used in masking 
            flag_interact = widgets.ToggleButton(
                value = True,
                description = 'Flag interact',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'True to interact with plots, False to close it automatically',
                icon = 'check'),

            background_plot = widgets.FloatText(
                value = 0.5,
                step = 0.01,
                max = 1,
                min = 0,
                description = 'Background plot:',
                tooltip = "In level of grey in [0,1], 0 being dark. For visual comfort during masking",
                readout = True,
                style = {'description_width': 'initial'},
                disabled = False),


            ### Parameters related to data cropping/padding/centering
            centering  = widgets.Dropdown(
                options = ["max", "com"],
                value = "max",
                description = 'Centering:',
                disabled = False,
                layout = Layout(width='15%'),
                tooltip = "Bragg peak determination: 'max' or 'com', 'max' is better usually. It will be overridden by 'fix_bragg' if not empty",
                style = {'description_width': 'initial'}),

            fix_bragg = widgets.Text(
                value = "[]",
                placeholder = "[z_bragg, y_bragg, x_bragg]",
                description = 'Bragg peak position',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'}),
            # fix the Bragg peak position [z_bragg, y_bragg, x_bragg] considering the full detector
            # It is useful if hotpixels or intense aliens. Leave it [] otherwise.

            fix_size = widgets.Text(
                value = "[]",
                placeholder = "[zstart, zstop, ystart, ystop, xstart, xstop]",
                description = 'Fix array size',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='40%'),
                style = {'description_width': 'initial'}),  
            # crop the array to predefined size considering the full detector,
            # leave it to [] otherwise [zstart, zstop, ystart, ystop, xstart, xstop]. ROI will be defaulted to []

            center_fft = widgets.Dropdown(
                options = ['crop_sym_ZYX','crop_asym_ZYX','pad_asym_Z_crop_sym_YX', 'pad_sym_Z_crop_asym_YX','pad_sym_Z', 'pad_asym_Z', 'pad_sym_ZYX','pad_asym_ZYX', 'skip'],
                value = "crop_asym_ZYX",
                description = 'Center FFT',
                disabled = False,
                style = {'description_width': 'initial'}),

            pad_size = widgets.Text(
                value = "[]",
                placeholder = "[256, 512, 512]",
                description = 'Array size after padding',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='50%'),
                style = {'description_width': 'initial'}), 
            # size after padding, e.g. [256, 512, 512]. Use this to pad the array.
            # used in 'pad_sym_Z_crop_sym_YX', 'pad_sym_Z', 'pad_sym_ZYX'


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


            ### Parameters used when reloading processed dat
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
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'True if the reloaded data is already intepolated in an orthonormal frame',
                icon = 'check'),

            preprocessing_binning = widgets.Text(
                value = "(1, 1, 1)",
                placeholder = "(1, 1, 1)",
                description = 'Binning used in data to be reloaded',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='30%'),
                style = {'description_width': 'initial'},
                tooltip = "binning that will be used for phasing (stacking dimension, detector vertical axis, detector horizontal axis)"),
            # binning factors in each dimension of the binned data to be reloaded


            ### Saving options
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


            ### Define beamline related parameters
            beamline = widgets.Dropdown(
                options = ['ID01', 'SIXS_2018', 'SIXS_2019', 'CRISTAL', 'P10', 'NANOMAX', '34ID'],
                value = "SIXS_2019",
                description = 'Beamline',
                disabled = False,
                tooltip = "Name of the beamline, used for data loading and normalization by monitor",
                style = {'description_width': 'initial'}),

            actuators = widgets.Text(
                value = "{}",
                placeholder = "{}",
                description = 'Actuators',
                tooltip = "Optional dictionary that can be used to define the entries corresponding to actuators in data files (useful at CRISTAL where the location of data keeps changing)",
                readout = True,
                style = {'description_width': 'initial'},
                disabled = False),

            is_series = widgets.ToggleButton(
                value = False,
                description = 'Is series (P10)',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'specific to series measurement at P10',
                icon = 'check'),

            custom_scan = widgets.ToggleButton(
                value = False,
                description = 'Custom scan',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'set it to True for a stack of images acquired without scan, e.g. with ct in a macro, or when there is no spec/log file available',
                icon = 'check'),

            custom_images = widgets.IntText(
                value = 3,
                description='Custom images',
                disabled = False,
                style = {'description_width': 'initial'}),
            # np.arange(11353, 11453, 1)  # list of image numbers for the custom_scan

            custom_monitor = widgets.IntText(
                value = 51,
                description='Custom monitor',
                disabled = False,
                style = {'description_width': 'initial'}),
            # np.ones(51),  # monitor values for normalization for the custom_scan

            rocking_angle = widgets.Dropdown(
                options = ['inplane', 'outofplane', 'energy'],
                value = "inplane",
                description = 'Rocking angle',
                disabled = False,
                tooltip = "Name of the beamline, used for data loading and normalization by monitor",
                style = {'description_width': 'initial'}),

            follow_bragg = widgets.ToggleButton(
                value = False,
                description = 'Follow bragg',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = 'Only for energy scans, set to True if the detector was also scanned to follow the Bragg peak',
                icon = 'check'),

            specfile_name = widgets.Text(
                placeholder = "alias_dict_2019.txt",
                value = "",
                description = 'Specfile name',
                disabled = False,
                continuous_update = False,
                tooltip = """For ID01: name of the spec file without, for SIXS_2018: full path of the alias dictionnary, typically root_folder + 'alias_dict_2019.txt',
                .fio for P10, not used for CRISTAL and SIXS_2019""",
                style = {'description_width': 'initial'}),


            ### Detector related parameters
            detector = widgets.Dropdown(
                options = ["Eiger2M", "Maxipix", "Eiger4M", "Merlin", "Timepix"],
                value = "Merlin",
                description = 'Detector',
                disabled = False,
                style = {'description_width': 'initial'}),

            x_bragg = widgets.IntText(
                value = 160,
                description = 'X Bragg:',
                disabled = False,
                tooltip = "Horizontal pixel number of the Bragg peak, can be used for the definition of the ROI",
                style = {'description_width': 'initial'}),

            y_bragg = widgets.IntText(
                value = 325,
                description = 'Y Bragg:',
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

            template_imagefile = widgets.Text(
                value = 'Pt_ascan_mu_%05d.nxs',
                description = 'Template imagefile',
                disabled = False,
                tooltip = """Template for ID01: 'data_mpx4_%05d.edf.gz' or 'align_eiger2M_%05d.edf.gz'; Template for SIXS_2018: 'align.spec_ascan_mu_%05d.nxs';
                            Template for SIXS_2019: 'spare_ascan_mu_%05d.nxs';
                            Template for Cristal: 'S%d.nxs';
                            Template for P10: '_master.h5'; 
                            Template for NANOMAX: '%06d.h5'; 
                            Template for 34ID: 'Sample%dC_ES_data_51_256_256.npz'""",
                layout = Layout(width='90%'),
                style = {'description_width': 'initial'}),

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
                disabled = False,
                # tooltip = "",
                style = {'description_width': 'initial'}),

            fill_value_mask = widgets.Dropdown(
                options = [0, 1],
                value = 0,
                description = 'Fill value mask',
                disabled = False,
                tooltip = "It will define how the pixels outside of the data range are processed during the interpolation. Because of the large number of masked pixels, phase retrieval converges better if the pixels are not masked (0 intensity imposed). The data is by default set to 0 outside of the defined range.",
                style = {'description_width': 'initial'}),

            beam_direction = widgets.Text(
                value = "(1, 0, 0)",
                placeholder = "(1, 0, 0)",
                description = 'Beam direction',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "Beam direction in the laboratory frame (downstream, vertical up, outboard), beam along z"),

            sample_offsets = widgets.Text(
                value = "(0, 0)",
                placeholder = "(0, 0, 90, 0)",
                description = 'Sample offsets',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='25%'),
                style = {'description_width': 'initial'},
                tooltip = """Tuple of offsets in degrees of the sample for each sample circle (outer first). 
                            Convention: the sample offsets will be subtracted to the motor values"""),

            sdd = widgets.FloatText(
                value = 1.18,
                description = 'Sample Detector Dist. (m):',
                disabled = False,
                tooltip = "sample to detector distance in m",
                style = {'description_width': 'initial'}),

            energy = widgets.IntText(
                value = 8500,
                description = 'X-ray energy in eV',
                disabled = False,
                style = {'description_width': 'initial'}),

            custom_motors = widgets.Text(
                value = "{}",
                placeholder = "{}",
                description = 'Custom motors',
                disabled = False,
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
            #xrayutilities uses the xyz crystal frame: for incident angle = 0, x is downstream, y outboard, and z vertical up
            align_q = widgets.ToggleButton(
                value = True,
                description = 'Align q',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                tooltip = """used only when interp_method is 'linearization', if True it rotates the crystal to align q along one axis of the array""",
                icon = 'check'),

            ref_axis_q = widgets.Dropdown(
                options = ["x", "y", "z"],
                value = "y",
                description = 'Ref axis q',
                disabled = False,
                tooltip = "q will be aligned along that axis",
                style = {'description_width': 'initial'}),

            outofplane_angle = widgets.FloatText(
                value = 0,
                description = 'Outofplane angle',
                disabled = False,
                style = {'description_width': 'initial'}),

            inplane_angle = widgets.FloatText(
                value = 0,
                description = 'Inplane angle',
                disabled = False,
                style = {'description_width': 'initial'}),

            sample_inplane = widgets.Text(
                value = "(1, 0, 0)",
                placeholder = "(1, 0, 0)",
                description = 'Sample inplane',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "Sample inplane reference direction along the beam at 0 angles"),

            sample_outofplane = widgets.Text(
                value = "(0, 0, 1)",
                placeholder = "(0, 0, 1)",
                description = 'Sample outofplane',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "Surface normal of the sample at 0 angles"),

            offset_inplane = widgets.FloatText(
                value = 0,
                step = 0.01,
                description = 'Offset inplane',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "Outer detector angle offset, not important if you use raw data"),

            cch1 = widgets.IntText(
                value = 271,
                description = 'cch1',
                disabled = False,
                layout = Layout(width='15%'),
                tooltip = "cch1 parameter from xrayutilities 2D detector calibration, vertical",
                style = {'description_width': 'initial'}),

            cch2 = widgets.IntText(
                value = 213,
                description = 'cch2',
                disabled = False,
                layout = Layout(width='15%'),
                tooltip = "cch2 parameter from xrayutilities 2D detector calibration, horizontal",
                style = {'description_width': 'initial'}),

            detrot = widgets.FloatText(
                value = 0,
                step = 0.01,
                description = 'Detector rotation',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='20%'),
                style = {'description_width': 'initial'},
                tooltip = "Detrot parameter from xrayutilities 2D detector calibration"),

            tiltazimuth = widgets.FloatText(
                value = 360,
                step = 0.01,
                description = 'Tilt azimuth',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='15%'),
                style = {'description_width': 'initial'},
                tooltip = "tiltazimuth parameter from xrayutilities 2D detector calibration"),

            tilt = widgets.FloatText(
                value = 0,
                step = 0.01,
                description = 'Tilt',
                disabled = False,
                continuous_update = False,
                layout = Layout(width='15%'),
                style = {'description_width': 'initial'},
                tooltip = "tilt parameter from xrayutilities 2D detector calibration"),


            run = widgets.ToggleButton(
                value = False,
                description = 'run BCDI',
                disabled = False,
                button_style = '', # 'success', 'info', 'warning', 'danger' or ''
                icon = 'check')
        )

        # Create the final window
        self.tab_scan = widgets.VBox([
            widgets.HBox(self._list_widgets.children[:2]),
            self._list_widgets.children[2],
            self._list_widgets.children[3],
            self._list_widgets.children[4],
            self._list_widgets.children[5],
            widgets.HBox([self._list_widgets.children[6], self._list_widgets.children[7]])
            ])

        self.tab_masking = widgets.HBox(self._list_widgets.children[8:10])

        self.tab_reduction = widgets.VBox([
            widgets.HBox(self._list_widgets.children[10:13]),
            widgets.HBox(self._list_widgets.children[13:15]),
            self._list_widgets.children[15],
            widgets.HBox(self._list_widgets.children[16:19])
            ])

        self.tab_save_load = widgets.VBox([
            widgets.HBox(self._list_widgets.children[19:22]), 
            widgets.HBox(self._list_widgets.children[22:27])
            ])

        self.tab_beamline = widgets.VBox([
            widgets.HBox(self._list_widgets.children[27:30]),
            widgets.HBox(self._list_widgets.children[30:33]),
            widgets.HBox(self._list_widgets.children[33:36]),
            ])
        
        self.tab_detector = widgets.VBox([
            widgets.HBox(self._list_widgets.children[36:39]),
            widgets.HBox(self._list_widgets.children[39:41]),
            self._list_widgets.children[41],
            self._list_widgets.children[42],
            self._list_widgets.children[43],
            self._list_widgets.children[44],
            widgets.HBox(self._list_widgets.children[45:47]),
            ])

        self.tab_ortho = widgets.VBox([
            widgets.HBox(self._list_widgets.children[47:51]),
            widgets.HBox(self._list_widgets.children[51:54]),
            self._list_widgets.children[54],
            ])
        
        self.tab_xru = widgets.VBox([
            widgets.HBox(self._list_widgets.children[55:59]),
            widgets.HBox(self._list_widgets.children[59:62]),
            widgets.HBox(self._list_widgets.children[62:67]),
            ])

        self.tab_run = widgets.VBox([widgets.HBox([self._list_widgets.children[-2]]), widgets.HBox([self._list_widgets.children[-1]])])

        self.window = widgets.Tab(children=[self.tab_scan, self.tab_masking, self.tab_reduction, self.tab_save_load, self.tab_beamline, self.tab_detector, self.tab_ortho, self.tab_xru, self.tab_run])
        self.window.set_title(0, 'Scan detail')
        self.window.set_title(1, "Masking")
        self.window.set_title(2, "Data reduction")
        self.window.set_title(3, "Load/Save")
        self.window.set_title(4, 'Beamline')
        self.window.set_title(5, 'Detector')
        self.window.set_title(6, 'Orthogonalization')
        self.window.set_title(7, 'X-ray utilities')
        self.window.set_title(8, 'Preprocess')

        display(self.window)

    def initialize_parameters(self,
        scans, sample_name, root_folder, save_dir, data_dirname, user_comment, debug, binning,
        flag_interact, background_plot,
        centering, fix_bragg, fix_size, center_fft, pad_size,
        normalize_flux, 
        mask_zero_event, flag_medianfilter, medfilt_order,
        reload_previous, reload_orthogonal, preprocessing_binning,
        save_rawdata, save_to_npz, save_to_mat, save_to_vti, save_asint,
        beamline, actuators, is_series, custom_scan, custom_images, custom_monitor, rocking_angle, follow_bragg, specfile_name,
        detector, x_bragg, y_bragg, photon_threshold, photon_filter, background_file, hotpixels_file, flatfield_file, template_imagefile, nb_pixel_x, nb_pixel_y,
        use_rawdata, interp_method, fill_value_mask, beam_direction, sample_offsets, sdd, energy, custom_motors,
        align_q, ref_axis_q, outofplane_angle, inplane_angle, 
        sample_inplane, sample_outofplane, offset_inplane, cch1, cch2, detrot, tiltazimuth, tilt,
        run):

        if run:

            # Save parameter values as attributes
            self.scans = scans
            self.sample_name = sample_name
            self.root_folder = root_folder
            self.save_dir = save_dir
            self.data_dirname = data_dirname
            self.user_comment = user_comment
            self.debug = debug
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
            self.template_imagefile = template_imagefile
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
            self.detrot = detrot
            self.tiltazimuth = tiltazimuth
            self.tilt = tilt

            for w in self._list_widgets.children[:-2]:
                w.disabled = True

            # Extract dict, list and tuple from strings
            print("Extracting parameters from strings ...")

            self.string_parameters = [
                "binning", "fix_bragg", "fix_size", "pad_size", "preprocessing_binning", "actuators", "beam_direction",
                "sample_offsets", "custom_motors", "sample_inplane", "sample_outofplane"]

            try:
                for p in self.string_parameters:
                    setattr(self, str(p), literal_eval(getattr(self, p)))
                    print(f"{p}:", getattr(self, p))
            except ValueError:
                print(f"Wrong syntax for {p}")

            # Empty parameters are set to None (bcdi syntax)
            if self.data_dirname == "":
                self.data_dirname = None

            if self.actuators == {}:
                self.actuators = None

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

            if self.custom_motors == {}:
                self.custom_motors = None

            self.roi_detector = [self.y_bragg - 160, self.y_bragg + 160, self.x_bragg - 160, self.x_bragg + 160]
            self.roi_detector = []
            # [Vstart, Vstop, Hstart, Hstop]
            # leave it as [] to use the full detector. Use with center_fft='skip' if you want this exact size.

            self.linearity_func = None

            # On lance BCDI
            print("BCDI logs : \n \n")
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

        if not run:
            clear_output(True)
            for w in self._list_widgets.children[:-2]:
                w.disabled = False

