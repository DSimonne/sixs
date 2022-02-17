import numpy as np
import tables as tb
import pandas as pd
import glob
import os
import inspect
import yaml

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import interact, Button, Layout, interactive, fixed

from scipy import interpolate

import sixs
from sixs_nxsread import ReadNxs4 as rn4

import xrayutilities as xu

from lmfit import minimize, Parameters, Parameter
from lmfit.models import *


class Reflectivity:
    """Contains methods to load reflectometry data collected at SixS"""

    def __init__(
        self,
        folder,
        scan_indices,
        data_format,
        var="l",
        configuration_file=False,
    ):
        """
        :param folder: path to data folder
        :param scan_indices: indices of reflectivity scans, list
        :param data_format: hdf5 (after binoculars) or nxs (raw data)
        :param var: "l", variable used as x axis to plot data
        :param configuration_file: False, .yml file that stores metadata
         specific to the reaction, if False, default to
         self.path_package + "experiments/ammonia.yml"
        """

        self.path_package = inspect.getfile(sixs).split("__")[0]

        # Load configuration file
        print("###########################################################")
        try:
            if os.path.isfile(configuration_file):
                self.configuration_file = configuration_file
            else:
                self.configuration_file = self.path_package + "experiments/ammonia.yml"
                print("Could not find configuration file.")
                print("Defaulted to ammonia configuration.")

        except TypeError:
            self.configuration_file = self.path_package + "experiments/ammonia.yml"
            print("Defaulted to ammonia configuration.")

        finally:
            with open(self.configuration_file) as filepath:
                yaml_parsed_file = yaml.load(
                    filepath,
                    Loader=yaml.FullLoader
                )

                for key in yaml_parsed_file:
                    setattr(self, key, yaml_parsed_file[key])
                print("Loaded configuration file.")
                print("###########################################################\n")

        # Init class arguments
        self.folder = folder

        if not isinstance(scan_indices, list):
            self.scan_indices = list(scan_indices)

        self.scan_indices = [str(s) for s in scan_indices]

        self.data_format = data_format
        self.var = var

        var_dict = {
            "hkl": "hkl",
            "h": "hkl",
            "k": "hkl",
            "l": "hkl",
            "qparqper": "qparqper",
            "qpar": "qparqper",
            "qper": "qparqper",
            "qxqyqz": "qxqyqz",
            "qx": "qxqyqz",
            "qy": "qxqyqz",
            "qz": "qxqyqz",
            "ang": "ang",
            "mu": "ang",
            "gamma": "ang",
            "delta": "ang",
            "omega": "ang",
            "beta": "ang",
            "basepitch": "ang",
        }

        try:
            self.data_type = var_dict[var]
        except:
            raise KeyError("Choose a variable according to", var_dict)

        # Find files in folder depending on data format
        files = [f.split("/")[-1]
                 for f in sorted(glob.glob(f"{folder}/*.{data_format}"))]
        print(f"Detected files in {folder}:")
        for f in files:
            print("\t", f)

        # Get files of interest based on scan_indices
        if self.data_type == "hkl" and self.data_format == "hdf5":
            self.scan_list = [f for f in files if any(
                ["hkl_" + n + ".hdf5" in f for n in self.scan_indices])]

        elif self.data_type == "qxqyqz" and self.data_format == "hdf5":
            self.scan_list = [f for f in files if any(
                ["qxqyqz_" + n + ".hdf5" in f for n in self.scan_indices])]

        elif self.data_type == "qparqper" and self.data_format == "hdf5":
            self.scan_list = [f for f in files if any(
                ["qparqper_" + n + ".hdf5" in f for n in self.scan_indices])]

        elif self.data_type == "ang" and self.data_format == "hdf5":
            self.scan_list = [f for f in files if any(
                ["ang_" + n + ".hdf5" in f for n in self.scan_indices])]

        # Data type not important for nxs files
        elif self.data_format == "nxs":
            self.scan_list = [f for f in files if any(
                [n + ".nxs" in f for n in self.scan_indices])]

        print("\n###########################################################")
        print("Working on the following files:")
        for f in self.scan_list:
            print("\t", f)
        print("###########################################################\n")

    def prep_nxs_data(
        self,
        roi,
        theta="mu",
        two_theta="gamma",
        detector="xpad140",
    ):
        """
        Change the integration roi to see if it has an impact on the data,
        then the data is intergrated on this roi.
        Many parameters are taken from the nxs file such as q, qpar, qper, mu,
        gamma, delta, omega, h, k and l.
        h, k and l depend on the orientation matrix used at that time.

        Each one of these parameters is defined FOR THE CENTRAL PIXEL of the
        detector !

        The reflectivity may not have the same length, and/ or amount of points

        We take the wavelength of the first dataset in the series

        :param roi: int or container
         if int, use this roi (e.g. if 3, roi3)
         if container of length 4, define roi as [roi[0], roi[1], roi[2], roi[3]]
        :param theta: angle used as theta in theta / two_theta geometry
        :param two_theta: angle used as two_theta in theta / two_theta geometry
        :param detector: detector used, can be xpad140, xpad70, ...
        """

        self.intensities = []
        self.theta = theta
        self.two_theta = two_theta
        self.detector = detector

        # Compute reflections, we use the wavelength of first dataset in list
        dataset = rn4.DataSet(
            filename=self.scan_list[0],
            directory=self.folder
        )
        self.wavel = dataset.waveL
        self.y_range = getattr(dataset, detector).shape[1]
        self.x_range = getattr(dataset, detector).shape[2]

        # self.compute_miller(
        #     wl=self.wavel,
        #     tt_cutoff=getattr(data, two_theta)[-1]
        # )

        # Possible x axes, key is the name here, value is the name in rn4
        self.x_axes = {
            "h": "h",
            "k": "k",
            "l": "l",
            "q": "q",
            "qpar": "qpar",
            "qper": "qper",
            "gamma": "gamma",
            "mu": "mu",
            "delta": "delta",
            "omega": "omega",
            "beta": "beta",
            "basepitch": "basepitch",
        }

        # Iterate on all scans
        for j, f in enumerate(self.scan_list):
            print("\n###########################################################")
            print("Opening " + f)

            # Andrea's script it corrects for attenuation
            data = rn4.DataSet(filename=f, directory=self.folder)

            if isinstance(roi, int):
                # calculate the roi corrected by attcoef, mask, filters,
                # acquisition_time
                data.calcROI_new2()

                # Do not take bad values due to attenuator change
                data_array = getattr(data, f"roi{roi}_{self.detector}_new")
                # self.intensities.append(data_array[self.common_mask])
                mask = np.log(data_array) > 0
                self.intensities.append(data_array[mask])

            else:
                # Create a roi, name it roi5
                data.calcROI(
                    stack=getattr(data, self.detector),
                    roiextent=self.roi,
                    maskname=f"_mask_{self.detector}",
                    acqTime=data._integration_time,
                    ROIname=f"roi5_{self.detector}_new",
                    coef='default',
                    filt='default'
                )

                data_array = getattr(data, f"roi5_{self.detector}_new")
                # self.intensities.append(data_array[self.common_mask])
                mask = np.log(data_array) > 0
                self.intensities.append(data_array[mask])

            # Get metadata if defined
            for key, value in self.x_axes.items():
                try:  # Get current list
                    p = getattr(self, key)
                except AttributeError:  # Create one
                    p = []

                # Append scan values for this x axis
                try:
                    # p.append(getattr(data, value)[self.common_mask])
                    p.append(getattr(data, value)[mask])
                    setattr(self, key, p)
                    print("Saved values for", value)
                except AttributeError:
                    pass

            print("###########################################################")

    def prep_binoc_data(
        self,
        wavel,
        tt_cutoff=20,
        interpol_step=0.03,
        CTR_range_h=None,
        CTR_range_k=None,
        qpar_range=None,
        qx_range=None,
        qy_range=None,
        delta_range=None,
        omega_range=None
    ):
        """
        Here we do not integrate over a roi of the detector but rather on a
        region of the reciprocal space, it is the same in the end but more clear
        In the future, implement integration over different axes than l
        for hkl for example
        """

        # Compute reflections for afterwards
        self.compute_miller(wl=wavel, tt_cutoff=tt_cutoff)
        self.wavel = wavel

        print("Finding smallest common range in L, careful, depends on the input of the initial map.")

        for i, fname in enumerate(self.scan_list):

            with tb.open_file(self.folder + fname, "r") as f:
                if self.data_type == "hkl":
                    x_axis = f.root.binoculars.axes.L[:]

                elif self.data_type == "qxqyqz":
                    x_axis = f.root.binoculars.axes.Qz[:]

                elif self.data_type == "qparqper":
                    x_axis = f.root.binoculars.axes.Qper[:]

                elif self.data_type == "ang":
                    x_axis = f.root.binoculars.axes.gamma[:]

                if i == 0:
                    x_axis_min = x_axis[1]
                    x_axis_max = x_axis[2]
                else:
                    x_axis_min = min(x_axis_min, x_axis[1])
                    x_axis_max = max(x_axis_max, x_axis[2])

        self.interpol_step = interpol_step

        self.binoc_x_axis = np.arange(
            x_axis_min, x_axis_max, self.interpol_step)

        # Store name of each file
        self.names = []
        self.intensities = []

        if self.data_type == "hkl":
            # For each file
            for i, fname in enumerate(self.scan_list):

                with tb.open_file(self.folder + fname, "r") as f:

                    ct = f.root.binoculars.counts[:]
                    cont = f.root.binoculars.contributions[:]

                    raw_data = np.divide(ct, cont, where=cont != 0)
                    print("Raw data shape :", raw_data.shape)

                    h = f.root.binoculars.axes.H[:]
                    k = f.root.binoculars.axes.K[:]
                    l = f.root.binoculars.axes.L[:]

                    scan_h_axe = np.linspace(
                        h[1], h[2], 1 + int(h[5] - h[4]))  # xaxe
                    scan_k_axe = np.linspace(
                        k[1], k[2], 1 + int(k[5] - k[4]))  # yaxe
                    scan_l_axe = np.linspace(
                        l[1], l[2], 1 + int(l[5] - l[4]))  # zaxe
                    print("Range in h:", h[1], h[2])
                    print("Range in k:", k[1], k[2])
                    print("Range in l:", l[1], l[2])

                    # CTR intensity, define roi indices
                    st_h_roi = ut3.find_nearest(scan_h_axe, CTR_range_h[0])[0]
                    end_h_roi = ut3.find_nearest(scan_h_axe, CTR_range_h[1])[0]
                    print("Indexes in h: ", st_h_roi, end_h_roi+1)

                    st_k_roi = ut3.find_nearest(scan_k_axe, CTR_range_k[0])[0]
                    end_k_roi = ut3.find_nearest(scan_k_axe, CTR_range_k[1])[0]
                    print("Indexes in k: ", st_k_roi, end_k_roi+1)

                    intensity = raw_data[st_h_roi:end_h_roi +
                                         1, st_k_roi:end_k_roi+1, :]

                    # Interpolate over common l axis
                    tck = interpolate.splrep(
                        scan_l_axe, intensity.sum(axis=(0, 1)), s=0)

                    self.intensities.append(
                        interpolate.splev(self.binoc_x_axis, tck))

        elif self.data_type == "qxqyqz":
            for i, fname in enumerate(self.scan_list):

                with tb.open_file(self.folder + fname, "r") as f:

                    ct = f.root.binoculars.counts[:]
                    cont = f.root.binoculars.contributions[:]

                    raw_data = np.divide(ct, cont, where=cont != 0)
                    print("Raw data shape :", raw_data.shape)

                    # swap axes for hqyl indices to follow miller convention (h,qy,l)
                    # self.hqyl_data = np.swapaxes(self.raw_data, 0, 2)

                    qx = f.root.binoculars.axes.Qx[:]
                    qy = f.root.binoculars.axes.Qy[:]
                    qz = f.root.binoculars.axes.Qz[:]

                    scan_qx_axe = np.linspace(
                        qx[1], qx[2], 1 + int(qx[5] - qx[4]))  # xaxe
                    scan_qy_axe = np.linspace(
                        qy[1], qy[2], 1 + int(qy[5] - qy[4]))  # yaxe
                    scan_qz_axe = np.linspace(
                        qz[1], qz[2], 1 + int(qz[5] - qz[4]))  # zaxe
                    print("Range in qx:", qx[1], qx[2])
                    print("Range in qy:", qy[1], qy[2])
                    print("Range in qz:", qz[1], qz[2])

                    # CTR intensity, define roi indices
                    st_qx_roi = ut3.find_nearest(scan_qx_axe, qx_range[0])[0]
                    end_qx_roi = ut3.find_nearest(scan_qx_axe, qx_range[1])[0]
                    print("Indexes in qx: ", st_qx_roi, end_qx_roi+1)

                    st_qy_roi = ut3.find_nearest(scan_qy_axe, qy_range[0])[0]
                    end_qy_roi = ut3.find_nearest(scan_qy_axe, qy_range[1])[0]
                    print("Indexes in qy: ", st_qy_roi, end_qy_roi+1)

                    intensity = raw_data[st_qx_roi:end_qx_roi +
                                         1, st_qy_roi:end_qy_roi+1, :]

                    # Interpolate over common l axis
                    tck = interpolate.splrep(
                        scan_qz_axe, intensity.sum(axis=(0, 1)), s=0)

                    self.intensities.append(
                        interpolate.splev(self.binoc_x_axis, tck))

        elif self.data_type == "qparqper":
            for i, fname in enumerate(self.scan_list):

                with tb.open_file(self.folder + fname, "r") as f:

                    ct = f.root.binoculars.counts[:]
                    cont = f.root.binoculars.contributions[:]

                    raw_data = np.divide(ct, cont, where=cont != 0)
                    print("Raw data shape :", raw_data.shape)

                    qpar = f.root.binoculars.axes.Qpar[:]
                    qper = f.root.binoculars.axes.Qper[:]

                    scan_qpar_axe = np.linspace(
                        qpar[1],  qpar[2], 1 + int(qpar[5] - qpar[4]))  # yaxe
                    scan_qper_axe = np.linspace(
                        qper[1], qper[2], 1 + int(qper[5] - qper[4]))  # zaxe
                    print("Range in qpar:", qpar[1], qpar[2])
                    print("Range in qper:", qper[1], qper[2])

                    # CTR intensity, define roi indices
                    st_qpar = ut3.find_nearest(scan_qpar_axe, qpar_range[0])[0]
                    end_qpar = ut3.find_nearest(
                        scan_qpar_axe, qpar_range[1])[0]
                    print("Indexes in qpar: ", st_qpar, end_qpar+1)

                    intensity = raw_data[st_qpar:end_qpar+1, :].sum(axis=(0))

                    f = interpolate.interp1d(scan_qper_axe, intensity)

                    self.intensities.append(f(self.binoc_x_axis))

        elif self.data_type == "ang":
            for i, fname in enumerate(self.scan_list):

                with tb.open_file(self.folder + fname, "r") as f:

                    ct = f.root.binoculars.counts[:]
                    cont = f.root.binoculars.contributions[:]

                    raw_data = np.divide(ct, cont, where=cont != 0)
                    print("Raw data shape :", raw_data.shape)

                    delta = f.root.binoculars.axes.delta[:]
                    gamma = f.root.binoculars.axes.gamma[:]
                    omega = f.root.binoculars.axes.omega[:]

                    scan_delta_axe = np.linspace(
                        delta[1], delta[2], 1 + int(delta[5] - delta[4]))  # xaxe
                    scan_omega_axe = np.linspace(
                        omega[1], omega[2], 1 + int(omega[5] - omega[4]))  # yaxe
                    scan_gamma_axe = np.linspace(
                        gamma[1], gamma[2], 1 + int(gamma[5] - gamma[4]))  # zaxe

                    print("Range in gamma:", gamma[1], gamma[2])
                    print("Range in delta:", delta[1], delta[2])
                    print("Range in omega:", omega[1], omega[2])

                    # CTR intensity, define roi indices
                    st_delta_roi = ut3.find_nearest(
                        scan_delta_axe, delta_range[0])[0]
                    end_delta_roi = ut3.find_nearest(
                        scan_delta_axe, delta_range[1])[0]
                    print("Indexes in delta: ", st_delta_roi, end_delta_roi+1)

                    st_omega_roi = ut3.find_nearest(
                        scan_omega_axe, omega_range[0])[0]
                    end_omega_roi = ut3.find_nearest(
                        scan_omega_axe, omega_range[1])[0]
                    print("Indexes in omega: ", st_omega_roi, end_omega_roi+1)

                    intensity = raw_data[st_delta_roi:end_delta_roi+1,
                                         :, st_omega_roi:end_omega_roi+1].sum(axis=(0, 2))

                    f = interpolate.interp1d(scan_gamma_axe, intensity)

                    self.intensities.append(f(self.binoc_x_axis))

    def compute_miller(
        self,
        wl,
        tt_cutoff=20
    ):
        """L given is not good because not the good surface orientation"""

        self.Pt_PD = xu.simpack.PowderDiffraction(
            xu.materials.Pt, wl=wl, tt_cutoff=tt_cutoff).data

        self.Al2O3_PD = xu.simpack.PowderDiffraction(
            xu.materials.Al2O3, wl=wl, tt_cutoff=tt_cutoff).data

        self.BN = xu.materials.Crystal.fromCIF(
            "/home/david/Documents/PhD_local/PhDScripts/Surface_Diffraction/BN_mp-984_conventional_standard.cif")
        self.BN_PD = xu.simpack.PowderDiffraction(
            self.BN, wl=wl, tt_cutoff=tt_cutoff).data

        del self.BN_PD[(0, 0, 1)]
        del self.BN_PD[(0, 0, 3)]

        self.theta_bragg_pos_BN = [
            (miller_indices, self.BN_PD[miller_indices]["ang"])
            for miller_indices in self.BN_PD
        ]
        self.theta_bragg_pos_Al2O3 = [
            (miller_indices, self.Al2O3_PD[miller_indices]["ang"])
            for miller_indices in self.Al2O3_PD
        ]
        self.theta_bragg_pos_Pt = [
            (miller_indices, self.Pt_PD[miller_indices]["ang"])
            for miller_indices in self.Pt_PD
        ]

        self.q_bragg_pos_BN = [
            (miller_indices, self.BN_PD[miller_indices]["qpos"])
            for miller_indices in self.BN_PD
        ]
        self.q_bragg_pos_Al2O3 = [
            (miller_indices, self.Al2O3_PD[miller_indices]["qpos"])
            for miller_indices in self.Al2O3_PD
        ]
        self.q_bragg_pos_Pt = [
            (miller_indices, self.Pt_PD[miller_indices]["qpos"])
            for miller_indices in self.Pt_PD
        ]

    def plot_refl(
        self,
        x_var,
        title=None,
        filename=None,
        figsize=(18, 9),
        ncol=2,
        color_dict=None,
        labels=False,
        y_zero=0,
        zoom=[None, None, None, None],
        x_tick_step=False,
        fill=False,
        fill_first=0,
        fill_last=-1,
        miller=False,
        critical_angles=False,
        background=False,
        log_intensities=True,
        normalisation_range=False,
    ):
        """
        Plot the reflectivity.

        :param x_var: choose x_axis in the self.x_axes list
        :param title: if string, set to figure title
        :param filename: if string, figure will be saved to this path.
        :param figsize: figure size, default is (18, 9)
        :param ncol: columns in label, default is 2
        :param color_dict: dict used for labels, keys are scan index, values are
         colours for matplotlib.
        :param labels: list of labels to use, defaulted to scan index if False
        :param y_zero: Generate a bolded horizontal line at y_zero to highlight
         background, default is 0
        :param zoom: values used for plot range, default is
         [None, None, None, None], order is left, right, bottom and top.
        :param x_tick_step: tick step used for x axis, default is False
        :param fill: if True, add filling between two plots
        :param fill_first: index of scan to use for filling
        :param fill_last: index of scan to use for filling
        :param miller: list containing nothing or "pt", 'al2o3' or "bn"
        :param critical_angles: if true, plot vertical line at 0.2540° (Pt)
        :param background: path to .npz background file to load and subtract,
         there must a data entry that corresponds to x_axis
        :param log_intensities: if True, y axis is logarithmic
        :param normalisation_range: normalizse by maximum on this range
        """

        # Get x axis
        if self.data_format == "nxs" and x_var in self.x_axes:
            x_axis = getattr(self, x_var)

        elif self.data_format == "nxs" and x_var not in self.x_axes:
            return("Choose x_axis in the following list :", self.x_axes)

        # x_var has no influence if we use binocular file for now
        elif self.data_format == "hdf5":
            x_axis = [self.binoc_x_axis for i in self.scan_list]

        # Load background
        if isinstance(background, str):
            print("Subtracting bck...")

            try:
                bck = np.load(background)[x_axis]
            except:
                print(f"Use background file with {x_axis} entry.")

        # Plot
        plt.figure(figsize=figsize)
        if log_intensities:
            plt.semilogy()
        plt.grid()

        for (i, y), x, scan_index in zip(
            enumerate(self.intensities),
            x_axis,
            self.scan_indices
        ):
            # Remove background
            try:
                # indices of x for which the value is defined on the background x
                idx = (x < max(self.bck[0])) * (x > min(self.bck[0]))

                x = x[idx]
                y = y[idx]

                # create ticks
                f = interpolate.interp1d(self.bck[0], self.bck[1])

                # create new bck
                new_bck = f(x)

                y_plot = np.where(
                    (y-new_bck < self.background_bottom),
                    self.background_bottom, y-new_bck
                )

            except (AttributeError, TypeError):
                y_plot = y

            # Normalize data
            if isinstance(normalisation_range, list) or isinstance(normalisation_range, tuple):
                print("\nScan index:", scan_index)
                start = self.find_nearest(x, normalisation_range[0])[1]
                end = self.find_nearest(x, normalisation_range[1])[1]
                max_ref = max(y_plot[start:end])
                print(f"\tmax(y[{start}: {end}])={max_ref}")
                y_plot = y_plot/max_ref
                print("\tNormalized the data by maximum on normalisation range.\n")

            # Add label
            if isinstance(labels, list):
                label = labels[i]
            elif isinstance(labels, dict):
                try:
                    label = labels[scan_index]
                except KeyError:
                    label = labels[int(scan_index)]
                except:
                    print("Dict not valid for labels, used scan_index")
                    label = scan_index
            else:
                label = scan_index

            # Add colour
            try:
                plt.plot(
                    x,
                    y_plot,
                    color=color_dict[int(scan_index)],
                    label=label,
                    linewidth=2,
                )

            except KeyError:
                # Take int(scan_index) in case keys are not strings in the dict
                try:
                    plt.plot(
                        x,
                        y_plot,
                        color=color_dict[scan_index],
                        label=label,
                        linewidth=2,
                    )
                except Exception as e:
                    raise e
            except TypeError:  # No special colour
                plt.plot(
                    x,
                    y_plot,
                    label=label,
                    linewidth=2,
                )

            # Get y values for filling
            if i == fill_first:
                y_first = y_plot

            elif i == len(self.intensities) + fill_last:
                y_last = y_plot

        # Generate a bolded horizontal line at y_zero to highlight background
        try:
            plt.axhline(y=y_zero, color='black', linewidth=1, alpha=0.5)
        except:
            pass

        # Generate a bolded vertical line at x = 0 to highlight origin
        plt.axvline(x=0, color='black', linewidth=1, alpha=0.5)

        if critical_angles:
            print("\nCritical angles position is angular (theta).\n")
            # Generate a bolded vertical line at x = 0.24 to highlightcritical angle of Pt
            # 0.2538
            plt.axvline(
                x=2*0.2540 if x_axis == "gamma" else 0.2540,
                color='red',
                linewidth=2,
                alpha=1,
                label="$\\alpha_c$ Pt",
                linestyle="--"
            )

            # Generate a bolded vertical line at x = 0.12 to highlightcritical angle of Al2O3
            # plt.axvline(
            #     x=2*0.130261 if x_axis == "gamma" else 0.130261,
            #     color='black',
            #     linewidth=1,
            #     alpha=1,
            #     label="$\\alpha_c$ $Al_2O_3$",
            #     linestyle="--"
            # )

        if miller:
            if self.data_format == "hdf5" and self.data_type == "qxqyqz":
                Pt_pos = self.q_bragg_pos_Pt[:1]
                BN_pos = self.q_bragg_pos_BN
                Al2P3_pos = self.q_bragg_pos_Al2O3
                plot_peaks = True

            elif self.data_format == "hdf5" and self.data_type == "qparqper":
                Pt_pos = self.q_bragg_pos_Pt[:1]
                BN_pos = self.q_bragg_pos_BN
                Al2P3_pos = self.q_bragg_pos_Al2O3
                plot_peaks = True

            elif self.data_format == "hdf5" and self.data_type == "ang":
                Pt_pos = self.theta_bragg_pos_Pt[:1]
                BN_pos = self.theta_bragg_pos_BN
                Al2P3_pos = self.theta_bragg_pos_Al2O3
                plot_peaks = True

            elif self.data_format == "nxs" and x_axis in "qparqper":
                Pt_pos = self.q_bragg_pos_Pt[:1]
                BN_pos = self.q_bragg_pos_BN
                Al2P3_pos = self.q_bragg_pos_Al2O3
                plot_peaks = True
                print("Careful, peak position in q.")

            elif self.data_format == "nxs" and x_axis == "mu":
                Pt_pos = self.theta_bragg_pos_Pt[:1]
                BN_pos = self.theta_bragg_pos_BN
                Al2P3_pos = self.theta_bragg_pos_Al2O3
                plot_peaks = True

            elif self.data_format == "nxs" and x_axis == "gamma":
                print("Shift of 0.5 in gamma")
                Pt_pos = [(a, 2*b + 0.5)
                          for (a, b) in self.theta_bragg_pos_Pt[:1]]
                BN_pos = [(a, 2*b + 0.5)
                          for (a, b) in self.theta_bragg_pos_BN]
                Al2P3_pos = [(a, 2*b + 0.5)
                             for (a, b) in self.theta_bragg_pos_Al2O3]
                plot_peaks = True

            else:
                print("Positions only in angle or q.")
                plot_peaks = False

            if plot_peaks:
                if "pt" in [z.lower() for z in miller]:
                    # Highlight Bragg peaks of Pt

                    for i, (miller_indices, bragg_angle) in enumerate(Pt_pos):
                        if bragg_angle > self.x_min and bragg_angle < self.x_max:
                            plt.axvline(x=bragg_angle,
                                        # label ="Pt",
                                        color=self.BP_colors["Pt"], linewidth=1, alpha=.5)
                            plt.text(x=bragg_angle, y=self.y_text, s=f"{miller_indices}", color=self.BP_colors["Pt"],
                                     weight='bold', rotation=60, backgroundcolor='#f0f0f0', fontsize=self.fontsize)

                if "al2o3" in [z.lower() for z in miller]:
                    # Highlight Bragg peaks of Al2O3

                    for i, (miller_indices, bragg_angle) in enumerate(Al2P3_pos):
                        if bragg_angle > self.x_min and bragg_angle < self.x_max:
                            if i == 0:
                                plt.axvline(x=bragg_angle, color=self.BP_colors["Al2O3"],
                                            linewidth=1, alpha=.5, label=("Al2O3"))
                                plt.text(x=bragg_angle, y=self.y_text, s=f"{miller_indices}", color=self.BP_colors["Al2O3"],
                                         weight='bold', rotation=60, backgroundcolor='#f0f0f0', fontsize=self.fontsize)
                            else:
                                plt.axvline(x=bragg_angle, color=self.BP_colors["Al2O3"],
                                            linewidth=1, alpha=.5)
                                plt.text(x=bragg_angle, y=self.y_text, s=f"{miller_indices}", color=self.BP_colors["Al2O3"],
                                         weight='bold', rotation=60, backgroundcolor='#f0f0f0', fontsize=self.fontsize)

                if "bn" in [z.lower() for z in miller]:
                    # Highlight Bragg peaks of BN

                    for i, (miller_indices, bragg_angle) in enumerate(BN_pos):
                        if bragg_angle > self.x_min and bragg_angle < self.x_max:
                            if i == 0:
                                plt.axvline(x=bragg_angle, color=self.BP_colors["BN"],
                                            linewidth=1, alpha=.5, label=("Boron nitride"))
                                plt.text(x=bragg_angle, y=self.y_text, s=f"{miller_indices}", color=self.BP_colors["BN"],
                                         weight='bold', rotation=60, backgroundcolor='#f0f0f0', fontsize=self.fontsize)
                            else:
                                plt.axvline(x=bragg_angle, color=self.BP_colors["BN"],
                                            linewidth=1, alpha=.5)
                                plt.text(x=bragg_angle, y=self.y_text, s=f"{miller_indices}", color=self.BP_colors["BN"],
                                         weight='bold', rotation=60, backgroundcolor='#f0f0f0', fontsize=self.fontsize)

        # Ticks
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)

        # Range
        plt.xlim(left=zoom[0], right=zoom[1])
        plt.ylim(bottom=zoom[2], top=zoom[3])

        # Filling
        if fill:
            try:
                # Add filling
                plt.fill_between(x_axis, y_first, y_last, alpha=0.1)
            except:
                print("Could not add filling.")

        # Legend and axis labels
        plt.legend(fontsize=self.fontsize, ncol=ncol)

        if self.data_format == "hdf5" and self.var == "hkl":
            plt.xlabel("L", fontsize=self.fontsize)
        elif self.data_format == "hdf5" and self.var == "qparqper":
            plt.xlabel("qper", fontsize=self.fontsize)
        elif self.data_format == "hdf5" and self.var == "qxqyqz":
            plt.xlabel("qz", fontsize=self.fontsize)
        elif self.data_format == "hdf5" and self.var == "qxqyqz":
            plt.xlabel("qz", fontsize=self.fontsize)
        elif self.data_format == "nxs":
            plt.xlabel(x_var, fontsize=self.fontsize)

        plt.ylabel("Intensity (a.u.)", fontsize=self.fontsize)
        if isinstance(title, str):
            plt.title(f"{title}", fontsize=20)

        plt.tight_layout()

        # Save
        if filename != None:
            plt.savefig(f"{filename}", bbox_inches='tight')
            print(f"Saved as {filename}")

        plt.show()

    def fit_bragg(
        self,
        scan_to_fit,
        fit_range=[2.75, 2.95],
        x_axis="q",
        peak_nb=1,
        peak_pos=[2.85],
        peak_amp=[1e5],
        peak_sigma=[0.0008],
        back=[10, 0, 100]
    ):
        """
        Fit Bragg peak that appear in reflectivity scans (00L).
        The peak shape is fixed to a lmfit.LorentzianModel for now.

        :param scan_to_fit: list of scans on which we fit the data
        :param fit_range: range on which we fit the data
        :param x_axis: axis to use for x
        :param peak_nb: amount of peaks in the range to fit
        :param peak_pos: peak positions, initial guess, list of length peak_nb
        :param peak_amp: peak amplitudes, initial guess, list of length peak_nb
        :param peak_sigma: peak sigmas, initial guess, list of length peak_nb
        :param back: initial guess, minimum and maximum values for background,
         constant in this model.
        """

        # Create a dictionnary for the peak to be able to iterate on their names
        peaks = dict()

        # Initialize the parameters and the data over the Bragg peak range
        x_axes = {"h": "cp_h",
                  "k": "cp_k",
                  "l": "cp_l",
                  "gamma": "cp_gamma",
                  "mu": "cp_mu",
                  "omega": "cp_omega",
                  "delta": "cp_delta",
                  "q": "cp_q",
                  "qpar": "cp_qpar",
                  "qper": "cp_qper"}

        # Find good x axis
        if self.data_format == "nxs" and x_axis in [
            "h", "k", "l", "q",
            "qpar", "qper", "gamma",
            "mu", "delta", "omega"
        ]:
            x_total = getattr(self, x_axes[x_axis])

        elif self.data_format == "nxs" and x_axis not in [
            "h", "k", "l",
            "q", "qpar", "qper",
            "qpar", "qper", "gamma",
            "mu", "delta", "omega"
        ]:
            return("Choose a x_axis in the following list :", [
                "h", "k", "l",
                "q", "qpar", "qper",
                "gamma", "mu",
                "delta", "omega"])

        # Iterate over each data set to find the Bragg peak
        for q_range, gamma_range, delta_range, x, y, scan_name in zip(
            self.cp_q,
            self.cp_gamma,
            self.cp_delta,
            x_total,
            self.intensities,
            self.scan_indices
        ):
            if scan_name in scan_to_fit:
                x_zoom = x[(x >= fit_range[0]) & (x <= fit_range[1])]
                y_zoom = y[(x >= fit_range[0]) & (x <= fit_range[1])]

                self.mod = ConstantModel(prefix='Bcgd_')
                self.pars = self.mod.guess(y, x=x)

                for i in range(peak_nb):
                    peaks[f"Peak_{i}"] = LorentzianModel(prefix=f"P{i}_")
                    self.pars.update(peaks[f"Peak_{i}"].make_params())
                    self.mod += peaks[f"Peak_{i}"]

                    self.pars[f"P{i}_center"].value = peak_pos[i]
                    self.pars[f"P{i}_amplitude"].value = peak_amp[i]
                    self.pars[f"P{i}_sigma"].value = peak_sigma[i]

                self.pars["Bcgd_c"].value = back[0]
                self.pars["Bcgd_c"].min = back[1]
                self.pars["Bcgd_c"].max = back[2]

                # display(pars)

                # Current guess
                self.init = self.mod.eval(self.pars, x=x_zoom)

                fig, axs = plt.subplots(1, 2, figsize=(
                    16, 5), sharex=True, sharey=True)
                axs[0].semilogy()

                axs[0].plot(x_zoom, y_zoom, label="Data")
                axs[0].plot(x_zoom, self.init, label='Current guess')
                axs[0].legend()

                # Launch fit
                self.out = self.mod.fit(y_zoom, self.pars, x=x_zoom)
                self.comps = self.out.eval_components(x=x_zoom)
                display(self.out.params)

                axs[1].plot(x_zoom, y_zoom, label="Data")
                axs[1].plot(x_zoom, self.out.best_fit, label='Best fit')
                axs[1].legend()

                plt.suptitle(scan_name)
                plt.show()
                q_hkl = self.out.params["P0_center"].value

                ind = ut3.find_nearest(q_range, q_hkl)[0]

                inplane_angle = gamma_range[ind]
                outofplane_angle = delta_range[ind]

                kin = 2*np.pi/self.wavel * np.asarray((1, 0, 0))

                kout = 2 * np.pi / self.wavel * np.array(  # in lab.frame z downstream, y vertical, x outboard
                    [np.cos(np.pi * inplane_angle / 180) * np.cos(np.pi * outofplane_angle / 180),  # z
                     np.sin(np.pi * outofplane_angle / 180),  # y
                     np.sin(np.pi * inplane_angle / 180) * np.cos(np.pi * outofplane_angle / 180)])  # x

                q = (kout - kin)  # convert from 1/m to 1/angstrom
                qnorm = np.linalg.norm(q)
                dist_plane = 2 * np.pi / qnorm
                self.lat_para[scan_name] = np.sqrt(3)*dist_plane

    def compare_roi(
        self,
        v_min_log=0.01,
        v_max_log=1000,
        central_pixel_x=94,
        central_pixel_y=278,
        figsize=(16, 9),
        data_range=300,
    ):
        """
        Widget function to assess the quality of the roi on the .nxs file.
        Assumes that the same detector is used for all the scans.

        :param v_min_log: minimum value in log scale
        :param v_max_log: maximum value in log scale
        :param central_pixel_x: central pixel (x usually delta) corresponding
         to the angular positions saved in nexus file. There is usually a
         shift of about ~1.7° that allows us to hide the direct beam to work
         with the attenuators.
        :param central_pixel_y: central pixel (y usually gamma) corresponding
         to the angular positions saved in nexus file.
        :param figsize: size of figure
        """

        try:
            self.x_range
            self.y_range

        except:
            return("You need to run Reflectivity.prep_nxs_data() before!")

        # Define detector image plotting function
        def plot2D(a, b, c, d, Refl, index):
            fig, ax = plt.subplots(2, 1, figsize=figsize)

            try:
                data = getattr(Refl, self.detector)
            except AttributeError:
                print("Could not load detector data ...")

            ax[0].imshow(
                data[index, :, :],
                norm=LogNorm(vmin=v_min_log, vmax=v_max_log)
            )
            ax[0].axhline(b, linewidth=1, color="red")
            ax[0].axhline(b + d, linewidth=1, color="red")

            ax[0].axvline(a, linewidth=1, color="red")
            ax[0].axvline(a + c, linewidth=1, color="red", label="ROI")

            ax[0].axvline(central_pixel_x, linestyle="--",
                          linewidth=1, color="black")
            ax[0].axhline(central_pixel_y, linestyle="--",
                          linewidth=1, color="black", label="Central pixel")
            ax[0].legend(fontsize=15)

            ax[0].set_title('Entire detector', fontsize=15)
            ax[0].set_ylabel("Y (Gamma)", fontsize=15)
            ax[0].set_xlabel("X (Delta)", fontsize=15)

            ax[1].set_title('Zoom in ROI', fontsize=15)
            ax[1].set_ylabel("Y (Gamma)", fontsize=15)
            ax[1].set_xlabel("X (Delta)", fontsize=15)

            # ax[0].text(x=1.5,
            #            y=-25,
            #            s="Parameters : gamma = {:.3f}, mu = {:.3f}, hkl = ({:.3f}, {:.3f}, {:.3f}), q = {:3f}, qpar, qper = ({:3f}, {:3f})".format(
            #                Refl.gamma[index], Refl.mu[index], Refl.h[index], Refl.k[index], Refl.l[index], Refl.q[index], Refl.qPar[index], Refl.qPer[index]),
            #            #color = diverging_colors_1["E"],
            #            weight='bold',
            #            rotation=0,
            #            backgroundcolor='#f0f0f0',
            #            fontsize="20")

            ax[1].imshow(
                data[index, b:b+d, a:a+c],
                norm=LogNorm(vmin=v_min_log, vmax=v_max_log)
            )

            plt.tight_layout()

        # Create widget list
        _list_widgets = interactive(
            plot2D,
            a=widgets.IntSlider(
                value=0,
                min=0,
                max=self.x_range,
                step=1,
                description='a:',
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='d'),
            b=widgets.IntSlider(
                value=0,
                min=0,
                max=self.y_range,
                step=1,
                description='b:',
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='d'),
            c=widgets.IntSlider(
                value=self.x_range,
                min=0,
                max=self.x_range,
                step=1,
                description='c:',
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='d'),
            d=widgets.IntSlider(
                value=self.y_range,
                min=0,
                max=self.y_range,
                step=1,
                description='d:',
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='d'),
            index=widgets.IntSlider(
                value=0,
                min=0,
                max=data_range,
                step=1,
                description='Index:',
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='d',
                style={'description_width': 'initial'},
                layout=Layout(width="30%")),
            Refl=widgets.Select(
                options=[rn4.DataSet(filename=f, directory=self.folder)
                         for f in self.scan_list],
                description='Scan:',
                style={'description_width': 'initial'},
                layout=Layout(width="70%")))
        window = widgets.VBox([
            widgets.HBox(_list_widgets.children[0:4]),
            widgets.HBox(_list_widgets.children[4:-1]),
            _list_widgets.children[-1]
        ])

        return window

    @ staticmethod
    def find_nearest(array, value):
        X = np.abs(array-value)
        idx = np.where(X == X.min())
        if len(idx) == 1:
            try:
                idx = idx[0][0]
                return array[idx], idx
            except IndexError:
                # print("Value is not in array")
                if all(array < value):
                    return array[-1], -1
                elif all(array > value):
                    return array[0], 0
