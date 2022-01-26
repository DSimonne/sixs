import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual
from ipywidgets import interact, Button, Layout, interactive, fixed

from scipy import interpolate

import tables as tb

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import numpy as np
import pandas as pd
import glob

from phdutils.binoculars import binUtil3 as bin3
from phdutils.sixs import ReadNxs4 as rn4
from phdutils.sixs import utilities3 as ut3

import xrayutilities as xu

import lmfit
from lmfit import minimize, Parameters, Parameter
from lmfit.models import LinearModel, ConstantModel, QuadraticModel, PolynomialModel, StepModel
from lmfit.models import GaussianModel, LorentzianModel, SplitLorentzianModel, VoigtModel, PseudoVoigtModel
from lmfit.models import MoffatModel, Pearson7Model, StudentsTModel, BreitWignerModel, LognormalModel, ExponentialGaussianModel, SkewedGaussianModel, SkewedVoigtModel, DonaichModel

plt.style.use('fivethirtyeight')
plt.rc('text', usetex=True)

np.seterr(divide='ignore')


class reflecto(object):
    """docstring for reflecto"""

    def __init__(self, folder, scan_indices, data_format, var="l"):
        # class arguments
        self.folder = folder
        self.scan_indices = [str(s) for s in scan_indices]
        self.data_format = data_format  # nxs or "hdf5"

        self.lat_para = dict()

        # implemented for now in case in the future we want to plot on other axes
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
            "omega": "ang"}

        try:
            self.data_type = var_dict[var]
        except:
            raise KeyError("Choose a variable according to", var_dict)

        all_files = [f.split("/")[-1]
                     for f in sorted(glob.glob(f"{folder}/*.hdf5"))]

        if self.data_type == "hkl" and self.data_format == "hdf5":
            self.scan_list = [f for f in all_files if any(
                ["hkl_" + n + ".hdf5" in f for n in self.scan_indices])]

        elif self.data_type == "qxqyqz" and self.data_format == "hdf5":
            self.scan_list = [f for f in all_files if any(
                ["qxqyqz_" + n + ".hdf5" in f for n in self.scan_indices])]

        elif self.data_type == "qparqper" and self.data_format == "hdf5":
            self.scan_list = [f for f in all_files if any(
                ["qparqper_" + n + ".hdf5" in f for n in self.scan_indices])]

        elif self.data_type == "ang" and self.data_format == "hdf5":
            self.scan_list = [f for f in all_files if any(
                ["ang_" + n + ".hdf5" in f for n in self.scan_indices])]

        # data type not important for nxs files
        elif self.data_format == "nxs":
            all_files = [f.split("/")[-1]
                         for f in sorted(glob.glob(f"{folder}/*.nxs"))]
            self.scan_list = [f for f in all_files if any(
                [n + ".nxs" in f for n in self.scan_indices])]

        # plotting variables
        self.linewidth = 2
        self.linewidth_hline = 1.6
        self.linewidth_vline = 1.6
        self.alpha_hline = 0.7
        self.alpha_vline = 0.7
        self.color_hline = "black"
        self.color_vline = "black"

        self.filling_alpha = 0.2

        self.fontsize = 25

        self.x_tick_step = 0.5
        self.y_tick_step = 10
        self.y_og = None
        self.tick_fontsize = 30
        self.y_text = 10e6

        self.title_fontsize = 30

        self.background_bottom = 0

    def prep_nxs_data(self, roi=False):
        """The goal of this function is to change the integration roi to see if it has an impact on the data, then the data is intergrated on this roi.
        Many parameters are taken from the nxs file such as q, qpar, qper, mu, gamma, delta, omega, h, k and l
        h, k and l depend on the orientation matrix used at that time.

        Each one of these parameters is defined FOR THE CENTRAL PIXEL of the detector !

        The reflecto may not have the same length, and/ or amount of points

        We take the wavelength of the first dataset in the series"""

        self.roi = roi
        self.intensities = []

        # cp stands for central pixel
        self.cp_h, self.cp_k, self.cp_l = [], [], []
        self.cp_gamma, self.cp_delta, self.cp_mu, self.cp_omega = [], [], [], []
        self.cp_q, self.cp_qper, self.cp_qpar = [], [], []

        # First loop to find shortest index list

        for f in self.scan_list:
            data = rn4.DataSet(self.folder + f)

            if not self.roi:
                data.calcROI_new2()

                # Do not take bad values due to attenuator change, long but necessary
                ind1 = np.where(np.log(data.roi3_xpad70_new) > 0)
                self.roi = tuple(data._roi_limits_xpad70[3])

            else:
                a, b, c, d = self.roi
                data.calcROI(data.xpad70, [a, b, c, d], "_mask_xpad70",
                             data._integration_time, "roi5_xpad70_new", coef='default', filt='default')

                # Do not take bad values due to attenuator change
                if "post" in f:
                    ind1_pm = np.where(np.log(5*data.roi5_xpad70_new) > 0)

                else:
                    ind1 = np.where(np.log(data.roi5_xpad70_new) > 0)

            if "post" in f:
                try:
                    ind_pm = np.intersect1d(ind_pm, ind1_pm)
                except:
                    ind_pm = ind1_pm
            else:
                try:
                    ind = np.intersect1d(ind, ind1)
                except:
                    ind = ind1

        print("Found common indices between all scans")
        try:
            self.ind_pm = ind_pm
        except:
            pass
        self.ind = ind

        dataset = rn4.DataSet(self.folder + self.scan_list[0])

        # Compute reflections for afterwards, we use the wavelength of last dataset in list
        self.compute_miller(wl=dataset.waveL, tt_cutoff=dataset.gamma[-1])
        self.wavel = dataset.waveL

        for f in self.scan_list:
            print("Opening " + f)

            # Keep andrea's script because it corrects for attenuation
            data = rn4.DataSet(self.folder + f)

            # Take good indices depending on scans
            if "post" in f:
                ind = self.ind_pm
            else:
                ind = self.ind

            try:
                self.roi_limits_xpad70 = data._roi_limits_xpad70

                # used for compare roi widget
                self.gamma_width = data.xpad_s70_image.shape[1]
                self.delta_width = data.xpad_s70_image.shape[2]
            except:
                # post mortem scans, there was a bog
                print("No roi for this dataset")

            if not self.roi:
                # We use the roi3, defined as the good one during the experiment, this can change for other experiments
                data.calcROI_new2()
                self.intensities.append(data.roi3_xpad70_new[ind])

            else:
                # We use our own roi
                a, b, c, d = self.roi
                data.calcROI(data.xpad70, [a, b, c, d], "_mask_xpad70",
                             data._integration_time, "roi5_xpad70_new", coef='default', filt='default')

                if "post" in f:
                    print("Post mortem data. Multiplying intensity by 5.")
                    self.intensities.append(5*data.roi5_xpad70_new[ind])

                else:
                    self.intensities.append(data.roi5_xpad70_new[ind])

            self.cp_h.append(data.h[ind])
            self.cp_k.append(data.k[ind])
            self.cp_l.append(data.l[ind])

            self.cp_gamma.append(data.gamma[ind]-0.5)
            self.cp_mu.append(data.mu[ind])
            self.cp_omega.append(data.omega[ind])
            self.cp_delta.append(data.delta[ind])

            self.cp_q.append(data.q[ind])
            self.cp_qpar.append(data.qPar[ind])
            self.cp_qper.append(data.qPer[ind])

    def prep_binoc_data(self, wavel, tt_cutoff=20, interpol_step=0.03, CTR_range_h=None, CTR_range_k=None, qpar_range=None, qx_range=None, qy_range=None, delta_range=None, omega_range=None):
        """Here we do not integrate over a roi of the detector but rather on a region of the reciprocal space,
        it is the same in the end but more clear
        In the future, implement integration over different axes than l for hkl for example"""

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

    def compute_miller(self, wl, tt_cutoff=20):
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
            (miller_indices, self.BN_PD[miller_indices]["ang"]) for miller_indices in self.BN_PD]
        self.theta_bragg_pos_Al2O3 = [
            (miller_indices, self.Al2O3_PD[miller_indices]["ang"]) for miller_indices in self.Al2O3_PD]
        self.theta_bragg_pos_Pt = [
            (miller_indices, self.Pt_PD[miller_indices]["ang"]) for miller_indices in self.Pt_PD]

        self.q_bragg_pos_BN = [
            (miller_indices, self.BN_PD[miller_indices]["qpos"]) for miller_indices in self.BN_PD]
        self.q_bragg_pos_Al2O3 = [
            (miller_indices, self.Al2O3_PD[miller_indices]["qpos"]) for miller_indices in self.Al2O3_PD]
        self.q_bragg_pos_Pt = [
            (miller_indices, self.Pt_PD[miller_indices]["qpos"]) for miller_indices in self.Pt_PD]

    def plot_refl(self, title, save_as, figsize=(18, 9), labels=False, ncol=2, scan_gas_dict=False, x_axis="l", y_max=False, y_min=False, x_min=False, x_max=False, fill=False, fill_first=0, fill_last=-1, miller=False, critical_angles=False, background=True, low_range=False, high_range=False):
        """miller must be a list containing nothing or "pt", 'al2o3' or "bn"
        x_axis will be set to var in the future
        """
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
        if self.data_format == "nxs" and x_axis in ["h", "k", "l", "q", "qpar", "qper", "gamma", "mu", "delta", "omega"]:
            self.x_axis = getattr(self, x_axes[x_axis])

        elif self.data_format == "nxs" and x_axis not in ["h", "k", "l", "q", "qpar", "qper", "gamma", "mu", "delta", "omega"]:
            return("Choose a x_axis in the following list :", ["h", "k", "l", "q", "qpar", "qper", "gamma", "mu", "delta", "omega"])

        # x_axis has no influence if we use binocular file for now
        elif self.data_format == "hdf5":
            self.x_axis = [self.binoc_x_axis for i in self.scan_list]

        # Take x axe limits for future use
        if not x_min:
            self.x_min = np.array(self.x_axis).min()
        else:
            self.x_min = x_min

        if not x_max:
            self.x_max = np.array(self.x_axis).max()
        else:
            self.x_max = x_max

        if not y_min:
            self.y_min = np.array(self.intensities).min()
        else:
            self.y_min = y_min

        if not y_max:
            self.y_max = np.array(self.intensities).max()
        else:
            self.y_max = y_max

        # Add dome background
        if isinstance(background, str):
            print("Subtracting bck...")

            bck_dict = {"h": "bck_h",
                        "k": "bck_k",
                        "l": "bck_l",
                        "hkl": "bck_l",
                        "gamma": "bck_gamma",
                        "mu": "bck_mu",
                        "omega": "bck_omega",
                        "delta": "bck_delta",
                        "ang": "bck_mu",
                        "q": "bck_q",
                        "qpar": "bck_qpar",
                        "qper": "bck_qper",
                        "qparqper": "bck_qper"}

            try:
                bck = np.load(background)
            except Exception as e:
                raise e

            if self.data_format == "hdf5":
                try:
                    self.bck = bck[bck_dict[self.data_type]]
                except:
                    print("Use x axis equal to l, mu, gamma, q or qper.")

            elif self.data_format == "nxs":
                try:
                    self.bck = bck[bck_dict[x_axis]]
                except:
                    print("Use x axis equal to l, mu, gamma, q or qper.")

        else:
            print("No background.")

        # Plot
        if not low_range and not high_range:
            plt.figure(figsize=figsize, dpi=150)
            plt.semilogy()

            for (i, y), x, scan_index in zip(enumerate(self.intensities), self.x_axis, self.scan_indices):
                if isinstance(background, str):
                    # indices of x for which the value is defined on the background x
                    idx = (x < max(self.bck[0])) * (x > min(self.bck[0]))

                    x = x[idx]
                    y = y[idx]

                    # create ticks
                    f = interpolate.interp1d(self.bck[0], self.bck[1])

                    # create new bck
                    new_bck = f(x)

                    y_plot = np.where(
                        (y-new_bck < self.background_bottom), self.background_bottom, y-new_bck)

                else:
                    y_plot = y

                if isinstance(scan_gas_dict, dict):
                    plt.plot(
                        x,
                        y_plot,
                        label=f"{scan_index}, {scan_gas_dict[scan_index]}",
                        color=self.ammonia_conditions_colors[scan_gas_dict[scan_index]],
                        linewidth=self.linewidth)

                else:
                    label = labels[i] if labels else scan_index

                    plt.plot(
                        x,
                        y_plot,
                        label=label,
                        linewidth=self.linewidth)

                if i == fill_first:
                    y_first = y_plot

                elif i == len(self.intensities) + fill_last:
                    y_last = y_plot

            # Generate a bolded horizontal line at y = self.y_og to highlight background
            try:
                plt.axhline(y=self.y_og, color='black',
                            linewidth=self.linewidth_hline, alpha=self.alpha_hline)
            except:
                pass

            # Generate a bolded vertical line at x = 0 to highlight origin
            plt.axvline(x=0, color='black',
                        linewidth=self.linewidth_vline, alpha=self.alpha_vline)

            if critical_angles:
                print("Critical angles position is angular (theta).")
                # Generate a bolded vertical line at x = 0.24 to highlightcritical angle of Pt
                plt.axvline(x=2*0.254509 if x_axis == "gamma" else 0.254509, color='red', linewidth=self.linewidth_vline, alpha=self.alpha_vline,
                            label="$\\alpha_c$ Pt",
                            linestyle="--")

                # Generate a bolded vertical line at x = 0.12 to highlightcritical angle of Al2O3
                plt.axvline(x=2*0.130261 if x_axis == "gamma" else 0.130261, color='black', linewidth=self.linewidth_vline, alpha=self.alpha_vline,
                            label="$\\alpha_c$ $Al_2O_3$",
                            linestyle="--")

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
                                            color=self.BP_colors["Pt"], linewidth=self.linewidth_vline, alpha=.5)
                                plt.text(x=bragg_angle, y=self.y_text, s=f"{miller_indices}", color=self.BP_colors["Pt"],
                                         weight='bold', rotation=60, backgroundcolor='#f0f0f0', fontsize=self.fontsize)

                    if "al2o3" in [z.lower() for z in miller]:
                        # Highlight Bragg peaks of Al2O3

                        for i, (miller_indices, bragg_angle) in enumerate(Al2P3_pos):
                            if bragg_angle > self.x_min and bragg_angle < self.x_max:
                                if i == 0:
                                    plt.axvline(x=bragg_angle, color=self.BP_colors["Al2O3"],
                                                linewidth=self.linewidth_vline, alpha=.5, label=("Al2O3"))
                                    plt.text(x=bragg_angle, y=self.y_text, s=f"{miller_indices}", color=self.BP_colors["Al2O3"],
                                             weight='bold', rotation=60, backgroundcolor='#f0f0f0', fontsize=self.fontsize)
                                else:
                                    plt.axvline(x=bragg_angle, color=self.BP_colors["Al2O3"],
                                                linewidth=self.linewidth_vline, alpha=.5)
                                    plt.text(x=bragg_angle, y=self.y_text, s=f"{miller_indices}", color=self.BP_colors["Al2O3"],
                                             weight='bold', rotation=60, backgroundcolor='#f0f0f0', fontsize=self.fontsize)

                    if "bn" in [z.lower() for z in miller]:
                        # Highlight Bragg peaks of BN

                        for i, (miller_indices, bragg_angle) in enumerate(BN_pos):
                            if bragg_angle > self.x_min and bragg_angle < self.x_max:
                                if i == 0:
                                    plt.axvline(x=bragg_angle, color=self.BP_colors["BN"],
                                                linewidth=self.linewidth_vline, alpha=.5, label=("Boron nitride"))
                                    plt.text(x=bragg_angle, y=self.y_text, s=f"{miller_indices}", color=self.BP_colors["BN"],
                                             weight='bold', rotation=60, backgroundcolor='#f0f0f0', fontsize=self.fontsize)
                                else:
                                    plt.axvline(x=bragg_angle, color=self.BP_colors["BN"],
                                                linewidth=self.linewidth_vline, alpha=.5)
                                    plt.text(x=bragg_angle, y=self.y_text, s=f"{miller_indices}", color=self.BP_colors["BN"],
                                             weight='bold', rotation=60, backgroundcolor='#f0f0f0', fontsize=self.fontsize)

            # Ticks
            plt.xticks(np.arange((self.x_min//self.x_tick_step)*self.x_tick_step, (self.x_max //
                       self.x_tick_step)*self.x_tick_step, self.x_tick_step), fontsize=self.tick_fontsize)
            plt.yticks(np.arange((self.x_min//self.x_tick_step)*self.x_tick_step, (self.x_max //
                       self.x_tick_step)*self.x_tick_step, self.x_tick_step), fontsize=self.tick_fontsize)

            # Range
            plt.xlim(left=self.x_min)
            plt.xlim(right=self.x_max)
            plt.ylim(bottom=self.y_min)
            plt.ylim(top=self.y_max)

            if fill:
                try:
                    # Add filling
                    plt.fill_between(self.x_axis, y_first, y_last, alpha=0.1)
                except:
                    print("Could not add filling.")
            else:
                print("No filling")

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
                plt.xlabel(x_axis, fontsize=self.fontsize)

            plt.ylabel("Intensity (a.u.)", fontsize=self.fontsize)
            plt.title(title, fontsize=self.title_fontsize)

        # TWO FIGURES ###########################################""

        elif low_range and high_range:
            fig, axes = plt.subplots(
                1, 2, sharey=True, figsize=figsize, dpi=150)
            fig.subplots_adjust(wspace=0.05)

            for (i, y), x, scan_index in zip(enumerate(self.intensities), self.x_axis, self.scan_indices):
                for ax in axes:

                    if isinstance(background, str):
                        # indices of x for which the value is defined on the background x
                        idx = (x < max(self.bck[0])) * (x > min(self.bck[0]))

                        x = x[idx]
                        y = y[idx]

                        # create ticks
                        f = interpolate.interp1d(self.bck[0], self.bck[1])

                        # create new bck
                        new_bck = f(x)

                        y_plot = np.where(
                            (y-new_bck < self.background_bottom), self.background_bottom, y-new_bck)

                    else:
                        y_plot = y

                    if isinstance(scan_gas_dict, dict):
                        ax.plot(
                            x,
                            y_plot,
                            label=f"{scan_index}, {scan_gas_dict[scan_index]}",
                            color=self.ammonia_conditions_colors[scan_gas_dict[scan_index]],
                            linewidth=self.linewidth)

                    else:
                        label = labels[i] if labels else scan_index

                        ax.plot(
                            x,
                            y_plot,
                            label=label,
                            linewidth=self.linewidth)

            if critical_angles:
                print("Critical angles position is angular (theta).")
                # Generate a bolded vertical line at x = 0.24 to highlightcritical angle of Pt
                axes[0].axvline(x=2*0.254509 if x_axis == "gamma" else 0.254509, color='red', linewidth=self.linewidth_vline, alpha=self.alpha_vline,
                                # label = "$\\alpha_c$ Pt",
                                linestyle="--")

                # Generate a bolded vertical line at x = 0.12 to highlightcritical angle of Al2O3
                axes[0].axvline(x=2*0.130261 if x_axis == "gamma" else 0.130261, color='black', linewidth=self.linewidth_vline, alpha=self.alpha_vline,
                                # label = "$\\alpha_c$ $Al_2O_3$",
                                linestyle="--")

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
                    # Highlight Bragg peaks of Pt
                    if "pt" in [z.lower() for z in miller]:

                        for i, (miller_indices, bragg_angle) in enumerate(Pt_pos):
                            if bragg_angle > self.x_min and bragg_angle < self.x_max:
                                axes[1].axvline(
                                    x=bragg_angle,
                                    # label ="Pt",
                                    color=self.BP_colors["Pt"],
                                    linewidth=self.linewidth_vline,
                                    alpha=.5)

                                axes[1].text(x=bragg_angle,
                                             y=self.y_text,
                                             s=f"{miller_indices}",
                                             color=self.BP_colors["Pt"],
                                             weight='bold',
                                             rotation=60,
                                             backgroundcolor='#f0f0f0',
                                             fontsize=self.fontsize)

            # Add respective ranges
            axes[0].set_xlim(low_range[0], low_range[1])
            axes[0].set_ylim(self.y_min, self.y_max)

            axes[1].set_xlim(high_range[0], high_range[1])
            axes[1].set_ylim(self.y_min, self.y_max)

            axes[0].legend(fontsize=self.fontsize, ncol=ncol)

            axes[0].tick_params(axis='both', labelsize=self.fontsize)
            axes[1].tick_params(axis='x', labelsize=self.fontsize)

            axes[0].set_ylabel("Intensity (a.u)", fontsize=30)
            axes[0].semilogy()
            axes[1].semilogy()

            # Generate a bolded vertical line at x = 0 to highlight origin
            axes[0].axvline(
                x=0.01, color='gray', linewidth=self.linewidth_vline, alpha=self.alpha_vline)

            if self.data_format == "hdf5" and self.var == "hkl":
                fig.text(x=0.5, y=0, s="L", fontsize=30, ha='center')
            elif self.data_format == "hdf5" and self.var == "qparqper":
                fig.text(x=0.5, y=0, s="qper", fontsize=30, ha='center')
            elif self.data_format == "hdf5" and self.var == "qxqyqz":
                fig.text(x=0.5, y=0, s="qz", fontsize=30, ha='center')
            elif self.data_format == "hdf5" and self.var == "qxqyqz":
                fig.text(x=0.5, y=0, s="qz", fontsize=30, ha='center')
            elif self.data_format == "nxs" and x_axis == "mu":
                fig.text(x=0.5, y=-0.04, s="Mu (°)", fontsize=30, ha='center')
            elif self.data_format == "nxs" and x_axis == "gamma":
                fig.text(x=0.5, y=-0.04, s="Gamma (°)",
                         fontsize=30, ha='center')

            fig.suptitle(title, fontsize=self.title_fontsize)

        plt.tight_layout()
        plt.savefig(f"{save_as}", bbox_inches='tight')

        print(f"Saved as {save_as}")

    def fit_bragg(self,
                  scan_to_fit,
                  fit_range=[2.75, 2.95],
                  x_axis="q",
                  peak_nb=1,
                  peak_pos=[2.85],
                  peak_amp=[1e5],
                  peak_sigma=[0.0008],
                  back=[10, 0, 100]):
        """docstring"""

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
        if self.data_format == "nxs" and x_axis in ["h", "k", "l", "q", "qpar", "qper", "gamma", "mu", "delta", "omega"]:
            x_total = getattr(self, x_axes[x_axis])

        elif self.data_format == "nxs" and x_axis not in ["h", "k", "l", "q", "qpar", "qper", "gamma", "mu", "delta", "omega"]:
            return("Choose a x_axis in the following list :", ["h", "k", "l", "q", "qpar", "qper", "gamma", "mu", "delta", "omega"])

        # Iterate over each data set to find the Bragg peak
        for q_range, gamma_range, delta_range, x, y, scan_name in zip(self.cp_q, self.cp_gamma, self.cp_delta, x_total, self.intensities, self.scan_indices):
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

    def compare_roi(self, index_range=1010, v_min_log=0.01, v_max_log=100, central_pixel_gamma=25, central_pixel_delta=285):
        """Define a widget function to be sure of the quality of the roi on the nxs file
        Assumes that the same detector is used for all the reflecto.
        """

        print("This uses a lot of RAM to facilitate the fluidity of the widget.")

        try:
            self.gamma_width
            self.delta_width

        except:
            return("You need to run reflecto.prep_nxs_data() before!")

        def plot2D(a, b, c, d, reflecto, index):
            fig, ax = plt.subplots(2, 1, figsize=(16, 9))

            # h_min = min(c, d)
            # h_max = max(c, d)
            # v_min = min(a, b)
            # v_max = max(a, b)

            # ax[0].imshow(np.log(reflecto.xpad_s70_image[index, :, :]))
            # ax[0].axhline(h_min, linewidth = 1)
            # ax[0].axhline(h_max, linewidth = 1)

            # ax[0].axvline(v_min, linewidth = 1)
            # ax[0].axvline(v_max, linewidth = 1)

            ax[0].imshow(
                # np.log(reflecto.xpad_s70_image[index, :, :]),
                reflecto.xpad_s70_image[index, :, :],
                norm=LogNorm(vmin=v_min_log, vmax=v_max_log)
            )
            ax[0].axhline(b, linewidth=1, color="red")
            ax[0].axhline(b + d, linewidth=1, color="red")

            ax[0].axvline(a, linewidth=1, color="red")
            ax[0].axvline(a + c, linewidth=1, color="red")

            ax[0].axvline(central_pixel_delta, linestyle="--",
                          linewidth=1, color="black")
            ax[0].axhline(central_pixel_gamma, linestyle="--",
                          linewidth=1, color="black")

            ax[0].set_title('Entire detector')
            ax[0].set_ylabel("gamma")
            ax[0].set_xlabel("delta")

            ax[1].set_title('Zoom in ROI')
            ax[1].set_ylabel("gamma")
            ax[1].set_xlabel("delta")

            ax[0].text(x=1.5,
                       y=-25,
                       s="Parameters : gamma = {:.3f}, mu = {:.3f}, hkl = ({:.3f}, {:.3f}, {:.3f}), q = {:3f}, qpar, qper = ({:3f}, {:3f})".format(
                           reflecto.gamma[index], reflecto.mu[index], reflecto.h[index], reflecto.k[index], reflecto.l[index], reflecto.q[index], reflecto.qPar[index], reflecto.qPer[index]),
                       #color = diverging_colors_1["E"],
                       weight='bold',
                       rotation=0,
                       backgroundcolor='#f0f0f0',
                       fontsize="20")

            # ax[1].imshow(np.log(reflecto.xpad_s70_image[index, h_min:h_max, v_min:v_max]))
            ax[1].imshow(
                # np.log(reflecto.xpad_s70_image[index, b:b+d, a:a+c]),
                reflecto.xpad_s70_image[index, b:b+d, a:a+c],
                norm=LogNorm(vmin=v_min_log, vmax=v_max_log))

        _list_widgets = interactive(plot2D,
                                    a=widgets.IntSlider(
                                        value=self.roi[0],
                                        min=0,
                                        max=self.delta_width,
                                        step=1,
                                        description='a:',
                                        continuous_update=False,
                                        orientation='horizontal',
                                        readout=True,
                                        readout_format='d'),
                                    b=widgets.IntSlider(
                                        value=self.roi[1],
                                        min=0,
                                        max=self.gamma_width,
                                        step=1,
                                        description='b:',
                                        continuous_update=False,
                                        orientation='horizontal',
                                        readout=True,
                                        readout_format='d'),
                                    c=widgets.IntSlider(
                                        value=self.roi[2],
                                        min=0,
                                        max=self.delta_width,
                                        step=1,
                                        description='c:',
                                        continuous_update=False,
                                        orientation='horizontal',
                                        readout=True,
                                        readout_format='d'),
                                    d=widgets.IntSlider(
                                        value=self.roi[3],
                                        min=0,
                                        max=self.gamma_width,
                                        step=1,
                                        description='d:',
                                        continuous_update=False,
                                        orientation='horizontal',
                                        readout=True,
                                        readout_format='d'),
                                    index=widgets.IntSlider(
                                        value=0,
                                        min=0,
                                        max=index_range-1,
                                        step=5,
                                        description='Index:',
                                        continuous_update=False,
                                        orientation='horizontal',
                                        readout=True,
                                        readout_format='d',
                                        style={'description_width': 'initial'},
                                        layout=Layout(width="30%")),
                                    reflecto=widgets.Select(
                                        options=[rn4.DataSet(
                                            self.folder + f) for f in self.scan_list],
                                        description='Scan:',
                                        style={'description_width': 'initial'},
                                        layout=Layout(width="70%")))
        window = widgets.VBox([
            widgets.HBox(_list_widgets.children[0:4]),
            widgets.HBox(_list_widgets.children[4:-1]),
            _list_widgets.children[-1]])

        return window
