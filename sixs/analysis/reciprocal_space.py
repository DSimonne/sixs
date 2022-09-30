"""
This module has functions to help with the analysis of reciprocal space data
during SXRD experiments at SixS.

The two main modules are the following:
    CTR()
    MAP()el

There are also functions that help with the simulations in ROD:
    simulate_rod()
    modify_surface_relaxation()
"""

import numpy as np
import tables as tb
import pandas as pd
import glob
import os
import inspect
import yaml
import sixs
import decimal
import shutil

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.interpolate import splrep, splev


class Map:
    """
    Loads an hdf5 file created by binoculars that represents a 3D map of the
    reciproal space and provides 2D plotting methods.

    Methods in class:
        * project_one_axis(): projects the data on one axis, with a given range
        * project_two_axes(): projects the data on two axes, with a given range
            (does not work yet)

    You can then plot the images with the plot_map() method.
    """

    def __init__(self, file_path):
        """
        Loads the binoculars file.

        The binocular data is loaded as follow:
            * Divide counts by contribution where cont != 0
            * Swap the h and k axes to be consistent with the indexing
                [h, k, l], or [Qx, Qy, Qz].
            * Flip k axis

        :param file_path: full path to .hdf5 file
        """

        self.file_path = file_path

        with tb.open_file(self.file_path) as f:

            # Get raw data
            ct = f.root.binoculars.counts.read()
            cont = f.root.binoculars.contributions.read()
            self.raw_data = np.divide(ct, cont, where=cont != 0)

            # Get which type of projection we are working with
            # HKL
            try:
                H = f.list_nodes('/binoculars/')[0].H
                hkl = True
            except tb.NoSuchNodeError:
                hkl = False

            ## Qpar, Qper
            try:
                Qpar = f.list_nodes('/binoculars/')[0].Qpar
                QparQper = True
            except tb.NoSuchNodeError:
                QparQper = False

            # Qindex
            try:
                Index = f.list_nodes('/binoculars/')[0].Index
                Qindex = True
            except tb.NoSuchNodeError:
                Qindex = False

            ## Qphi (not tested)
            try:
                Index = f.list_nodes('/binoculars/')[0].Phi
                QxQy = False  # also Qphi can have Qz (or Qx, Qy)
                Qphi = True
            except tb.NoSuchNodeError:
                QxQy = True
                Qphi = False

            if Qphi == False:  # also Qphi can have Qz (or Qx, Qy)
                try:
                    Qz = f.list_nodes('/binoculars/')[0].Qz
                except tb.NoSuchNodeError:
                    QxQy = False

            if Qphi == True:
                self.data = self.raw_data
                self.Phi = f.list_nodes('/binoculars/')[0].Phi[:]
                self.Q = f.list_nodes('/binoculars/')[0].Q[:]
                try:
                    self.Qxyz = f.list_nodes('/binoculars/')[0].Qx[:]
                except:
                    pass
                try:
                    self.Qxyz = f.list_nodes('/binoculars/')[0].Qy[:]
                except:
                    pass
                try:
                    self.Qxyz = f.list_nodes('/binoculars/')[0].Qz[:]
                except:
                    pass

            # Load data
            if Qindex == True:
                self.data = self.raw_data
                self.index = f.list_nodes('/binoculars/')[0].Index[:]
                self.Q = f.list_nodes('/binoculars/')[0].Q[:]

            if hkl == True:
                self.data = np.swapaxes(self.raw_data, 0, 2)
                self.H = f.list_nodes('/binoculars/')[0].H[:]
                self.K = f.list_nodes('/binoculars/')[0].K[:]
                self.L = f.list_nodes('/binoculars/')[0].L[:]

            if QxQy == True:

                self.data = self.raw_data
                self.Z = f.list_nodes('/binoculars/')[0].Qz[:]
                self.X = f.list_nodes('/binoculars/')[0].Qx[:]
                self.Y = f.list_nodes('/binoculars/')[0].Qy[:]

            elif QparQper == True:
                self.data = self.raw_data
                self.Y = f.list_nodes('/binoculars/')[0].Qper[:]
                self.X = f.list_nodes('/binoculars/')[0].Qpar[:]

            if Qphi == True:
                x_axis = np.linspace(
                    self.Q[1], self.Q[2], 1+self.Q[5]-self.Q[4])
                self.Q_axis = x_axis
                y_axis = np.linspace(
                    self.Qxyz[1], self.Qxyz[2], 1+self.Qxyz[5]-self.Qxyz[4])
                self.Qxyz_axis = y_axis
                z_axis = np.linspace(
                    self.Phi[1], self.Phi[2], 1+self.Phi[5]-self.Phi[4])
                self.Phi_axis = z_axis

            if Qindex == True:
                self.q_axis = np.linspace(
                    self.Q[1], self.Q[2], 1+self.Q[5]-self.Q[4])
                self.ind_axis = np.linspace(
                    self.index[1], self.index[2], 1+self.index[5]-self.index[4])

            if hkl == True:
                x_axis = np.arange(self.H[1], self.H[2], 1+self.H[5]-self.H[4])
                self.h_axis = np.linspace(
                    self.H[1], self.H[2], 1 + int(self.H[5] - self.H[4]))
                y_axis = np.arange(self.K[1], self.K[2], 1+self.K[5]-self.K[4])
                self.k_axis = np.linspace(
                    self.K[1], self.K[2], 1 + int(self.K[5] - self.K[4]))
                z_axis = np.arange(self.L[1], self.L[2], 1+self.L[5]-self.L[4])
                self.l_axis = np.round(np.linspace(
                    self.L[1], self.L[2], 1 + int(self.L[5] - self.L[4])), 3)

            if QxQy == True:
                x_axis = np.linspace(
                    self.X[1], self.X[2], 1 + int(self.X[5]-self.X[4]))
                self.Qx_axis = x_axis
                y_axis = np.linspace(
                    self.Y[1], self.Y[2], 1 + int(self.Y[5]-self.Y[4]))
                self.Qy_axis = y_axis
                z_axis = np.linspace(
                    self.Z[1], self.Z[2], 1 + int(self.Z[5]-self.Z[4]))
                self.Qz_axis = z_axis

            if QparQper == True:
                x_axis = np.linspace(
                    self.X[1], self.X[2], 1+self.X[5]-self.X[4])
                self.Qpar = x_axis
                y_axis = np.linspace(
                    self.Y[1], self.Y[2], 1+self.Y[5]-self.Y[4])
                self.Qper = y_axis

            print("\n###########################################################")
            print("Data shape:", self.data.shape)
            print("\tHKL data:", hkl)
            print("\tQxQy data:", QxQy)
            print("\tQparQper data:", QparQper)
            print("\tQphi data:", Qphi)
            print("\tQindex:", Qindex)
            print("###########################################################")

    def project_data(
        self,
        axis1,
        axis2=None,
        axis_range_1=[None, None],
        axis_range_2=[None, None],
    ):
        """
        Project the data on one or two of the measured axes, the result is saved
        as attribute `.two_d_data`.

        :param axis1: string in ("H", "K", "L", "Qx", "Qy", "Qz")
        :param axis2: None or string in ("H", "K", "L", "Qx", "Qy", "Qz")
        :param axis_range_1: list or tuple of length two, defines the positions
            of the value to be used in the array on the desired axis, use [None,
            None] to use the whole range.
        :param axis_range_2: list or tuple of length two, defines the positions
            of the value to be used in the array on the desired axis, use [None,
            None] to use the whole range.
        """
        # Start with first axis
        axis1_index = {
            "H": 2,
            "K": 1,
            "L": 0,
            "Qz": 2,
            "Qy": 1,
            "Qx": 0,
        }[axis1]

        axis1_name = {
            "H": "h_axis",
            "K": "k_axis",
            "L": "l_axis",
            "Qz": "Qz_axis",
            "Qy": "Qy_axis",
            "Qx": "Qx_axis",
        }[axis1]

        axis1_values = getattr(self, axis1_name)

        if axis1_range[0] != None:
            start_value = find_value_in_array(
                axis_1_values, axis_1_range[0])[0]

        if axis1_range[1] != None:
            end_value = find_value_in_array(axis_1_values, axis_1_range[1])[0]

        if axis_1 == 'H':
            datanan = self.data[:, :, start_value:end_value]

        elif axis_1 == 'K':
            datanan = self.data[:, start_value:end_value, :]

        elif axis_1 == 'L':
            datanan = self.data[start_value:end_value, :, :]

        elif axis_1 == 'Qz':
            # swap_data = np.swapaxes(self.raw_data, 0, 1)
            datanan = self.data[:, :, start_value:end_value]

        elif axis_1 == 'Qy':
            # swap_data = np.swapaxes(self.raw_data, 0, 2)
            datanan = self.data[:, start_value:end_value, :]

        elif axis_1 == 'Qx':
            # swap_data = np.swapaxes(self.raw_data, 1, 2)
            datanan = self.data[start_value:end_value, :, :]

        self.projected_data = np.nanmean(datanan, axis=axis_1_index)

        # Now second axis if necessary
        if axis2 != None:
            axis2_index = {
                "H": 2,
                "K": 1,
                "L": 0,
                "Qz": 2,
                "Qy": 1,
                "Qx": 0,
            }[axis2]

            axis2_name = {
                "H": "h_axis",
                "K": "k_axis",
                "L": "l_axis",
                "Qz": "Qz_axis",
                "Qy": "Qy_axis",
                "Qx": "Qx_axis",
            }[axis2]

            axis1_values = getattr(self, axis1_name)

            if axis1_range[0] != None:
                start_value = find_value_in_array(
                    axis_2_values, axis_2_range[0])[0]

            if axis1_range[1] != None:
                end_value = find_value_in_array(
                    axis_2_values, axis_2_range[1])[0]

            if axis_2 == 'H':
                datanan = self.projected_data[:, :, start_value:end_value]

            elif axis_2 == 'K':
                datanan = self.projected_data[:, start_value:end_value, :]

            elif axis_2 == 'L':
                datanan = self.projected_data[start_value:end_value, :, :]

            elif axis_2 == 'Qz':
                # swap_data = np.swapaxes(self.raw_data, 0, 1)
                datanan = self.projected_data[:, :, start_value:end_value]

            elif axis_2 == 'Qy':
                # swap_data = np.swapaxes(self.raw_data, 0, 2)
                datanan = self.projected_data[:, start_value:end_value, :]

            elif axis_2 == 'Qx':
                # swap_data = np.swapaxes(self.raw_data, 1, 2)
                datanan = self.projected_data[start_value:end_value, :, :]

            self.projected_data = np.nanmean(datanan, axis=axis_2_index)

    def plot_map(
        self,
        axis,
        axis_range=[None, None],
        interpolation="none",
        vmin=0.1,
        vmax=2000,
        figsize=(16, 9),
        title=None,
        cmap="jet",
        save_path=False,
        three_d_plot=False,
    ):
        """
        Plot/save a hdf5 map.

        You can use the command `%matplotlib notebook` before to use a cursor
        in the notebook cell (change figsize to (8,8))

        :param axis: string in ("H", "K", "L")
        :param axis_range: list or tuple of length two, defines the positions of
            the value to be used in the array on the desired axis
        :param interpolation: default is 'none'. See plt.imshow? for options,
            e.g. 'nearest'
        :param vmin: default to 0.1
        :param vmax: default to 2000
        :param figsize: default to (16, 9)
        :param title: figure title
        :param cmap: color map used, pick from
            https://matplotlib.org/stable/tutorials/colors/colormaps.html
        :param save_path: path to save file at
        :param three_d_plot: True to show a 3D plot
        """
        try:
            img = self.projected_data
        except NameError:
            print("Use the methods `project_data` to define the data first.")

        if axis == 'H':
            axis1 = self.k_axis
            axis2 = self.l_axis
            axis_name1 = 'K (rlu)'
            axis_name2 = 'L (rlu)'

        elif axis == 'K':
            axis1 = self.h_axis
            axis2 = self.l_axis
            axis_name1 = 'H (rlu)'
            axis_name2 = 'L (rlu)'

        elif axis == 'L':
            axis1 = self.h_axis
            axis2 = self.k_axis
            axis_name1 = 'H (rlu)'
            axis_name2 = 'K (rlu)'

        elif axis == 'Qxyz':
            axis1 = self.Q_axis
            axis2 = self.Phi_axis
            axis_name1 = 'Q'
            axis_name2 = 'Phi (deg)'

        elif axis == 'Qx':
            axis1 = self.Qy_axis
            axis2 = self.Qz_axis
            axis_name1 = 'Qy'
            axis_name2 = 'Qz'

        elif axis == 'Qy':
            axis1 = self.Qx_axis
            axis2 = self.Qz_axis
            axis_name1 = 'Qx'
            axis_name2 = 'Qz'

        elif axis == 'Qz':
            axis1 = self.Qx_axis
            axis2 = self.Qy_axis
            axis_name1 = 'Qx'
            axis_name2 = 'Qy'

        # Plot
        if three_d_plot:
            X, Y = np.meshgrid(axis1, axis2)
            Z = np.where(img > vmin, np.log(img), 0)

            fig, ax = plt.subplots(
                figsize=figsize,
                subplot_kw={'projection': '3d'}
            )
            plotted_img = ax.plot_surface(
                X,
                Y,
                Z,
                cmap=cmap,
                # cstride=40,
                # rstride=40,
            )

        else:
            fig, ax = plt.subplots(figsize=figsize)
            plotted_img = ax.imshow(
                img,
                cmap=cmap,
                interpolation=interpolation,
                origin="lower",
                # aspect = 'auto',
                norm=LogNorm(vmin=vmin, vmax=vmax),
                extent=[axis1.min(), axis1.max(), axis2.min(), axis2.max()]
            )

        # Labels and ticks
        ax.set_xlabel(axis_name1, fontsize=20)
        ax.set_ylabel(axis_name2, fontsize=20)
        ax.tick_params(axis=('both'), labelsize=20)

        # Colorbar
        cbar = fig.colorbar(plotted_img, shrink=0.5)
        cbar.ax.tick_params(labelsize=20)

        plt.tight_layout()

        if isinstance(title, str):
            ax.set_title(title, fontsize=20)

        if save_path:
            plt.savefig(save_path)

        plt.show()


class CTR:
    """
    Loads an hdf5 file created by binoculars that represents a 3D map of the
    reciproal space and provides integration methods to analyse the diffracted
    intensity along one direction.

    For now the classical workflow is the following:
    * process the data with binoculars, creating hdf5 files
    * integrate RODs with binoculars-fitaid, creating .txt files
    * fit these RODs with the `ROD` program (https://www.esrf.fr/computing/
        scientific/joint_projects/ANA-ROD/RODMAN2.html)

    Since binoculars-fitaid is not reliable at all, I tried to rewrite that part
    to be able to integrate the RODs with a python function (integrate_CTR()).
    It does not currently give the same results as when using fitaid, idk why,
    so it's better to use fitaid at the moment.

    Use one the following three methods to load the data:
        * integrate_CTR()
        * load_fitaid_data()
        * load_ROD_data()

    All these functions create numpy arrays that can then be plotted with the
    plot_CTR() method.
    """

    def __init__(
        self,
        configuration_file=False,
    ):
        """
        Init the class with configuration file

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

    def integrate_CTR(
        self,
        folder,
        scan_indices,
        save_name,
        glob_string_match="*.hdf5",
        interpol_step=False,
        CTR_width_H=0.02,
        CTR_width_K=0.02,
        background_width_H=0.01,
        background_width_K=0.01,
        HK_peak=[-1, 1],
        center_background=False,
        verbose=False,
    ):
        """
        Prepare the data for plots by interpolating on the smallest common range
        center_background can be different from HK_peak, the goal is to avoid
        summing the CTR intensity at the same time as the diffraction rings.

        The binocular data is loaded as follow:
            Divide counts by contribution where cont != 0
            Swap the h and k axes to have a good convention
            Flip k axis

        The data is computed as follow:
            integral = sum of pixels in CTR roi /nb of pixels in CTR roi
            bckd_per_pixel = sum of pixels in bckd roi /nb of pixels in bckd roi
            ctr = integral - bckd_per_pixel

        Saves the result as a numpy array on disk.

        :param folder: path to data folder
        :param scan_indices: indices of maps scans, list
        :param save_name: name of file in which the results are saved, saved in
            folder.
        :param glob_string_match: string pattern used for file matching
        :param interpol_step: step size in interpolation along L, to avoid
            problem with analysing CTR with different steps. No interpolation if
            False, default.
        :param CTR_width_H: width in h of CTR in reciprocal space, default is
            0.02.
        :param CTR_width_K: width in k of CTR in reciprocal space, default is
            0.02.
        :param background_width_H: width in h of background region taken in
            reciprocal space, default is 0.01.
        :param background_width_K: width in k of background region taken in
            reciprocal space, default is 0.01.
        :param HK_peak: node in reciprocal space around which the CTR is taken,
            default is [-1, 1].
        :param center_background: node in reciprocal space around which the
            background is taken. If equal to HK_peak, the background width is
            added to the width of the CTR. If False, no background is subtracted
        :param verbose: if True, print more details.
        """

        # Load data
        scan_indices = [str(s) for s in scan_indices]

        # Get all files first
        files = [f.split("/")[-1]
                 for f in sorted(glob.glob(f"{folder}/{glob_string_match}"))]

        # Get scans specified with scan_indices
        self.scan_files = [f for f in files if any(
            [n in f for n in scan_indices])]

        if verbose:
            print("\n###########################################################")
            print(f"Detected files in {folder}:")
            for f in files:
                print("\t", f)
            print("###########################################################\n")

            print("\n###########################################################")
            print("Working on the following files:")
            for f in self.scan_files:
                print("\t", f)
            print("###########################################################\n")

        # Parameters of rod
        self.HK_peak = HK_peak
        self.interpol_step = interpol_step

        self.CTR_width_H = CTR_width_H
        self.CTR_width_K = CTR_width_K

        self.CTR_range_H = [
            np.round(self.HK_peak[0] - self.CTR_width_H, 3),
            np.round(self.HK_peak[0] + self.CTR_width_H, 3)
        ]
        self.CTR_range_K = [
            np.round(self.HK_peak[1] - self.CTR_width_K, 3),
            np.round(self.HK_peak[1] + self.CTR_width_K, 3)
        ]

        print("\n###########################################################")
        print(f"Range in H is [{self.CTR_range_H[0]} : {self.CTR_range_H[1]}]")
        print(f"Range in K is [{self.CTR_range_K[0]} : {self.CTR_range_K[1]}]")
        print("###########################################################")

        # Background parameter
        self.center_background = center_background

        if center_background == HK_peak:
            self.background_width_H = CTR_width_H + background_width_H
            self.background_width_K = CTR_width_K + background_width_K

            self.background_range_H = [
                np.round(self.HK_peak[0] - self.background_width_H, 3),
                np.round(self.HK_peak[0] + self.background_width_H, 3)
            ]
            self.background_range_K = [
                np.round(self.HK_peak[1] - self.background_width_K, 3),
                np.round(self.HK_peak[1] + self.background_width_K, 3)
            ]

            print("\n###########################################################")
            print(
                f"Background range in H is [{self.background_range_H[0]} : {self.background_range_H[1]}]")
            print(
                f"Background range in K is [{self.background_range_K[0]} : {self.background_range_K[1]}]")
            print("###########################################################")

        elif isinstance(center_background, list) and center_background != HK_peak:
            self.background_width_H = background_width_H
            self.background_width_K = background_width_K

            self.background_range_H = [
                self.center_background[0] - self.background_width_H,
                self.center_background[0] + self.background_width_H
            ]
            self.background_range_K = [
                self.center_background[1] - self.background_width_K,
                self.center_background[1] + self.background_width_K
            ]

            print("\n###########################################################")
            print(
                f"Background range in H is [{self.background_range_H[0]} : {self.background_range_H[1]}]")
            print(
                f"Background range in K is [{self.background_range_K[0]} : {self.background_range_K[1]}]")
            print("###########################################################")

        # Start iterating on file to see the shape
        print("\n###########################################################")
        print("Finding smallest common range in L")
        print("Depends on the config file in binoculars-process.")
        print("###########################################################")

        for i, fname in enumerate(self.scan_files):
            with tb.open_file(folder + fname, "r") as f:
                H = f.root.binoculars.axes.H[:]
                K = f.root.binoculars.axes.K[:]
                L = f.root.binoculars.axes.L[:]

            if verbose:
                print(
                    "\n###########################################################")
                print(f"Opening file {fname} ...")
                print("\tRange and stepsize in H: [{0:.3f}: {1:.3f}: {2:.3f}]".format(
                    H[1], H[2], H[3]))
                print("\tRange and stepsize in K: [{0:.3f}: {1:.3f}: {2:.3f}]".format(
                    K[1], K[2], K[3]))
                print("\tRange and stepsize in L: [{0:.3f}: {1:.3f}: {2:.3f}]".format(
                    L[1], L[2], L[3]))
                print("###########################################################")

            if i == 0:
                l_min = L[1]
                l_max = L[2]
                l_shape = 1 + int(L[5] - L[4])
            else:
                l_min = max(l_min, L[1])
                l_max = min(l_max, L[2])
                l_shape = max(l_shape, 1 + int(L[5] - L[4]))

        print("\n###########################################################")
        print(f"Smallest common range in L is [{l_min} : {l_max}]")
        print("###########################################################")

        # Create a common l axis if we interpolate
        if isinstance(self.interpol_step, float):
            l_axe = np.arange(l_min, l_max, self.interpol_step)

            # Save final data as numpy array
            # 0 is x axis, 1 is data, 2 is background
            data = np.nan * np.empty((len(self.scan_files), 3, (len(l_axe))))
        else:
            # Save final data as numpy array
            # 0 is x axis, 1 is data, 2 is background
            data = np.nan * np.empty((len(self.scan_files), 3, l_shape))

        # Iterate on each file
        for i, fname in enumerate(self.scan_files):
            if verbose:
                print("\n###########################################################")
                print(f"Opening file {fname} ...")

            with tb.open_file(folder + fname, "r") as f:

                ct = f.root.binoculars.counts.read()
                cont = f.root.binoculars.contributions.read()

                raw_data = np.divide(ct, cont, where=cont != 0)

                # Need to swap the axes
                raw_data = np.swapaxes(raw_data, 0, 1)

                # Need to flip K axis
                raw_data = np.flip(raw_data, axis=(0))

                H = f.root.binoculars.axes.H[:]
                K = f.root.binoculars.axes.K[:]
                L = f.root.binoculars.axes.L[:]

            scan_h_axe = np.round(np.linspace(
                H[1], H[2], 1 + int(H[5] - H[4])), 3)  # xaxe
            scan_k_axe = np.round(np.linspace(
                K[1], K[2], 1 + int(K[5] - K[4])), 3)  # yaxe
            scan_l_axe = np.round(np.linspace(
                L[1], L[2], 1 + int(L[5] - L[4])), 3)  # zaxe

            # Need to flip K axis again
            scan_k_axe = np.flip(scan_k_axe)

            # CTR intensity, define roi indices
            st_H_roi = find_value_in_array(scan_h_axe, self.CTR_range_H[0])
            end_H_roi = find_value_in_array(scan_h_axe, self.CTR_range_H[1])

            st_K_roi = find_value_in_array(scan_k_axe, self.CTR_range_K[1])
            end_K_roi = find_value_in_array(scan_k_axe, self.CTR_range_K[0])

            if verbose:
                print(
                    f"""Data ROI: [start_k, end_K, start_H, end_H] = \
                    \n\t[{st_K_roi[0]}, {end_K_roi[0]}, {st_H_roi[0]}, {end_H_roi[0]}]\
                    \n\t[{st_K_roi[1]}, {end_K_roi[1]}, {st_H_roi[1]}, {end_H_roi[1]}]\
                    """)

            # Get data only in specific ROI
            # CAREFUL WITH THE ORDER OF H AND K HERE
            roi_2D = raw_data[st_K_roi[1]:end_K_roi[1],
                              st_H_roi[1]:end_H_roi[1], :]
            nb_pixel_roi = roi_2D.shape[0] * roi_2D.shape[1]

            # Interpolate over common l axis
            if isinstance(self.interpol_step, float):
                # Save x axis
                data[i, 0, :] = l_axe

                # Save intensities
                tck = splrep(scan_l_axe, roi_2D.sum(axis=(0, 1)), s=0)
                roi_2D_sum = splev(l_axe, tck)
                data[i, 1, :] = roi_2D_sum / nb_pixel_roi

            # No interpolation
            else:
                # Save x axis
                data[i, 0, :len(scan_l_axe)] = scan_l_axe

                # Save intensities
                roi_2D_sum = roi_2D.sum(axis=(0, 1))
                data[i, 1, :len(scan_l_axe)] = roi_2D_sum / nb_pixel_roi

            # Get background
            if center_background == HK_peak:
                # Background intensity, define roi indices
                st_H_background = find_value_in_array(
                    scan_h_axe, self.background_range_H[0])
                end_H_background = find_value_in_array(
                    scan_h_axe, self.background_range_H[1])

                st_K_background = find_value_in_array(
                    scan_k_axe, self.background_range_K[0])
                end_K_background = find_value_in_array(
                    scan_k_axe, self.background_range_K[1])

                if verbose:
                    print(
                        f"Background ROI = [{st_H_background[1]}, {end_H_background[1]}, {st_K_background[1]}, {end_K_background[1]}]")
                    print(
                        "###########################################################")

                # CAREFUL WITH THE ORDER OF H AND K HERE
                background_H = raw_data[st_K_roi[1]:end_K_roi[1],
                                        st_H_background[1]:end_H_background[1],
                                        :  # all data in L
                                        ]
                nb_pixel_background_H = background_H.shape[0] * \
                    background_H.shape[1]

                background_K = raw_data[st_K_background[1]:end_K_background[1],
                                        st_H_roi[1]:end_H_roi[1],
                                        :  # all data in L
                                        ]
                nb_pixel_background_K = background_H.shape[0] * \
                    background_H.shape[1]

                # Interpolate
                if isinstance(self.interpol_step, float):
                    tck_H = splrep(
                        scan_l_axe, background_H.sum(axis=(0, 1)), s=0)
                    tck_K = splrep(
                        scan_l_axe, background_K.sum(axis=(0, 1)), s=0)

                    background_H_sum = splev(l_axe, tck_H)
                    background_K_sum = splev(l_axe, tck_K)

                    # Save background
                    # Subtract twice here because we are using two rectangles
                    # that overlap our data
                    # Need to divide by nb of pixels
                    data[i, 2, :] = background_H_sum / nb_pixel_background_H + \
                        background_K_sum / nb_pixel_background_K - \
                        2 * data[i, 1, :]
                else:
                    background_H_sum = background_H.sum(axis=(0, 1))
                    background_K_sum = background_K.sum(axis=(0, 1))

                    # Save background
                    data[i, 2, :len(scan_l_axe)] = background_H_sum / nb_pixel_background_H + \
                        background_K_sum / nb_pixel_background_K - \
                        2 * data[i, 1, :len(scan_l_axe)]

            elif isinstance(center_background, list) and center_background != HK_peak:
                # Background intensity, define roi indices
                st_H_background = find_value_in_array(
                    scan_h_axe, self.background_range_H[0])
                end_H_background = find_value_in_array(
                    scan_h_axe, self.background_range_H[1])

                st_K_background = find_value_in_array(
                    scan_k_axe, self.background_range_K[0])
                end_K_background = find_value_in_array(
                    scan_k_axe, self.background_range_K[1])

                if verbose:
                    print(
                        f"Background ROI = [{st_H_background[1]}, {end_H_background[1]}, {st_K_background[1]}, {end_K_background[1]}]")
                    print(
                        "###########################################################")

                background_2D = raw_data[st_K_background[1]:end_K_background[1],
                                         st_H_background[1]:end_H_background[1],
                                         :
                                         ]
                nb_pixel_background_2D = background_2D.shape[0] * \
                    background_2D.shape[1]

                # Interpolate
                if isinstance(self.interpol_step, float):
                    tck_2D = splrep(
                        scan_l_axe, background_2D.sum(axis=(0, 1)), s=0)
                    background_2D_sum = splev(l_axe, tck_2D)

                    # Save background
                    data[i, 2, :] = background_2D_sum / nb_pixel_background_2D
                else:
                    background_2D_sum = background_2D.sum(axis=(0, 1))

                    # Save background
                    data[i, 2, :len(scan_l_axe)] = background_2D_sum / \
                        nb_pixel_background_2D

            else:
                print("No background subtracted")
                print("###########################################################")

            # Resume with a plot
            if verbose:
                plt.figure(figsize=(8, 8))
                plt.imshow(
                    np.sum(raw_data, axis=2),
                    norm=LogNorm(),
                    cmap="cividis",
                    extent=(H[1], H[2], K[1], K[2]),
                )
                plt.xlabel("H", fontsize=15)
                plt.ylabel("K", fontsize=15)
                plt.title("Intensity summed over L", fontsize=15)

                # Plot data ROI
                plt.axvline(x=st_H_roi[0], color='red', linestyle="--")
                plt.axvline(x=end_H_roi[0], color='red', linestyle="--")

                plt.axhline(y=st_K_roi[0], color='red', linestyle="--")
                plt.axhline(y=end_K_roi[0], color='red',
                            linestyle="--", label="ROI")

                if center_background != False:
                    # Plot background ROI
                    plt.axvline(
                        x=st_H_background[0], color='blue', linestyle="--")
                    plt.axvline(
                        x=end_H_background[0], color='blue', linestyle="--")

                    plt.axhline(
                        y=st_K_background[0], color='blue', linestyle="--")
                    plt.axhline(
                        y=end_K_background[0], color='blue', linestyle="--",
                        label="Bckd")

                # Legend
                plt.legend()
                plt.show()
                plt.close()

        # Saving
        print("\n###########################################################")
        print(f"Saving data as: {folder}{save_name}.npy")
        print("###########################################################")
        np.save(folder + save_name, data)

    def load_fitaid_data(
        self,
        folder,
        scan_indices,
        save_name,
        glob_string_match="nisf*.txt",
        interpol_step=False,
        verbose=False,
    ):
        """
        Load CTR integrated via binoculars-fitaid

        :param folder: path to data folder
        :param scan_indices: list of CTR scans indices
        :param save_name: name of file in which the results are saved, saved in
         folder.
        :param glob_string_match: string pattern used for file matching
        :param data_type: type of data to load from binoculars, usually the
         possibilities are "nisf" or "sf". Prefer nisf data, detail here the
         differences
        :param interpol_step: step size in interpolation along L, to avoid
         problem with analysing CTR with different steps. No interpolation if
         False, default.
        :param verbose: True for additional informations printed during function
        """

        # Get files
        scan_indices = [str(s) for s in scan_indices]

        # Get all txt files first
        files = [f.split("/")[-1]
                 for f in sorted(glob.glob(f"{folder}/{glob_string_match}"))]

        # Get scans specified with scan_indices
        self.scan_files = [f for f in files if any(
            [n in f for n in scan_indices])]
        if verbose:
            print("\n###########################################################")
            print(f"Detected files in {folder}:")
            for f in files:
                print("\t", f)
            print("###########################################################\n")

            print("\n###########################################################")
            print("Working on the following files:")
            for f in self.scan_files:
                print("\t", f)
            print("###########################################################\n")

        # Iterating on all files to create l axis
        for i, fname in enumerate(self.scan_files):
            # Load data
            fitaid_data = np.loadtxt(folder + fname)

            # L axis
            L = fitaid_data[:, 0]

            if verbose:
                print(
                    "\n###########################################################")
                print(f"Opening file {fname} ...")
                print("\tRange and stepsize in L: [{0:.3f}: {1:.3f}: {2:.3f}]".format(
                    min(L), max(L), len(L)))
                print("###########################################################")

            if i == 0:
                l_min = np.round(min(L), 3)
                l_max = np.round(max(L), 3)
                l_shape = len(L)
            else:
                l_min = np.round(max(l_min, min(L)), 3)
                l_max = np.round(min(l_max, max(L)), 3)
                l_shape = max(l_shape, len(L))

        print("\n###########################################################")
        print(f"Smallest common range in L is [{l_min} : {l_max}]")
        print("###########################################################")

        # Create new x axis for interpolation
        self.interpol_step = interpol_step
        if isinstance(self.interpol_step, float):
            l_axe = np.arange(l_min, l_max, self.interpol_step)

            # Save final data as numpy array
            # 0 is x axis, 1 is data, 2 is background
            data = np.nan * \
                np.empty((len(self.scan_files), 3, (len(l_axe))))
        else:
            # Save final data as numpy array
            # 0 is x axis, 1 is data, 2 is background
            data = np.nan * np.empty((len(self.scan_files), 3, l_shape))

        # Background already subtracted, left as nan
        # Get l axis and CTR intensity for each file
        for i, fname in enumerate(self.scan_files):

            # Load data
            fitaid_data = np.loadtxt(folder + fname)
            scan_l_axe = fitaid_data[:, 0]
            ctr_data = fitaid_data[:, 1]

            # Interpolate
            if isinstance(self.interpol_step, float):
                data[i, 0, :] = l_axe

                tck = splrep(scan_l_axe, ctr_data, s=0)
                data[i, 1, :] = splev(l_axe, tck)

            else:
                data[i, 0, :len(scan_l_axe)] = scan_l_axe
                data[i, 1, :len(scan_l_axe)] = ctr_data

        # Saving
        print("\n###########################################################")
        print(f"Saving data as: {folder}{save_name}.npy")
        print("###########################################################")
        np.save(folder + save_name, data)

    def load_ROD_data(
        self,
        folder,
        scan_indices,
        save_name,
        data_column=7,
        glob_string_match="*.dat",
        interpol_step=False,
        verbose=False,
    ):
        """
        Load CTR simulated with ROD, at least 5 columns, e.g.:
        h      k      l   f-bulk   f-surf    f-mol    f-liq    f-sum    phase

        We are only interested in two: l and a data column.

        :param folder: path to data folder
        :param scan_indices: indices of maps scans, list
        :param save_name: name of file in which the results are saved, saved in
         folder.
        :param data_type: type of data to load from binoculars, usually the
         possibilities are "nisf" or "sf". Prefer nisf data, detail here the
         differences
        :param interpol_step: step size in interpolation along L, to avoid
         problem with analysing CTR with different steps. No interpolation if
         False, default.
        :param verbose: True for additional informations printed during function
        """

        # Get files
        scan_indices = [str(s) for s in scan_indices]

        # Get all txt files first
        files = [f.split("/")[-1]
                 for f in sorted(glob.glob(f"{folder}/{glob_string_match}"))]

        # Get scans specified with scan_indices
        self.scan_files = [f for f in files if any(
            [n in f for n in scan_indices])]
        if verbose:
            print("\n###########################################################")
            print(f"Detected files in {folder}:")
            for f in files:
                print("\t", f)
            print("###########################################################\n")

            print("\n###########################################################")
            print("Working on the following files:")
            for f in self.scan_files:
                print("\t", f)
            print("###########################################################\n")

        # Iterating on all files to create l axis
        for i, fname in enumerate(self.scan_files):
            # Load data
            rod_data = np.loadtxt(folder + fname, skiprows=2)

            # L axis
            L = rod_data[:, 2]

            if verbose:
                print(
                    "\n###########################################################")
                print(f"Opening file {fname} ...")
                print("\tRange and stepsize in L: [{0:.3f}: {1:.3f}: {2:.3f}]".format(
                    min(L), max(L), len(L)))
                print("###########################################################")

            if i == 0:
                l_min = np.round(min(L), 3)
                l_max = np.round(max(L), 3)
                l_shape = len(L)
            else:
                l_min = np.round(max(l_min, min(L)), 3)
                l_max = np.round(min(l_max, max(L)), 3)
                l_shape = max(l_shape, len(L))

        print("\n###########################################################")
        print(f"Smallest common range in L is [{l_min} : {l_max}]")
        print("###########################################################")

        # Create new x axis for interpolation
        self.interpol_step = interpol_step
        if isinstance(self.interpol_step, float):
            l_axe = np.arange(l_min, l_max, self.interpol_step)

            # Save final data as numpy array
            # 0 is x axis, 1 is data, 2 is background
            data = np.nan * \
                np.empty((len(self.scan_files), 3, (len(l_axe))))
        else:
            # Save final data as numpy array
            # 0 is x axis, 1 is data, 2 is background
            data = np.nan * np.empty((len(self.scan_files), 3, l_shape))

        # Background already subtracted, left as nan
        # Get l axis and CTR intensity for each file
        for i, fname in enumerate(self.scan_files):

            # Load data
            rod_data = np.loadtxt(folder + fname, skiprows=2)
            scan_l_axe = rod_data[:, 2]
            ctr_data = rod_data[:, data_column]

            # Interpolate
            if isinstance(self.interpol_step, float):
                data[i, 0, :] = l_axe

                tck = splrep(scan_l_axe, ctr_data, s=0)
                data[i, 1, :] = splev(l_axe, tck)

            else:
                data[i, 0, :len(scan_l_axe)] = scan_l_axe
                data[i, 1, :len(scan_l_axe)] = ctr_data

        # Saving
        print("\n###########################################################")
        print(f"Saving data as: {folder}{save_name}.npy")
        print("###########################################################")
        np.save(folder + save_name, data)

    @ staticmethod
    def plot_CTR(
        numpy_array,
        scan_indices,
        title=None,
        filename=None,
        figsize=(18, 9),
        ncol=2,
        color_dict=None,
        labels=None,
        zoom=[None, None, None, None],
        fill=False,
        fill_first=0,
        fill_last=-1,
        log_intensities=True,
        line_plot=True,
        s=None,
        marker=None,
        fontsize=15,
    ):
        """
        Plot the CTRs together

        :param numpy_array: path to .npy file on disk.
        TODO problem here, inverting indices in array
            - l
            - data
            - background
        :param scan_indices: scan indices of files plotted, in order, used for
         labelling, mandatory because we need to know what we plot!
        :param title: if string, set to figure title
        :param filename: if string, figure will be saved to this path.
        :param figsize: figure size, default is (18, 9)
        :param ncol: columns in label, default is 2
        :param color_dict: dict used for labels, keys are scan index, values are
         colours for matplotlib.
        :param labels: dict of labels to use, defaulted to scan index if None
        :param zoom: values used for plot range, default is
         [None, None, None, None], order is left, right, bottom and top.
        :param fill: if True, add filling between two plots
        :param fill_first: index of scan to use for filling
        :param fill_last: index of scan to use for filling
        :param log_intensities: if True, y axis is logarithmic
        :param line_plot: if False, scatter plot
        :param s: scatter size in scatter plot
        :param marker: marker used for scatter plot
        :param fontsize: fontsize in plots
        """

        # Create figure
        plt.figure(figsize=figsize)
        if log_intensities:
            plt.semilogy()
        plt.grid()

        # Load np array on disk
        data = np.load(numpy_array)
        print("Loaded", numpy_array)

        # Iterate on data
        for (i, arr), scan_index in zip(enumerate(data), scan_indices):
            # take l again but still or better to keep x values in the same array with y
            l = arr[0, :]  # x axis
            y = arr[1, :]  # data
            b = arr[2, :]  # background

            # Remove background
            # Replace nan by zeroes for background, makes more sense
            y_plot = y-np.nan_to_num(b)

            # Add label
            if isinstance(labels, dict):
                try:
                    label = labels[scan_index]
                except KeyError:
                    label = labels[int(scan_index)]
                except:
                    print("Dict not valid for labels, using scan_indices")
                    label = scan_index
            elif labels == None:
                label = scan_index
            else:
                print("Labels must be a dictionnary with keys = scan_indices")
                label = scan_index

            # Add colour
            try:
                if line_plot:
                    plt.plot(
                        l,
                        y_plot,
                        color=color_dict[int(scan_index)],
                        label=label,
                        linewidth=2,
                    )
                else:
                    plt.scatter(
                        x=l,
                        y=y_plot,
                        c=color_dict[int(scan_index)],
                        label=label,
                        s=s,
                        marker=marker
                    )

            except (KeyError, ValueError):
                # Take int(scan_index) in case keys are not strings in the dict
                try:
                    if line_plot:
                        plt.plot(
                            l,
                            y_plot,
                            color=color_dict[scan_index],
                            label=label,
                            linewidth=2,
                        )
                    else:
                        plt.scatter(
                            x=l,
                            y=y_plot,
                            c=color_dict[scan_index],
                            label=label,
                            s=s,
                            marker=marker
                        )
                except TypeError:  # no color
                    if line_plot:
                        plt.plot(
                            l,
                            y_plot,
                            label=label,
                            linewidth=2,
                        )
                    else:
                        plt.scatter(
                            x=l,
                            y=y_plot,
                            label=label,
                            s=s,
                            marker=marker
                        )
            except TypeError:  # No special colour
                if line_plot:
                    plt.plot(
                        l,
                        y_plot,
                        label=label,
                        linewidth=2,
                    )
                else:
                    plt.scatter(
                        x=l,
                        y=y_plot,
                        label=label,
                        s=s,
                        marker=marker
                    )

            # For filling
            if i == fill_first:
                y_first = y_plot

            elif i == len(data) + fill_last:
                y_last = y_plot

        # Ticks
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)

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
        plt.legend(fontsize=fontsize, ncol=ncol, markerscale=1.2)

        plt.xlabel("L", fontsize=fontsize)
        plt.ylabel("Intensity (a.u.)", fontsize=fontsize)

        # Title
        if isinstance(title, str):
            plt.title(f"{title}", fontsize=20)

        # Save
        plt.tight_layout()
        if filename != None:
            plt.savefig(f"{filename}", bbox_inches='tight')
            print(f"Saved as {filename}")

        plt.show()


def simulate_rod(
    filename,
    bulk_file=None,
    surface_file=None,
    rod_hk=[2, 2],
    l_start=0,
    l_end=3,
    nb_points=251,
    l_bragg=0,
    nb_layers_bulk=2,
    attenuation=0.001,
    beta=0,
    error_bars=True,
    save_folder=None,
    comment=None,
):
    """
    This function uses ROD, that must be installed on your computer.
    Help document:
        https://www.esrf.fr/computing/scientific/joint_projects/
        ANA-ROD/RODMAN2.html

    Will generate the following files the first time it is run in a folder:
        - pgplot.ps
        - plotinit.mac
        - rod_init.mac
    These files will be used during subsequent runs, they initialize a bunch of
        arguments used for plotting.

    :param filename: str, used in the names of the output files, not a path
    :param bulk_file: str, path to bulk file (.bul)
    :param surface_file: str, path to surface file (.sur)
    :param rod_hk: list, position in h and k of the rod
    :param l_start: beginning of the rod in l
    :param l_end: end of the rod in l
    :param nb_points: nb of points in the rod
    :param l_bragg: position in l of the first bragg peak
    :param nb_layers_bulk: number of layers in the bulk
    :param attenuation: attenuation of the beam
    :param beta: beta used for roughness in beta model
    :param error_bars: bool, use error bars or not in the plot
    :param save_folder: folder in which output files are saved
    :param comment: str, comment to add to file
    """

    # Save folder is either os.getcwd() or specified
    if save_folder == None:
        save_folder = os.getcwd()

    elif not os.path.exists(save_folder):
        print(f"{save_folder} does not exist, defaulting to {os.getcwd()}")
        save_folder = os.getcwd()

    # Remove file extension if one was provided to avoid bugs
    filename, _ = os.path.splitext(filename)
    macro_file = filename + '.mac'
    save_file = filename + '.dat'

    # Create list of lines
    lines = [
        f"set calc LStart {l_start}",
        f"\nLEnd {l_end}",
        f"\nNpoints {nb_points}",
        f"\nLBragg {l_bragg}",
        f"\nAtten {attenuation}",
        f"\nBeta {beta}",
        f"\nNLayers {nb_layers_bulk} return return",
    ]

    # Read files
    if isinstance(bulk_file, str):
        if os.path.isfile(bulk_file):
            # Copy bulk file to save_folder for execution
            shutil.copy2(
                bulk_file,
                save_folder,
            )
            lines.append(f"\nread bul {os.path.basename(bulk_file)}")

    if isinstance(surface_file, str):
        if os.path.isfile(surface_file):
            # Copy surface file to save_folder for execution
            shutil.copy2(
                surface_file,
                save_folder,
            )
            lines.append(f"\nread sur {os.path.basename(surface_file)}")

    # Plotting options
    if error_bars:
        lines.append(f"\nplot errors y return")

    # Calculate rod at [h, k]
    lines.append(f"\ncalc rod {rod_hk[0]} {rod_hk[1]}")

    # Save data
    lines.append(f"\nlist all {save_file} {comment}")

    # Add one more to avoid bogs
    lines.append("\nquit\n\n")

    # Create file
    print("Saving macro in file", save_folder + "/" + macro_file, end="\n\n")
    with open(save_folder + "/" + macro_file, "w") as m:
        for line in lines:
            m.write(line)

    # Run macro in folder
    os.system(f'cd {save_folder} && rod < {macro_file}')


def modify_surface_relaxation(
    base_file,
    save_as,
    lines_to_edit=[3],
    columns_to_edit=["z"],
    relaxation=0.99,
    round_order=3,
    sep=" ",
    print_old_file=False,
    print_new_file=True,
):
    """
    The files must use only one space between characters to split properly !!!

    :param base_file: file to edit
    :param save_as: save new file at this path
    :param lines_to_edit: list of lines to edit in the file
    :param columns_to_edit: list of columns to edit in the file,
     e.g. ["x", "y", "z"]
    :param relaxation: values are multipled by this float number
    :param round_order: to avoid weird float values, the relaxation is rounded
    :param sep: str, separator between the columns, e.g. " " is one space
    :param print_old_file: bool, True to see lines in old file
    :param print_new_file: bool, True to see lines in new file
    """
    # Open old file
    with open(base_file) as f:
        old_file_lines = f.readlines()

    # Print old file
    if print_old_file:
        print("############### Old surface file ###############\n")
        for line in old_file_lines:
            print(line, end="")

        print("\n############### Old surface file ###############")

    # Make copy
    new_file_lines = old_file_lines.copy()

    # Modify lines
    for l in lines_to_edit:

        # Split line
        try:
            line = old_file_lines[l].split(sep)
        except IndexError:
            print("l out of range, try to change lines_to_edit")

        # Modify parameter
        for c in columns_to_edit:
            c_index = {"x": 1, "y": 2, "z": 3}[c]
            line[c_index] = str(float(line[c_index]) *
                                np.round(relaxation, round_order))

        # Join line
        line = sep.join(line)

        # Bog when changing the last column
        if not line.endswith("\n"):
            line += "\n"

        # Save changes in new lines
        new_file_lines[l] = line

    # Print new file
    if print_new_file:
        print("\n############### New surface file ###############")
        print(f"################## r = {relaxation:.3f}  ##################\n")
        for line in new_file_lines:
            print(line, end="")

        print("\n############### New surface file ###############")

    # Create new file
    with open(save_as, "w") as f:
        f.writelines(new_file_lines)


def find_value_in_array(array, value):
    try:
        if all(array < value):
            return array[-1], -1
        elif all(array > value):
            return array[0], 0
        else:  # value in array
            mask, = np.where(array == value)
            if len(mask) != 1:
                print("There are multiple values in the array")
            else:
                return array[mask[0]], mask[0]
    except TypeError:
        print("Use a numerical value")
