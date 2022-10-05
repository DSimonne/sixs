"""
This module has functions to help with the analysis of reciprocal space data
during SXRD experiments at SixS.

The two main modules are the following:
    CTR()
    MAP()

There are also functions that help with the simulations in ROD:
    simulate_rod()
    modify_surface_relaxation()

Which version to use for binoculars ?
    * processing data: https://salsa.debian.org/science-team/binoculars
    * fitting data: https://github.com/picca/binoculars

NOTES ON BINOCULARS-FITAID:
The resolution parameter in L is very important and gives the range of values
which will be averaged during the HK projection calculations.
A smaller value will give a better resolution in the final integrated CTR curve,
but it may be that some of the voxels will not have any intensity assigned and
need to be interpolated. This is due to the resolution of the original scan
which might not have been sufficient (especially for high Q) and therefore not
every voxel in reciprocal space map has an assigned value (Drnec et al.,
J. Appl. Cryst., 47 (2014), 365-377).
You can either increase the resolution parameter or rely on the interpolation.
The best is to use a larger resolution directly in binoculars-process to avoid
such effects, and then use the same step when integrating the data.

The structure factor for each L can be determined by integrating the reflection
intensity in the slice. Fitaid determines the structure factors by:
* Fitting the reflection with the function chosen in the dropdown list below the
    slider. The fitting is also used for the determination of the center of the
    reflection.
* Interpolating the slice in the selected ROI and then integrating within the
    ROI with proper background subtraction.
* Integrating within the selected ROI without the interpolation.
Q: What is the range around the point used for interpolation ? 2D or 3D ?

The first step is to try to fit the data with the selected function so that the
peak tracker, which depends on the center of the reflection determined by the
fits, is activated.

The next step is to perform the integration of the reflections on each slice.
We first choose the ROI and background regions.
When the peak tracker tick box is selected, the center of the ROI is taken from
the fits obtained in the previous step. However, sometimes the fit fails and the
peak tracking is not working correctly. In this case the center of the ROI is
selected by unticking the peak tracker tick box and clicking on the slice in the
position of desired center of the ROI.
Selecting the center in one slice selects the same center for all other slices
within the same rod.

The background selection is important and it is a crucial part for obtaining
correct structure factors. Three values are calculated for establishing the
structure factors:

* sf (structure factor): all voxels with no value are interpolated with a scipy
    interpolate function (in the slice ?). Then, the sum is taken over all
    the values in the area indicated by the ROI.
    The background is calculated by taking the sum of all the values of selected
    background regions, corrected by the number of voxels.
    If we have:
        roi: the integrated intensity in the ROI
        #roi: the nb of voxels in the ROI
        bkg: the integrated background
        #roi: the nb of voxels in the background
    We have for the structure factor:
        sf = sqrt[roi - (#roi / #bkg) * bkg]
* nisf (no interpolation structure factor): the same calculation is performed as
    the sf except that it omits the interpolation of the empty voxels.
* fitsf: same calculation as with sf except that instead of the raw data, the
    result from the fit is used to do the calculation. It is numerically
    integrating the result of the fit and subtracting the background as selected
    by the ROI and the background regions.
    Q: What does it exactly ? Use the resulting parameter of the fit to create
    a peak and then integrate this peak ?

It is also a good idea to check for few L values if the reflections are within
the ROI and don't spill to the background.
For the rocking scans, the reciprocal space is much better sampled at low L and
the structure factor there is more reliable.

Plotting possibilities:
* sf (structure factor after interpolation)
* nisf (structure factor without interpolation)
* fitsf (structure factor from fit)
* I (intensity)


If nisf and fitsf are overlapping with the sf values at low L, it means that the
accuracy of the integration is good. If those three values are wildly different,
one needs to carefully analyse the slices to understand where the problem is.

The counters related to the fit are:
* loc: location of the peak of the fit.
* guessloc: value calculated from the results of the fit. It is a polynomial fit
    of degree two weighed by the variance of the fit of the loc values. guessloc
    is used as input for a new round of fitting. This is also the value used by
    the peaktracker in the integration widget. guessloc can be modified by
    clicking on the peak in the integration widget.
* gamma: related to the width of the peak
* slope: slope for a linear background with offset as offset.
* th: angle of the two axes if using the polar lorentzian fit function. All the
    var counters represent the variance of the corresponding fit parameters as
    calculated from the covariance returned by the fitting algorithm.

For details see:
https://github.com/id03/binoculars/wiki/
"""

import numpy as np
import tables as tb
import pandas as pd
import glob
import os
import inspect
import yaml
import sixs
import shutil

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.interpolate import splrep, splev


class Map:
    """
    Loads an hdf5 file created by binoculars that represents a 3D map of the
    reciprocal space and provides 2D plotting methods.

    Methods in class:
        * project_data(): projects the data on one axis, with a given range.
        * plot_map(): plot the data with matplotlib.
    """

    def __init__(self, file_path):
        """
        Loads the binoculars file.

        The binocular data is loaded as follow:
            * Divide counts by contribution where cont != 0
            * Swap the h and k axes to be consistent with the indexing
                [h, k, l], or [Qx, Qy, Qz].

        :param file_path: full path to .hdf5 file
        """

        self.file_path = file_path

        with tb.open_file(self.file_path) as f:
            # Get raw data
            ct = f.root.binoculars.counts[...]
            cont = f.root.binoculars.contributions[...]
            self.raw_data = np.where(cont != 0, ct/cont, np.nan)

            # Get which type of projection we are working with
            # HKL
            try:
                H = f.root.binoculars.axes.H[...]
                hkl = True
            except tb.NoSuchNodeError:
                hkl = False

            ## Qpar, Qper
            try:
                Qpar = f.root.binoculars.axes.Qpar[...]
                QparQper = True
            except tb.NoSuchNodeError:
                QparQper = False

            # Qindex
            try:
                Index = f.root.binoculars.axes.Index[...]
                Qindex = True
            except tb.NoSuchNodeError:
                Qindex = False

            # Qphi
            try:
                Index = f.root.binoculars.axes.Phi[...]
                QxQy = False  # also Qphi can have Qz (or Qx, Qy)
                Qphi = True
            except tb.NoSuchNodeError:
                QxQy = True
                Qphi = False

            if Qphi == False:  # also Qphi can have Qz (or Qx, Qy)
                try:
                    Qz = f.root.binoculars.axes.Qz[...]
                except tb.NoSuchNodeError:
                    QxQy = False

            if Qphi == True:
                self.data = self.raw_data
                self.Phi = f.root.binoculars.axes.Phi[...]
                self.Q = f.root.binoculars.axes.Q[...]
                try:
                    self.Qxyz = f.root.binoculars.axes.Qx[...]
                except:
                    pass
                try:
                    self.Qxyz = f.root.binoculars.axes.Qy[...]
                except:
                    pass
                try:
                    self.Qxyz = f.root.binoculars.axes.Qz[...]
                except:
                    pass

            # Load data
            if Qindex == True:
                self.data = self.raw_data
                self.index = f.root.binoculars.axes.Index[...]
                self.Q = f.root.binoculars.axes.Q[...]

            if hkl == True:
                self.data = np.swapaxes(self.raw_data, 0, 2)  # l, k, h
                self.H = f.root.binoculars.axes.H[...]
                self.K = f.root.binoculars.axes.K[...]
                self.L = f.root.binoculars.axes.L[...]

            if QxQy == True:
                self.data = self.raw_data
                self.Z = f.root.binoculars.axes.Qz[...]
                self.X = f.root.binoculars.axes.Qx[...]
                self.Y = f.root.binoculars.axes.Qy[...]

            elif QparQper == True:
                self.data = self.raw_data
                self.Y = f.root.binoculars.axes.Qper[...]
                self.X = f.root.binoculars.axes.Qpar[...]

            if Qphi == True:
                self.Q_axis = np.linspace(
                    self.Q[1], self.Q[2], 1+self.Q[5]-self.Q[4])
                self.Qxyz_axis = np.linspace(
                    self.Qxyz[1], self.Qxyz[2], 1+self.Qxyz[5]-self.Qxyz[4])
                self.Phi_axis = np.linspace(
                    self.Phi[1], self.Phi[2], 1+self.Phi[5]-self.Phi[4])

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
                self.Qx_axis = np.linspace(
                    self.X[1], self.X[2], 1 + int(self.X[5]-self.X[4]))
                self.Qy_axis = np.linspace(
                    self.Y[1], self.Y[2], 1 + int(self.Y[5]-self.Y[4]))
                self.Qy_axis = y_axis
                self.Qz_axis = np.linspace(
                    self.Z[1], self.Z[2], 1 + int(self.Z[5]-self.Z[4]))
                self.Qz_axis = z_axis

            if QparQper == True:
                self.Qpar = np.linspace(
                    self.X[1], self.X[2], 1+self.X[5]-self.X[4])
                self.Qper = np.linspace(
                    self.Y[1], self.Y[2], 1+self.Y[5]-self.Y[4])

            print(
                "\n############################################################"
                f"\nData shape: {self.data.shape}"
                f"\n\tHKL data: {hkl}"
                f"\n\tQxQy data: {QxQy}"
                f"\n\tQparQper data: {QparQper}"
                f"\n\tQphi data: {Qphi}"
                f"\n\tQindex: {Qindex}"
                f"\n###########################################################"
            )

    def project_data(
        self,
        projection_axis,
        projection_axis_range=[None, None],
    ):
        """
        Project the data on one of the measured axis, the result is saved as 
        attribute `.projected_data`.

        :param projection_axis: string in ("H", "K", "L", "Qx", "Qy", "Qz")
        :param axis_range_1: list or tuple of length two, defines the positions
            of the value to be used in the array on the desired axis, use [None,
            None] to use the whole range.
        """
        # Start with first axis
        self.projection_axis = projection_axis
        projection_axis_index = {
            "H": 2,
            "K": 1,
            "L": 0,
            "Qz": 2,
            "Qy": 1,
            "Qx": 0,
        }[self.projection_axis]

        projection_axis_name = {
            "H": "h_axis",
            "K": "k_axis",
            "L": "l_axis",
            "Qz": "Qz_axis",
            "Qy": "Qy_axis",
            "Qx": "Qx_axis",
        }[self.projection_axis]

        projection_axis_values = getattr(self, projection_axis_name)

        if projection_axis_range[0] != None:
            start_value = find_value_in_array(
                projection_axis_values,
                projection_axis_range[0]
            )[0]
        else:
            start_value = 0

        if projection_axis_range[1] != None:
            end_value = find_value_in_array(
                projection_axis_values,
                projection_axis_range[1]
            )[0]
        else:
            end_value = -1

        if self.projection_axis == 'H':
            datanan = self.data[:, :, start_value:end_value]

        elif self.projection_axis == 'K':
            datanan = self.data[:, start_value:end_value, :]

        elif self.projection_axis == 'L':
            datanan = self.data[start_value:end_value, :, :]

        elif self.projection_axis == 'Qz':
            # swap_data = np.swapaxes(self.raw_data, 0, 1)
            datanan = self.data[:, :, start_value:end_value]

        elif self.projection_axis == 'Qy':
            # swap_data = np.swapaxes(self.raw_data, 0, 2)
            datanan = self.data[:, start_value:end_value, :]

        elif self.projection_axis == 'Qx':
            # swap_data = np.swapaxes(self.raw_data, 1, 2)
            datanan = self.data[start_value:end_value, :, :]

        self.projected_data = np.nanmean(datanan, axis=projection_axis_index)

    def plot_map(
        self,
        zoom_axis1=[None, None],
        zoom_axis2=[None, None],
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

        You can use the command `%matplotlib notebook` to use a cursor in the 
        notebook cell (change figsize to (8,8)).

        :param zoom_axis1: container of length two, defines the zoom in the x 
         axis.
        :param zoom_axis2: container of length two, defines the zoom in the y 
         axis.
        :param interpolation: default is 'none'. See plt.imshow? for options,
            other options are 'nearest', 'gouraud', ...
        :param vmin: default is 0.1
        :param vmax: default is 2000
        :param figsize: default is (16, 9)
        :param title: figure title
        :param cmap: color map used, pick from
            https://matplotlib.org/stable/tutorials/colors/colormaps.html
        :param save_path: path to save file at
        :param three_d_plot: True to show a 3D plot
        """
        try:
            img = self.projected_data
        except AttributeError:
            return("Use the methods `project_data` to define the data first.")

        if self.projection_axis == 'H':
            axis1 = self.k_axis
            axis2 = self.l_axis
            axis_name1 = 'K (rlu)'
            axis_name2 = 'L (rlu)'

        elif self.projection_axis == 'K':
            axis1 = self.h_axis
            axis2 = self.l_axis
            axis_name1 = 'H (rlu)'
            axis_name2 = 'L (rlu)'

        elif self.projection_axis == 'L':
            axis1 = self.h_axis
            axis2 = self.k_axis
            axis_name1 = 'H (rlu)'
            axis_name2 = 'K (rlu)'

        elif self.projection_axis == 'Qxyz':
            axis1 = self.Q_axis
            axis2 = self.Phi_axis
            axis_name1 = 'Q'
            axis_name2 = 'Phi (deg)'

        elif self.projection_axis == 'Qx':
            axis1 = self.Qy_axis
            axis2 = self.Qz_axis
            axis_name1 = 'Qy'
            axis_name2 = 'Qz'

        elif self.projection_axis == 'Qy':
            axis1 = self.Qx_axis
            axis2 = self.Qz_axis
            axis_name1 = 'Qx'
            axis_name2 = 'Qz'

        elif self.projection_axis == 'Qz':
            axis1 = self.Qx_axis
            axis2 = self.Qy_axis
            axis_name1 = 'Qx'
            axis_name2 = 'Qy'

        # Zoom
        if zoom_axis1[0] != None:
            zoom_axis1[0] = find_value_in_array(axis1, zoom_axis1[0])[-1]
        if zoom_axis1[1] != None:
            zoom_axis1[1] = find_value_in_array(axis1, zoom_axis1[1])[-1]

        if zoom_axis2[0] != None:
            zoom_axis2[0] = find_value_in_array(axis2, zoom_axis2[0])[-1]
        if zoom_axis2[1] != None:
            zoom_axis2[1] = find_value_in_array(axis2, zoom_axis2[1])[-1]

        img = img[zoom_axis2[0]:zoom_axis2[1], zoom_axis1[0]:zoom_axis1[1]]
        axis1 = axis1[zoom_axis1[0]:zoom_axis1[1]]
        axis2 = axis2[zoom_axis2[0]:zoom_axis2[1]]

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
    Loads a hdf5 file created by binoculars that represents a 3D map of the
    reciprocal space and provides integration methods to analyse the diffracted
    intensity along one direction.

    The binocular data is loaded as follow:
        * Divide counts by contribution where cont != 0
        * Swap the h and k axes to be consistent with the indexing
            [h, k, l], or [Qx, Qy, Qz].

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
        * integrate_CTR(): integrate rods on the data with python
        * load_fitaid_data(): load the rods integrated via fitaid
        * load_ROD_data(): load the rods simulated with the ROD program

    All these functions create numpy arrays that can then be plotted with the
    plot_CTR() method.
    """

    def __init__(
        self,
        configuration_file=False,
    ):
        """
        Init the class with configuration file

        :param configuration_file: default is False. `.yml` file that stores
            metadata specific to the reaction, if False, defaults to
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
                print(
                    "Could not find configuration file."
                    "Defaulted to ammonia configuration."
                )

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
                print(
                    "Loaded configuration file."
                    "\n###########################################################\n"
                )

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
        Integrate 3D data along two directions with a specific ROI.

        The data is computed as follow:
            * the background intensity is computed from four regions of interest
                that are squares lying around the data region of interest
            * the background intensity is subtracted to each pixel in the data
                region of interest
            * the data is integrated in the ROI

        If we have:
            roi: the integrated intensity in the ROI
            #roi: the nb of voxels in the ROI
            bkg: the integrated background
            #roi: the nb of voxels in the background
        We have for the structure factor:
            sf = sqrt[roi - (#roi / #bkg) * bkg]

        Saves the result as a numpy array on disk:
            * dim 0: L
            * dim 1: Structure factor F(L) along the ROD, np.sqrt(I(L))
            * dim 2: Background removed from each pixel in the ROD ROI.

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
        :param center_background: list, node in reciprocal space around which
            the background is taken. If equal to HK_peak, the background width
            is added to the width of the CTR. If False, no background is
            subtracted.
        :param verbose: if True, print more details.
        """

        # Load data
        scan_indices = [str(s) for s in scan_indices]

        # Get all files first
        files = [f.split("/")[-1]
                 for f in sorted(glob.glob(f"{folder}/{glob_string_match}"))]

        # Get scans specified with scan_indices
        scan_files = [f for f in files if any([n in f for n in scan_indices])]

        if verbose:
            print(
                "\n###########################################################"
                f"\nDetected files in {folder}:"
            )
            for f in files:
                print("\t", f)

            print(
                "###########################################################\n"
                "\n###########################################################"
                "\nWorking on the following files:"
            )
            for f in scan_files:
                print("\t", f)
            print(
                "###########################################################\n")

        # ROD range
        CTR_range_H = [
            np.round(HK_peak[0] - CTR_width_H, 3),
            np.round(HK_peak[0] + CTR_width_H, 3)
        ]
        CTR_range_K = [
            np.round(HK_peak[1] - CTR_width_K, 3),
            np.round(HK_peak[1] + CTR_width_K, 3)
        ]

        print(
            "\n###########################################################"
            f"\nRange in H: [{CTR_range_H[0]} : {CTR_range_H[1]}]"
            f"\nRange in K: [{CTR_range_K[0]} : {CTR_range_K[1]}]"
            "\n###########################################################"
        )

        # Background range
        if center_background == HK_peak:
            background_range_H = [
                np.round(CTR_range_H[0] - background_width_H, 3),
                np.round(CTR_range_H[1] + background_width_H, 3)
            ]
            background_range_K = [
                np.round(CTR_range_K[0] - background_width_K, 3),
                np.round(CTR_range_K[1] + background_width_K, 3)
            ]

        elif isinstance(center_background, list) \
                and center_background != HK_peak:

            background_range_H = [
                center_background[0] - background_width_H,
                center_background[0] + background_width_H
            ]
            background_range_K = [
                center_background[1] - background_width_K,
                center_background[1] + background_width_K
            ]

        if verbose and isinstance(center_background, list):
            print(
                "\n###########################################################"
                f"\nBackground range in H: [{background_range_H[0]}"
                f" : {background_range_H[1]}]"
                f"\nBackground range in K: [{background_range_K[0]}"
                f" : {background_range_K[1]}]"
                "\n###########################################################"
            )

        # Start iterating on file to see the shape
        print(
            "\n###########################################################"
            "\nFinding smallest common range in L"
            "\nDepends on the config file in binoculars-process."
            "\n###########################################################"
        )

        for i, fname in enumerate(scan_files):
            with tb.open_file(folder + fname, "r") as f:
                H = f.root.binoculars.axes.H[:]
                K = f.root.binoculars.axes.K[:]
                L = f.root.binoculars.axes.L[:]

            if verbose:
                print(
                    "\n###########################################################"
                    f"\nOpening file {fname} ..."
                    f"\n\tAxis number, range and stepsize in H: [{H[0]:.3f}: {H[1]:.3f}: {H[2]:.3f}]"
                    f"\n\tAxis number, range and stepsize in K: [{K[0]:.3f}: {K[1]:.3f}: {K[2]:.3f}]"
                    f"\n\tAxis number, range and stepsize in L: [{L[0]:.3f}: {L[1]:.3f}: {L[2]:.3f}]"
                    "\n###########################################################"
                )

            if i == 0:
                l_min = np.round(L[1], 3)
                l_max = np.round(L[2], 3)
                l_length = 1 + int(L[5] - L[4])
            else:
                l_min = np.round(max(l_min, L[1]), 3)
                l_max = np.round(min(l_max, L[2]), 3)
                l_length = max(l_length, 1 + int(L[5] - L[4]))

        print(
            "\n###########################################################"
            f"\nSmallest common range in L is [{l_min} : {l_max}]"
            "\n###########################################################"
        )

        # Create new L axis for interpolation
        if isinstance(interpol_step, float):
            l_axis = np.arange(l_min, l_max, interpol_step)
            l_length = len(l_axis)

        # Save final data as numpy array
        # 0 is x axis, 1 is data, 2 is background (compatible with ROD)
        data = np.nan * np.empty((len(scan_files), 3, l_length))

        # Iterate on each file now to get the data
        for i, fname in enumerate(scan_files):
            if verbose:
                print(
                    "\n###########################################################"
                    f"\nOpening file {fname} ..."
                )

            with tb.open_file(folder + fname, "r") as f:
                ct = f.root.binoculars.counts[...]
                cont = f.root.binoculars.contributions[...]

                raw_data = np.where(cont != 0, ct/cont, np.nan)
                raw_data = np.swapaxes(raw_data, 0, 2)  # originally l, k, h

                H = f.root.binoculars.axes.H[:]
                K = f.root.binoculars.axes.K[:]
                L = f.root.binoculars.axes.L[:]

            scan_h_axis = np.round(np.linspace(
                H[1], H[2], 1 + int(H[5] - H[4])), 3)
            scan_k_axis = np.round(np.linspace(
                K[1], K[2], 1 + int(K[5] - K[4])), 3)
            scan_l_axis = np.round(np.linspace(
                L[1], L[2], 1 + int(L[5] - L[4])), 3)

            # Define ROI indices
            start_H_ROI = find_value_in_array(scan_h_axis, CTR_range_H[0])
            end_H_ROI = find_value_in_array(scan_h_axis, CTR_range_H[1])
            start_K_ROI = find_value_in_array(scan_k_axis, CTR_range_K[0])
            end_K_ROI = find_value_in_array(scan_k_axis, CTR_range_K[1])

            if verbose:
                print(
                    f"Data ROI (H, K): [{start_H_ROI[0]}, {end_H_ROI[0]}, "
                    f"{start_K_ROI[0]}, {end_K_ROI[0]}] ; [{start_H_ROI[1]}, "
                    f"{end_H_ROI[1]}, {start_K_ROI[1]}, {end_K_ROI[1]}]"
                )

            # Get data only in specific ROI
            ROI_2D = raw_data[:,
                              start_K_ROI[1]:end_K_ROI[1],
                              start_H_ROI[1]:end_H_ROI[1],
                              ]
            self.ROI_2D = ROI_2D

            # Integrate the data in the ROI, replace nan by zeroes otherwise
            # the total is equal to np.nan
            intensity = np.sum(np.nan_to_num(ROI_2D), axis=(1, 2))

            # Count number of non-np.nan pixels in the ROI
            # These pixels do not have an intensity (!= from zero intensity),
            # they were not recorded.
            roi_pixel_count = np.sum(~np.isnan(ROI_2D), axis=(1, 2))

            # Compute background
            if center_background == HK_peak:
                # Define background ROIs indices
                start_H_background = find_value_in_array(
                    scan_h_axis,
                    background_range_H[0]
                )
                end_H_background = find_value_in_array(
                    scan_h_axis,
                    background_range_H[1]
                )

                start_K_background = find_value_in_array(
                    scan_k_axis,
                    background_range_K[0]
                )
                end_K_background = find_value_in_array(
                    scan_k_axis,
                    background_range_K[1]
                )

                if verbose:
                    print(
                        f"Background ROI (H, K): [{start_H_background[0]}, "
                        f"{end_H_background[0]}, {start_K_background[0]}, "
                        f"{end_K_background[0]}] ; [{start_H_background[1]}, "
                        f"{end_H_background[1]}, {start_K_background[1]}, "
                        f"{end_K_background[1]}]"
                        "\n###########################################################"
                    )

                # Define background ROIs
                background_ROI_0 = raw_data[
                    :,
                    start_K_ROI[1]:end_K_ROI[1],
                    start_H_background[1]:start_H_ROI[1],
                ]

                background_ROI_1 = raw_data[
                    :,
                    start_K_ROI[1]:end_K_ROI[1],
                    end_H_ROI[1]:end_H_background[1],
                ]

                background_ROI_2 = raw_data[
                    :,
                    start_K_background[1]:start_K_ROI[1],
                    start_H_ROI[1]:end_H_ROI[1],
                ]

                background_ROI_3 = raw_data[
                    :,
                    end_K_ROI[1]:end_K_background[1],
                    start_H_ROI[1]:end_H_ROI[1],
                ]

                # Integrate the data in the ROIs, replace nan by zeroes
                # otherwise the total is equal to np.nan
                background_values = \
                    np.sum(np.nan_to_num(background_ROI_0), axis=(1, 2)) + \
                    np.sum(np.nan_to_num(background_ROI_1), axis=(1, 2)) + \
                    np.sum(np.nan_to_num(background_ROI_2), axis=(1, 2)) + \
                    np.sum(np.nan_to_num(background_ROI_3), axis=(1, 2))

                # Count number of non-np.nan pixels in the background
                # These pixels do not # have an intensity (!= from zero
                # intensity), they were not recorded.
                background_pixel_count = \
                    np.sum(~np.isnan(background_ROI_0), axis=(1, 2)) + \
                    np.sum(~np.isnan(background_ROI_1), axis=(1, 2)) + \
                    np.sum(~np.isnan(background_ROI_2), axis=(1, 2)) + \
                    np.sum(~np.isnan(background_ROI_3), axis=(1, 2))

                # Remove background
                structure_factor = np.nan_to_num(np.where(
                    background_pixel_count > 0,
                    np.sqrt(
                        intensity - ((roi_pixel_count / background_pixel_count) * background_values)),
                    0
                ))

            elif isinstance(center_background, list) \
                    and center_background != HK_peak:
                # Background intensity, define ROI indices
                start_H_background = find_value_in_array(
                    scan_h_axis,
                    background_range_H[0]
                )
                end_H_background = find_value_in_array(
                    scan_h_axis,
                    background_range_H[1]
                )

                start_K_background = find_value_in_array(
                    scan_k_axis,
                    background_range_K[0]
                )
                end_K_background = find_value_in_array(
                    scan_k_axis,
                    background_range_K[1]
                )

                if verbose:
                    print(
                        f"Background ROI (H, K): [{start_H_background[0]}, "
                        f"{end_H_background[0]}, {start_K_background[0]}, "
                        f"{end_K_background[0]}] ; [{start_H_background[1]}, "
                        f"{end_H_background[1]}, {start_K_background[1]}, "
                        f"{end_K_background[1]}]"
                        "\n###########################################################"
                    )

                # Define the background ROI
                background_ROI = raw_data[
                    :,
                    start_K_background[1]:end_K_background[1],
                    start_H_background[1]:end_H_background[1],
                ]

                background_values = np.sum(background_ROI, axis=(1, 2))

                # Count number of non-np.nan pixels in the background
                # These pixels do not # have an intensity (!= from zero
                # intensity), they were not recorded.
                background_pixel_count = np.sum(
                    ~np.isnan(background_ROI_0) + ~np.isnan(background_ROI_1) +
                    ~np.isnan(background_ROI_2) + ~np.isnan(background_ROI_3)
                )

                # Remove background
                structure_factor = np.nan_to_num(np.where(
                    background_pixel_count > 0,
                    np.sqrt(
                        intensity - ((roi_pixel_count / background_pixel_count) * background_values)),
                    0
                ))

            else:
                structure_factor = np.sqrt(intensity)

            # Save data
            if isinstance(interpol_step, float):
                # Save x axis
                data[i, 0, :] = l_axis

                # Save structure factor
                tck = splrep(scan_l_axis, structure_factor)
                ROI_2D_sum = splev(l_axis, tck)
                data[i, 1, :] = ROI_2D_sum

            else:
                # Save x axis
                data[i, 0, :len(scan_l_axis)] = scan_l_axis

                # Save structure factor
                data[i, 1, :len(scan_l_axis)] = structure_factor

            # Save background
            try:
                if isinstance(interpol_step, float):
                    tck = splrep(scan_l_axis, background_values /
                                 background_pixel_count)
                    background = splev(l_axis, tck)

                    data[i, 2, :] = background
                else:
                    data[i, 2, :len(scan_l_axis)] = background

            except NameError:
                print(
                    "No background subtracted"
                    "\n###########################################################"
                )

            if verbose:
                # Resume with a plot
                img = np.sum(np.nan_to_num(raw_data),
                             axis=0)  # sum over L axis
                # need to flip to have a plot coherent with binoculars
                img = np.flip(img, axis=0)

                plt.figure(figsize=(8, 8))
                plt.imshow(
                    img,
                    norm=LogNorm(),
                    cmap="jet",
                    extent=(H[1], H[2], K[1], K[2]),
                )
                plt.xlabel("H", fontsize=15)
                plt.ylabel("K", fontsize=15)
                plt.title("Intensity summed over L", fontsize=15)

                # Plot data ROI
                plt.axvline(
                    x=start_H_ROI[0],
                    color='red',
                    linestyle="--"
                )
                plt.axvline(
                    x=end_H_ROI[0],
                    color='red',
                    linestyle="--"
                )

                plt.axhline(
                    y=start_K_ROI[0],
                    color='red',
                    linestyle="--"
                )
                plt.axhline(
                    y=end_K_ROI[0],
                    color='red',
                    linestyle="--",
                    label="ROI"
                )

                if center_background != False:
                    # Plot background ROI
                    plt.axvline(
                        x=start_H_background[0],
                        color='blue',
                        linestyle="--"
                    )
                    plt.axvline(
                        x=end_H_background[0],
                        color='blue',
                        linestyle="--"
                    )

                    plt.axhline(
                        y=start_K_background[0],
                        color='blue',
                        linestyle="--"
                    )
                    plt.axhline(
                        y=end_K_background[0],
                        color='blue',
                        linestyle="--",
                        label="Background"
                    )

                # Legend
                plt.legend()
                plt.colorbar()
                plt.show()
                plt.close()

        # Saving
        print(
            "\n###########################################################"
            f"\nSaving data as: {folder}{save_name}.npy"
            "\n###########################################################"
        )
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
        Load CTR integrated via binoculars-fitaid, two columns data with L
        and the absolute structure factor values.

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
        scan_files = [f for f in files if any(
            [n in f for n in scan_indices])]
        if verbose:
            print(
                "\n###########################################################"
                f"\nDetected files in {folder}:"
            )
            for f in files:
                print("\t", f)
            print("###########################################################\n")

            print(
                "\n###########################################################"
                "\nWorking on the following files:"
            )
            for f in scan_files:
                print("\t", f)
            print("###########################################################\n")

        # Iterating on all files to create l axis
        for i, fname in enumerate(scan_files):
            # Load data
            fitaid_data = np.loadtxt(folder + fname)

            # L axis
            L = fitaid_data[:, 0]

            if verbose:
                print(
                    "\n###########################################################"
                    f"\nOpening file {fname} ..."
                    f"\n\tRange and stepsize in L: [{min(L):.3f}: {max(L):.3f}: {len(L):.3f}]"
                    "\n###########################################################"
                )

            if i == 0:
                l_min = np.round(min(L), 3)
                l_max = np.round(max(L), 3)
                l_length = len(L)
            else:
                l_min = np.round(max(l_min, min(L)), 3)
                l_max = np.round(min(l_max, max(L)), 3)
                l_length = max(l_length, len(L))

        print(
            "\n###########################################################"
            f"\nSmallest common range in L is [{l_min} : {l_max}]"
            "\n###########################################################"
        )

        # Create new x axis for interpolation
        if isinstance(interpol_step, float):
            l_axis = np.arange(l_min, l_max, interpol_step)
            l_length = len(l_axis)

        # Save final data as numpy array
        # 0 is x axis, 1 is data, 2 is background
        data = np.nan * np.empty((len(scan_files), 3, l_length))

        # Background already subtracted
        # Get l axis and CTR intensity for each file
        for i, fname in enumerate(scan_files):
            # Load data
            fitaid_data = np.loadtxt(folder + fname)
            scan_l_axis = fitaid_data[:, 0]
            ctr_data = fitaid_data[:, 1]

            # Interpolate
            if isinstance(interpol_step, float):
                data[i, 0, :] = l_axis

                tck = splrep(scan_l_axis, ctr_data, s=0)
                data[i, 1, :] = splev(l_axis, tck)

            else:
                data[i, 0, :len(scan_l_axis)] = scan_l_axis
                data[i, 1, :len(scan_l_axis)] = ctr_data

        # Saving
        print(
            "\n###########################################################"
            f"\nSaving data as: {folder}{save_name}.npy"
            "\n###########################################################"
        )
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
        :param data_column: index of column to plot
        :param glob_string_match: string used in glob matching of files
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
        scan_files = [f for f in files if any(
            [n in f for n in scan_indices])]
        if verbose:
            print(
                "\n###########################################################"
                f"\nDetected files in {folder}:"
            )
            for f in files:
                print("\t", f)
            print("###########################################################\n")

            print(
                "\n###########################################################"
                "\nWorking on the following files:"
            )
            for f in scan_files:
                print("\t", f)
            print("###########################################################\n")

        # Iterating on all files to create l axis
        for i, fname in enumerate(scan_files):
            # Load data
            rod_data = np.loadtxt(folder + fname, skiprows=2)

            # L axis
            L = rod_data[:, 2]

            if verbose:
                print(
                    "\n###########################################################"
                    f"\nOpening file {fname} ..."
                    f"\n\tRange and stepsize in L: [{min(L):.3f}: {max(L):.3f}: {len(L):.3f}]"
                    "\n###########################################################"
                )

            if i == 0:
                l_min = np.round(min(L), 3)
                l_max = np.round(max(L), 3)
                l_length = len(L)
            else:
                l_min = np.round(max(l_min, min(L)), 3)
                l_max = np.round(min(l_max, max(L)), 3)
                l_length = max(l_length, len(L))

        print(
            "\n###########################################################"
            f"\nSmallest common range in L is [{l_min} : {l_max}]"
            "###########################################################"
        )

        # Create new x axis for interpolation
        if isinstance(interpol_step, float):
            l_axis = np.arange(l_min, l_max, interpol_step)
            l_length = len(l_axis)

        # Save final data as numpy array
        # 0 is x axis, 1 is data, 2 is background
        data = np.nan * np.empty((len(scan_files), 3, l_length))

        # Get l axis and CTR intensity for each file
        for i, fname in enumerate(scan_files):

            # Load data
            rod_data = np.loadtxt(folder + fname, skiprows=2)
            scan_l_axis = rod_data[:, 2]
            ctr_data = rod_data[:, data_column]

            # Interpolate
            if isinstance(interpol_step, float):
                data[i, 0, :] = l_axis

                tck = splrep(scan_l_axis, ctr_data, s=0)
                data[i, 1, :] = splev(l_axis, tck)

            else:
                data[i, 0, :len(scan_l_axis)] = scan_l_axis
                data[i, 1, :len(scan_l_axis)] = ctr_data

        # Saving
        print(
            "\n###########################################################"
            f"\nSaving data as: {folder}{save_name}.npy"
            "###########################################################"
        )
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
            - l
            - data - background
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
            y_plot = arr[1, :]  # data - background

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
        print(
            "\n############### New surface file ###############"
            f"################## r = {relaxation:.3f}  ##################\n"
        )
        for line in new_file_lines:
            print(line, end="")

        print("\n############### New surface file ###############")

    # Create new file
    with open(save_as, "w") as f:
        f.writelines(new_file_lines)


def find_value_in_array(array, value):
    array = np.round(array, 4)  # bc of float errors
    try:
        if all(array < value):
            return array[-1], -1
        elif all(array > value):
            return array[0], 0
        else:  # value in array
            mask, = np.where(array == value)
            if len(mask) > 1:
                print("There are multiple values in the array")
            elif len(mask) == 0:
                print("Value not in array")
            else:
                return array[mask[0]], mask[0]
    except TypeError:
        print("Use a numerical value")
