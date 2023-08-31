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
need to be interpolated.
This is due to the resolution of the original scan
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
from sixs.scripts.print_binoculars_axes import print_binocular_axes
import shutil
from h5glance import H5Glance
from IPython.display import display, clear_output
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import patches

import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, Layout

from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Legend, RangeTool, HoverTool, WheelZoomTool, CrosshairTool
from bokeh.plotting import figure, show
from bokeh.io import output_notebook, export_png

output_notebook()


class Map:
    """
    Loads an hdf5 file created by binoculars that represents a 3D map of the
    reciprocal space and provides 2D plotting methods.

    Methods in class:
        * project_data(): projects the data on one axis, with a given range.
        * plot_map(): plot the data with matplotlib.
        * view_space(): interactive plot with widgets
    """

    def __init__(
        self,
        file_path,
        show_tree=False,
        verbose=False,
    ):
        """
        Loads the data arrays from the binoculars file.

        Two arrays are loaded:
            * `ct`: counts, total intensity count in a region of reciprocal 
                space 
            * `cont`: contribution, number of voxels that contributed to 
                this region in reciprocal space

        :param file_path: full path to .hdf5 file
        :param show_tree: True to display the hdf5 tree in a Jupyter Notebook
        :param verbose: True to print informations about the file
        """

        self.file_path = file_path

        if show_tree:
            display(H5Glance(file_path))

        try:
            if verbose:
                print_binocular_axes(self.file_path)
        except AttributeError:
            print("Data type not supported")

        with tb.open_file(self.file_path) as f:
            # Get raw data
            self.ct = f.root.binoculars.counts[...]
            self.cont = f.root.binoculars.contributions[...]

            ####################### 1) Get projection type #######################
            # Qphi
            try:
                Phi = f.root.binoculars.axes.Phi[...]
                Qphi = True
            except tb.NoSuchNodeError:
                Qphi = False

            # Qindex
            try:
                Index = f.root.binoculars.axes.Index[...]
                Qindex = True
            except tb.NoSuchNodeError:
                Qindex = False

            # HKL
            try:
                H = f.root.binoculars.axes.H[...]
                hkl = True
            except tb.NoSuchNodeError:
                hkl = False

            # qxqyqz
            try:
                qx = f.root.binoculars.axes.qx[...]
                qy = f.root.binoculars.axes.qy[...]
                qxqy = True
            except tb.NoSuchNodeError:
                try:
                    qx = f.root.binoculars.axes.Qx[...]
                    qy = f.root.binoculars.axes.Qy[...]
                    qxqy = True
                except tb.NoSuchNodeError:
                    qxqy = False

            # Q, xp, yp
            try:
                Q = f.root.binoculars.axes.Q[...]
                xp = f.root.binoculars.axes.xp[...]
                yp = f.root.binoculars.axes.yp[...]
                qxpYp = True
            except tb.NoSuchNodeError:
                qxpYp = False

            # Qpar, Qper
            try:
                Qpar = f.root.binoculars.axes.Qpar[...]
                QparQper = True
            except tb.NoSuchNodeError:
                QparQper = False

            # Angles
            try:
                delta = f.root.binoculars.axes.delta[...]
                Angles = True
            except tb.NoSuchNodeError:
                Angles = False

            ############################ 2) Load data ############################
            if Qphi:  # also Qphi can have qz (or qx, qy)
                self.Phi = f.root.binoculars.axes.Phi[...]
                self.Q = f.root.binoculars.axes.Q[...]
                try:  # one of the three
                    self.qxyz = f.root.binoculars.axes.qx[...]
                except:
                    try:
                        self.qxyz = f.root.binoculars.axes.Qx[...]
                    except:
                        pass
                try:
                    self.qxyz = f.root.binoculars.axes.qy[...]
                except:
                    try:
                        self.qxyz = f.root.binoculars.axes.Qy[...]
                    except:
                        pass
                try:
                    self.qxyz = f.root.binoculars.axes.qz[...]
                except:
                    try:
                        self.qxyz = f.root.binoculars.axes.Qz[...]
                    except:
                        pass

            elif Qindex:
                self.Index = f.root.binoculars.axes.Index[...]
                self.Q = f.root.binoculars.axes.Q[...]

            elif hkl:
                self.ct = np.swapaxes(self.ct, 0, 2)  # l, k, h
                self.cont = np.swapaxes(self.cont, 0, 2)  # l, k, h
                self.H = f.root.binoculars.axes.H[...]
                self.K = f.root.binoculars.axes.K[...]
                self.L = f.root.binoculars.axes.L[...]

            elif qxqy:
                self.ct = np.swapaxes(self.ct, 0, 2)  # qz, qy, qx
                self.cont = np.swapaxes(self.cont, 0, 2)  # qz, qy, qx
                try:
                    self.qz = f.root.binoculars.axes.qz[...]
                    self.qx = f.root.binoculars.axes.qx[...]
                    self.qy = f.root.binoculars.axes.qy[...]
                except:
                    self.qz = f.root.binoculars.axes.Qz[...]
                    self.qx = f.root.binoculars.axes.Qx[...]
                    self.qy = f.root.binoculars.axes.Qy[...]

            elif qxpYp:
                self.Q = f.root.binoculars.axes.Q[...]
                self.xp = f.root.binoculars.axes.xp[...]
                self.yp = f.root.binoculars.axes.yp[...]

            elif QparQper:
                self.Qper = f.root.binoculars.axes.Qper[...]
                self.Qpar = f.root.binoculars.axes.Qpar[...]

            elif Angles:
                self.delta = f.root.binoculars.axes.delta[...]
                self.gamma = f.root.binoculars.axes.gamma[...]
                try:
                    self.mu = f.root.binoculars.axes.mu[...]
                except tb.NoSuchNodeError:
                    # omega scan
                    self.omega = f.root.binoculars.axes.omega[...]

            ########################### 3) Update axes ###########################
            if Qphi:
                self.Q_axis = np.linspace(
                    self.Q[1], self.Q[2], 1+self.Q[5]-self.Q[4])
                self.qxyz_axis = np.linspace(
                    self.qxyz[1], self.qxyz[2], 1+self.qxyz[5]-self.qxyz[4])
                self.Phi_axis = np.linspace(
                    self.Phi[1], self.Phi[2], 1+self.Phi[5]-self.Phi[4])

            elif Qindex:
                self.Q_axis = np.linspace(
                    self.Q[1], self.Q[2], 1+self.Q[5]-self.Q[4])
                self.Index_axis = np.linspace(
                    self.Index[1], self.Index[2], 1+self.Index[5]-self.Index[4])

            elif hkl:
                self.H_axis = np.round(np.linspace(
                    self.H[1], self.H[2], 1 + int(self.H[5] - self.H[4])), 3)
                self.K_axis = np.round(np.linspace(
                    self.K[1], self.K[2], 1 + int(self.K[5] - self.K[4])), 3)
                self.L_axis = np.round(np.linspace(
                    self.L[1], self.L[2], 1 + int(self.L[5] - self.L[4])), 3)

            elif qxqy:
                self.qz_axis = np.linspace(
                    self.qz[1], self.qz[2], 1 + int(self.qz[5]-self.qz[4]))
                self.qx_axis = np.linspace(
                    self.qx[1], self.qx[2], 1 + int(self.qx[5]-self.qx[4]))
                self.qy_axis = np.linspace(
                    self.qy[1], self.qy[2], 1 + int(self.qy[5]-self.qy[4]))

            elif qxpYp:
                self.Q_axis = np.linspace(
                    self.Q[1], self.Q[2], 1 + int(self.Q[5]-self.Q[4]))
                self.xp_axis = np.linspace(
                    self.xp[1], self.xp[2], 1 + int(self.xp[5]-self.xp[4]))
                self.yp_axis = np.linspace(
                    self.yp[1], self.yp[2], 1 + int(self.yp[5]-self.yp[4]))

            elif QparQper:
                self.Qper_axis = np.linspace(
                    self.Qper[1], self.Qper[2], 1+self.Qper[5]-self.Qper[4])
                self.Qpar_axis = np.linspace(
                    self.Qpar[1], self.Qpar[2], 1+self.Qpar[5]-self.Qpar[4])

            elif Angles:
                self.delta_axis = np.round(np.linspace(
                    self.delta[1], self.delta[2], 1 + int(self.delta[5] - self.delta[4])), 3)
                self.gamma_axis = np.round(np.linspace(
                    self.gamma[1], self.gamma[2], 1 + int(self.gamma[5] - self.gamma[4])), 3)
                try:
                    self.mu_axis = np.round(np.linspace(
                        self.mu[1], self.mu[2], 1 + int(self.mu[5] - self.mu[4])), 3)
                except AttributeError:
                    self.omega_axis = np.round(np.linspace(
                        self.omega[1], self.omega[2], 1 + int(self.omega[5] - self.omega[4])), 3)

    def project_data(
        self,
        projection_axis,
        projection_axis_range=[None, None],
    ):
        """
        Project the data on one of the measured axis, the result is saved as 
        a numpy.array() attribute :`projected_data`.

        The projection is done by FIRST summing the counts and contribution over
        the defined range and THEN by dividing the summed counts by the summed
        contribution.

        Be very careful of the selected range to not drown signal. Especially in 
        L.

        In general, the binning should be done in binoculars process.

        This is exactly the same as what is performed in binoculars for the 
        version installed at SixS: `0.0.11-1~bpo11+1soleil1`.

        If you want to save the data manually, you can use :

        ```python
        import numpy as np
        np.save("my_data.npy", Map.projected_data)
        ```

        For ASCII file, use np.savetxt instead.

        :param projection_axis: string in ("H", "K", "L", "qx", "qy", "qz")
        :param axis_range_1: list or tuple of length two, defines the positions
            of the value to be used in the array on the desired axis, use [None,
            None] to use the whole range.
        """
        # Get axis
        self.projection_axis = projection_axis
        projection_axis_index = {
            "H": 2,
            "K": 1,
            "L": 0,
            "qx": 2,
            "qy": 1,
            "qz": 0,
            "delta": 0,
            "gamma": 1,
            "mu": 2,
            "omega": 2,
        }[self.projection_axis]

        projection_axis_name = {
            "H": "H_axis",
            "K": "K_axis",
            "L": "L_axis",
            "qx": "qx_axis",
            "qy": "qy_axis",
            "qz": "qz_axis",
            "delta": "delta_axis",
            "gamma": "gamma_axis",
            "mu": "mu_axis",
            "omega": "omega_axis",
        }[self.projection_axis]

        projection_axis_values = getattr(self, projection_axis_name)

        # Get start and end indices
        if projection_axis_range[0] != None:
            start_index = find_value_in_array(
                projection_axis_values,
                projection_axis_range[0]
            )[1]
        else:
            start_index = 0

        if projection_axis_range[1] != None:
            end_index = find_value_in_array(
                projection_axis_values,
                projection_axis_range[1]
            )[1]
        else:
            end_index = len(projection_axis_values)-1

        # Only take values that are within the axis range
        if self.projection_axis in ('H', "qx", "omega", "mu"):
            sliced_ct = self.ct[:, :, start_index:end_index+1]
            sliced_cont = self.cont[:, :, start_index:end_index+1]

        elif self.projection_axis in ('K', "qy", "gamma"):
            sliced_ct = self.ct[:, start_index:end_index+1, :]
            sliced_cont = self.cont[:, start_index:end_index+1, :]

        elif self.projection_axis in ('L', "qz", "delta"):
            sliced_ct = self.ct[start_index:end_index+1, :, :]
            sliced_cont = self.cont[start_index:end_index+1, :, :]

        # Sum the ct and cont
        summed_ct = np.sum(sliced_ct, axis=projection_axis_index)
        summed_cont = np.sum(sliced_cont, axis=projection_axis_index)

        # Compute the final data over the selected range
        self.projected_data = np.where(
            summed_cont != 0, summed_ct/summed_cont, np.nan)

    def plot_map(
        self,
        zoom_axis1=[None, None],
        zoom_axis2=[None, None],
        interpolation="none",
        vmin=None,
        vmax=None,
        figsize=(10, 8),
        title=None,
        cmap="jet",
        three_d_plot=False,
        circles=None,
        arcs=False,
        lines=False,
        grid=True,
        save_path=False,
        x_labels_rotation=None,
        y_labels_rotation=90,
        x_ticks_rotation=None,
        y_ticks_rotation=None,
        text=False,
    ):
        """
        Plot/save a hdf5 map.

        Saves the zoomed image as `img` attribute.

        You can use the command `%matplotlib notebook` to use a cursor in the 
        notebook cell (change figsize to (8,8)).

        :param zoom_axis1: container of length two, defines the zoom in the x 
         axis.
        :param zoom_axis2: container of length two, defines the zoom in the y 
         axis.
        :param interpolation: default is 'none'. See plt.imshow? for options,
            other options are 'nearest', 'gouraud', ...
        :param vmin: default is 0.1, you can also use None
        :param vmax: default is max, you can also use None
        :param figsize: default is (16, 9)
        :param title: figure title
        :param cmap: color map used, pick from
            https://matplotlib.org/stable/tutorials/colors/colormaps.html
        :param three_d_plot: True to show a 3D plot
        :param circles: list of tuples of length 5 that follows:
            (x, y, radius, color, alpha, linewidth)
            e.g.: [(1, 1, 0.1, 'r', 0.5, 2),]
        :param arcs: list of tuples of length 9 that follows:
            (x, y, width, height, rotation_angle, theta1, theta2, color, alpha)
            e.g.: [(0, 0, 1, 1, 0 ,270, 360, "r", 0.8),]
        :param lines: list of tuples of length 7 that follows:
            (x1, y1, x2, y2, color, linestyle, linewidth, alpha),
            e.g.: [(0, 0, 1, 1, 'r', "--", 1, 0.5)]
        :param grid: True to show a grid
        :param save_path: path to save file
        :param x_labels_rotation:
        :param y_labels_rotation:
        :param x_ticks_rotation:
        :param y_ticks_rotation:
        :param text: use to put a label on the image, tuple of length 4
            that follows (x, y, string, fontsize) e.g., (1, 1, a), 15)
        """
        try:
            img = self.projected_data
        except AttributeError:
            return ("Use the methods `project_data` to define the data first.")

        axis1, axis2, axis_name1, axis_name2 = self._get_axes()

        # Zoom
        if zoom_axis1[0] != None:
            zoom_axis1[0] = find_value_in_array(axis1, zoom_axis1[0])[-1]
        if zoom_axis1[1] != None:
            zoom_axis1[1] = find_value_in_array(axis1, zoom_axis1[1])[-1]

        if zoom_axis2[0] != None:
            zoom_axis2[0] = find_value_in_array(axis2, zoom_axis2[0])[-1]
        if zoom_axis2[1] != None:
            zoom_axis2[1] = find_value_in_array(axis2, zoom_axis2[1])[-1]

        self.img = img[zoom_axis2[0]:zoom_axis2[1],
                       zoom_axis1[0]:zoom_axis1[1]]
        if self.img.shape == (0, 0):
            raise ValueError("Try zoom_axis = [b, a] instead of [a, b]")
        axis1 = axis1[zoom_axis1[0]:zoom_axis1[1]]
        axis2 = axis2[zoom_axis2[0]:zoom_axis2[1]]

        # Plot
        if three_d_plot:
            X, Y = np.meshgrid(axis1, axis2)
            if vmin is None:
                vmin = 0.1
            Z = np.where(self.img > vmin, np.log(self.img), 0)

            fig, ax = plt.subplots(
                figsize=figsize,
                subplot_kw={'projection': '3d'}
            )
            plotted_img = ax.plot_surface(
                X,
                Y,
                Z,
                cmap=cmap,
            )

        else:
            fig, ax = plt.subplots(figsize=figsize)

            # Circles
            if isinstance(circles, list):
                circles_patches = [
                    patches.Circle(
                        xy=(x, y),
                        radius=r,
                        color=c,
                        alpha=al,
                        fill=False,
                        lw=lw,
                    )
                    for (x, y, r, c, al, lw) in circles
                ]

                for cp in circles_patches:
                    ax.add_patch(cp)

            # Lines
            if isinstance(lines, list):
                for (x1, y1, x2, y2, c, ls, ln, al) in lines:
                    ax.plot([x1, x2], [y1, y2], color=c,
                            linestyle=ls, alpha=al, linewidth=ln)

            # Arc
            if isinstance(arcs, list):
                arc_patches = [
                    patches.Arc(
                        xy=(x, y),
                        width=w,
                        height=h,
                        angle=ang,
                        theta1=t1,
                        theta2=t2,
                        color=c,
                        alpha=al,
                    )
                    for (x, y, w, h, ang, t1, t2, c, al) in arcs
                ]

                for ap in arc_patches:
                    ax.add_patch(ap)

            # Grid
            if grid:
                ax.grid(alpha=0.5, which='both', axis="both")

            plotted_img = ax.imshow(
                self.img,
                cmap=cmap,
                interpolation=interpolation,
                origin="lower",
                norm=LogNorm(vmin=vmin, vmax=vmax),
                extent=[axis1.min(), axis1.max(), axis2.min(), axis2.max()]
            )

        if text:
            ax.text(text[0], text[1], text[2], fontsize=text[3], color=text[4])

        # Labels and ticks
        ax.set_xlabel(axis_name1, fontsize=20, rotation=x_labels_rotation)
        ax.set_ylabel(axis_name2, fontsize=20, rotation=y_labels_rotation)
        ax.tick_params(axis=('x'), labelsize=20,
                       labelrotation=x_ticks_rotation)
        ax.tick_params(axis=('y'), labelsize=20,
                       labelrotation=y_ticks_rotation)

        # Colorbar
        try:
            cbar = fig.colorbar(plotted_img, ax=ax)
            cbar.ax.tick_params(labelsize=20)
        except ValueError:
            print("Could not display colorbar, change scale values.")
            pass

        fig.tight_layout()

        if isinstance(title, str):
            ax.set_title(title, fontsize=20)

        if save_path:
            plt.savefig(save_path)

        try:
            plt.show()
        except ValueError:
            pass

    def view_space(
        self,
        projection_axis,
        figsize=(10, 10),
        cmap="jet",
    ):
        """
        """

        # Choose the first axis
        axis = getattr(self, f"{projection_axis}_axis")

        # Interact with widgets only to zoom in the first axis
        @ interact(
            projection_axis_range=widgets.FloatRangeSlider(
                value=[axis[0], axis[-2]],
                min=min(axis[:-1]),
                max=max(axis[:-1]),
                step=np.mean(axis[1:] - axis[:-1]),
                description='Projection axis range:',
                continuous_update=False,
                orientation='horizontal',
                readout=True,
                readout_format='.3f',
                layout=Layout(width="50%"),
                style={'description_width': 'initial'},
            ),
            projection_axis=fixed(projection_axis),
            figsize=fixed(figsize),
            cmap=fixed(cmap),
        )
        def change_file_and_axis(
            projection_axis_range,
            projection_axis,
            figsize,
            cmap,
        ):
            """Update the space range"""
            self.project_data(
                projection_axis_range=projection_axis_range,
                projection_axis=projection_axis,
            )

            # Get the two other axes for the widgets range
            axis1, axis2, axis_name1, axis_name2 = self._get_axes()

            # Interact with widgets to zoom in the two other axes
            @ interact(
                zoom_axis1=widgets.FloatRangeSlider(
                    value=[axis1[0], axis1[-1]],
                    min=min(axis1),
                    max=max(axis1),
                    step=np.mean(axis1[1:] - axis1[:-1]),
                    description=f'{axis_name1} range:',
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.3f',
                    layout=Layout(width="50%"),
                    style={'description_width': 'initial'},
                ),
                zoom_axis2=widgets.FloatRangeSlider(
                    value=[axis2[0], axis2[-1]],
                    min=min(axis2),
                    max=max(axis2),
                    step=np.mean(axis2[1:] - axis2[:-1]),
                    description=f'{axis_name2} range:',
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.3f',
                    layout=Layout(width="50%"),
                    style={'description_width': 'initial'},
                ),
                data_range=widgets.FloatRangeSlider(
                    value=[
                        np.nanmin(self.projected_data),
                        np.nanmax(self.projected_data),
                    ],
                    min=np.nanmin(self.projected_data),
                    max=np.nanmax(self.projected_data),
                    description='Data range:',
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.1f',
                    layout=Layout(width="50%"),
                    style={'description_width': 'initial'},
                ),
                figsize=fixed(figsize),
                cmap=fixed(cmap),
            )
            def interactive_plot(
                zoom_axis1,
                zoom_axis2,
                data_range,
                figsize,
                cmap,
            ):
                """Plot the space"""

                vmin, vmax = data_range
                self.plot_map(
                    zoom_axis1=list(zoom_axis1),
                    zoom_axis2=list(zoom_axis2),
                    vmin=vmin,
                    vmax=vmax,
                    figsize=figsize,
                    cmap=cmap,
                )

    def _get_axes(self):
        if self.projection_axis == 'H':
            axis1 = self.K_axis
            axis2 = self.L_axis
            axis_name1 = 'K (rlu)'
            axis_name2 = 'L (rlu)'

        elif self.projection_axis == 'K':
            axis1 = self.H_axis
            axis2 = self.L_axis
            axis_name1 = 'H (rlu)'
            axis_name2 = 'L (rlu)'

        elif self.projection_axis == 'L':
            axis1 = self.H_axis
            axis2 = self.K_axis
            axis_name1 = 'H (rlu)'
            axis_name2 = 'K (rlu)'

        elif self.projection_axis == 'qxyz':
            axis1 = self.Q_axis
            axis2 = self.Phi_axis
            axis_name1 = 'Q'
            axis_name2 = 'Phi (deg)'

        elif self.projection_axis == 'qx':
            axis1 = self.qy_axis
            axis2 = self.qz_axis
            axis_name1 = r"$q_y \, (\AA^{-1})$"
            axis_name2 = r"$q_z \, (\AA^{-1})$"

        elif self.projection_axis == 'qy':
            axis1 = self.qx_axis
            axis2 = self.qz_axis
            axis_name1 = r"$q_x \, (\AA^{-1})$"
            axis_name2 = r"$q_z \, (\AA^{-1})$"

        elif self.projection_axis == 'qz':
            axis1 = self.qx_axis
            axis2 = self.qy_axis
            axis_name1 = r"$q_x \, (\AA^{-1})$"
            axis_name2 = r"$q_y \, (\AA^{-1})$"

        elif self.projection_axis == 'delta':
            axis1 = self.gamma_axis
            axis_name1 = 'Gamma'
            try:
                axis2 = self.mu_axis
                axis_name2 = 'Mu'
            except AttributeError:
                axis2 = self.omega_axis
                axis_name2 = 'Omega'

        elif self.projection_axis == 'gamma':
            axis2 = self.delta_axis
            axis_name2 = 'Delta'
            try:
                axis1 = self.mu_axis
                axis_name1 = 'Mu'
            except AttributeError:
                axis1 = self.omega_axis
                axis_name1 = 'Omega'

        elif self.projection_axis == 'mu':
            axis1 = self.delta_axis
            axis2 = self.gamma_axis
            axis_name1 = 'Delta'
            axis_name2 = 'Gamma'

        elif self.projection_axis == 'omega':
            axis1 = self.delta_axis
            axis2 = self.gamma_axis
            axis_name1 = 'Delta'
            axis_name2 = 'Gamma'

        return axis1, axis2, axis_name1, axis_name2

    def _get_3D_data(self):
        return np.where(self.cont != 0, self.ct/self.cont, np.nan)


class CTR:
    """
    Loads a hdf5 file created by binoculars that represents a 3D map of the
    reciprocal space and provides (optional) integration and plotting methods to
    analyse the diffracted intensity along one direction.

    For now the classical workflow is the following:
    * process the data with binoculars, creating hdf5 files
    * integrate RODs with binoculars-fitaid, creating .txt files
    * fit these RODs with the `ROD` program (https://www.esrf.fr/computing/
        scientific/joint_projects/ANA-ROD/RODMAN2.html)

    Use one the following three methods to load the data:
        * load_fitaid_data(): load the rods integrated via fitaid (best option)
        * integrate_CTR(): integrate rods on the data with python (for tests)
        * load_ROD_data(): load the rods simulated with the ROD program

    All these functions create numpy arrays that can then be plotted with the
    plot_CTR() method.


    The binocular data is loaded as followin the integrate_CTR() method:
        * Divide counts by contribution where cont != 0
        * Swap the h and k axes to be consistent with the indexing
            [h, k, l], or [qx, qy, qz].
    If you use this method, to have the best results you should define your step
    in L directly in binoculars process and use the same step afterwards.
    """

    def __init__(
        self,
        configuration_file=False,
    ):
        """
        Init the class with a configuration file that is usefull to keep the 
        same colors depending on the conditions.

        :param configuration_file: default is False. `.yml` file that stores
            metadata specific to the reaction, if False, defaults to
            path_package + "experiments/ammonia.yml"
        """

        path_package = inspect.getfile(sixs).split("__")[0]

        # Load configuration file
        try:
            if os.path.isfile(configuration_file):
                self.configuration_file = configuration_file
            else:
                self.configuration_file = path_package + "experiments/ammonia.yml"
                print("Defaulted to ammonia configuration.")

        except TypeError:
            self.configuration_file = path_package + "experiments/ammonia.yml"
            print("Defaulted to ammonia configuration.")

        finally:
            with open(self.configuration_file) as filepath:
                yaml_parsed_file = yaml.load(
                    filepath,
                    Loader=yaml.FullLoader
                )

                for key in tqdm(yaml_parsed_file):
                    setattr(self, key, yaml_parsed_file[key])
                print("Loaded configuration file.")

    def integrate_CTR(
        self,
        folder,
        scan_indices,
        save_name,
        glob_string_match="*.hdf5",
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
        The structure factor is defined as:
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
        # Get all files first
        files = [f.split("/")[-1]
                 for f in sorted(glob.glob(f"{folder}/{glob_string_match}"))]

        # Get scans specified with scan_indices
        self.scan_files, good_scan_indices = [], []
        for f in files:
            if any([str(n) in f for n in scan_indices]):
                self.scan_files.append(f)
                good_scan_indices.append(
                    [n for n in scan_indices if str(n) in f][0])

        if verbose:
            print(
                "\n###########################################################"
                f"\nDetected files in {folder}:"
            )
            for f in files:
                print("\t", f)

            print(
                "\nWorking on the following files:"
            )
            for f in self.scan_files:
                print("\t", f)
            print(
                "###########################################################")

        # Define ROD range
        CTR_range_H = [
            np.round(HK_peak[0] - CTR_width_H/2, 3),
            np.round(HK_peak[0] + CTR_width_H/2, 3)
        ]
        CTR_range_K = [
            np.round(HK_peak[1] - CTR_width_K/2, 3),
            np.round(HK_peak[1] + CTR_width_K/2, 3)
        ]

        print(
            "\n###########################################################"
            f"\nRange in H: [{CTR_range_H[0]} : {CTR_range_H[1]}]"
            f"\nRange in K: [{CTR_range_K[0]} : {CTR_range_K[1]}]"
            "\n###########################################################"
        )

        # Define background range
        if center_background == HK_peak:
            background_range_H = [
                np.round(CTR_range_H[0] - background_width_H/2, 3),
                np.round(CTR_range_H[1] + background_width_H/2, 3)
            ]
            background_range_K = [
                np.round(CTR_range_K[0] - background_width_K/2, 3),
                np.round(CTR_range_K[1] + background_width_K/2, 3)
            ]

        elif isinstance(center_background, list) \
                and center_background != HK_peak:

            background_range_H = [
                center_background[0] - background_width_H/2,
                center_background[0] + background_width_H/2
            ]
            background_range_K = [
                center_background[1] - background_width_K/2,
                center_background[1] + background_width_K/2
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

        # Define the lines around the ROI and the background
        ROI_lines = [
            (HK_peak[0]-CTR_width_H/2, HK_peak[1]-CTR_width_K/2, HK_peak[0] +
             CTR_width_H/2, HK_peak[1]-CTR_width_K/2, "r", "--", 0.8),
            (HK_peak[0]-CTR_width_H/2, HK_peak[1]-CTR_width_K/2, HK_peak[0] -
             CTR_width_H/2, HK_peak[1]+CTR_width_K/2, "r", "--", 0.8),
            (HK_peak[0]+CTR_width_H/2, HK_peak[1]+CTR_width_K/2, HK_peak[0] +
             CTR_width_H/2, HK_peak[1]-CTR_width_K/2, "r", "--", 0.8),
            (HK_peak[0]+CTR_width_H/2, HK_peak[1]+CTR_width_K/2, HK_peak[0] -
             CTR_width_H/2, HK_peak[1]+CTR_width_K/2, "r", "--", 0.8),
        ]

        if isinstance(center_background, list):
            ROI_lines += [
                (HK_peak[0]-CTR_width_H/2-background_width_H/2, HK_peak[1]-CTR_width_K/2, HK_peak[0] -
                 CTR_width_H/2-background_width_H/2, HK_peak[1]+CTR_width_K/2, "b", "--", 0.8),
                (HK_peak[0]-CTR_width_H/2-background_width_H/2, HK_peak[1]-CTR_width_K /
                 2, HK_peak[0]-CTR_width_H/2, HK_peak[1]-CTR_width_K/2, "b", "--", 0.8),
                (HK_peak[0]-CTR_width_H/2-background_width_H/2, HK_peak[1]+CTR_width_K /
                 2, HK_peak[0]-CTR_width_H/2, HK_peak[1]+CTR_width_K/2, "b", "--", 0.8),

                (HK_peak[0]+CTR_width_H/2+background_width_H/2, HK_peak[1]-CTR_width_K/2, HK_peak[0] +
                 CTR_width_H/2+background_width_H/2, HK_peak[1]+CTR_width_K/2, "b", "--", 0.8),
                (HK_peak[0]+CTR_width_H/2+background_width_H/2, HK_peak[1]-CTR_width_K /
                 2, HK_peak[0]+CTR_width_H/2, HK_peak[1]-CTR_width_K/2, "b", "--", 0.8),
                (HK_peak[0]+CTR_width_H/2+background_width_H/2, HK_peak[1]+CTR_width_K /
                 2, HK_peak[0]+CTR_width_H/2, HK_peak[1]+CTR_width_K/2, "b", "--", 0.8),

                (HK_peak[0]+CTR_width_H/2, HK_peak[1]+CTR_width_K/2+background_width_K/2, HK_peak[0] -
                 CTR_width_H/2, HK_peak[1]+CTR_width_K/2+background_width_K/2, "b", "--", 0.8),
                (HK_peak[0]-CTR_width_H/2, HK_peak[1]+CTR_width_K/2, HK_peak[0]-CTR_width_H /
                 2, HK_peak[1]+CTR_width_K/2+background_width_K/2, "b", "--", 0.8),
                (HK_peak[0]+CTR_width_H/2, HK_peak[1]+CTR_width_K/2, HK_peak[0]+CTR_width_H /
                 2, HK_peak[1]+CTR_width_K/2+background_width_K/2, "b", "--", 0.8),

                (HK_peak[0]+CTR_width_H/2, HK_peak[1]-CTR_width_K/2-background_width_K/2, HK_peak[0] -
                 CTR_width_H/2, HK_peak[1]-CTR_width_K/2-background_width_K/2, "b", "--", 0.8),
                (HK_peak[0]-CTR_width_H/2, HK_peak[1]-CTR_width_K/2, HK_peak[0]-CTR_width_H /
                 2, HK_peak[1]-CTR_width_K/2-background_width_K/2, "b", "--", 0.8),
                (HK_peak[0]+CTR_width_H/2, HK_peak[1]-CTR_width_K/2, HK_peak[0]+CTR_width_H /
                 2, HK_peak[1]-CTR_width_K/2-background_width_K/2, "b", "--", 0.8),
            ]

        # Start iterating on the files to see the shape
        print(
            "\n###########################################################"
            "\nFinding smallest common range in L"
            "\nDepends on the config file in binoculars-process."
            "\n###########################################################"
        )

        for i, fname in tqdm(enumerate(self.scan_files)):
            with tb.open_file(folder + fname, "r") as f:
                H = f.root.binoculars.axes.H[:]
                K = f.root.binoculars.axes.K[:]
                L = f.root.binoculars.axes.L[:]

            if verbose:
                print_binocular_axes(folder + fname)

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
            f"\nMaximum number of points is {l_length}"
            "\n###########################################################"
        )

        # Save final data as numpy array
        # 0 is x axis, 1 is data, 2 is background (compatible with ROD)
        data = np.nan * np.empty((
            len(self.scan_files),
            3,
            l_length-1,  # FITAID CORRECTION
        ))

        # Iterate on each file now to get the data
        for i, fname in tqdm(enumerate(self.scan_files)):
            if verbose:
                print(
                    "\n###########################################################"
                    f"\nOpening file {fname} ..."
                )

            # Use Map class to avoid mistakes
            dataset = Map(folder + fname, verbose=False)

            # FITAID CORRECTION
            scan_l_length = len(dataset.L_axis)
            scan_l_axis = (dataset.L_axis[:-1]+dataset.L_axis[1:])/2

            # Define ROI indices
            start_H_ROI = find_value_in_array(dataset.H_axis, CTR_range_H[0])
            end_H_ROI = find_value_in_array(dataset.H_axis, CTR_range_H[1])
            start_K_ROI = find_value_in_array(dataset.K_axis, CTR_range_K[0])
            end_K_ROI = find_value_in_array(dataset.K_axis, CTR_range_K[1])

            if verbose:
                print(
                    f"Data ROI (H, K): [{start_H_ROI[0]}, {end_H_ROI[0]}, "
                    f"{start_K_ROI[0]}, {end_K_ROI[0]}] ; [{start_H_ROI[1]}, "
                    f"{end_H_ROI[1]}, {start_K_ROI[1]}, {end_K_ROI[1]}]"
                )

            # Get data only in specific ROI
            intensity = np.empty(scan_l_length-1)
            roi_pixel_count = np.empty(scan_l_length-1)
            background_values = np.empty(scan_l_length-1)
            background_pixel_count = np.empty(scan_l_length-1)
            structure_factor = np.empty(scan_l_length-1)

            for (l_index_min, l_index_max) in zip(
                range(0, scan_l_length-1),
                range(1, scan_l_length),
            ):
                dataset.project_data(
                    "L",
                    projection_axis_range=[
                        dataset.L_axis[l_index_min],
                        dataset.L_axis[l_index_max],
                    ]
                )

                # Integrate the data in the ROI, replace nan by zeroes otherwise
                # the total is equal to np.nan
                intensity[l_index_min] = np.nansum(
                    dataset.projected_data[
                        start_K_ROI[1]:end_K_ROI[1],
                        start_H_ROI[1]:end_H_ROI[1],
                    ],
                    axis=(0, 1),
                )

                # Count number of np.nan pixels in the ROI
                # These pixels do not have an intensity (!= from zero intensity),
                # they were not recorded.
                roi_pixel_count[l_index_min] = np.sum(
                    ~np.isnan(
                        dataset.projected_data[
                            start_K_ROI[1]:end_K_ROI[1],
                            start_H_ROI[1]:end_H_ROI[1],
                        ],
                    ),
                    axis=(0, 1),
                )

                # Compute background
                if center_background == HK_peak:
                    # Define background ROIs indices
                    start_H_background = find_value_in_array(
                        dataset.H_axis,
                        background_range_H[0]
                    )
                    end_H_background = find_value_in_array(
                        dataset.H_axis,
                        background_range_H[1]
                    )

                    start_K_background = find_value_in_array(
                        dataset.K_axis,
                        background_range_K[0]
                    )
                    end_K_background = find_value_in_array(
                        dataset.K_axis,
                        background_range_K[1]
                    )

                    if verbose and l_index_min == 0:
                        print(
                            f"Background ROI (H, K): [{start_H_background[0]}, "
                            f"{end_H_background[0]}, {start_K_background[0]}, "
                            f"{end_K_background[0]}] ; [{start_H_background[1]}, "
                            f"{end_H_background[1]}, {start_K_background[1]}, "
                            f"{end_K_background[1]}]"
                            "\n###########################################################"
                        )

                    # Define background ROIs
                    background_ROI_0 = dataset.projected_data[
                        start_K_ROI[1]:end_K_ROI[1],
                        start_H_background[1]:start_H_ROI[1],
                    ]

                    background_ROI_1 = dataset.projected_data[
                        start_K_ROI[1]:end_K_ROI[1],
                        end_H_ROI[1]:end_H_background[1],
                    ]

                    background_ROI_2 = dataset.projected_data[
                        start_K_background[1]:start_K_ROI[1],
                        start_H_ROI[1]:end_H_ROI[1],
                    ]

                    background_ROI_3 = dataset.projected_data[
                        end_K_ROI[1]:end_K_background[1],
                        start_H_ROI[1]:end_H_ROI[1],
                    ]

                    # Integrate the data in the ROIs, replace nan by zeroes
                    # otherwise the total is equal to np.nan
                    background_values[l_index_min] = \
                        np.nansum(background_ROI_0, axis=(0, 1)) + \
                        np.nansum(background_ROI_1, axis=(0, 1)) + \
                        np.nansum(background_ROI_2, axis=(0, 1)) + \
                        np.nansum(background_ROI_3, axis=(0, 1))

                    # Count number of non-np.nan pixels in the background
                    # These pixels do not # have an intensity (!= from zero
                    # intensity), they were not recorded.
                    background_pixel_count[l_index_min] = \
                        np.sum(~np.isnan(background_ROI_0), axis=(0, 1)) + \
                        np.sum(~np.isnan(background_ROI_1), axis=(0, 1)) + \
                        np.sum(~np.isnan(background_ROI_2), axis=(0, 1)) + \
                        np.sum(~np.isnan(background_ROI_3), axis=(0, 1))

                    # Remove background
                    structure_factor[l_index_min] = np.nan_to_num(np.where(
                        background_pixel_count[l_index_min] > 0,
                        np.sqrt(
                            intensity[l_index_min] -
                            roi_pixel_count[l_index_min] * (background_values[l_index_min] /
                                                            background_pixel_count[l_index_min])),
                        0
                    ))

                elif isinstance(center_background, list) \
                        and center_background != HK_peak:
                    # Background intensity, define ROI indices
                    start_H_background = find_value_in_array(
                        dataset.H_axis,
                        background_range_H[0]
                    )
                    end_H_background = find_value_in_array(
                        dataset.H_axis,
                        background_range_H[1]
                    )

                    start_K_background = find_value_in_array(
                        dataset.K_axis,
                        background_range_K[0]
                    )
                    end_K_background = find_value_in_array(
                        dataset.K_axis,
                        background_range_K[1]
                    )

                    if verbose and l_index_min == 0:
                        print(
                            f"Background ROI (H, K): [{start_H_background[0]}, "
                            f"{end_H_background[0]}, {start_K_background[0]}, "
                            f"{end_K_background[0]}] ; [{start_H_background[1]}, "
                            f"{end_H_background[1]}, {start_K_background[1]}, "
                            f"{end_K_background[1]}]"
                            "\n###########################################################"
                        )

                    # Define the background ROI
                    background_ROI = dataset.projected_data[
                        start_K_background[1]:end_K_background[1],
                        start_H_background[1]:end_H_background[1],
                    ]

                    background_values = np.nansum(background_ROI, axis=(0, 1))

                    # Count number of non-np.nan pixels in the background
                    # These pixels do not # have an intensity (!= from zero
                    # intensity), they were not recorded.
                    background_pixel_count = np.sum(
                        ~np.isnan(background_ROI), axis=(0, 1))

                    # Remove background
                    structure_factor[l_index_min] = np.nan_to_num(np.where(
                        background_pixel_count[l_index_min] > 0,
                        np.sqrt(
                            intensity[l_index_min] -
                            roi_pixel_count[l_index_min] * (background_values[l_index_min] /
                                                            background_pixel_count[l_index_min])),
                        0
                    ))

                else:
                    structure_factor = np.sqrt(intensity)

            # Save x axis
            # TODO, assumes same starting value
            data[i, 0, :scan_l_length-1] = scan_l_axis

            # Save structure factor
            data[i, 1, :scan_l_length-1] = structure_factor

            # Save background
            try:
                data[i, 2, :scan_l_length-1] = background_values / \
                    background_pixel_count

            except NameError:
                print(
                    "No background subtracted"
                    "\n###########################################################"
                )

            if verbose:
                # Resume with a plot for the last dataset
                mappo = Map(folder + fname)
                mappo.project_data("L")
                mappo.plot_map(
                    lines=ROI_lines,
                    title="CTR data projected on L axis",
                )

        # Saving
        print(
            "\n###########################################################"
            f"\nSaving data as: {folder}{save_name}"
            "\n###########################################################"
        )
        np.save(folder + save_name, data)

    def load_fitaid_data(
        self,
        folder,
        scan_indices,
        save_name,
        glob_string_match="nisf*.txt",
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
        :param verbose: True for additional informations printed during function
        """
        # Get all txt files first
        files = [f.split("/")[-1]
                 for f in sorted(glob.glob(f"{folder}/{glob_string_match}"))]

        # Get scans specified with scan_indices
        scan_files, good_scan_indices = [], []
        for f in files:
            if any([str(n) in f for n in scan_indices]):
                scan_files.append(f)
                good_scan_indices.append(
                    [n for n in scan_indices if str(n) in f][0])

        if len(scan_files) == 0:
            return ("No matching files found in folder.")

        if verbose:
            print(
                "\n###########################################################"
                f"\nDetected files in {folder}:"
            )
            for f in files:
                print("\t", f)
            print(
                "\nWorking on the following files:"
            )
            for f in scan_files:
                print("\t", f)
            print("###########################################################\n")

        # Iterating on all files to create l axis
        for i, fname in tqdm(enumerate(scan_files)):
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

        # Save final data as numpy array
        # 0 is x axis, 1 is data, 2 is background
        data_dict = {
            str(g): np.nan * np.empty((3, l_length))
            for g in good_scan_indices
        }

        # Background already subtracted
        # Get l axis and CTR intensity for each file
        for g, fname in tqdm(zip(good_scan_indices, scan_files)):
            # Load data
            fitaid_data = np.loadtxt(folder + fname)
            scan_l_axis = fitaid_data[:, 0]
            ctr_data = fitaid_data[:, 1]

            data_dict[str(g)][0, :len(scan_l_axis)] = scan_l_axis
            data_dict[str(g)][1, :len(scan_l_axis)] = ctr_data

        # Saving
        print(
            "\n###########################################################"
            f"\nSaving data as: {folder}{save_name}"
            "\n###########################################################"
        )
        np.savez(
            folder + save_name,
            **data_dict,
        )

    def load_simulated_ROD_data(
        self,
        folder,
        scan_indices,
        save_name,
        data_column=7,
        glob_string_match="*.dat",
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
        :param verbose: True for additional informations printed during function
        """
        # Get all txt files first
        files = [f.split("/")[-1]
                 for f in sorted(glob.glob(f"{folder}/{glob_string_match}"))]

        # Get scans specified with scan_indices
        scan_files = []
        for f in files:
            if any([str(n) in f for n in scan_indices]):
                scan_files.append(f)

        if len(scan_files) == 0:
            return ("No matching files found in folder.")

        if verbose:
            print(
                "\n###########################################################"
                f"\nDetected files in {folder}:"
            )
            for f in files:
                print("\t", f)
            print(
                "\nWorking on the following files:"
            )
            for f in scan_files:
                print("\t", f)
            print("###########################################################\n")

        # Iterating on all files to create l axis
        for i, fname in tqdm(enumerate(scan_files)):
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
            "\n###########################################################"
        )

        # Save final data as numpy array
        # 0 is x axis, 1 is data, 2 is background
        data_dict = {
            str(g): np.nan * np.empty((3, l_length))
            for g in scan_indices
        }

        # Get l axis and CTR intensity for each file
        for g, fname in tqdm(zip(scan_indices, scan_files)):
            # Load data
            rod_data = np.loadtxt(folder + fname, skiprows=2)
            scan_l_axis = rod_data[:, 2]
            ctr_data = rod_data[:, data_column]

            data_dict[str(g)][0, :len(scan_l_axis)] = scan_l_axis
            data_dict[str(g)][1, :len(scan_l_axis)] = ctr_data

        # Saving
        print(
            "\n###########################################################"
            f"\nSaving data as: {folder}{save_name}"
            "\n###########################################################"
        )
        np.savez(
            folder + save_name,
            **data_dict,
        )

    @ staticmethod
    def plot_CTR(
        numpy_array,
        scan_indices,
        title=None,
        color_dict=None,
        labels=None,
        y_scale="log",
        plot_style="scatter",
        size=4,
        legend=True,
        legend_position="right",
        figure_width=900,
        figure_height=500,
        label_text_font_size="20pt",
        axis_label_text_font_size="20pt",
        axis_major_label_text_font_size="20pt",
        title_text_font_size="25pt",
        x_range=None,
        y_range=None,
    ):
        """
        Plot the CTRs together, using Bokeh

        :param numpy_array: path to .npy file on disk.
            - l
            - data - background
            - background
        :param scan_indices: scan indices of files plotted, in order, used for
            labelling, mandatory because we need to know what we plot!
        :param title: if string, set to figure title
        :param color_dict: dict used for labels, keys are scan index, values are
            colours for matplotlib.
        :param labels: dict of labels to use, defaults to scan_index list if None
        :param y_scale: if "log", y axis is logarithmic, else 'lin'
        :param plot_style: if "scatter", scatter plot, else "line"
        :param size: size of markers
        :param legend: add a legend to the plot
        :param legend_position: choose in ('left', 'right', 'center', 'above', 
            'below')
        :param figure_width: in pixels, default is 900
        :param figure_height: in pixels, default is 500
        :param label_text_font_size: e.g. '15pt'
        :param axis_label_text_font_size: e.g '15pt'
        :param axis_major_label_text_font_size: e.g '15pt'
        :param title_text_font_size: e.g '15pt'
        """
        matplotlib_colours = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22',
            '#17becf'
        ]

        # Create figure
        TOOLTIPS = [
            (f"L", "$x"),
            ("Intensity", "$y"),
        ]

        p = figure(
            height=figure_height,
            width=figure_width,
            tools="xpan, pan, wheel_zoom, box_zoom, reset, undo, redo, crosshair, hover, save",
            tooltips=TOOLTIPS,
            title=title,
            x_axis_label="L",
            y_axis_label="Intensity",
            active_scroll="wheel_zoom",
            y_axis_type=y_scale,
            x_range=x_range,
            y_range=y_range,
        )

        p.add_layout(
            Legend(
                click_policy="mute",
                label_text_font_size=label_text_font_size,
            ),
            legend_position
        )

        # Load np array on disk
        data = np.load(numpy_array)
        print(
            "###########################################################"
            f"\nLoaded {numpy_array}"
            f"\nFiles in array: {data.files}"
            "\n###########################################################"
        )

        # Iterate on array
        for i, scan_index in enumerate(scan_indices):
            arr = data[str(scan_index)]
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
                label = str(scan_index)
            else:
                print("Labels must be a dictionnary with keys = scan_indices")
                label = scan_index

            # Create source
            source = ColumnDataSource(
                data=dict(
                    x=l,
                    y=y_plot,
                ))

            # Get color
            try:
                color = color_dict[int(scan_index)]
            except (KeyError, ValueError):
                color = color_dict[scan_index]
            except TypeError:
                color = matplotlib_colours[i]

            if plot_style == "scatter":
                # Add scatter plot
                p.scatter(
                    x='x',
                    y='y',
                    source=source,
                    legend_label=label,
                    marker="circle",
                    size=size,
                    alpha=0.7,
                    fill_color=color,
                    line_color=color,
                    muted_alpha=0,
                    hover_alpha=1,
                )

            if plot_style == "line":
                # Add line plot
                p.line(
                    x='x',
                    y='y',
                    source=source,
                    legend_label=label,
                    line_alpha=0.7,
                    line_color=color,
                    muted_alpha=0,
                    hover_alpha=1,
                )

        # Show figure
        p.xaxis.axis_label_text_font_size = axis_label_text_font_size
        p.xaxis.major_label_text_font_size = axis_major_label_text_font_size
        p.yaxis.axis_label_text_font_size = axis_label_text_font_size
        p.yaxis.major_label_text_font_size = axis_major_label_text_font_size
        if not legend:
            p.legend.visible = False
        if isinstance(title, str):
            p.title.text_font_size = title_text_font_size

        show(p)


def change_nb_unit_cells_pt3o4(
    save_as,
    nb_surf_unit_cells=1,
    comment="",
    spacing=None,
):
    """
    Add cubic Pt3O4 unit cells on top of a .bul file.

    The structure is hardcoded to be:
        5.66568400 5.66568400 3.9254 90.0 90.0 90.0

        Pt 0.50 0 0 0 0 0.25 0 0 0 0 {z0} 1 1 1 0
        Pt 0.50 0 0 0 0 0.75 0 0 0 0 {z0} 1 1 1 0

        Pt 0.00 0 0 0 0 0.50 0 0 0 0 {z1} 2 1 1 0
        O 0.75 0 0 0 0 0.25 0 0 0 0 {z1} 2 1 1 0
        O 0.25 0 0 0 0 0.75 0 0 0 0 {z1} 2 1 1 0
        O 0.75 0 0 0 0 0.75 0 0 0 0 {z1} 2 1 1 0
        O 0.25 0 0 0 0 0.25 0 0 0 0 {z1} 2 1 1 0

        Pt 0.75 0 0 0 0 0.00 0 0 0 0 {z2} 3 1 1 0
        Pt 0.25 0 0 0 0 0.00 0 0 0 0 {z2} 3 1 1 0

        O 0.25 0 0 0 0 0.25 0 0 0 0 {z3} 4 1 1 0
        O 0.75 0 0 0 0 0.75 0 0 0 0 {z3} 4 1 1 0
        O 0.25 0 0 0 0 0.75 0 0 0 0 {z3} 4 1 1 0
        O 0.75 0 0 0 0 0.25 0 0 0 0 {z3} 4 1 1 0
        Pt 0.00 0 0 0 0 0.50 0 0 0 0 {z3} 4 1 1 0

    With z = (5.6657/3.9254) and z0 = 0.25*z, z1 = 0.5*z, z2 = 0.75*z.

    :param nb_surf_unit_cells: number of unit cells on top of the bulk
    :param spacing: spacing (Angström) between bulk and surface structures
    """

    # Keep same lattice parameter as bulk Pt
    lines = [comment]
    lines.append(f", {nb_surf_unit_cells} unit cells.\n")

    # crystal lattice
    lines.append("5.6657 5.6657 3.9254 90.0 90.0 90.0\n")
    z = (5.6657/3.9254)

    for n in range(nb_surf_unit_cells):
        # non z = 0 layers inside the first unit cell
        # values in z
        z0 = np.round(n*z, 4)
        z1 = np.round((n + 0.25) * z, 4)
        z2 = np.round((n + 0.5) * z, 4)
        z3 = np.round((n + 0.75) * z, 4)

        unit_cell = [
            f"Pt 0.50 0 0 0 0 0.25 0 0 0 0 {z0} {4*n+1} 1 1 0\n",
            f"Pt 0.50 0 0 0 0 0.75 0 0 0 0 {z0} {4*n+1} 1 1 0\n",

            f"Pt 0.00 0 0 0 0 0.50 0 0 0 0 {z1} {4*n+2} 1 1 0\n",
            f"O 0.75 0 0 0 0 0.25 0 0 0 0 {z1} {4*n+2} 1 1 0\n",
            f"O 0.25 0 0 0 0 0.75 0 0 0 0 {z1} {4*n+2} 1 1 0\n",
            f"O 0.75 0 0 0 0 0.75 0 0 0 0 {z1} {4*n+2} 1 1 0\n",
            f"O 0.25 0 0 0 0 0.25 0 0 0 0 {z1} {4*n+2} 1 1 0\n",

            f"Pt 0.75 0 0 0 0 0.00 0 0 0 0 {z2} {4*n+3} 1 1 0\n",
            f"Pt 0.25 0 0 0 0 0.00 0 0 0 0 {z2} {4*n+3} 1 1 0\n",

            f"O 0.25 0 0 0 0 0.25 0 0 0 0 {z3} {4*n+4} 1 1 0\n",
            f"O 0.75 0 0 0 0 0.75 0 0 0 0 {z3} {4*n+4} 1 1 0\n",
            f"O 0.25 0 0 0 0 0.75 0 0 0 0 {z3} {4*n+4} 1 1 0\n",
            f"O 0.75 0 0 0 0 0.25 0 0 0 0 {z3} {4*n+4} 1 1 0\n",
            f"Pt 0.00 0 0 0 0 0.50 0 0 0 0 {z3} {4*n+4} 1 1 0\n",
        ]
        for l in unit_cell:
            lines.append(l)

    with open(save_as, "w") as f:
        f.writelines(lines)


def modify_surface_relaxation(
    base_file,
    save_as,
    lines_to_edit=[3],
    columns_to_edit=["z"],
    relaxation=0.99,
    round_order=4,
    sep=" ",
    print_old_file=False,
    print_new_file=True,
):
    """
    Edit .sur file for ROD. Play here with the relaxation by multiplying the z
    parameter of each atom by the same value. !!Simple model where there is no
    strain between the successive unit cells but just between the bulk and
    surface files.!!
    The files must use only one space between characters to split properly !

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
            c_index = {"x": 1, "y": 6, "z": 11}[c]
            line[c_index] = str(np.round(float(line[c_index]) *
                                relaxation, round_order))

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
            "\n############### New surface file ###############\n"
            f"################## r = {relaxation:.3f}  ##################\n"
        )
        for line in new_file_lines:
            print(line, end="")

        print("\n############### New surface file ###############")

    # Create new file
    with open(save_as, "w") as f:
        f.writelines(new_file_lines)


def simulate_rod(
    filename,
    bulk_file=None,
    surface_file=None,
    fit_file=None,
    rod_hk=[2, 2],
    l_start=0,
    l_end=3,
    nb_points=251,
    l_bragg=0,
    nb_layers_bulk=2,
    attenuation=0.001,
    beta=0,
    scale=1,
    error_bars=True,
    save_folder=None,
    comment=None,
):
    """
    This function uses ROD, that must be installed on your computer.
    Help document:
        https://www.esrf.fr/computing/scientific/joint_projects/
        ANA-ROD/RODMAN2.html

    It will generate the following files the first time it is run in a folder:
        - pgplot.ps
        - plotinit.mac
        - rod_init.mac
    These files will be used during subsequent runs, they initialize a bunch of
        arguments used for plotting.

    :param filename: str, used in the names of the output files, not a path
    :param bulk_file: str, path to bulk file (.bul)
    :param surface_file: str, path to surface file (.sur)
    :param fit_file: str, path to fit file (.sur)
    :param rod_hk: list, position in h and k of the rod
    :param l_start: beginning of the rod in l
    :param l_end: end of the rod in l
    :param nb_points: nb of points in the rod
    :param l_bragg: position in l of the first bragg peak
    :param nb_layers_bulk: number of layers in the bulk unit cell
    :param attenuation: attenuation of the beam
    :param beta: beta used for roughness in beta model
    :param scale: scale factor between theory and experiment
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
        f"\nScale {scale}",
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

    if isinstance(fit_file, str):
        if os.path.isfile(fit_file):
            # Copy surface file to save_folder for execution
            shutil.copy2(
                fit_file,
                save_folder,
            )
            lines.append(f"\nread fit {os.path.basename(fit_file)}")

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


def save_as_dat(
    fitaid_file,
    h,
    k,
    l_bragg,
    save_as,
    f_threshold=None,
    l_shift=None,
    l_range=None,
    sigma=None,
):
    """
    Save data in `.dat format to then be used in ROD.
    Will automatically mask np.nan values

    :param fitaid_file: 2D array containing l and structure factor values,
        output of binoculars fitaid
    :param h: position in h
    :param k: position in k
    :param l_bragg: first bragg peak position in L
    :param save_as: final file name
    :param f_threshold: threshold for structure factor above which the values
        are masked
    :param l_shift: rigid shift in l to apply
    :param l_range: container of len 2, mask points outside this range
    :param sigma: if int, saved as sigma for each row, if array of same length
        as the data, used for sigma, if "sqrt", uses sigma=np.sqrt(i), if "xx%"
        will save error at xx % of I.
    """

    data = np.loadtxt(fitaid_file)
    # Mask np.nan
    data = data[~np.isnan(data[:, 1])]

    # Mask data if above threshold
    if f_threshold is not None:
        data = data[
            data[:, 1] < f_threshold
        ]

    # Mask data if outside l_range
    if isinstance(l_range, (list, tuple)):
        if len(l_range) == 2:
            data = data[
                (data[:, 0] >= l_range[0]) * (data[:, 0] <= l_range[1])
            ]
        else:
            print("`l_range` must be of length 2.")

    # Create final array
    final_data = np.ones((data.shape[0], 6))
    final_data[:, 0] = h
    final_data[:, 1] = k
    final_data[:, 2:4] = data
    final_data[:, 5] = l_bragg

    if l_shift is not None:
        fig, ax = plt.subplots()
        ax.scatter(
            final_data[:, 2],
            final_data[:, 3],
            label="No shift",
            s=2,
        )
        ax.scatter(
            final_data[:, 2]+l_shift,
            final_data[:, 3],
            label="With l_shift",
            s=2,
        )
        ax.grid()
        ax.legend()
        ax.semilogy()

        final_data[:, 2] += l_shift

    if isinstance(sigma, np.ndarray):
        final_data[:, 4] = sigma
    elif isinstance(sigma, int):
        final_data[:, 4] = np.ones(data.shape[0])*sigma
    elif isinstance(sigma, str):
        if sigma == "sqrt":
            final_data[:, 4] = np.sqrt(data[:, 1])
        elif sigma.endswith("%"):
            error_percentage = int(sigma.split("%")[0])/100
            final_data[:, 4] = error_percentage * data[:, 1]
    else:
        final_data[:, 4] = 0

    np.savetxt(
        save_as,
        final_data,
        fmt="%10.4f",
        header=f"h, k, l, f, sigma, l_bragg. File used for creation: {fitaid_file}"
    )


def merge_data_files(*files, save_as):
    """
    Stack arrays vertically to fit together in ROD.
    """

    print("Merging files:", files)

    arrays = [np.loadtxt(f) for f in files]

    stacked_array = np.vstack((arrays))

    np.savetxt(
        save_as,
        stacked_array,
        fmt="%10.4f",
        header=f"h, k, l, f, sigma, l_bragg. File used for creation: {files}"
    )


def read_par_file(file):
    """
    Read parameter file, output of ROD fitting, and extract the parameters
    values and chi_square.

    :param file: path to `.par` file
    """
    chi_square = None

    with open(file) as f:
        lines = f.readlines()

        parameter_line = False
        parameters, min_value, value, max_value, refined = [], [], [], [], []
        min_value = []
        for line in lines:

            if line.startswith("!chisqr"):
                chi_square = float(line[10:])

            if line.startswith("return"):
                parameter_line = False

            if parameter_line:
                parameters.append(line[:13].replace(" ", ""))
                value.append(float(line[13:23].replace(" ", "")))
                min_value.append(float(line[23:33].replace(" ", "")))
                max_value.append(float(line[33:39].replace(" ", "")))
                refined.append(True if "YES" in line[39:] else False)

            if line.startswith("set par"):
                parameter_line = True

    df = pd.DataFrame({
        "Parameters": parameters,
        "Min": min_value,
        "Value": value,
        "Max": max_value,
        "Refined": refined,
    })

    return chi_square, df


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


def show_par_files(
    files,
    disp_pars=3,
    debye_waller_pars=3,
    save_as=None,
    figsize=(10, 10),
):
    """
    Read parameter file, output of ROD fitting, and extract the parameters
    values and chi_square.

    Plots a figure that shows parameter values vs chi^2

    Returns df with lowest chi^2 value.

    :param files: list of path to `.par` files
    :param save_as: path to save figure
    """
    chi_squares = [read_par_file(f)[0] for f in files]
    dfs = [read_par_file(f)[1] for f in files]

    arr = np.empty((len(dfs), len(dfs[0])))
    for j, df in enumerate(dfs):
        arr[j, :] = df.Value.values

    # Nb of rows
    disp_rows = int(np.ceil(disp_pars/3))
    debye_waller_rows = int(np.ceil(debye_waller_pars/3))
    rows = 1+disp_rows+debye_waller_rows

    # Create figure
    fig, axs = plt.subplots(rows, 3, figsize=figsize, sharey=True)

    axs[0, 0].scatter(arr[:, 0], chi_squares)  # scale
    axs[0, 0].set_xlabel(dfs[0].Parameters.values[0])
    axs[0, 1].scatter(arr[:, 1], chi_squares)  # beta
    axs[0, 1].set_xlabel(dfs[0].Parameters.values[1])
    axs[0, 2].scatter(arr[:, 2], chi_squares)  # surf frac
    axs[0, 2].set_xlabel(dfs[0].Parameters.values[2])

    # Disp
    for i in range(disp_rows):
        for j in range(3):
            if 3*i+j < disp_pars:
                axs[i+1, j].scatter(arr[:, 3+i*3+j], chi_squares)
                axs[i+1, j].set_xlabel(dfs[0].Parameters.values[3+i*3+j])

    # DWF
    for i in range(debye_waller_rows):
        for j in range(3):
            if 3*i+j < debye_waller_pars:
                axs[i+1+disp_rows,
                    j].scatter(arr[:, 3+disp_pars+i*3+j], chi_squares)
                axs[i+1+disp_rows,
                    j].set_xlabel(dfs[0].Parameters.values[3+disp_pars+i*3+j])

    for j, ax in enumerate(axs.ravel()):
        ax.grid()
        ax.set_ylabel(r"$\chi^2$")

    plt.tight_layout()
    if isinstance(save_as, str):
        plt.savefig(save_as)
    plt.show()

    best_index = np.array(chi_squares).argmin()
    print(
        f"Lowest Chi^2 value for {files[best_index]}: {chi_squares[best_index]}")

    return dfs[best_index]
