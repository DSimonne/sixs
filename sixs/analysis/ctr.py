import numpy as np
import tables as tb
import pandas as pd
import glob
import os
import inspect
import yaml

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.interpolate import splrep, splev

import sixs
from sixs_nxsread import ReadNxs4 as rn4


class CTR:
    """
    Loads a Map of the reciprocal space created via binoculars, and provides
    integration methods to analyse the diffracted intensity.
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

    def prep_CTR_data(
        self,
        folder,
        scan_indices,
        save_name,
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
         background is taken, default is [-1, 1]. If equal to HK_peak, the
         background width is added to the width of the CTR.
        :param verbose: if True, print more details.
        """

        # Load data
        self.scan_indices = [str(s) for s in scan_indices]

        # Get all hdf5 files first
        files = [f.split("/")[-1]
                 for f in sorted(glob.glob(f"{folder}/*.hdf5"))]

        # Get scans specified with scan_indices
        self.scan_files = [f for f in files if any(
            ["-" + n + ".hdf5" in f for n in self.scan_indices])]

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
        if isinstance(self.interpol_step, float):
            print("\n###########################################################")
            print("Finding smallest common range in L")
            print("Depends on the config file in binoculars-process.")
            print("###########################################################")
        else:
            # No interpolation, we assume that all the scan have the same l axis
            print("\n###########################################################")
            print("No interpolation")
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
                l_min = min(l_min, L[1])
                l_max = max(l_max, L[2])
                l_shape = max(l_shape, 1 + int(L[5] - L[4]))

        if isinstance(self.interpol_step, float):
            # Determine interpolation range
            print("\n###########################################################")
            print(f"Smallest common range in L is [{l_min} : {l_max}]")
            print("###########################################################")

            l_axe = np.arange(l_min, l_max, self.interpol_step)

            # Save final data as numpy array
            # 0 is x axis, 1 is data, 2 is background
            data = np.nan * \
                np.empty((len(self.scan_files), 3, (len(l_axe))))
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

                ct = f.root.binoculars.counts[:]
                cont = f.root.binoculars.contributions[:]

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

            if verbose:
                print(
                    "Range and stepsize in H: [{0:.3f}: {1:.3f}: {2:.3f}]".format(
                        H[1], H[2], H[3]))
                print(
                    "Range and stepsize in K: [{0:.3f}: {1:.3f}: {2:.3f}]".format(
                        K[1], K[2], K[3]))
                print(
                    "Range and stepsize in L: [{0:.3f}: {1:.3f}: {2:.3f}]".format(
                        L[1], L[2], L[3]))

            # CTR intensity, define roi indices
            st_H_roi = self.find_nearest(
                scan_h_axe, self.CTR_range_H[0])
            end_H_roi = self.find_nearest(
                scan_h_axe, self.CTR_range_H[1])

            st_K_roi = self.find_nearest(
                scan_k_axe, self.CTR_range_K[1])
            end_K_roi = self.find_nearest(
                scan_k_axe, self.CTR_range_K[0])

            if verbose:
                print(
                    f"""Data ROI: [start_k, end_K, start_H, end_H] = \
                    \n\t[{st_K_roi[0]}, {end_K_roi[0]}, {st_H_roi[0]}, {end_H_roi[0]}]\
                    \n\t[{st_K_roi[1]}, {end_K_roi[1]}, {st_H_roi[1]}, {end_H_roi[1]}]\
                    """)
                # f"Data ROI: [start_k, end_K, start_H, end_H] = [{st_K_roi}, {end_K_roi}, {st_H_roi}, {end_H_roi}]")
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
            else:
                # Save x axis
                data[i, 0, :len(scan_l_axe)] = scan_l_axe

                # Save intensities
                roi_2D_sum = roi_2D.sum(axis=(0, 1))
                data[i, 1, :len(scan_l_axe)] = roi_2D_sum / nb_pixel_roi

            # Get background
            if center_background == HK_peak:
                # Background intensity, define roi indices
                st_H_background = self.find_nearest(
                    scan_h_axe, self.background_range_H[0])
                end_H_background = self.find_nearest(
                    scan_h_axe, self.background_range_H[1])

                st_K_background = self.find_nearest(
                    scan_k_axe, self.background_range_K[0])
                end_K_background = self.find_nearest(
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
                st_H_background = self.find_nearest(
                    scan_h_axe, self.background_range_H[0])
                end_H_background = self.find_nearest(
                    scan_h_axe, self.background_range_H[1])

                st_K_background = self.find_nearest(
                    scan_k_axe, self.background_range_K[0])
                end_K_background = self.find_nearest(
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
                plt.figure()
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

    def prep_CTR_data_fitaid(
        self,
        folder,
        scan_indices,
        save_name,
        data_type="nisf",
        interpol_step=False,
        verbose=False,
    ):
        """
        Load data prepared with fitaid

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
        self.scan_indices = [str(s) for s in scan_indices]

        # Get all txt files first
        files = [f.split("/")[-1]
                 for f in sorted(glob.glob(f"{folder}/{data_type}*.txt"))]

        # Get scans specified with scan_indices
        self.scan_files = [f for f in files if any(
            ["-" + n + ".txt" in f for n in self.scan_indices])]
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
            fitaid_data = np.loadtxt(fname)

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
                l_min = min(L)
                l_max = max(L)
                l_shape = len(L)
            else:
                l_min = min(l_min, min(L))
                l_max = max(l_max, max(L))
                l_shape = max(l_shape, len(L))

        print("\n###########################################################")
        print(f"Smallest common range in L is [{l_min} : {l_max}]")
        print("###########################################################")

        # Create new x axis for interpolation
        self.interpol_step = interpol_step
        if isinstance(self.interpol_step, float):
            print(f"\nSmallest common range is [{l_min} : {l_max}]")
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
            fitaid_data = np.loadtxt(fname)
            scan_l_axe = fitaid_data[:, 0]
            ctr_data = fitaid_data[:, 1]

            # Interpolate
            if isinstance(self.interpol_step, float):
                data[i, 0, :] = l_axe

                tck = splrep(scan_l_axe, ctr_data, s=0)
                data[i, 1, :] = splev(l_axe, tck)

            else:
                data[i, 0, :] = scan_l_axe
                data[i, 1, :] = ctr_data

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
        labels=False,
        zoom=[None, None, None, None],
        fill=False,
        fill_first=0,
        fill_last=-1,
        log_intensities=True,
        fontsize=15,
    ):
        """
        Plot the CTRs together

        :param numpy_array: path to .npy file on disk.
        :param scan_indices: scan indices of files plotted, in order, used for
         labelling, mandatory because we need to know what we plot!
        :param title: if string, set to figure title
        :param filename: if string, figure will be saved to this path.
        :param figsize: figure size, default is (18, 9)
        :param ncol: columns in label, default is 2
        :param color_dict: dict used for labels, keys are scan index, values are
         colours for matplotlib.
        :param labels: list of labels to use, defaulted to scan index if False
        :param zoom: values used for plot range, default is
         [None, None, None, None], order is left, right, bottom and top.
        :param fill: if True, add filling between two plots
        :param fill_first: index of scan to use for filling
        :param fill_last: index of scan to use for filling
        :param log_intensities: if True, y axis is logarithmic
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
                    l,
                    y_plot,
                    color=color_dict[int(scan_index)],
                    label=label,
                    linewidth=2,
                )

            except KeyError:
                # Take int(scan_index) in case keys are not strings in the dict
                try:
                    plt.plot(
                        l,
                        y_plot,
                        color=color_dict[scan_index],
                        label=label,
                        linewidth=2,
                    )
                except Exception as e:
                    raise e
            except TypeError:  # No special colour
                plt.plot(
                    l,
                    y_plot,
                    label=label,
                    linewidth=2,
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
        plt.legend(fontsize=fontsize, ncol=ncol)

        plt.xlabel("L", fontsize=fontsize)
        plt.ylabel("Intensity (a.u.)", fontsize=fontsize)

        # Title
        if isinstance(title, str):
            plt.title(f"{title}.png", fontsize=20)

        # Save
        plt.tight_layout()
        if filename != None:
            plt.savefig(f"{filename}", bbox_inches='tight')
            print(f"Saved as {filename}")

        plt.show()

    @ staticmethod
    def find_nearest(array, value):
        mask = np.where(array == value)
        if len(mask) == 1:
            try:
                mask = mask[0][0]
                return array[mask], mask
            except IndexError:
                # print("Value is not in array")
                if all(array < value):
                    return array[-1], -1
                elif all(array > value):
                    return array[0], 0

    # def plot_map(
    #     self,
    #     scan_index,
    #     axe,
    #     save = False,
    #     axerange = None
    # ):
    #     """2D plotting tool"""
    #     if axerange != None:
    #         self.prjaxe_range(axe, axerange)
    #         img = self.imgr

    #     if axe == 'H':
    #        img = data[scan_index, ]
    #     if axe == 'K':
    #        img = data[scan_index, ]
    #     if axe == 'L':
    #        img = data[scan_index, ]

    #     if plot == 'YES':
    #         plt.figure(figsize=(16, 9))

    #         plt.imshow(img,
    #             cmap='jet',
    #             #interpolation="nearest",
    #             origin="lower",
    #             #aspect = 'auto',
    #             norm = LogNorm(vmin = 0.01, vmax = 2000),
    #             extent=[axe1.min(),axe1.max(),axe2.min(),axe2.max()])
    #         plt.title(self.fn[-5], fontsize = 20)
    #         plt.xlabel(axe_name1, fontsize = 20)
    #         plt.ylabel(axe_name2, fontsize = 20)
    #         plt.colorbar()
    #         plt.tight_layout()

    #     if save =='YES':
    #         plt.savefig(self.directory+self.fn[:-5]+'_prj'+axe+'.png')
