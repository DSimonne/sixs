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


class Map:
    """
    Loads a Map of the reciprocal space created via binoculars, and provides
    integration methods to analyse the diffracted intensity.
    """

    def __init__(
        self,
        folder,
        scan_indices,
        configuration_file=False,
    ):
        """
        Loads .hdf5 files in param folder that are included in param
        scan_indices.
        :param folder: path to data folder
        :param scan_indices: indices of maps scans, list
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

        # Load data
        self.folder = folder
        self.scan_indices = [str(s) for s in scan_indices]

        # Get all hdf5 files first
        files = [f.split("/")[-1]
                 for f in sorted(glob.glob(f"{folder}/*.hdf5"))]
        print(f"Detected files in {folder}:")
        for f in files:
            print("\t", f)

        # Get scans specified with scan_indices
        self.scan_list = [f for f in files if any(
            ["-" + n + ".hdf5" in f for n in self.scan_indices])]

        print("\n###########################################################")
        print("Working on the following files:")
        for f in self.scan_list:
            print("\t", f)
        print("###########################################################\n")

    def prep_data(
        self,
        save_name,
        interpol_step=False,
        CTR_width_H=0.02,
        CTR_width_K=0.02,
        background_width_H=0.01,
        background_width_K=0.01,
        HK_peak=[-1, 1],
        center_background=[-1, 1],
        verbose=False,
    ):
        """
        Prepare the data for plots by interpolating on the smallest common range
        If center_background is different from HK_peak, the goal is to avoid
        summing the CTR intensity at the same time as the diffraction rings

        Saves the result as a numpy array on disk.

        :param save_name: name of file in which the results are saved, saved in
         self.folder.
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

        # Parameters of rod
        self.HK_peak = HK_peak

        self.CTR_width_H = CTR_width_H
        self.CTR_width_K = CTR_width_K

        self.CTR_range_H = [
            self.HK_peak[0] - self.CTR_width_H,
            self.HK_peak[0] + self.CTR_width_H
        ]
        self.CTR_range_K = [
            self.HK_peak[1] - self.CTR_width_K,
            self.HK_peak[1] + self.CTR_width_K
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
                self.HK_peak[0] - self.background_width_H,
                self.HK_peak[0] + self.background_width_H
            ]
            self.background_range_K = [
                self.HK_peak[1] - self.background_width_K,
                self.HK_peak[1] + self.background_width_K
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
        if isinstance(interpol_step, float):
            print("\n###########################################################")
            print("Finding smallest common range in L")
            print("Depends on the config file in binoculars-process.")
            print("###########################################################")
        else:
            # No interpolation, we assume that all the scan have the same l axis
            print("\n###########################################################")
            print("No interpolation")
            print("###########################################################")

        for i, fname in enumerate(self.scan_list):

            with tb.open_file(self.folder + fname, "r") as f:
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

        if isinstance(interpol_step, float):
            # Determine interpolation range
            print("\n###########################################################")
            print(f"Smallest common range in L is [{l_min} : {l_max}]")
            print("###########################################################")

            self.interpol_step = interpol_step
            self.l_axe = np.arange(l_min, l_max, self.interpol_step)

            # Save final data as numpy array
            # 0 is x axis, 1 is data, 2 is background
            data = np.nan * \
                np.empty((len(self.scan_list), 3, (len(self.l_axe))))
        else:
            # Save final data as numpy array
            # 0 is x axis, 1 is data, 2 is background
            data = np.nan * np.empty((len(self.scan_list), 3, l_shape))

        # Iterate on each file
        for i, fname in enumerate(self.scan_list):
            if verbose:
                print("\n###########################################################")
                print(f"Opening file {fname} ...")

            with tb.open_file(self.folder + fname, "r") as f:

                ct = f.root.binoculars.counts[:]
                cont = f.root.binoculars.contributions[:]

                raw_data = np.divide(ct, cont, where=cont != 0)

                H = f.root.binoculars.axes.H[:]
                K = f.root.binoculars.axes.K[:]
                L = f.root.binoculars.axes.L[:]

                scan_h_axe = np.round(np.linspace(
                    H[1], H[2], 1 + int(H[5] - H[4])), 3)  # xaxe
                scan_k_axe = np.round(np.linspace(
                    K[1], K[2], 1 + int(K[5] - K[4])), 3)  # yaxe
                # scan_l_axe = np.round(np.linspace(L[1], L[2], 1 + int(L[5] - L[4])), 3) #zaxe
                scan_l_axe = np.round(np.linspace(
                    L[1], L[2], 1 + int(L[5] - L[4])), 3)  # zaxe

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
                    scan_h_axe, self.CTR_range_H[0])[1]
                end_H_roi = self.find_nearest(
                    scan_h_axe, self.CTR_range_H[1])[1]

                st_K_roi = self.find_nearest(
                    scan_k_axe, self.CTR_range_K[0])[1]
                end_K_roi = self.find_nearest(
                    scan_k_axe, self.CTR_range_K[1])[1]

                if verbose:
                    print(
                        f"Data ROI = [{st_H_roi}, {end_H_roi}, {st_K_roi}, {end_K_roi}]")

                # Get data only in specific ROI
                roi_2D = raw_data[st_H_roi:end_H_roi, st_K_roi:end_K_roi, :]

                # Interpolate over common l axis
                if isinstance(interpol_step, float):
                    # Save x axis
                    data[i, 0, :] = self.l_axe

                    # Save intensities
                    tck = splrep(scan_l_axe, roi_2D.sum(axis=(0, 1)), s=0)
                    roi_2D_sum = splev(self.l_axe, tck)
                    data[i, 1, :] = roi_2D_sum
                else:
                    # Save x axis
                    data[i, 0, :len(scan_l_axe)] = scan_l_axe

                    # Save intensities
                    roi_2D_sum = roi_2D.sum(axis=(0, 1))
                    data[i, 1, :len(scan_l_axe)] = roi_2D_sum

                # Get background
                if center_background == HK_peak:
                    # Background intensity, define roi indices
                    st_H_background = self.find_nearest(
                        scan_h_axe, self.background_range_H[0])[1]
                    end_H_background = self.find_nearest(
                        scan_h_axe, self.background_range_H[1])[1]

                    st_K_background = self.find_nearest(
                        scan_k_axe, self.background_range_K[0])[1]
                    end_K_background = self.find_nearest(
                        scan_k_axe, self.background_range_K[1])[1]

                    if verbose:
                        print(
                            f"Background ROI = [{st_H_background}, {end_H_background}, {st_K_background}, {end_K_background}]")
                        print(
                            "###########################################################")

                    background_H = raw_data[st_H_background:end_H_background,
                                            st_K_roi:end_K_roi,
                                            :  # all data in L
                                            ]
                    background_K = raw_data[st_H_roi:end_H_roi,
                                            st_K_background:end_K_background,
                                            :  # all data in L
                                            ]

                    # Interpolate
                    if isinstance(interpol_step, float):
                        tck_H = splrep(
                            scan_l_axe, background_H.sum(axis=(0, 1)), s=0)
                        tck_K = splrep(
                            scan_l_axe, background_K.sum(axis=(0, 1)), s=0)

                        background_H_sum = splev(self.l_axe, tck_H)
                        background_K_sum = splev(self.l_axe, tck_K)

                        # Save background
                        # Subtract twice here because we are using two rectangles
                        # that overlap our data
                        data[i, 2, :] = background_H_sum + \
                            background_K_sum - 2 * data[i, 1, :]
                    else:
                        background_H_sum = background_H.sum(axis=(0, 1))
                        background_K_sum = background_K.sum(axis=(0, 1))

                        # Save background
                        data[i, 2, :len(scan_l_axe)] = background_H_sum + \
                            background_K_sum - 2 * data[i, 1, :len(scan_l_axe)]

                elif isinstance(center_background, list) and center_background != HK_peak:
                    # Background intensity, define roi indices
                    st_H_background = self.find_nearest(
                        scan_h_axe, self.background_range_H[0])[1]
                    end_H_background = self.find_nearest(
                        scan_h_axe, self.background_range_H[1])[1]

                    st_K_background = self.find_nearest(
                        scan_k_axe, self.background_range_K[0])[1]
                    end_K_background = self.find_nearest(
                        scan_k_axe, self.background_range_K[1])[1]

                    if verbose:
                        print(
                            f"Background ROI = [{st_H_background}, {end_H_background}, {st_K_background}, {end_K_background}]")
                        print(
                            "###########################################################")

                    background_2D = raw_data[st_H_background:end_H_background,
                                             st_K_background:end_K_background,
                                             :
                                             ]

                    # Interpolate
                    if isinstance(interpol_step, float):
                        tck_2D = splrep(
                            scan_l_axe, background_2D.sum(axis=(0, 1)), s=0)
                        background_2D_sum = splev(self.l_axe, tck_2D)

                        # Save background
                        data[i, 2, :] = background_2D_sum
                    else:
                        background_2D_sum = background_2D.sum(axis=(0, 1))

                        # Save background
                        data[i, 2, :len(scan_l_axe)] = background_2D_sum

                else:
                    print("No background subtracted")
                    print("###########################################################")

        # Saving
        self.save_name = save_name
        print("\n###########################################################")
        print(f"Saving data as: {self.folder}{self.save_name}.npy")
        print("###########################################################")
        np.save(self.folder + self.save_name, data)

    def prep_data_fitaid(
        self,
        constant,
        variable,
        data_type="nisf",
        CTR_width_H=0.02,
        CTR_width_K=0.02,
        background_width_H=0.02,
        background_width_K=0.01,
        HK_peak=[-1, 1],
        interpol_step=0.03
    ):
        """
        Load data prepared with fitaid (nisf data)

        :param constant:
        :param variable:
        :param data_type:
        :param CTR_width_H:
        :param CTR_width_K:
        :param background_width_H:
        :param background_width_K:
        :param HK_peak:
        :param interpol_step:
        """

        # Parameters of rod
        self.CTR_width_H = CTR_width_H
        self.CTR_width_K = CTR_width_K

        self.background_width_H = background_width_H + CTR_width_H
        self.background_width_K = background_width_K + CTR_width_K

        self.HK_peak = HK_peak

        self.CTR_range_H = [self.HK_peak[0] -
                            self.CTR_width_H, self.HK_peak[0] + self.CTR_width_H]
        self.CTR_range_K = [self.HK_peak[1] -
                            self.CTR_width_H, self.HK_peak[1] + self.CTR_width_K]

        self.background_range_H = [
            self.HK_peak[0] - self.background_width_H, self.HK_peak[0] + self.background_width_H]
        self.background_range_K = [
            self.HK_peak[1] - self.background_width_K, self.HK_peak[1] + self.background_width_K]

        self.interpol_step = interpol_step

        print("Finding smallest common range in L, careful, depends on the input of the initial map.")

        for i, fname in enumerate(self.scan_list):

            fname = self.folder + data_type + "_" + fname

            data = np.loadtxt(fname)

            L = data[:, 0]

            if i == 0:
                l_min = L[1]
                l_max = L[-1]
            else:
                l_min = min(l_min, L[1])
                l_max = max(l_max, L[-1])

        print(f"\nSmallest common range is [{l_min} : {l_max}]")

        self.l_axe = np.arange(l_min, l_max, self.interpol_step)

        # round to step
        self.x_min = self.l_axe[0] // self.x_tick_step * self.x_tick_step
        self.x_max = self.l_axe[-1] // self.x_tick_step * self.x_tick_step

        # too big so save as numpy array
        self.data = np.zeros((len(self.scan_list), 3, (len(self.l_axe))))

        # background already subtracted; but still in big array for plotting function, just equal to zeros

        for i, fname in enumerate(self.scan_list):

            fname = self.folder + data_type + "_" + fname

            data = np.loadtxt(fname)
            scan_l_axe = data[:, 0]
            ctr_data = data[:, 1]

            # Use 3D arrays
            self.data[i, 0, :] = self.l_axe

            # Interpolate
            tck = splrep(scan_l_axe, ctr_data, s=0)
            self.data[i, 1, :] = splev(self.l_axe, tck)

    def plot_CTR(
        self,
        numpy_array=False,
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
    ):
        """
        Plot the CTR together
        :param numpy_array: False if fitaid data (data is saved as
         self.scan_data), otherwise path to .npy file on disk.
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
        """

        # Create figure
        plt.figure(figsize=figsize)
        if log_intensities:
            plt.semilogy()
        plt.grid()

        # Only data saved as attribute if not too big
        try:
            data = self.data
            print("Loaded self.data")

        # Otherwise as np array on disk
        except:
            data = np.load(numpy_array)
            print("Loaded", numpy_array)

        # Iterate on data
        for (i, arr), scan_index in zip(enumerate(data), self.scan_indices):
            # take l again but still or better to keep x values in the same array with y
            l = arr[0, :]  # x axis
            y = arr[1, :]  # data
            b = arr[2, :]  # background

            # Remove background
            # y_plot = y-b
            y_plot = y

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

        plt.xlabel("L", fontsize=self.fontsize)
        plt.ylabel("Intensity (a.u.)", fontsize=self.fontsize)

        # Title
        if isinstance(title, str):
            plt.title(f"{title}.png", fontsize=20)

        # Save
        plt.tight_layout()
        if filename != None:
            plt.savefig(f"{filename}", bbox_inches='tight')
            print(f"Saved as {filename}")

        plt.show()

    # More methods below

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
                    return None, -1
                elif all(array > value):
                    return None, 0

    # def hdf2png(self, scan_index, axe, plot = 'YES', save = 'NO', axerange = None):
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
