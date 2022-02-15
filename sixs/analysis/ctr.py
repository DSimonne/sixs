import numpy as np
import tables as tb
import pandas as pd
import glob
import os
import inspect
import yaml

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy import interpolate

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
        interpol_L_step=0.01,
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
        :param interpol_L_step: step size in interpolation along L, to avoid
         problem with analysing CTR with different steps.
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

        else:
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

        # Start interpolation
        print("###########################################################")
        print("Finding smallest common range in L, careful, depends on the input of the initial map.")
        print("###########################################################")

        for i, fname in enumerate(self.scan_list):

            with tb.open_file(self.folder + fname, "r") as f:
                H = f.root.binoculars.axes.H[:]
                K = f.root.binoculars.axes.K[:]
                L = f.root.binoculars.axes.L[:]

            if verbose:
                print("\n###########################################################")
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
            else:
                l_min = min(l_min, L[1])
                l_max = max(l_max, L[2])

        # Determine interpolation range
        print("\n###########################################################")
        print(f"Smallest common range is [{l_min} : {l_max}]")
        print("###########################################################")

        self.interpol_L_step = interpol_L_step
        self.l_axe = np.arange(l_min, l_max, self.interpol_L_step)

        # Round future plot limits to  actual stepsize
        # self.x_min = self.l_axe[0] // self.x_tick_step * self.x_tick_step
        # self.x_max = self.l_axe[-1] // self.x_tick_step * self.x_tick_step

        # Save final data as numpy array
        span_data = np.empty((len(self.scan_list), 2, (len(self.l_axe))))

        # Iterate on each file
        for i, fname in enumerate(self.scan_list):
            if verbose:
                print("\n###########################################################")
                print(f"Opening file {fname} ...")

            with tb.open_file(self.folder + fname, "r") as f:

                ct = f.root.binoculars.counts[:]
                cont = f.root.binoculars.contributions[:]

                raw_data = np.divide(ct, cont, where=cont != 0)

                # swap axes for hkl indices to follow miller convention (h,k,l)
                # self.hkl_data = np.swapaxes(self.raw_data, 0, 2)

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
                tck = interpolate.splrep(
                    scan_l_axe, roi_2D.sum(axis=(0, 1)))
                span_data[i, 0, :] = interpolate.splev(self.l_axe, tck)

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
                    tck_H = interpolate.splrep(
                        scan_l_axe, background_H.sum(axis=(0, 1)))
                    tck_K = interpolate.splrep(
                        scan_l_axe, background_K.sum(axis=(0, 1)))

                    # Subtract twice here because we are using two rectangles
                    # that overlap our data
                    span_data[i, 1, :] = interpolate.splev(
                        self.l_axe, tck_H) + interpolate.splev(self.l_axe, tck_K) - 2 * span_data[i, 1, :]

                else:
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
                    tck_2D = interpolate.splrep(
                        scan_l_axe, background_2D.sum(axis=(0, 1)))

                    # Only subtract once bc not on the peak
                    span_data[i, 1, :] = interpolate.splev(self.l_axe, tck_2D)

        # Saving
        print("\n###########################################################")
        print(f"Saving data as: {self.folder}{save_name}.npy")
        print("###########################################################")

        np.save(self.folder + save_name, span_data)

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
        interpol_L_step=0.03
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
        :param interpol_L_step:
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

        self.interpol_L_step = interpol_L_step

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

        self.l_axe = np.arange(l_min, l_max, self.interpol_L_step)

        # round to step
        self.x_min = self.l_axe[0] // self.x_tick_step * self.x_tick_step
        self.x_max = self.l_axe[-1] // self.x_tick_step * self.x_tick_step

        # too big so save as numpy array
        self.span_data = np.zeros((len(self.scan_list), 3, (len(self.l_axe))))

        # background already subtracted; but still in big array for plotting function, just equal to zeros

        for i, fname in enumerate(self.scan_list):

            fname = self.folder + data_type + "_" + fname

            data = np.loadtxt(fname)
            scan_l_axe = data[:, 0]
            ctr_data = data[:, 1]

            # Use 3D arrays
            self.span_data[i, 0, :] = self.l_axe

            # Interpolate
            tck = interpolate.splrep(scan_l_axe, ctr_data)
            self.span_data[i, 1, :] = interpolate.splev(self.l_axe, tck)

    def plot_CTR(
        self,
        scan_gas_dict,
        scan_temp_dict,
        figsize=(18, 6),
        fill_first=0,
        fill_last=-1,
        zoom=[None, None, None, None],
        title="CTR",
        save_as="CTR.png"
    ):
        """
        Variable is gas or temp

        :param scan_gas_dict:
        :param scan_temp_dict:
        :param figsize:
        :param fill_first:
        :param fill_last:
        :param zoom:
        :param title:
        :param save_as:
        """

        if self.variable == "temp":
            self.labels = {
                scan_nb: f"{scan_temp_dict[scan_nb]} °C" for scan_nb in self.names}
            self.colors = {
                scan_nb: self.qualitative_colors_2[scan_temp_dict[scan_nb]] for scan_nb in self.names}

        elif self.variable == "gas":
            self.labels = {
                scan_nb: f"{scan_gas_dict[scan_nb]}" for scan_nb in self.names}
            self.colors = {
                scan_nb: self.fivethirtyeight_colors[scan_gas_dict[scan_nb]] for scan_nb in self.names}

        plt.figure(figsize=figsize, dpi=150)
        plt.semilogy()

        # Only data saved as attribute if not too big
        try:
            span_data = self.span_data
            print("fitaid data")

        # Otherwise as np array on disk
        except:
            span_data = np.load(
                self.folder + self.constant + self.variable + ".npy")

        for (i, arr), scan_nb in zip(enumerate(span_data), self.names):
            # print(arr.shape)
            # no need really to take l again but still it's better to keep x values in the same array with y
            l = self.l_axe
            y = arr[0, :]
            b = arr[1, :]

            plt.plot(l,
                     y,
                     label=self.labels[scan_nb],
                     color=self.colors[scan_nb],
                     linewidth=self.linewidth)

            # if i==fill_first:
            #     y_first = norm_y

            # elif i==len(span_data) + fill_last:
            #     y_last = norm_y

        # Generate a bolded horizontal line at y = 5 to highlight background
        # plt.axhline(y = self.y_og, color = self.color_hline, linewidth = self.linewidth_hline, alpha = self.alpha_hline)

        # Generate a bolded vertical line at x = 0 to highlight origin
        # plt.axvline(x = l[0], color = self.color_vline, linewidth = self.linewidth_vline, alpha = self.alpha_vline)

        # Ticks
        plt.xticks(fontsize=self.fontsize)
        plt.yticks(fontsize=self.fontsize)

        # Range
        plt.xlim(left=zoom[0], right=zoom[1])
        plt.ylim(bottom=zoom[2], top=zoom[3])

        # Adding a title and a subtitle
        plt.title(title, fontsize=self.title_fontsize)

        # if self.variable == "temp":
        #     plt.text(x = self.subtitle_x, y = self.subtitle_y,
        #         s = f"""Comparison of the integrated CTR intensity under condition {self.constant} at different temperature as a function of the miller index L.
        #             The intensity is integrated for H $\in$ {self.CTR_range_H} and K $\in$ {self.CTR_range_K}.
        #             The background is integrated for H $\in$ {self.background_range_H} and K $\in$ {self.background_range_K}, minus the smaller ROI.""",
        #         fontsize = self.fontsize, alpha = self.subtitle_alpha)

        # elif self.variable == "gas":
        #     plt.text(x = self.subtitle_x, y = self.subtitle_y,
        #         s = f"""Comparison of the integrated CTR intensity at {self.constant} °C as a function of the miller index L for different gas atmosphere.
        #             The intensity is integrated for H $\in$ {self.CTR_range_H} and K $\in$ {self.CTR_range_K}.
        #             The background is integrated for H $\in$ {self.background_range_H} and K $\in$ {self.background_range_K}, minus the smaller ROI.""",
        #         fontsize = self.fontsize, alpha = self.subtitle_alpha)

        # Add filling
        # try:
        #     plt.fill_between(l, y_first, y_last, alpha = self.filling_alpha)
        # except:
        #     print('Shapes do not coincide')

        plt.legend(bbox_to_anchor=(1, 1), loc="upper left",
                   fontsize=self.fontsize)

        plt.ylabel("Normalized Intensity", fontsize=self.fontsize)
        plt.xlabel("L", fontsize=self.fontsize)
        plt.tight_layout()
        # plt.savefig(f"Images/ctr/CTR_{self.constant}.svg")
        plt.savefig(f"Images/ctr/{save_as}", bbox_inches='tight')
        print(f"Saved as Images/ctr/{save_as}")
        plt.show()

    @staticmethod
    def find_nearest(array, value,):
        mask = np.where(array == value)
        if len(mask) == 1:
            mask = mask[0][0]
        return array[mask], mask

    # def hdf2png(self, scan_index, axe, plot = 'YES', save = 'NO', axerange = None):
    #     """2D plotting tool"""
    #     if axerange != None:
    #         self.prjaxe_range(axe, axerange)
    #         img = self.imgr

    #     if axe == 'H':
    #        img = span_data[scan_index, ]
    #     if axe == 'K':
    #        img = span_data[scan_index, ]
    #     if axe == 'L':
    #        img = span_data[scan_index, ]

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
