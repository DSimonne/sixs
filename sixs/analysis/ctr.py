from matplotlib.colors import LogNorm
import warnings
from importlib import reload
import tables as tb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob
from phdutils.binoculars import binUtil3 as bin3
from phdutils.sixs import utilities3 as ut3
from phdutils.sixs import ReadNxs4 as rn4
from scipy import interpolate

class Map(object):
    """docstring for Map"""

    def __init__(self, folder, scan_indices):

        super(Map, self).__init__()

        # class arguments
        self.folder = folder
        self.scan_indices = [str(s) for s in scan_indices]

        all_files = [f.split("/")[-1] for f in glob.glob(f"{folder}/*.hdf5")]

        # self.scan_list = [f for f in all_files if any(["_" + n + ".hdf5" in f for n in self.scan_indices])]

        # Out of plane maps
        self.scan_list = [f for f in all_files if any(
            ["-" + n + ".hdf5" in f for n in self.scan_indices])]

        # plotting variables
        self.linewidth = 2
        self.linewidth_hline = 1.3
        self.linewidth_vline = 1.3
        self.alpha_hline = 0.7
        self.alpha_vline = 0.7
        self.color_hline = "black"
        self.color_vline = "black"

        self.filling_alpha = 0.2

        self.fontsize = 20

        self.y_og = 0

        self.title_fontsize = 26

    @classmethod
    def get_class_name(cls):
        print(cls.__name__)
        return cls.__name__

    def prep_data(self, constant, variable, interpol_L_step=0.01, CTR_width_H=0.02, CTR_width_K=0.02,
                  background_width_H=0.01, background_width_K=0.01, HK_peak=[-1, 1], center_background=[-1, 1]):
        """Prepare the data for plots by interpolating on the smallest common range
        If center_background is different from HK_peak, the goal is to avoid summing the CTR intensity at the same time 
        as the diffraction rings """

        # Parameters of rod
        self.HK_peak = HK_peak

        self.CTR_width_H = CTR_width_H
        self.CTR_width_K = CTR_width_K

        self.CTR_range_H = [self.HK_peak[0] -
                            self.CTR_width_H, self.HK_peak[0] + self.CTR_width_H]
        self.CTR_range_K = [self.HK_peak[1] -
                            self.CTR_width_H, self.HK_peak[1] + self.CTR_width_K]

        # Background parameter
        self.center_background = center_background

        if center_background == HK_peak:
            self.background_width_H = CTR_width_H + background_width_H
            self.background_width_K = CTR_width_K + background_width_K

            self.background_range_H = [
                self.HK_peak[0] - self.background_width_H, self.HK_peak[0] + self.background_width_H]
            self.background_range_K = [
                self.HK_peak[1] - self.background_width_K, self.HK_peak[1] + self.background_width_K]

        else:
            self.background_width_H = background_width_H
            self.background_width_K = background_width_K

            self.background_range_H = [
                self.center_background[0] - self.background_width_H, self.center_background[0] + self.background_width_H]
            self.background_range_K = [
                self.center_background[1] - self.background_width_K, self.center_background[1] + self.background_width_K]

        print("Finding smallest common range in L, careful, depends on the input of the initial map.")

        self.constant = constant
        self.variable = variable

        for i, fname in enumerate(self.scan_list):

            with tb.open_file(self.folder + fname, "r") as f:
                # print(f"\nOpening file {fname} ...")
                # H = f.root.binoculars.axes.H[:]
                # K = f.root.binoculars.axes.K[:]
                L = f.root.binoculars.axes.L[:]

            # print("Range and stepsize in H: [{0:.3f}: {1:.3f}: {2:.3f}]".format(f.H[1], f.H[2], f.H[3]))
            # print("Range and stepsize in K: [{0:.3f}: {1:.3f}: {2:.3f}]".format(f.K[1], f.K[2], f.K[3]))
            # print("Range and stepsize in L: [{0:.3f}: {1:.3f}: {2:.3f}]".format(f.L[1], f.L[2], f.L[3]))

            if i == 0:
                l_min = L[1]
                l_max = L[2]
            else:
                l_min = min(l_min, L[1])
                l_max = max(l_max, L[2])

        print(f"\nSmallest common range is [{l_min} : {l_max}]")
        self.interpol_L_step = interpol_L_step

        self.l_axe = np.arange(l_min, l_max, self.interpol_L_step)

        # Round future plot limits to  actual stepsize
        # self.x_min = self.l_axe[0] // self.x_tick_step * self.x_tick_step
        # self.x_max = self.l_axe[-1] // self.x_tick_step * self.x_tick_step

        # Store name of each file
        self.names = []

        # Save as numpy array
        span_data = np.empty((len(self.scan_list), 2, (len(self.l_axe))))

        # For each file
        for i, scan in enumerate(self.scan_list):

            with tb.open_file(self.folder + scan, "r") as f:
                print(f"\nOpening file {scan} ...")

                ct = f.root.binoculars.counts[:]
                cont = f.root.binoculars.contributions[:]

                raw_data = np.divide(ct, cont, where=cont != 0)

                # swap axes for hkl indices to follow miller convention (h,k,l)
                # self.hkl_data = np.swapaxes(self.raw_data, 0, 2)

                H = f.root.binoculars.axes.H[:]
                K = f.root.binoculars.axes.K[:]
                L = f.root.binoculars.axes.L[:]

                scan_h_axe = np.linspace(
                    H[1], H[2], 1 + int(H[5] - H[4]))  # xaxe
                scan_k_axe = np.linspace(
                    K[1], K[2], 1 + int(K[5] - K[4]))  # yaxe
                # scan_l_axe = np.round(np.linspace(L[1], L[2], 1 + int(L[5] - L[4])), 3) #zaxe
                scan_l_axe = np.linspace(
                    L[1], L[2], 1 + int(L[5] - L[4]))  # zaxe

                # print("Range and stepsize in H: [{0:.3f}: {1:.3f}: {2:.3f}]".format(H[1], H[2], H[3]))
                # print("Range and stepsize in K: [{0:.3f}: {1:.3f}: {2:.3f}]".format(K[1], K[2], K[3]))
                # print("Range and stepsize in L: [{0:.3f}: {1:.3f}: {2:.3f}]".format(L[1], L[2], L[3]))

                # CTR intensity, define roi indices
                st_H_roi = ut3.find_nearest(scan_h_axe, self.CTR_range_H[0])[0]
                end_H_roi = ut3.find_nearest(
                    scan_h_axe, self.CTR_range_H[1])[0]

                st_K_roi = ut3.find_nearest(scan_k_axe, self.CTR_range_K[0])[0]
                end_K_roi = ut3.find_nearest(
                    scan_k_axe, self.CTR_range_K[1])[0]

                print(st_H_roi, end_H_roi)
                print(st_K_roi, end_K_roi)

                print("roi 2D")
                roi_2D = raw_data[st_H_roi:end_H_roi, st_K_roi:end_K_roi, :]

                # Interpolate over common l axis
                tck = interpolate.splrep(
                    scan_l_axe, roi_2D.sum(axis=(0, 1)), s=0)
                span_data[i, 0, :] = interpolate.splev(self.l_axe, tck)

                if center_background == HK_peak:
                    # Background intensity, define roi indices
                    st_H_background = ut3.find_nearest(
                        scan_h_axe, self.background_range_H[0])[0]
                    end_H_background = ut3.find_nearest(
                        scan_h_axe, self.background_range_H[1])[0]

                    st_K_background = ut3.find_nearest(
                        scan_k_axe, self.background_range_K[0])[0]
                    end_K_background = ut3.find_nearest(
                        scan_k_axe, self.background_range_K[1])[0]

                    background_H = raw_data[st_H_background:end_H_background,
                                            st_K_roi:end_K_roi, :]
                    background_K = raw_data[st_H_roi:end_H_roi,
                                            st_K_background:end_K_background, :]

                    print(st_H_background, end_H_background)
                    print(st_K_background, end_K_background)

                    # Interpolate
                    tck_H = interpolate.splrep(
                        scan_l_axe, background_H.sum(axis=(0, 1)), s=0)
                    tck_K = interpolate.splrep(
                        scan_l_axe, background_K.sum(axis=(0, 1)), s=0)

                    span_data[i, 1, :] = interpolate.splev(
                        self.l_axe, tck_H) + interpolate.splev(self.l_axe, tck_K) - 2 * span_data[i, 1, :]

                else:
                    print("background")
                    # Background intensity, define roi indices
                    st_H_background = ut3.find_nearest(
                        scan_h_axe, self.background_range_H[0])[0]
                    end_H_background = ut3.find_nearest(
                        scan_h_axe, self.background_range_H[1])[0]

                    st_K_background = ut3.find_nearest(
                        scan_k_axe, self.background_range_K[0])[0]
                    end_K_background = ut3.find_nearest(
                        scan_k_axe, self.background_range_K[1])[0]

                    background_2D = raw_data[st_H_background:end_H_background,
                                             st_K_background:end_K_background, :]

                    # Interpolate
                    tck_2D = interpolate.splrep(
                        scan_l_axe, background_2D.sum(axis=(0, 1)), s=0)

                    span_data[i, 1, :] = interpolate.splev(self.l_axe, tck_2D)

                self.names.append(scan.split(".hdf5")[
                                  0].split("_")[-1].replace("-", ":"))

        self.names = [x for _, x in sorted(
            zip([int(f.split(":")[-1]) for f in self.names], self.names))]
        np.save(self.folder + self.constant +
                self.variable + ".npy", span_data)

    def prep_data_fitaid(self, constant, variable, data_type="nisf", CTR_width_H=0.02, CTR_width_K=0.02,
                         background_width_H=0.02, background_width_K=0.01, HK_peak=[-1, 1], interpol_L_step=0.03):
        """Load data prepared with fitaid"""

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

        self.constant = constant
        self.variable = variable

        # store name of each file
        self.names = []

        for i, scan in enumerate(self.scan_list):
            self.names.append(scan.split(".txt")[0].split(
                "_")[-1].replace("-", ":"))

            fname = self.folder + data_type + "_" + scan

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

        for i, scan in enumerate(self.scan_list):

            fname = self.folder + data_type + "_" + scan

            data = np.loadtxt(fname)
            scan_l_axe = data[:, 0]
            ctr_data = data[:, 1]

            # Use 3D arrays
            self.span_data[i, 0, :] = self.l_axe

            # Interpolate
            tck = interpolate.splrep(scan_l_axe, ctr_data, s=0)
            self.span_data[i, 1, :] = interpolate.splev(self.l_axe, tck)

    def plot_CTR(self, scan_gas_dict, scan_temp_dict, figsize=(18, 6), fill_first=0, fill_last=-1, zoom=[None, None, None, None], title="CTR", save_as="CTR.png"):
        """Variable is gas or temp"""

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
