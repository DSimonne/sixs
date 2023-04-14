import numpy as np
import tables as tb
import glob
import os
import inspect
import yaml
import sixs
from scipy import interpolate

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import ipywidgets as widgets
from ipywidgets import interactive, fixed, Layout


class Reflectivity:
    """
    Contains methods to load reflectivity data collected at SixS. To analyze the
    data, proceed as follows:

    1) Initiaze the Class
    2) Import the `.nxs` files with the `prep_nxs_data` method. Be careful about
    the attenuator values as well as the ROI, you can see if the ROI is good
    with the `compare_roi` method.
    3) Normalize the data with the `normalize_data` method.
    3) Extract the data as .dot files, it can then be analyzed using GenX.

    You can plot the data with the `plot_refl` method.

    X-ray reflectivity (XRR) is a technique used in materials science and
    surface science to study the structure and properties of thin films and
    multilayered structures. It is a non-destructive, non-contact method that
    involves shining a beam of X-rays onto a sample and measuring the intensity
    of the reflected X-rays as a function of the incident angle.

    XRR is based on the principle of interference of X-rays that are reflected
    from different interfaces within a layered structure. When X-rays strike an
    interface between two materials with different electron densities, such as a
    film and a substrate, a portion of the X-rays is reflected back while the
    rest penetrates into the material. The reflected X-rays interfere with each
    other, resulting in a pattern of constructive and destructive interference
    that can be detected and analyzed to obtain information about the structure
    and properties of the sample.

    By varying the incident angle of the X-rays, XRR can provide information
    about the thickness, density, and roughness of thin films, as well as the
    composition and interface roughness of multilayered structures. XRR is
    widely used in fields such as thin film deposition, surface chemistry,
    nanoscience, and materials characterization to study a wide range of
    materials, including metals, polymers, semiconductors, and thin biological
    films.

    X-ray reflectivity is a powerful technique that provides valuable insights
    into the structural properties of thin films and multilayered structures,
    making it a valuable tool in materials science research and industrial
    applications.
    """

    def __init__(
        self,
        folder,
        scan_indices,
        configuration_file=False,
        verbose=True,
    ):
        """
        Initializes the Class by finding the .nxs files in the `folder` that
        match the values in `scan_indices`.

        :param folder: path to data folder
        :param scan_indices: indices of reflectivity scans, list
        :param configuration_file: str, .yml file that stores metadata
         specific to the reaction, if False, default to path_package
         + "experiments/ammonia.yml"
        :param verbose: True to print extra informations
        """
        path_package = inspect.getfile(sixs).split("__")[0]

        # Load configuration file
        print("###########################################################")
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

                for key in yaml_parsed_file:
                    setattr(self, key, yaml_parsed_file[key])
                print("Loaded configuration file.")
                print("###########################################################\n")

        # Init class arguments
        self.folder = folder
        self.scan_indices = [str(s) for s in scan_indices]

        # Find files in folder depending on data format
        files = [f.split("/")[-1]
                 for f in sorted(glob.glob(f"{self.folder}/*.nxs"))]
        if verbose:
            print("\n###########################################################")
            print(f"Detected files in {self.folder}:")
            for f in files:
                print("\t", f)
            print("###########################################################\n")

        # Get files of interest based on scan_indices
        self.scan_list = [f for f in files if any(
            [n in f for n in self.scan_indices])]

        self.data_key = "data_14"
        self.piezo_attenuators_key = "data_25"
        self.pneumatic_attenuators_key = "data_26"
        self.acquisition_time_key = "i14-c-c00-ex-config-global"
        self.detector_key = "i14-c-c00-ex-config-xpads140"
        self.mu_key = "data_40"
        self.delta_key = "data_41"
        self.gamma_key = "data_42"
        self.etaa_key = "data_43"
        self.beta_key = "data_44"
        self.wavelength_key = "i14-c-c02-op-mono"
        # /com/SIXS/i14-c-cx1-ex-med-h-dif-group.1

        print("\n###########################################################")
        print("Working on the following files:")
        for f in self.scan_list:
            print("\t", f)
        print("###########################################################\n")

    def prep_nxs_data(
        self,
        roi,
        piezo_attenuators_coef=None,
        pneumatic_attenuators_coef=None,
    ):
        """
        Load the nexus files and integrate the reflectivity intensity inside the
        given Region Of Interest (ROI).
        Use the attenuators coefficient to correct the data if you see that
        values before and after an attenuator change do not match.

        Careful, when loading the nexus file, the values of the motors, ROI, the
        detector image, etc ... are found thanks to a hard coded part of the code.
        You must change that if there is a NoSuchNodeError error.
        The hardcoded values are attributes of the class:
            self.pneumatic_attenuators_key
            self.piezo_attenuators_key
            self.data_key
            self.detector_key
            self.detector_key
            self.acquisition_time_key
            self.mu_key
            self.beta_key
            self.delta_key
            self.etaa_key
            self.gamma_key
            self.wavelength_key

        :param roi: int or container
            if int, use this roi, if container of length 4, define roi as
            [roi[0], roi[1], roi[2], roi[3]]
        :param piezo_attenuators_coef: None, if not specified the value is
            extracted from the file, otherwise provide float
        :param pneumatic_attenuators_coef: None, if not specified the value is
            extracted from the file, otherwise provide float
        """
        self.intensities = []
        self.mu, self.beta, self.delta, self.etaa, self.gamma = [], [], [], [], []
        self.wavelength = []

        # Iterate on all scans
        for j, file in enumerate(self.scan_list):
            print("###########################################################")
            print("Opening", file)
            with tb.open_file(self.folder + file) as f:
                if pneumatic_attenuators_coef is None:
                    pneumatic_attenuators_coef = f.root.com.SIXS["i14-c-c00-ex-config-att-old"].att_coef[...]
                pneumatic_attenuators_amounts = f.root.com.scan_data[
                    self.pneumatic_attenuators_key][...]
                if piezo_attenuators_coef is None:
                    piezo_attenuators_coef = f.root.com.SIXS["i14-c-c00-ex-config-att"].att_coef[...]
                piezo_attenuators_amounts = f.root.com.scan_data[self.piezo_attenuators_key][...]
                detector_images = f.root.com.scan_data[self.data_key][...]
                detector_mask = f.root.com.SIXS[self.detector_key].mask[...]
                roi_limits = f.root.com.SIXS[self.detector_key].roi_limits[...]
                acquisition_time = f.root.com.SIXS[self.acquisition_time_key].integration_time[...]
                mu = f.root.com.scan_data[self.mu_key][...]
                beta = f.root.com.scan_data[self.beta_key][...]
                delta = f.root.com.scan_data[self.delta_key][...]
                etaa = f.root.com.scan_data[self.etaa_key][...]
                gamma = f.root.com.scan_data[self.gamma_key][...]
                wavelength = f.root.com.SIXS[self.wavelength_key]["lambda"][0]

            # Save angles
            self.mu.append(mu)
            self.beta.append(beta)
            self.delta.append(delta)
            self.etaa.append(etaa)
            self.gamma.append(gamma)
            self.wavelength.append(wavelength)

            # create 3d mask array
            mask_array = np.ones(detector_images.shape) * detector_mask

            # mask the data
            masked_images = np.where(
                mask_array == 0,
                detector_images,
                np.nan
            )

            # Get ROI
            if isinstance(roi, int):
                roi = roi_limits[3]

            # Compute reflectivity by summing data in ROI
            reflectivity_data = np.nansum(
                masked_images[:, roi[1]:roi[1]+roi[3], roi[0]:roi[0]+roi[2]],
                axis=(1, 2)
            )

            # Compute attenuator coefficient
            # There are two types of attenuators with two different coefficients
            correction_coef = (piezo_attenuators_coef**piezo_attenuators_amounts) * \
                (pneumatic_attenuators_coef**pneumatic_attenuators_amounts)

            # Correct the data for the acquisition time and attenuators
            corrected_reflectivity_data = reflectivity_data * \
                correction_coef / acquisition_time

            # Do not take bad values due to attenuator change
            attenuator_change = np.where(
                (correction_coef[1:]-correction_coef[:-1]) != 0
            )[0]
            np.put(corrected_reflectivity_data, attenuator_change+1, np.nan)
            self.intensities.append(corrected_reflectivity_data)
            print("###########################################################\n")

    def compute_q(self, theta):
        """
        Compute values of the scattering vector from the values of theta.

        :param theta: string corresponding to angle (degrees) used following:
            Q = (4pi/lambda) * sin(theta)
            Should be mu (vertical configuration) or beta (horizontal
            configuration)
        """
        # Append Q values for this scan, assume delta and gamma equal to zero
        self.q = []
        for theta, wavelength in zip(getattr(self, theta), self.wavelength):
            try:
                self.q.append(
                    (4*np.pi / wavelength) * np.sin(np.deg2rad(theta))
                )
            except:
                print("\tCould not compute Q.")

    def normalize_data(
        self,
        x_var,
        normalisation_range=False,
    ):
        """
        Normalize the data by maximum intensity on `normalisation_range`.

        :param x_var: choose x_axis in the self.x_axes list
        :param normalisation_range: normalize by maximum on this range
        """
        # Get x axis
        x_axis = getattr(self, x_var)

        print("\n###########################################################")
        print("Normalizing data ...")
        for (i, y), x, scan_index in zip(
            enumerate(self.intensities),
            x_axis,
            self.scan_indices
        ):
            # Normalize data
            if isinstance(normalisation_range, list) \
                    or isinstance(normalisation_range, tuple):
                print("\nScan index:", scan_index)
                start = self.find_nearest(x, normalisation_range[0])[1]
                end = self.find_nearest(x, normalisation_range[1])[1]
                max_ref = np.nanmax(y[start:end])
                print(f"\tmax(y[{start}: {end}])={max_ref}")
                self.intensities[i] = y/max_ref
                print("\tNormalized the data by maximum on normalisation range.\n")
            else:
                print("Provide a list or tuple for the normalisation range.\n")
        print("###########################################################")

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
        marker_size=2,
        zoom=[None, None, None, None],
        fill=False,
        fill_first=0,
        fill_last=-1,
        background=False,
        log_intensities=True,
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
        :param marker_size: size of markers in plt.scatter
        :param zoom: values used for plot range, default is
            [None, None, None, None], order is left, right, bottom and top.
        :param fill: if True, add filling between two plots
        :param fill_first: index of scan to use for filling
        :param fill_last: index of scan to use for filling
        :param log_intensities: if True, y axis is logarithmic
        """
        x_axis = getattr(self, x_var)

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

            # Add label
            if isinstance(labels, list):
                label = labels[i]
            elif isinstance(labels, dict):
                try:
                    label = labels[scan_index]
                except KeyError:
                    label = labels[int(scan_index)]
                except TypeError:
                    print("Dict not valid for labels, used scan_index")
                    label = scan_index
            else:
                label = scan_index

            # Add colour
            try:
                color = color_dict[int(scan_index)]
            except KeyError:
                color = color_dict[scan_index]
            except TypeError:
                color = None
            plt.scatter(
                x,
                y_plot,
                s=marker_size,
                color=color,
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
        if x_var == "q":
            x_label = "Q (A-1)"
        else:
            x_label = f"{x_var} (°)"
        plt.xlabel(x_label, fontsize=self.fontsize)
        plt.ylabel("Intensity (a.u.)", fontsize=self.fontsize)
        if isinstance(title, str):
            plt.title(f"{title}", fontsize=20)

        plt.tight_layout()

        # Save
        if filename != None:
            plt.savefig(f"{filename}", bbox_inches='tight')
            print(f"Saved as {filename}")

        plt.show()

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
            return ("You need to run Reflectivity.prep_nxs_data() before!")

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

    def extract_data(
        self,
        x_var,
        convert_theta_two_theta=False,
        folder=None,
        y_errors=None,
    ):
        """
        Extract data as .dat files to be used in Genx.
        Be careful that GenX asks for two-theta or Q as x input.
        removes np.nan values from y and corresponding x values.

        :param x_var: variable to save as x, can be an angle or q
        :param convert_theta_two_theta: True to multiply by two the values of
            x_var to save two_theta instead of theta.
        :param folder: None, folder in which the files are saved, if None then
            uses the same folder as defined in __init__
        :param y_errors: not supported yet
        """

        x_axis = getattr(self, x_var)

        save_folder = self.folder if folder is None else folder

        for x, y, scan_index in zip(
            x_axis,
            self.intensities,
            self.scan_indices
        ):
            if convert_theta_two_theta:
                x *= 2

            data_array = np.array([x[~np.isnan(y)], y[~np.isnan(y)]]).T

            np.savetxt(
                f"{save_folder}/reflectivity_{x_var}_{scan_index}.dat",
                data_array
            )
            print(
                f"Saved as {save_folder}/reflectivity_{x_var}_{scan_index}.dat"
            )

    @ staticmethod
    def find_nearest(array, value):
        X = np.abs(array-value)
        idx = np.where(X == np.nanmin(X))
        if len(idx) == 1:
            try:
                idx = idx[0][0]
                return array[idx], idx
            except IndexError:
                if all(array < value):
                    return array[-1], -1
                elif all(array > value):
                    return array[0], 0
