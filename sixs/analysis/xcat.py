import numpy as np
import pandas as pd
from datetime import datetime
import sixs
import inspect
import yaml
import os
from tqdm import tqdm
import h5py

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from IPython.display import display

from scipy import interpolate
from scipy import special
from scipy.optimize import curve_fit

hashtag_line = "################################################################################"


class XCAT:
    """
    Combines data from the mass flow controller and the mass spectrometer
    to compare the input flow in the reactor cell and the products of the
    reaction.
    All the data is transformed to pandas.DataFrame() objects.

    What's left to do?
    * See if we can create a routine to normalize the data by the coefficient of
      the leak that is linked to the temperature. It can be extracted from the
      experiment without any products but just a temperature change.
    """

    def __init__(
        self,
        configuration_file=False,
        time_shift=502,
        gaz_travel_time=12,
    ):
        """
        Initialize the Class with a configuration file

        Typical workflow after init (Class methods):
        * load_mass_flow_controller_file()
        * separate_mass_flow_dataframes()
        * load_mass_spectrometer_file()
        * truncate_mass_flow_df()
        * interpolate_mass_flow_df()
        * plot_mass_flow_entry()
        * plot_mass_flow_valves()
        * plot_mass_spec()

        :param configuration_file: `.yml` file, stores metadata
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

        # Time for the gaz to travel from cell to detector
        self.gaz_travel_time = gaz_travel_time
        print(
            f"\n{hashtag_line}\n"
            f"Travel time from cell to detector set to {self.gaz_travel_time} secs."
        )

        # Time shift between computers
        self.time_shift = time_shift
        # jan 2022: 502 sec
        # 2021: 1287
        print(
            f"Travel shift between computers set to {self.time_shift} secs.\n"
            f"{hashtag_line}\n"
        )

    def load_mass_flow_controller_file(
        self,
        mass_flow_file,
        MRS_pos=None,
        MIX_pos=None,
        mass_flow_file_columns=None,
        time_columns=None,
    ):
        """Initialize the module with a log file, output from XCAT
        Automatically runs self.separate_mass_flow_dataframes()

        :param mass_flow_file: .txt, file from the mass flow controller
        :param MRS_pos: Meaning of the positions of the MRS valve, default
         values are printed on execution.
        :param MIX_pos: Meaning of the positions of the MIX valve, default
         values are printed on execution.
        :param mass_flow_file_columns: Columns of the mass flow file to rename,
         default values are printed on execution.
        :param time_columns: Columns on the mass flow that give the time,
         default value is printed on execution.
        """
        self.mass_flow_file = mass_flow_file
        print(f"Using {self.mass_flow_file} as filename for data frame input.")
        print(f"\n{hashtag_line}\n")

        if MRS_pos == None:
            self.MRS_pos = [
                "All", "Ar", "Ar+Shu", "All", "Rea+Ar", "Closed", "All", "Closed",
                "Ar", "Ar+Shu", "Shu", "Closed", "All", "Rea", "Rea", "Rea+Ar",
                "Rea+Ar", "Closed", "All", "Shu", "Rea+Shu", "Rea+Shu", "Shu",
                "Closed"
            ]
            print(f"Defaulting MRS valve positions to:\n{self.MRS_pos}.")
            print(f"\n{hashtag_line}\n")
        else:
            self.MRS_pos = MRS_pos

        if MIX_pos == None:
            self.MIX_pos = [
                "NO", "All", "Closed", "H2+CO", "H2+O2+CO", "H2+O2", "H2", "All",
                "Closed", "NO+O2", "NO+O2+CO", "O2+CO", "O2", "All", "Closed",
                "H2+CO", "NO+H2+CO", "NO+CO", "CO", "All", "Closed", "NO+O2",
                "NO+H2+O2", "NO+H2"
            ]
            print(f"Defaulting MIX valve positions to:\n{self.MIX_pos}.")
            print(f"\n{hashtag_line}\n")
        else:
            self.MIX_pos = MIX_pos

        if mass_flow_file_columns == None:
            self.mass_flow_file_columns = [
                "time_no", "flow_no", "setpoint_no", "valve_no",
                "time_h2", "flow_h2", "setpoint_h2", "valve_h2",
                "time_o2", "flow_o2", "setpoint_o2", "valve_o2",
                "time_co", "flow_co", "setpoint_co", "valve_co",
                "time_ar", "flow_ar", "setpoint_ar", "valve_ar",
                "time_shunt", "flow_shunt", "setpoint_shunt", "valve_shunt",
                "time_reactor", "flow_reactor", "setpoint_reactor", "valve_reactor",
                "time_drain", "flow_drain", "setpoint_drain", "valve_drain",
                "time_valve", "valve_MRS", "valve_MIX"
            ]
            print(
                f"Defaulted to log file columns (in order):",
                f"\n{self.mass_flow_file_columns}.")
            print(f"\n{hashtag_line}\n")
        else:
            self.mass_flow_file_columns = mass_flow_file_columns

        if time_columns == None:
            self.time_columns = [
                "time_no", "time_h2", "time_o2", "time_co",
                "time_ar", "time_shunt", "time_reactor",
                "time_drain", "time_valve"
            ]
            print(
                f"Defaulted to the following time columns (in order):\n{self.time_columns}.")
            print(f"\n{hashtag_line}\n")
        else:
            self.time_columns = time_columns

        # Create pandas.DataFrame
        try:
            self.mass_flow_df = pd.read_csv(
                self.mass_flow_file,
                header=None,
                delimiter="\t",
                skiprows=1,
                names=self.mass_flow_file_columns
            )

        except Exception as E:
            raise E

        # Change time to unix epoch
        for column_name in self.time_columns:
            column = getattr(self.mass_flow_df, column_name)

            column -= 2082844800
        print("Changed time to unix epoch for all time columns")
        self.mass_flow_start_time_epoch = self.mass_flow_df.iloc[0, 0]
        self.mass_flow_start_time = datetime.fromtimestamp(
            self.mass_flow_df.iloc[0, 0]).strftime('%Y-%m-%d %H:%M:%S')
        self.mass_flow_end_time_epoch = self.mass_flow_df.iloc[0, 0]
        self.mass_flow_end_time = datetime.fromtimestamp(
            self.mass_flow_df.iloc[-1, 0]).strftime('%Y-%m-%d %H:%M:%S')

        print(
            f"Mass flow. starting time: {self.mass_flow_start_time_epoch} "
            f"(unix epoch), {self.mass_flow_start_time}."
        )
        print(
            f"Mass flow. end time: {self.mass_flow_end_time_epoch} "
            f"(unix epoch), {self.mass_flow_end_time}."
            "\nCareful, there are two hours added regarding utc time."
            f"\n{hashtag_line}\n"
        )

        # Show preview of pandas.DataFrame
        display(self.mass_flow_df.head())
        display(self.mass_flow_df.tail())

        # two times shunt valve ? TODO
        # INJ ?
        # OUT ?
        self.separate_mass_flow_dataframes()

    def separate_mass_flow_dataframes(
        self,
        mass_flow_controller_list=False,
    ):
        """
        Create new pandas.DataFrames based on the gases that were analyzed, important
        otherwise the original pandas.DataFrame is too big. The position of the
        columns are hardcoded based on the usual rga files.
        Each entry has 4 columns, time, flow, setpoint, valve.

        The mass flow controller (Bronkhorst High-Tech B.V. series) is
        calibrated for these gases. When we use NH3 for example, it is flowed
        through the H2 mass flow controller (see page 17 of mass flow docs).

        Shunt: pressure (bar) at the exit. Différence de pression liée à la
         perte de charges. Exit of MRS.
        Reactor: pressure (bar) in the reactor.
        Drain: pressure (bar) in the drain. Exit of MIX.
        Valve: Settings at the MIX and MRS valves, see self.MRS_pos and
         self.MIX_pos

        :param mass_flow_controller_list: list of entries set up in the XCAT,
         defaults to ["no", "h2", "o2", "co", "ar", "shunt", "reactor",
         "drain", "valve"]
        """
        if mass_flow_controller_list:
            self.mass_flow_controller_list = [g.lower()
                                              for g in mass_flow_controller_list]

        else:
            self.mass_flow_controller_list = [
                "no", "h2", "o2", "co", "ar", "shunt",
                "reactor", "drain", "valve"
            ]
            print("Defaulted entry list to:", self.mass_flow_controller_list)

        try:
            for entry in self.mass_flow_controller_list:
                if entry == "no":
                    self.no_df = self.mass_flow_df.iloc[:, 0:4].copy()
                    print(f"New pandas.DataFrame created starting  or NO.")

                if entry == "h2":
                    self.h2_df = self.mass_flow_df.iloc[:, 4:8].copy()
                    print(f"New pandas.DataFrame created starting for H2.")

                if entry == "o2":
                    self.o2_df = self.mass_flow_df.iloc[:, 8:12].copy()
                    print(f"New pandas.DataFrame created starting for O2.")

                if entry == "co":
                    self.co_df = self.mass_flow_df.iloc[:, 12:16].copy()
                    print(f"New pandas.DataFrame created starting for CO.")

                if entry == "ar":
                    self.ar_df = self.mass_flow_df.iloc[:, 16:20].copy()
                    print(f"New pandas.DataFrame created starting for Ar.")

                if entry == "shunt":
                    self.shunt_df = self.mass_flow_df.iloc[:, 20:24].copy()
                    print(f"New pandas.DataFrame created starting for shunt.")

                if entry == "reactor":
                    self.reactor_df = self.mass_flow_df.iloc[:, 24:28].copy()
                    print(f"New pandas.DataFrame created starting for reactor.")

                if entry == "drain":
                    self.drain_df = self.mass_flow_df.iloc[:, 28:32].copy()
                    print(f"New pandas.DataFrame created starting for drain.")

                if entry == "valve":
                    self.valve_df = self.mass_flow_df.iloc[:, 32:35].copy()
                    print(f"New pandas.DataFrame created starting for valve.")

        except Exception as e:
            raise e

    def load_mass_spectrometer_file(
        self,
        mass_spec_file,
        skiprows=33,
    ):
        """
        Find timestamp of the mass spectrometer (Resifual Gas Analyzer - RGA)
            file and load the file as a pandas.DataFrame.
        Careful, all the data extraction from the file is hardcoded due to its
            specific architecture.
        Also interpolates the data in seconds using scipy.interpolate.interp1d
        Defines the time range of the experiment.

        The self.truncate_mass_flow_df and self.interpolate_mass_flow_df
            methods are automatically launched.

        :param mass_spec_file: absolute path to the mass_spec_file (converted to
            .txt beforehand in the RGA software)
        :param skiprows: nb of rows to skip when loading the data. Defaults to
            33.
        """
        self.mass_spec_file = mass_spec_file

        # Find experiment starting time based on file content
        try:
            with open(self.mass_spec_file) as f:
                lines = f.readlines()[:skiprows]

                # Create channel index to detect columns
                channel_index = [f"{i+1}  " for i in range(9)]\
                    + [f"{i+1} " for i in range(9, 20)]
                channels = ["Time"]

                # Iterate on lines up to skiprows
                for line in lines:
                    print(line, end="")

                    # Get starting time
                    if "Start time, " in line:
                        timeline = line.rstrip()  # removes spaces
                        self.mass_spec_start_time_epoch = int(datetime.strptime(
                            timeline[12:], "%b %d, %Y  %I:%M:%S  %p").timestamp())
                        self.mass_spec_start_time = datetime.fromtimestamp(
                            self.mass_spec_start_time_epoch).strftime('%Y-%m-%d %H:%M:%S')

                    # Get mass spectrometer channels
                    elif line[:3] in channel_index:
                        channels.append(line[25:35].replace(" ", ""))
            print(
                f"\n{hashtag_line}"
                f"\nMass spec. starting time: {self.mass_spec_start_time_epoch}"
                f" (unix epoch), {self.mass_spec_start_time}."
            )

        except TypeError:
            self.mass_spec_start_time_epoch = None
            self.mass_spec_start_time = None
            print(
                f"\n{hashtag_line}"
                "\nCould not find start time in the file. \nPlay with the "
                "amount of columns names and the amount of rows skipped."
                f"\n{hashtag_line}\n"
            )

        # Create pandas.DataFrame
        self.mass_spec_df = pd.read_csv(
            self.mass_spec_file,
            delimiter=',',
            index_col=False,
            names=channels,
            skiprows=skiprows
        )

        # Interpolate the data of the mass spectrometer in seconds
        try:
            # Get time axis
            x = self.mass_spec_df["Time"]

            # Get time range in seconds of the experiment linked to that dataset
            self.mass_spectrometer_file_duration = int(float(x.values[-1]))
            h = self.mass_spectrometer_file_duration//3600
            m = (self.mass_spectrometer_file_duration - (h*3600))//60
            s = (self.mass_spectrometer_file_duration - (h*3600) - (m*60))
            print(f"Experiment time range: {h}h:{m}m:{s}s")

            self.mass_spec_end_time_epoch = int(
                self.mass_spectrometer_file_duration + self.mass_spec_start_time_epoch)
            self.mass_spec_end_time = datetime.fromtimestamp(
                self.mass_spec_end_time_epoch).strftime('%Y-%m-%d %H:%M:%S')
            print(
                f"Mass spec. end time: {self.mass_spec_end_time_epoch} (unix epoch), ",
                self.mass_spec_end_time
            )
            print(
                "Careful, there are two hours added regarding utc time."
                f"\n{hashtag_line}\n"
            )

            # Create new time column in integer seconds
            new_time_column = np.round(
                np.linspace(
                    0,
                    self.mass_spectrometer_file_duration,
                    self.mass_spectrometer_file_duration + 1
                ),
                0
            )

            # Proceed to interpolation over the new time axis
            interpolated_df = pd.DataFrame({
                "Time": new_time_column
            })

            # Iterate over all the columns
            for column in self.mass_spec_df.columns[self.mass_spec_df.columns != "Time"]:

                f = interpolate.interp1d(x, self.mass_spec_df[column].values)
                interpolated_df[column] = f(new_time_column)

            # Make sure that the time type is integer
            interpolated_df["Time"] = interpolated_df["Time"].astype(int)

            # Save
            setattr(self, "mass_spec_df_interpolated", interpolated_df)
            print(
                f"\n{hashtag_line}"
                f"\nNew pandas.DataFrame created for the mass spectrometer, "
                "interpolated on its time range (in s)."
                f"\n{hashtag_line}"
            )

            display(self.mass_spec_df_interpolated.head())
            display(self.mass_spec_df_interpolated.tail())

        except (ValueError, NameError):
            print(
                f"\n{hashtag_line}"
                "\nCould not interpolate the mass spectrometer data."
                "\nPlay with the amount of columns names and the amount of "
                "rows skipped."
                f"\n{hashtag_line}"
            )

        # Directly truncate the data of the mass flow spectrometer
        self.truncate_mass_flow_df()

        # Directly interpolate the data of the mass flow spectrometer
        self.interpolate_mass_flow_df()

    def truncate_mass_flow_df(
        self,
    ):
        """
        Truncate mass flow pandas.DataFrames based on starting time and duration of
        mass spectrometer file to have smaller pandas.DataFrames on the range of
        interest.
        """
        print(f"\n{hashtag_line}")
        try:
            # Iterate on mass
            for mass in self.mass_flow_controller_list:

                # Find starting time for this pandas.DataFrame
                temp_df = getattr(self, f"{mass}_df").copy()
                index_start_time = (
                    temp_df[f"time_{mass}"] - self.mass_spec_start_time_epoch).abs().argsort()[0]

                # Reset time
                temp_df = temp_df.iloc[index_start_time:, :].reset_index(
                    drop=True)
                temp_df[f"time_{mass}"] -= temp_df[f"time_{mass}"].values[0]

                # Truncate based on time range, take one more sec to be sure
                # to have enough data for interpolation
                temp_df = temp_df[temp_df[f"time_{mass}"]
                                  < self.mass_spectrometer_file_duration + 1]

                setattr(self, f"{mass}_df_truncated", temp_df)

                print(
                    "New truncated pandas.DataFrame created starting from mass "
                    f"spectrometer starting time for {mass} and for "
                    f"{self.mass_spectrometer_file_duration} seconds."
                )

        except AttributeError:
            print(
                "Could not proceed to the truncation of the mass flow pandas.DataFrame"
                "\nMake sure that the mass spectrometer file was loaded."
            )
        print(f"{hashtag_line}")

    def interpolate_mass_flow_df(
        self,
    ):
        """
        Interpolate the data in seconds for the mass flow pandas.DataFrames, makes it
        easier to compare the data with the mass spectrometer.
        """

        print(f"{hashtag_line}")
        try:
            for mass in self.mass_flow_controller_list:

                # Get mass pandas.DataFrame
                temp_df = getattr(self, f"{mass}_df_truncated").copy()

                # Get bad time axis
                x = temp_df[f"time_{mass}"]

                x_0 = int(temp_df[f"time_{mass}"].values[0])
                x_1 = int(temp_df[f"time_{mass}"].values[-1])

                # Create new time column in integer seconds
                new_time_column = np.linspace(x_0, x_1, x_1 + 1)

                # Proceed to interpolation over the new time axis
                interpolated_df = pd.DataFrame({
                    f"time_{mass}": new_time_column
                })

                # Iterate over all the columns
                for column in temp_df.columns[temp_df.columns != f"time_{mass}"]:

                    tck = interpolate.splrep(x, temp_df[column].values, s=0)
                    y_new = interpolate.splev(new_time_column, tck)
                    interpolated_df[column] = y_new

                # Make sure that the time is integer
                interpolated_df[f"time_{mass}"] = interpolated_df[f"time_{mass}"
                                                                  ].astype(int)

                # Save
                setattr(self, f"{mass}_df_interpolated", interpolated_df)
                print(f"New interpolated pandas.DataFrame created for {mass}.")

        except NameError:
            print(
                "You need to get the experiment time range first, run XCAT.load_mass_spectrometer_file()")

        except TypeError:
            print("Do these files overlap on the same experiment ?")

        except AttributeError:
            print(
                "Could not proceed to the interpolation of the mass flow pandas.DataFrame"
                "\nMake sure that the mass spectrometer file was loaded."
            )
        print(f"{hashtag_line}")

    def plot_mass_flow_entry(
        self,
        mass_flow_controller_list=[
            "NO", "H2", "O2", "CO", "Ar",
            "shunt", "reactor", "drain"
        ],
        df="interpolated",
        figsize=(10, 6),
        fontsize=15,
        zoom1=None,
        zoom2=None,
        color_dict="ammonia_reaction_colors",
        cursor_positions=[None],
        cursor_labels=[None],
        text_dict=None,
        hours=True,
        save_as=False,
        plot_valve=True,
    ):
        """
        Plot the evolution of the input of the reactor
        Each mass corresponds to one channel controlled by the mass flow
        controller.

        :param mass_flow_controller_list: list of mass to be plotted
        :param df: pandas.DataFrame from which the data will be plotted. Default is
         "interpolated" which corresponds to the data truncated on the mass
         spectrometer time range and in seconds.
        :param figsize: size of each figure, defaults to (16, 9)
        :param fontsize: size of labels, defaults to 15, title has +2.
        :param zoom1: list of 4 integers to zoom on ax1
        :param zoom2: list of 4 integers to zoom on ax2
        :param color_dict: str, name of dict from the configuration file that
         will be used for the colors.
        :param cursor_positions: add cursors using these positions
        :param cursor_labels: colour of the cursors, same length as
         cursor_positions
        :param text_dict: Add text on plot, same length as
         cursor_positions, e.g. {"ar": [45, 3.45], "argon": [45, 3.45],
         "o2": [5, 2], "h2": [5, 2]}, not used for now !! TODO
        :param hours: True to show x scale in hours instead of seconds
        :param save_as: figure name when saving, no saving if False
        :param plot_valve: True to also see the valve positions
        """

        # Get coloring dictionnary
        try:
            color_dict = getattr(self, color_dict)
        except AttributeError:
            print("Wrong name for color dict.")

        mass_flow_controller_list = [g.lower()
                                     for g in mass_flow_controller_list]

        # Iterate over each valve
        for entry in mass_flow_controller_list:
            print(
                f"{hashtag_line}"
                f"\nPlotting for {entry}"
                f"\n{hashtag_line}"
            )
            plt.close()

            # Create figure
            if plot_valve:
                fig, axes = plt.subplots(2, 1, figsize=figsize)
            else:
                fig, axes = plt.subplots(1, 1, figsize=figsize)
                axes = [axes]

            # Get pandas.DataFrame
            try:
                plot_df = getattr(self, f"{entry}_df_interpolated").copy()
            except AttributeError:
                try:
                    plot_df = getattr(self, f"{entry}_df_truncated").copy()
                    print(
                        f"No attribute `{entry}_df_interpolated`,",
                        f" defaulted to `{entry}_df_truncated`"
                    )
                except AttributeError:
                    plot_df = getattr(self, f"{entry}_df").copy()
                    print(
                        f"No attribute `{entry}_df_truncated`,",
                        f" defaulted to `{entry}_df`"
                    )

            # Change to hours
            if hours:
                x_label = "Time (h)"
                plot_df[f"time_{entry}"] = plot_df[f"time_{entry}"].values/3600
            else:
                x_label = "Time (s)"

            # Plot flow
            axes[0].plot(
                plot_df[f"time_{entry}"],
                plot_df[f"flow_{entry}"],
                label=f"flow_{entry}"
            )

            # Plot setpoint
            try:
                axes[0].plot(
                    plot_df[f"time_{entry}"],
                    plot_df[f"setpoint_{entry}"],
                    linestyle="--",
                    label=f"setpoint_{entry}"
                )
            except:
                pass

            # Plot valve position
            try:
                axes[1].plot(
                    plot_df[f"time_{entry}"],
                    plot_df[f"valve_{entry}"],
                    label=f"valve_{entry}"
                )
            except:
                pass

            # Zoom
            try:
                axes[0].set_xlim([zoom1[0], zoom1[1]])
                axes[0].set_ylim([zoom1[2], zoom1[3]])
            except TypeError:
                pass

            try:
                axes[1].set_xlim([zoom2[0], zoom2[1]])
                axes[1].set_ylim([zoom2[2], zoom2[3]])
            except:
                pass

            # Plot cursors
            for cursor_pos, cursor_label in zip(
                cursor_positions,
                cursor_labels
            ):
                for ax in axes:
                    try:
                        ax.axvline(
                            cursor_pos,
                            linestyle="--",
                            color=color_dict[cursor_label],
                            linewidth=1.5,
                            label=cursor_label
                        )
                    except (KeyError, TypeError):
                        print("No cursors.")

            # Dict for y positions of text depending on the valve
            for x1, x2, cursor_label in zip(
                cursor_positions[:-1],
                cursor_positions[1:],
                cursor_labels
            ):
                for j, ax in enumerate(axes):

                    # Vertical span color to show conditions
                    try:
                        ax.axvspan(
                            x1,
                            x2,
                            alpha=0.2,
                            facecolor=color_dict[cursor_label],
                        )
                    except Exception as E:
                        raise E

            axes[0].set_ylabel("Flow", fontsize=fontsize+2)
            axes[0].set_xlabel(x_label, fontsize=fontsize+2)
            axes[0].legend(ncol=3, fontsize=fontsize)
            axes[0].grid()
            axes[0].set_title(entry.upper())

            try:
                axes[1].set_ylabel("Valve position", fontsize=fontsize+2)
                axes[1].set_xlabel(x_label, fontsize=fontsize+2)
                axes[1].grid()
                axes[1].legend(ncol=3, fontsize=fontsize)
            except:
                pass

            # Save figure
            plt.tight_layout()
            if isinstance(save_as, str):
                plt.savefig(f"{save_as}_{entry}.png")
                plt.savefig(f"{save_as}_{entry}.pdf")
                print(
                    f"Saved as {save_as}_{entry} in (png, pdf) formats.")

            plt.show()

    def plot_mass_flow_valves(
        self,
        df="interpolated",
        figsize=(10, 6),
        fontsize=15,
        zoom1=None,
        zoom2=None,
        color_dict="ammonia_reaction_colors",
        cursor_positions=[None],
        cursor_labels=[None],
        text_dict=None,
        hours=True,
        save_as=False,
    ):
        """Plot the evolution of the input of the reactor, for both the MIX and
         the MRS valve. The gases besided Argon (carrier gas) are mixed
         depending on the position of the MIX valve. At the MRS is then added
         Argon.

        :param df: pandas.DataFrame from which the data will be plotted. Default is
         "interpolated" which corresponds to the data truncated on the mass
         spectrometer time range and in seconds.
        :param figsize: size of each figure, defaults to (16, 9)
        :param fontsize: size of labels, defaults to 15, title have +2.
        :param zoom1: list of 4 integers to zoom on ax1
        :param zoom2: list of 4 integers to zoom on ax2
        :param color_dict: str, name of dict from the configuration file that
         will be used for the colors.
        :param cursor_positions: add cursors using these positions
        :param cursor_labels: colour of the cursors, same length as
         cursor_positions
        :param text_dict: Add text on plot, same length as
         cursor_positions, e.g. {"ar": [45, 3.45], "argon": [45, 3.45],
         "o2": [5, 2], "h2": [5, 2]}
        :param hours: True to show x scale in hours instead of seconds
        :param save_as: figure name when saving, no saving if False
        """

        try:
            color_dict = getattr(self, color_dict)
        except AttributeError:
            print("Wrong name for color dict.")

        fig, axes = plt.subplots(2, 1, figsize=figsize)

        # Get pandas.DataFrame
        try:
            if df == "interpolated":
                plot_df = getattr(self, f"valve_df_interpolated").copy()

            elif df == "truncated":
                plot_df = getattr(self, f"valve_df_truncated").copy()

            else:
                plot_df = getattr(self, f"valve_df").copy()
        except AttributeError:
            raise NameError(
                "This pandas.DataFrame does not exist yet. Try df=\"default\"")

        # Change to hours
        if hours:
            x_label = "Time (h)"
            plot_df["time_valve"] = plot_df["time_valve"].values/3600
        else:
            x_label = "Time (s)"

        axes[0].plot(plot_df["time_valve"],
                     plot_df["valve_MRS"],
                     label="valve_MRS")
        axes[1].plot(plot_df["time_valve"],
                     plot_df["valve_MIX"],
                     label="valve_MIX")

        # Zoom
        try:
            axes[0].set_xlim([zoom1[0], zoom1[1]])
            axes[0].set_ylim([zoom1[2], zoom1[3]])

            axes[1].set_xlim([zoom2[0], zoom2[1]])
            axes[1].set_ylim([zoom2[2], zoom2[3]])
        except TypeError:
            pass

        # Plot cursors
        for cursor_pos, cursor_label in zip(
            cursor_positions,
            cursor_labels
        ):
            for j, ax in enumerate(axes):
                try:
                    ax.axvline(
                        cursor_pos,
                        linestyle="--",
                        color=color_dict[cursor_label],
                        linewidth=1.5,
                        label=cursor_label
                    )
                except (TypeError, KeyError):
                    pass

        # Dict for y positions of text depending on the valve
        for x1, x2, cursor_label in zip(
            cursor_positions[:-1],
            cursor_positions[1:],
            cursor_labels
        ):
            for j, ax in enumerate(axes):
                try:
                    ax.text(
                        x1 + (x2-x1)/2,
                        y=text_dict[cursor_label],
                        s=cursor_label,
                        fontsize=fontsize+5
                    )
                except TypeError:
                    pass
                except Exception as E:
                    raise E

                # Vertical span color to show conditions
                try:
                    ax.axvspan(
                        x1,
                        x2,
                        alpha=0.2,
                        facecolor=color_dict[cursor_label],
                    )
                except TypeError:
                    pass
                except Exception as E:
                    raise E

        axes[0].set_ylabel("Valve MRS position", fontsize=fontsize+2)
        axes[0].set_xlabel(x_label, fontsize=fontsize+2)
        axes[1].set_ylabel("Valve MIX position", fontsize=fontsize+2)
        axes[1].set_xlabel(x_label, fontsize=fontsize+2)

        axes[0].legend(fontsize=fontsize)
        axes[1].legend(fontsize=fontsize)

        # Save figure
        plt.tight_layout()
        if isinstance(save_as, str):
            plt.savefig(f"{save_as}.png")
            plt.savefig(f"{save_as}.pdf")
            print(
                f"Saved as {save_as} in (png, pdf) formats.")

        plt.show()

    def plot_mass_spec(
        self,
        mass_spectrometer_list=None,
        normalize_data=False,
        p_tot=False,
        ptot_mass_list=False,
        carrier_gaz=False,
        carrier_pressure=False,
        figsize=(16, 9),
        fontsize=15,
        zoom=None,
        color_dict="ammonia_reaction_colors",
        cursor_positions=[None],
        cursor_labels=[None],
        text_dict=None,
        hours=True,
        logscale=True,
        heater_file=None,
        save_as=False,
        title="Pressure in XCAT reactor cell for each mass",
    ):
        """
        Plot the evolution of the gas detected by the mass spectrometer.
        Each mass corresponds to one channel detected. Careful, some mass can
         overlap.

        :param mass_spectrometer_list: list of mass to be plotted (if set up prior to the
            experiment for detection).
        :param normalize_data: False, choose in [False, "reactor_pressure",
            "carrier_pressure"].
        :param p_tot: None, total pressure in the cell (in bar).
            Used if normalize_data = "reactor_pressure".
        :param ptot_mass_list: None, list of mass to use for normalization,
            defaults as same as mass_spectrometer_list, otherwise provide list
            of str, e.g. ["Ar", "NH3", "O2"]
            Used if normalize_data = "reactor_pressure".
        :param carrier_gaz: None, carrier gas, str to choose in
            mass_spectrometer_list. Used if normalize_data = "carrier_pressure".
        :param carrier_pressure: None, carrier pressure in bar, e.g. 0.5
            Used if normalize_data = "carrier_pressure".
        :param figsize: size of each figure, defaults to (16, 9)
        :param fontsize: size of labels, defaults to 15, title have +2.
        :param zoom: list of 4 integers to zoom, [xmin, xmax, ymin, ymax]
        :param color_dict: str, name of dict from the configuration file that
            will be used for the colors.
        :param cursor_positions: add cursors using these positions
        :param cursor_labels: colour of the cursors, same length as
            cursor_positions
        :param text_dict: Add text on plot, same length as
            cursor_positions, combine labels and (x_pos, y_pos), e.g. {"ar":
            (1, 3.45)}
        :param hours: True to show x scale in hours instead of seconds
        :param logscale: True to have logscale on y
        :param heater_file: path to heater data, in csv format, must contain
            a time and a temperature column.
        :param save_as: str instance to save the plot, figure name when saving
        :param title: figure title
        """
        # Save important parameters as attributes
        self.mass_spectrometer_list = mass_spectrometer_list
        self.normalize_data = normalize_data
        self.p_tot = p_tot
        self.ptot_mass_list = ptot_mass_list
        self.carrier_gaz = carrier_gaz
        self.carrier_pressure = carrier_pressure
        self.cursor_positions = cursor_positions
        self.cursor_labels = cursor_labels

        # Get coloring dictionnary
        try:
            color_dict = getattr(self, color_dict)
        except AttributeError:
            print("Wrong name for color dict.")

        # Get pandas.DataFrame
        try:
            self.norm_df = self.mass_spec_df_interpolated.copy()
        except AttributeError:
            print(
                "No attribute `mass_spec_df_interpolated`,",
                " defaulted to `mass_spec_df`"
            )
            self.norm_df = self.mass_spec_df.copy()

        # Use all columns if none specified
        if self.mass_spectrometer_list is None:
            self.mass_spectrometer_list = list(self.norm_df.columns[1:-1])
            print("Defaulted mass_spectrometer_list to all pandas.DataFrame columns.")

        # Normalize data by total pressure
        # Making the hypothesis that the sum of the pressure of all the gases
        # in mass_spectrometer_list is equal to the reactor pressure, which is
        # controlled independently, dangerous bc we are looking at partial
        # pressures in the UHV chamber and not at the pressures in the reactor
        # cell
        if normalize_data == "reactor_pressure":
            # Total pressure
            if isinstance(self.ptot, bool):
                raise TypeError("Precise a value for ptot.")
            print(
                f"Setting P={self.ptot} bar as total pressure in the reactor.")

            # Gases used for the normalization
            if isinstance(ptot_mass_list, list):
                self.ptot_mass_list = ptot_mass_list
                print("Using columns specified with ptot_mass_list for normalization.")

            elif isinstance(self.mass_spectrometer_list, list):
                self.ptot_mass_list = self.mass_spectrometer_list
                print(
                    "Using columns specified with mass_spectrometer_list for "
                    "normalization. \nIf you want to use other gases, use "
                    "the `ptot_mass_list` parameter."
                )

            # Get data
            used_arr = self.norm_df[self.ptot_mass_list].values

            # Normalize, sum columns to have total pressure per second
            ptot_col = self.ptot/used_arr.sum(axis=1)
            ptot_arr = (np.ones(used_arr.shape).T * ptot_col).T

            # Careful units are of self.ptot
            norm_arr = used_arr*ptot_arr

            # Create new pandas.DataFrame
            norm_df_ptot = pd.DataFrame(norm_arr, columns=self.ptot_mass_list)
            norm_df_ptot["Time"] = self.norm_df["Time"]
            self.norm_df = norm_df_ptot

            print("Normalized RGA pressure by total pressure.")
            display(self.norm_df.head())

            # Changed y units to bar because ptot is in bars
            y_units = "bar"

        # Normalize data by carrier gas
        elif normalize_data == "carrier_pressure":
            if isinstance(self.carrier_pressure, bool):
                raise TypeError("Precise a value for carrier_pressure")
            print(
                f"Using {self.carrier_gaz} as carrier gas for normalization.")

            # Check if it is in the list, and remove it otherwise there is a bog
            if self.carrier_gaz in self.norm_df.columns and self.carrier_gaz in self.mass_spectrometer_list:
                self.mass_spectrometer_list.remove(self.carrier_gaz)
            else:
                raise KeyError("This carrier gas is not in the list.")

            # Normalize by carrier gas partial pressure
            for mass in self.mass_spectrometer_list:
                self.norm_df[mass] /= self.norm_df[self.carrier_gaz]

            self.norm_df[self.carrier_gaz] = np.ones(self.norm_df.shape[0])

            # Put the carrier gas back in the list of mass to plot
            self.mass_spectrometer_list.append(self.carrier_gaz)

            # We must know the different conditions to properly correct the data
            # Otherwise the pressure is not correct
            correction_ratio_values = np.ones(self.norm_df.shape[0])

            # Apply a correction ratio for each condition
            for x1, x2, cursor_label in zip(
                cursor_positions[:-1],
                cursor_positions[1:],
                cursor_labels,
            ):
                x1 = x1*3600  # in secs
                x2 = x2*3600  # in secs

                idx1 = find_nearest(self.norm_df.Time.values, x1)
                idx2 = find_nearest(self.norm_df.Time.values, x2)

                correction_ratio_values[idx1:idx2] = self.carrier_pressure_correction_ratio[cursor_label]

                for mass in self.mass_spectrometer_list:
                    self.norm_df[mass][idx1:idx2] *= self.carrier_pressure_correction_ratio[cursor_label] * carrier_pressure

            # Create a new column for the correction ratio
            self.norm_df["correction_ratio"] = correction_ratio_values

            print("Normalized RGA pressure by carrier gas pressure.")
            display(self.norm_df.head())

            # Change y scale
            y_units = "bar"

        # Normalize data by leak pressure
        # TODO

        # No normalisation, meaning we are looking at the partial pressure
        # after the leak in the UHV chamber
        else:
            print("No normalization performed on data.")
            y_units = "mbar"

        print("Saved data as self.norm_df pandas.DataFrame.")

        # Change to hours
        if hours:
            x_label = "Time (h)"
            self.norm_df["Time"] = self.norm_df["Time"].values/3600
        else:
            x_label = "Time (s)"

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        if logscale:
            ax.semilogy()
        ax.set_title(
            title,
            fontsize=fontsize+5
        )

        # If only plotting a subset of the masses
        for mass in self.mass_spectrometer_list:
            try:
                ax.plot(
                    self.norm_df.Time.values,
                    self.norm_df[mass].values,
                    linewidth=2,
                    label=f"{mass}",
                    color=color_dict[mass]
                )

            except (KeyError, TypeError):
                print("Is there an entry on the color dict for", mass)
                ax.plot(
                    self.norm_df.Time.values,
                    self.norm_df[mass].values,
                    linewidth=2,
                    label=f"{mass}",
                )

        # Temperature on second ax
        try:
            self.norm_df, mask = self.add_temperature_column(
                df=self.norm_df,
                heater_file=heater_file,
                time_shift=self.time_shift,
            )

            twin_ax = ax.twinx()
            twin_ax.plot(
                self.norm_df.Time[mask],
                self.norm_df.temperature[mask],
                label="Temperature",
                linestyle="--",
                color="r"
            )

            twin_ax.legend(fontsize=fontsize)
            twin_ax.set_ylabel('Temperature (°C)', fontsize=fontsize)
            twin_ax.grid(which="both", color="r", alpha=0.5)
            twin_ax.tick_params(axis='both', labelsize=fontsize)

        except (TypeError, ValueError):
            pass

        # Plot cursors
        for cursor_pos, cursor_label in zip(
            cursor_positions,
            cursor_labels
        ):
            try:
                ax.axvline(
                    cursor_pos,
                    linestyle="--",
                    color=color_dict[cursor_label],
                    linewidth=1.5,
                    label=cursor_label
                )
            except (TypeError, KeyError):
                pass

        # Zoom
        try:
            ax.set_xlim(zoom[0], zoom[1])
            ax.set_ylim(zoom[2], zoom[3])
        except TypeError:
            pass

        # Vertical span color to show conditions
        for x1, x2, cursor_label in zip(
            cursor_positions[:-1],
            cursor_positions[1:],
            cursor_labels,
        ):
            try:
                facecolor = color_dict[cursor_label]
            except KeyError:
                facecolor = "white"

            try:
                ax.axvspan(
                    x1,
                    x2,
                    alpha=0.2,
                    facecolor=color_dict[cursor_label],
                )
            except TypeError:
                pass

            # Dict for y positions of text depending on the valve
            try:
                ax.text(
                    x=text_dict[cursor_label][0],
                    y=text_dict[cursor_label][1],
                    s=cursor_label,
                    fontsize=fontsize+5
                )
            except TypeError:
                pass

        ax.set_xlabel(x_label, fontsize=fontsize)
        ax.set_ylabel(f'Pressure ({y_units})', fontsize=fontsize)
        ax.tick_params(axis='both', labelsize=fontsize)
        ax.legend(
            bbox_to_anchor=(0., -0.2, 1., .102),
            loc=3,
            ncol=7,
            mode="expand",
            borderaxespad=0.,
            fontsize=fontsize-2
        )
        ax.grid(True, which='major', linestyle='-')
        ax.grid(True, which='minor', linestyle='--')

        fig.tight_layout()
        if isinstance(save_as, str):
            plt.savefig(f"{save_as}.png")
            plt.savefig(f"{save_as}.pdf")
            print(f"Saved as {save_as} in (png, pdf) formats.")

        plt.show()

    # Extra functions

    def remove_background(
        self,
        df,
        x_col,
        y_col,
        peak_start,
        peak_end,
        degree=2,
        data_start=0,
        data_end=-1
    ):
        """
        Background reduction routine around a peak on a given range using
        np.polyfit()

        y- background is saved in the column f"{y_col}_bckg"

        :param df: pandas.DataFrame to use
        :param x_col: str, column in df to use for x axis
        :param y_col: str, column in df to use for y axis
        :param peak_start: beginning of range to ignore for background
            subtraction, 0 is equal to data_start
        :param peak_end: end of range to ignore for background subtraction,
            0 is equal to data_end
        :param degree: degree of np.polyfit
        :param data_start: beginning of data range of interest (includes peak
            and background around it)
        :param data_end: end of data range of interest (includes peak and
            background around it)
        """
        # Init figure
        fig, axs = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

        # Range of interest
        # axs[0].plot(df[x_col],
        #             df[y_col],
        #             label="Data"
        #             )
        x = df[x_col][data_start:data_end].values
        y = df[y_col][data_start:data_end].values

        # Peak area in this range
        x_peak = x[peak_start:peak_end]
        y_peak = y[peak_start:peak_end]

        axs[0].plot(x_peak, y_peak, label="Peak")
        axs[0].set_ylabel(y_col)
        axs[0].set_xlabel(x_col)

        # No peak area in this range
        x_no_peak = np.concatenate([x[:peak_start], x[peak_end:]])
        y_no_peak = np.concatenate([y[:peak_start], y[peak_end:]])

        axs[0].plot(x_no_peak, y_no_peak, label="No peak")

        # Fit range without peak to get background
        poly = np.polyfit(x_no_peak, y_no_peak, degree)
        poly_fit = np.poly1d(poly)

        # Plot data on range of interest minus background
        axs[0].plot(x, poly_fit(x), label="Background fit")
        axs[0].legend(fontsize=15)

        # Save y - background
        new_y = np.zeros(len(df))
        new_y[data_start:data_end] = y - poly_fit(x)
        df[f"{y_col}_bckg"] = new_y

        axs[1].plot(df[x_col], new_y, label="Background subtracted data")
        axs[1].legend(fontsize=15)
        axs[1].set_ylabel(y_col)
        axs[1].set_xlabel(x_col)
        plt.show()

    def add_temperature_column(
        self,
        df,
        heater_file,
        time_shift=0,
    ):
        """Add a temperature column to the pandas.DataFrame from a csv file
        The csv file must have a unix timestamp "time" column that will be used
        to add the temperature values at the right time in the pandas.DataFrame.

        :param df: pandas.DataFrame object to which we assign the new df,
         usually rga_interpolated, must be created before the heater df
        :param heater_file: path to heater data, in csv format, must contain
         a time and a temperature column.
        :param time_shift: rigid time shift in seconds to apply.
        """

        # Get heater df
        self.heater_df = pd.read_csv(heater_file)

        # Starting time
        self.heater_starting_time = self.heater_df.time[0]

        # Find delta in seconds between beginning of heater df and experiment
        delta_s = self.heater_starting_time - self.mass_spec_start_time_epoch

        # Reset time
        self.heater_df.time = self.heater_df.time \
            - self.heater_starting_time + delta_s + time_shift

        # Create temp column
        df["temperature"] = [np.nan for i in range(len(df))]

        # Find closest value and assign them
        for j, row in self.heater_df.iterrows():
            df.iloc[int(row.time), -1] = row.temperature

        # Create mask for plots
        mask = ~np.isnan(df.iloc[:, -1].values)

        return df, mask

    def fit_error_function(
        self,
        initial_guess,
        new_amper_vect,
        interpolated_data=True,
        fitted_columns=None,
        binning=False,
        zoom=[None, None, None, None],
        cursor_positions=[None],
        cursor_labels=[None],
    ):
        """
        Fit pressure in the reactor as a function of the temperature with an
        error function
        """

        def error_function(z, a, b, c, d):
            return a * (1 + special.erf((z - b) / c)) + d

        longer_temp_vect = self.heater_poly_fit(new_amper_vect)

        if not binning:
            used_df = self.rga_during_heating.copy()
            xdata = used_df.temperature
        else:
            used_df = self.binned_data.copy()
            xdata = used_df.temperature

        fig, axes = plt.subplots(1, 1, figsize=(
            18, 13), dpi=80, facecolor='w', edgecolor='k')

        try:
            for element in fitted_columns:
                ydata = used_df[element]
                popt, pcov = curve_fit(
                    error_function, xdata, ydata, p0=initial_guess)
                # axes.plot(xdata, func(xdata, *guessErf))
                axes.plot(xdata, ydata, linewidth=2,
                          linestyle="dashdot", label=f"{element}")
                # axes.plot(xdata, func(xdata, *popt))
                axes.plot(longer_temp_vect, error_function(
                    longer_temp_vect, *popt), linewidth=2, linestyle="dashdot", label=f"{element}")

        except:
            for element in used_df.columns[1:]:
                ydata = used_df[element]
                popt, pcov = curve_fit(
                    error_function, xdata, ydata, p0=initial_guess)
                # axes.plot(xdata, func(xdata, *guessErf))
                axes.plot(xdata, ydata, linewidth=2,
                          linestyle="dashdot", label=f"{element}")
                # axes.plot(xdata, func(xdata, *popt))
                axes.plot(longer_temp_vect, error_function(
                    longer_temp_vect, *popt), linewidth=2, linestyle="dashdot", label=f"{element}")

        # cursor
        try:
            # mcolors.CSS4_COLORS["teal"]
            for cursor_pos in cursor_positions:
                axes.axvline(x=cursor_pos, linestyle="--", color="#bb1e10")
        except TypeError:
            print("No cursor")

        finally:
            axes.set_xlim(zoom[0], zoom[1])
            axes.set_ylim(zoom[2], zoom[3])

            axes.set_xlabel('Temperature (°C)', fontsize=16)
            axes.set_ylabel('Pressure (mBar)', fontsize=16)
            axes.legend(bbox_to_anchor=(0., -0.1, 1., .102), loc=3,
                        ncol=5, mode="expand", borderaxespad=0.)
            axes.grid(b=True, which='major', color='b', linestyle='-')
            axes.grid(b=True, which='minor',
                      color=mcolors.CSS4_COLORS["teal"], linestyle='--')

            plt.title(
                "Temperature vs pressure graphs fitted with error functions")
            plt.show()

    def save_xcat_data_as_hdf5(
        self,
        filename,
    ):
        """
        Save class data in a hdf5 file.

        Attributes saved under the Mass_flow group":
            mass_flow_file, MRS_pos, MIX_pos, mass_flow_start_time,
            mass_flow_end_time, gaz_travel_time, time_shift

        DataFrames saved under the Mass_flow.Dataframes group (try to save the
        interpolated pandas.DataFrame first, then the truncated pandas.DataFrame and finally
        the base pandas.DataFrame:
            no_df, h2_df, o2_df, co_df, ar_df, shunt_df, reactor_df, drain_df,
            valve_df
        The axes labels are fixed to Time (s), Flow (mL/min), Setpoint (mL/min),
        Valve position.

        Attributes saved under the Mass_spectrometer group":
            mass_spec_file, mass_spec_start_time, mass_spec_end_time,
            mass_spectrometer_list, normalize_data, p_tot, ptot_mass_list,
            carrier_gaz, carrier_pressure, cursor_positions, cursor_labels,
            carrier_pressure_correction_ratio

        Dataframes saved under the Mass_spectrometer.Dataframes group:
            mass_spec_df, mass_spec_df_interpolated, norm_df
        The axes labels are fixed to mass_spec_df.columns

        :param filename: full path to hdf5 file
        """

        # Overwite existing file
        if os.path.isfile(filename):
            os.remove(filename)

        mass_flow_attributes = [
            "mass_flow_file",
            "mass_flow_controller_list",
            "mass_flow_start_time",
            "mass_flow_end_time",
            "MRS_pos",
            "MIX_pos",
            "gaz_travel_time",
            "time_shift",
        ]

        mass_flow_dataframes = [
            "no_df",
            "h2_df",
            "o2_df",
            "co_df",
            "ar_df",
            "shunt_df",
            "reactor_df",
            "drain_df",
            "valve_df",
        ]

        mass_spectrometer_attributes = [
            "mass_spec_file",
            "mass_spec_start_time",
            "mass_spec_end_time",
            "mass_spectrometer_list",
            "normalize_data",
            "p_tot",
            "ptot_mass_list",
            "carrier_gaz",
            "carrier_pressure",
            "cursor_positions",
            "cursor_labels",
            "carrier_pressure_correction_ratio",
        ]

        mass_spectrometer_dataframes = [
            "mass_spec_df",
            "mass_spec_df_interpolated",
            "norm_df",
        ]

        with h5py.File(filename, "a") as f:
            print(
                f"{hashtag_line}"
                "\nSaving mass flow attributes"
                f"\n{hashtag_line}"
            )
            f.create_group("Mass_flow")

            for key in tqdm(mass_flow_attributes):
                try:
                    f["Mass_flow"].create_dataset(
                        key, data=getattr(self, key)
                    )
                except TypeError:
                    f["Mass_flow"].create_dataset(
                        key, data=str(getattr(self, key))
                    )

            print(
                f"{hashtag_line}"
                "\nSaving mass flow pandas.DataFrames"
                f"\n{hashtag_line}"
            )
            f["Mass_flow"].create_group("Dataframes")
            f["Mass_flow"]["Dataframes"].create_dataset(
                "Axes_labels",
                data=[
                    "Time (s)", "Flow (mL/min)",
                    "Setpoint (mL/min)", "Valve position"
                ]
            )

            for key in tqdm(mass_flow_dataframes):
                try:
                    f["Mass_flow"]["Dataframes"].create_dataset(
                        key, data=getattr(self, key+"_interpolated")
                    )
                except AttributeError:
                    try:
                        f["Mass_flow"]["Dataframes"].create_dataset(
                            key, data=getattr(self, key+"_truncated")
                        )
                    except AttributeError:
                        try:
                            f["Mass_flow"]["Dataframes"].create_dataset(
                                key, data=getattr(self, key)
                            )
                        except AttributeError:
                            print(f"Could not save {key}.")
                        except TypeError:
                            f["Mass_flow"]["Dataframes"].create_dataset(
                                key, data=str(getattr(self, key))
                            )

            print(
                f"{hashtag_line}"
                "\nSaving mass spectrometer attributes"
                f"\n{hashtag_line}"
            )
            f.create_group("Mass_spectrometer")

            for key in tqdm(mass_spectrometer_attributes):
                try:
                    f["Mass_spectrometer"].create_dataset(
                        key, data=getattr(self, key)
                    )
                except TypeError:
                    f["Mass_spectrometer"].create_dataset(
                        key, data=str(getattr(self, key))
                    )
                except AttributeError:
                    print(
                        f"Normalize the data when plotting to save {key}."
                    )

            print(
                f"{hashtag_line}"
                "\nSaving mass spectrometer pandas.DataFrames"
                f"\n{hashtag_line}"
            )
            f["Mass_spectrometer"].create_group("Dataframes")
            try:
                f["Mass_spectrometer"]["Dataframes"].create_dataset(
                    "Axes_labels", data=list(self.mass_spec_df.columns)
                )
            except AttributeError:
                raise AttributeError("Load mass spectrometer file first.")

            for key in tqdm(mass_spectrometer_dataframes):
                try:
                    f["Mass_spectrometer"]["Dataframes"].create_dataset(
                        key, data=getattr(self, key)
                    )
                except AttributeError:
                    print(
                        f"Could not save {key}."
                    )


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_error(array):
    """
    Compute error with the statistical method for a 1D
    array of length l that contain l measurements of the
    same variable.

    :param array: 1D np.array
    """
    l = array.shape[0]

    error = np.sqrt(1/(l*(l-1)) * np.sum((array-array.mean())**2))

    return error
