import numpy as np
import pandas as pd
import datetime
import phdutils
import inspect
import yaml

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from IPython.display import display

from scipy import interpolate
from scipy import special
from scipy.optimize import curve_fit


class XCAT():
    """Combines data from the mass flow controller and the mass spectrometer
    to compare the input flow in the reactor cell and the products of the
    reaction.
    All the data is transformed to pandas.DataFrame objects.
    """

    def __init__(self, configuration_file=None):
        """Initialize the module with a configuration file

        :param configuration_file: .yml file,
         stores metadata specific to the reaction
        """

        self.path_package = inspect.getfile(phdutils).split("__")[0]

        # Load configuration file
        try:
            with open(configuration_file) as filepath:
                yaml_parsed_file = yaml.load(
                    filepath,
                    Loader=yaml.FullLoader
                )

                for key in yaml_parsed_file:
                    setattr(self, key, yaml_parsed_file[key])
                print("Loaded configuration file.")
        except TypeError:
            with open(self.path_package + "experiments/ammonia.yml") as filepath:
                yaml_parsed_file = yaml.load(
                    filepath,
                    Loader=yaml.FullLoader
                )

                for key in yaml_parsed_file:
                    setattr(self, key, yaml_parsed_file[key])
            print("Defaulted to ammonia configuration.")

        except FileNotFoundError:
            with open(self.path_package + "experiments/ammonia.yml") as filepath:
                yaml_parsed_file = yaml.load(
                    filepath,
                    Loader=yaml.FullLoader
                )

                for key in yaml_parsed_file:
                    setattr(self, key, yaml_parsed_file[key])
            print("Could not find configuration file.")
            print("Defaulted to ammonia configuration.")

        self.gaz_travel_time = 12  # time for the gaz to travel from cell to detector
        print(
            f"Travel time from cell to detector fixed to {self.gaz_travel_time} seconds.")
        print("\n#####################################################################\n")

        self.time_shift = 502  # experiment in 2022
        # self.time_shift = 1287 # experiment in 2021
        print(f"Time shift fixed to {self.time_shift} seconds.")
        print("\n#####################################################################\n")

    def load_mass_flow_controller_file(
        self,
        mass_flow_file,
        MRS_pos=None,
        MIX_pos=None,
        mass_flow_file_columns=None,
        time_columns=None,
    ):
        """Initialize the module with a log file, output from XCAT

        :param mass_flow_file: .txt, file from the mass flow controller
        :param MRS_pos: Meaning of the positions of the MRS valve
        :param MIX_pos: Meaning of the positions of the MIX valve
        :param mass_flow_file_columns: Columns of the mass flow file to rename.
        :param time_columns: Columns on the mass flow that give the time

        """
        self.mass_flow_file = mass_flow_file
        print(f"Using {self.mass_flow_file} as filename for data frame input.")
        print("\n#####################################################################\n")

        self.MRS_pos = [
            "All", "Ar", "Ar+Shu", "All", "Rea+Ar", "Closed", "All", "Closed",
            "Ar", "Ar+Shu", "Shu", "Closed", "All", "Rea", "Rea", "Rea+Ar",
            "Rea+Ar", "Closed", "All", "Shu", "Rea+Shu", "Rea+Shu", "Shu",
            "Closed"
        ]
        print(f"Defaulting MRS valve positions to:\n{self.MRS_pos}.")
        print("\n#####################################################################\n")

        self.MIX_pos = [
            "NO", "All", "Closed", "H2+CO", "H2+O2+CO", "H2+O2", "H2", "All",
            "Closed", "NO+O2", "NO+O2+CO", "O2+CO", "O2", "All", "Closed",
            "H2+CO", "NO+H2+CO", "NO+CO", "CO", "All", "Closed", "NO+O2",
            "NO+H2+O2", "NO+H2"
        ]
        print(f"Defaulting MIX valve positions to:\n{self.MIX_pos}.")
        print("\n#####################################################################\n")

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
            f"Defaulted to log file columns (in order):\n{self.mass_flow_file_columns}.")
        print("\n#####################################################################\n")

        # Create DataFrame
        try:
            self.df = pd.read_csv(
                self.mass_flow_file,
                header=None,
                delimiter="\t",
                skiprows=1,
                names=self.mass_flow_file_columns
            )

        except Exception as E:
            raise E

        # Change time to unix epoch
        self.time_columns = [
            "time_no", "time_h2", "time_o2", "time_co",
            "time_ar", "time_shunt", "time_reactor",
            "time_drain", "time_valve"
        ]
        print(
            f"Defaulted to the following time columns (in order):\n{self.time_columns}.")
        print("\n#####################################################################\n")

        for column_name in self.time_columns:
            column = getattr(self.df, column_name)

            column -= 2082844800
        print("Changed time to unix epoch for all time columns")
        print("\n#####################################################################\n")

        # Show preview of DataFrame
        display(self.df.head())
        display(self.df.tail())

        # two times shunt valve ? TODO
        # INJ ?
        # OUT ?

    def separate_mass_flow_dataframes(self, mass_list=False):
        """
        Create new dataframes based on the gases that were analyzed
         Important otherwise the original data frame is too big
         The position of the columns are hardcoded (see defaults).
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

        :param mass_list: list of entries set up in the XCAT,
         defaults to ["no", "h2", "o2", "co", "ar", "shunt", "reactor",
         "drain", "valve"]
        """
        if mass_list:
            self.mass_list = [g.lower() for g in mass_list]

        else:
            self.mass_list = [
                "no", "h2", "o2", "co", "ar", "shunt",
                "reactor", "drain", "valve"
            ]
            print("Defaulted entry list to:", self.mass_list)

        try:
            for entry in self.mass_list:
                if entry == "no":
                    self.no_df = self.df.iloc[:, 0:4].copy()
                    print(f"New DataFrame created starting  or NO.")

                if entry == "h2":
                    self.h2_df = self.df.iloc[:, 4:8].copy()
                    print(f"New DataFrame created starting for H2.")

                if entry == "o2":
                    self.o2_df = self.df.iloc[:, 8:12].copy()
                    print(f"New DataFrame created starting for O2.")

                if entry == "co":
                    self.co_df = self.df.iloc[:, 12:16].copy()
                    print(f"New DataFrame created starting for CO.")

                if entry == "ar":
                    self.ar_df = self.df.iloc[:, 16:20].copy()
                    print(f"New DataFrame created starting for Ar.")

                if entry == "shunt":
                    self.shunt_df = self.df.iloc[:, 20:24].copy()
                    print(f"New DataFrame created starting for shunt.")

                if entry == "reactor":
                    self.reactor_df = self.df.iloc[:, 24:28].copy()
                    print(f"New DataFrame created starting for reactor.")

                if entry == "drain":
                    self.drain_df = self.df.iloc[:, 28:32].copy()
                    print(f"New DataFrame created starting for drain.")

                if entry == "valve":
                    self.valve_df = self.df.iloc[:, 32:35].copy()
                    print(f"New DataFrame created starting for valve.")

        except Exception as e:
            raise e

    def load_mass_spectrometer_file(self, mass_spec_file, skiprows=33):
        """
        Find timestamp of the mass spectrometer (Resifual Gas Analyzer - RGA)
         file and load the file as a DataFrame.
         Careful, all the data extraction from the file is hardcoded due to its
         specific architecture.
        Also interpolates the data in seconds using scipy.interpolate.interp1d
        Defines the time range of the experiment.

        :param mass_spec_file: absolute path to the mass_spec_file (converted to
         .txt beforehand in the RGA software)
        :param skiprows: nb of rows to skip when loading the data. Defaults to
         33.
        """

        # Find starting time
        try:
            with open(mass_spec_file) as f:
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
                        self.time_stamp = int(datetime.datetime.strptime(
                            timeline[12:], "%b %d, %Y  %I:%M:%S  %p").timestamp())
                        self.start_time_utc = timeline[12:]

                    # Get mass spectrometer channels
                    elif line[:3] in channel_index:
                        channels.append(line[25:35].replace(" ", ""))

        except Exception as E:
            self.time_stamp = None
            self.start_time_utc = None
            raise E  # TODO

        print(
            f"Experiment starting time: {self.time_stamp} (unix epoch), {self.start_time_utc}.")

        # Create DataFrame
        try:
            self.rga_data = pd.read_csv(
                mass_spec_file,
                delimiter=',',
                index_col=False,
                names=channels,
                skiprows=skiprows
            )

        except OSError:
            raise OSError

        # Interpolate the data of the mass spectrometer in seconds
        try:
            # Get bad time axis
            x = self.rga_data["Time"]

            # Get time range in seconds of the experiment linked to that dataset
            self.time_range = int(float(x.values[-1]))
            h = self.time_range//3600
            m = (self.time_range - (h*3600))//60
            s = (self.time_range - (h*3600) - (m*60))

            print(f"Experiment time range: {h}h:{m}m:{s}s")

            self.end_time = int(self.time_range + self.time_stamp)
            self.end_timeutc = datetime.datetime.fromtimestamp(
                self.end_time).strftime("%b %d, %Y  %I:%M:%S  %p")
            print(
                f"Experiment end time: {self.end_time} (unix epoch), {self.end_timeutc}.")
            print("Careful, there are two hours added regarding utc time.")

            # Create new time column in integer seconds
            new_time_column = np.round(np.linspace(
                0, self.time_range, self.time_range + 1), 0)

            # Proceed to interpolation over the new time axis
            interpolated_df = pd.DataFrame({
                "Time": new_time_column
            })

            # Iterate over all the columns
            for column in self.rga_data.columns[self.rga_data.columns != "Time"]:

                # Interpolate
                f = interpolate.interp1d(x, self.rga_data[column].values)
                interpolated_df[column] = f(new_time_column)

            # Make sure that the time is integer
            interpolated_df["Time"] = interpolated_df["Time"].astype(int)

            # Save
            setattr(self, "rga_df_interpolated", interpolated_df)
            print(
                f"New DataFrame created for RGA, interpolated on its time range (in s).")

            display(self.rga_df_interpolated.head())
            display(self.rga_df_interpolated.tail())

        except NameError:
            print(
                "To interpolate also the RGA data, you need to load the mass spectrometer file first")

        except Exception as e:
            raise e
            print("Play with the amount of columns names and the amount of rows skipped.")

        # Directly truncate df to avoid errors if forgotten
        self.truncate_mass_flow_df()

    def truncate_mass_flow_df(self):
        """Truncate mass-specific mass flow DataFrame based on timestamp and
         time range if given (otherwise from timestamp to end).
        """

        try:
            # Iterate on mass
            for mass in self.mass_list:

                # Find starting time for this DataFrame
                temp_df = getattr(self, f"{mass}_df").copy()
                index_start_time = (
                    temp_df[f"time_{mass}"] - self.time_stamp).abs().argsort()[0]

                # Reset time
                temp_df = temp_df.iloc[index_start_time:, :].reset_index(
                    drop=True)
                temp_df[f"time_{mass}"] -= temp_df[f"time_{mass}"].values[0]

                # Truncate based on time range, take one more sec to be sure
                # to have enough data for interpolation
                temp_df = temp_df[temp_df[f"time_{mass}"]
                                  < self.time_range + 1]

                setattr(self, f"{mass}_df_truncated", temp_df)

                print(
                    f"New truncated DataFrame created starting from timestamp for {mass} and for {self.time_range} seconds.")

        except Exception as e:
            raise e

        # Directly interpolate the data of the mass flow spectrometer
        self.interpolate_mass_flow_df()

    def interpolate_mass_flow_df(self):
        """Interpolate the data in seconds."""

        try:
            for mass in self.mass_list:

                # Det mass DataFrame
                temp_df = getattr(self, f"{mass}_df_truncated").copy()

                # Get bad time axis
                x = temp_df[f"time_{mass}"]

                x_0 = int(temp_df[f"time_{mass}"].values[0])
                x_1 = int(temp_df[f"time_{mass}"].values[-1])

                # # get time range in seconds of the experiment linked to that dataset from RGA if exist, or just length oh dataset
                # try:
                # 	exp_duration = self.time_range
                # except:
                # 	exp_duration = int(x.values[-1])

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
                interpolated_df[f"time_{mass}"] = interpolated_df[f"time_{mass}"].astype(
                    int)

                # Save
                setattr(self, f"{mass}_df_interpolated", interpolated_df)
                print(f"New interpolated DataFrame created for {mass}.")

        except NameError:
            print(
                "You need to get the experiment time range first, run XCAT.load_mass_spectrometer_file()")

        except TypeError:
            print("Do these files overlap on the same experiment ?")

        except Exception as e:
            raise e

    #### plotting functions for mass flow controller ###

    def plot_mass_flow_entry(
        self,
        mass_list,
        color_dict="ammonia_reaction_colors",
        df="interpolated",
        fig_name="Mass flow data",
        zoom1=None,
        zoom2=None,
        cursor_positions=[None],
        cursor_colours=[None],
        y_pos_text_valve=None,
        hours=True,
        save=False,
        figsize=(10, 6),
        fontsize=15,
    ):
        """Plot the evolution of the input of the reactor
        Each mass corresponds to one channel controlled by the mass flow
        controller.

        :param mass_list: list of mass to be plotted
        :param color_dict: str, name of dict from the configuration file that
         will be used for the colors.
        :param df: DataFrame from which the data will be plotted. Default is
         "interpolated" which corresponds to the data truncated on the mass
         spectrometer time range and in seconds.
        :param fig_name: figure name
        :param zoom1: list of 4 integers to zoom
        :param zoom2: list of 4 integers to zoom
        :param cursor_positions: add cursors using these positions
        :param cursor_colours: colour of the cursors, same length as
         cursor_positions
        :param y_pos_text_valve: Add text on plot, same length as
         cursor_positions
         e.g. {"ar": [45, 3.45], "argon": [45, 3.45], "o2": [5, 2], "h2": [5, 2]},
        :param hours: True to show x scale in hours instead of seconds
        :param save: True to save the plot:
        :param figsize: size of each figure, defaults to (16, 9)
        :param fontsize: size of labels, defaults to 15, title have +2.
        """

        # Get coloring dictionnary
        try:
            color_dict = getattr(self, color_dict)
        except AttributeError:
            print("Wrong name for color dict.")

        try:
            mass_list = [g.lower() for g in mass_list]

            for entry in mass_list:
                plt.close()
                fig, axes = plt.subplots(2, 1, figsize=figsize)

                # Get dataframe
                if df == "interpolated":
                    plot_df = getattr(self, f"{entry}_df_interpolated").copy()

                elif df == "truncated":
                    plot_df = getattr(self, f"{entry}_df_truncated").copy()

                else:
                    raise NameError("Wrong df.")

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
                    label=f"flow_{entry}")

                # Plot setpoint
                axes[0].plot(
                    plot_df[f"time_{entry}"],
                    plot_df[f"setpoint_{entry}"],
                    linestyle="--",
                    label=f"setpoint_{entry}")

                # Plot valve position
                axes[1].plot(
                    plot_df[f"time_{entry}"],
                    plot_df[f"valve_{entry}"],
                    linestyle="--",
                    label=f"valve_{entry}")

                # Zoom
                try:
                    if entry != "ar":
                        axes[0].set_xlim([zoom1[0], zoom1[1]])
                        axes[0].set_ylim([zoom1[2], 10])

                        axes[1].set_xlim([zoom2[0], zoom2[1]])
                        axes[1].set_ylim([zoom2[2], zoom2[3]])

                    else:
                        axes[0].set_xlim([zoom1[0], zoom1[1]])
                        axes[0].set_ylim([zoom1[2], zoom1[3]])

                        axes[1].set_xlim([zoom2[0], zoom2[1]])
                        axes[1].set_ylim([zoom2[2], zoom2[3]])
                except TypeError:
                    pass

                # Plot cursors
                try:
                    for cursor_pos, cursor_colour in zip(
                        cursor_positions,
                        cursor_colours
                    ):
                        axes[0].axvline(
                            cursor_pos,
                            linestyle="--",
                            color=color_dict[cursor_colour],
                            linewidth=1.5,
                            label=cursor_colour
                        )
                        axes[1].axvline(
                            cursor_pos,
                            linestyle="--",
                            color=color_dict[cursor_colour],
                            linewidth=1.5,
                            label=cursor_colour
                        )

                except (TypeError, KeyError):
                    pass

                # Vertical span color to show conditions
                # dict for y positions of text depending on the valve
                try:
                    for x1, x2, cursor_colour in zip(
                        cursor_positions[:-1],
                        cursor_positions[1:],
                        cursor_colours
                    ):
                        for j, ax in enumerate(axes):
                            ax.axvspan(
                                x1,
                                x2,
                                alpha=0.2,
                                facecolor=color_dict[cursor_colour],
                            )

                            # ax.text(
                            #     x1 + (x2-x1)/2,
                            #     y=y_pos_text_valve[entry.lower()][j],
                            #     s=cursor_colour,
                            #     fontsize=fontsize+5
                            # )

                finally:
                    axes[0].set_ylabel("Flow", fontsize=fontsize+2)
                    axes[0].set_xlabel(x_label, fontsize=fontsize+2)
                    axes[1].set_ylabel("Valve position", fontsize=fontsize+2)
                    axes[1].set_xlabel(x_label, fontsize=fontsize+2)

                    axes[0].legend(ncol=len(mass_list), fontsize=fontsize)
                    axes[1].legend(ncol=len(mass_list), fontsize=fontsize)

                    plt.tight_layout()
                    if save:
                        plt.savefig(f"{fig_name}_{entry}.png")
                        print(f"Saved as {fig_name}_{entry} in png formats.")

                    plt.show()

        except KeyError as e:
            raise KeyError("Plot valves with Valves.plot_valves")

    def plot_mass_flow_valves(
        self,
        color_dict="ammonia_reaction_colors",
        df="interpolated",
        fig_name="Mass flow valves",
        zoom1=None,
        zoom2=None,
        cursor_positions=[None],
        cursor_colours=[None],
        y_pos_text_valve=None,
        hours=True,
        save=False,
        figsize=(10, 6),
        fontsize=15,
    ):
        """Plot the evolution of the input of the reactor, for both the MIX and
         the MRS valve. The gases besided Argon (carrier gas) are mixed
         depending on the position of the MIX valve. At the MRS is then added
         Argon.

        :param color_dict: str, name of dict from the configuration file that
         will be used for the colors.
        :param df: DataFrame from which the data will be plotted. Default is
         "interpolated" which corresponds to the data truncated on the mass
         spectrometer time range and in seconds.
        :param fig_name: figure name
        :param zoom1: list of 4 integers to zoom
        :param cursor_positions: add cursors using these positions
        :param cursor_colours: colour of the cursors, same length as
         cursor_positions
        :param y_pos_text_valve: Add text on plot, same length as
         cursor_positions
         e.g. {"ar": [45, 3.45], "argon": [45, 3.45], "o2": [5, 2], "h2": [5, 2]},
        :param hours: True to show x scale in hours instead of seconds
        :param save: True to save the plot:
        :param figsize: size of each figure, defaults to (16, 9)
        :param fontsize: size of labels, defaults to 15, title have +2.
        """

        try:
            color_dict = getattr(self, color_dict)
        except AttributeError:
            print("Wrong name for color dict.")

        try:
            fig, axes = plt.subplots(2, 1, figsize=figsize)

            # Get dataframe
            if df == "interpolated":
                plot_df = getattr(self, f"valve_df_interpolated").copy()

            elif df == "truncated":
                plot_df = getattr(self, f"valve_df_truncated").copy()

            else:
                raise NameError("Wrong df.")

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
            try:
                for cursor_pos, cursor_colour in zip(
                    cursor_positions,
                    cursor_colours
                ):
                    axes[0].axvline(
                        cursor_pos,
                        linestyle="--",
                        color=color_dict[cursor_colour],
                        linewidth=1.5,
                        label=cursor_colour
                    )
                    axes[1].axvline(
                        cursor_pos,
                        linestyle="--",
                        color=color_dict[cursor_colour],
                        linewidth=1.5,
                        label=cursor_colour
                    )

            except (TypeError, KeyError):
                pass

            # Vertical span color to show conditions
            # dict for y positions of text depending on the valve
            try:
                for x1, x2, cursor_colour in zip(
                    cursor_positions[:-1],
                    cursor_positions[1:],
                    cursor_colours
                ):
                    for j, ax in enumerate(axes):
                        ax.axvspan(
                            x1,
                            x2,
                            alpha=0.2,
                            facecolor=color_dict[cursor_colour],
                        )

                        # ax.text(
                        #     x1 + (x2-x1)/2,
                        #     y=y_pos_text_valve[entry.lower()][j],
                        #     s=cursor_colour,
                        #     fontsize=fontsize+5
                        # )

            finally:
                axes[0].set_ylabel("Valve MRS position",
                                   fontsize=fontsize+2)
                axes[0].set_xlabel(x_label,
                                   fontsize=fontsize+2)
                axes[1].set_ylabel("Valve MIX position",
                                   fontsize=fontsize+2)
                axes[1].set_xlabel(x_label,
                                   fontsize=fontsize+2)

                axes[0].legend(fontsize=fontsize)
                axes[1].legend(fontsize=fontsize)

                plt.tight_layout()
                if save:
                    plt.savefig(f"{fig_name}.png")
                    print(f"Saved as {fig_name} in png formats.")

                plt.show()

        except KeyError as e:
            raise KeyError("Plot valves with Valves.plot_valves")

    ##### plotting functions for mass spectrometer #####

    def plot_mass_spec(
        self,
        mass_list=None,
        normalization=False,
        color_dict="ammonia_reaction_colors",
        df="interpolated",
        fig_name="RGA",
        zoom=None,
        cursor_positions=[None],
        cursor_colours=[None],
        y_pos_text_valve=None,
        hours=True,
        save=False,
        figsize=(16, 9),
        fontsize=15,
    ):
        """Plot the evolution of the gas detected by the mass spectrometer.
        Each mass corresponds to one channel detected. Careful, some mass can
         overlap.

        :param mass_list: list of mass to be plotted (if set up prior to the
         experiment for detection).
        :param normalization: False, choose how to normalize the data. Options
         are False, "ptot", "leak", "carrier_gas"
        :param color_dict: str, name of dict from the configuration file that
         will be used for the colors.
        :param df: DataFrame from which the data will be plotted. Default is
         "interpolated" which corresponds to the data truncated on the mass
         spectrometer time range and in seconds.
        :param fig_name: figure name
        :param zoom: list of 4 integers to zoom
        :param cursor_positions: add cursors using these positions
        :param cursor_colours: colour of the cursors, same length as
         cursor_positions
        :param y_pos_text_valve: Add text on plot, same length as
         cursor_positions
         e.g. {"ar": [45, 3.45], "argon": [45, 3.45], "o2": [5, 2], "h2": [5, 2]},
        :param hours: True to show x scale in hours instead of seconds
        :param save: True to save the plot:
        :param figsize: size of each figure, defaults to (16, 9)
        :param fontsize: size of labels, defaults to 15, title have +2.
        """

        # Get coloring dictionnary
        try:
            color_dict = getattr(self, color_dict)
        except AttributeError:
            print("Wrong name for color dict.")

        # Get dataframe
        if df == "interpolated":
            used_df = self.rga_df_interpolated.copy()

        else:
            used_df = self.rga_data.copy()

        # Normalize data
        if normalization == "ptot":

            # Get total pressure
            try:
                print(
                    f"Using P={self.ptot} (bar)for total pressure in the reactor.")
            except:
                self.ptot = 0.3
                print(f"Defaulted to P={self.ptot} (bar)for total pressure in the reactor.\
                \nTo change, add a ptot entry in the configuration file.")

            # Get data
            try:
                print("Using columns specified with ptot_names list.")
                used_arr = used_df[[self.ptot_names]].values
                used_columns = used_df[[self.ptot_names]].columns

            except (TypeError, AttributeError):
                print("Using all the columns to compute the total pressure.\
                    To change, create a ptot_names list in conf file")
                used_arr = used_df.iloc[:, 1:].values
                used_columns = used_df.iloc[:, 1:].columns

            # Normalize
            # Sum columns to have total pressure per second
            ptot_col = self.ptot/used_arr.sum(axis=1)
            ptot_arr = (np.ones(used_arr.shape).T * ptot_col).T

            # Careful units are of self.ptot
            norm_arr = used_arr*ptot_arr

            # Create new DataFrame
            self.norm_df_ptot = pd.DataFrame(norm_arr, columns=used_columns)
            self.norm_df_ptot["Time"] = used_df["Time"]

            print("Created a data frame normalized by total pressure, XCAT.norm_df_ptot")
            display(self.norm_df_ptot.head())

            # Use new df for plots
            used_df = self.norm_df_ptot

            # Changed y scale
            y_units = "bar"

        else:
            y_units = "mbar"

        # Change to hours
        if hours:
            x_label = "Time (h)"
            used_df["Time"] = used_df["Time"].values/3600
        else:
            x_label = "Time (s)"

        # Create figure
        plt.figure(figsize=figsize)
        plt.semilogy()
        plt.title(f"Pressure for each element", fontsize=fontsize+5)

        # If only plotting a subset of the masses
        try:
            for mass in mass_list:
                plt.plot(
                    used_df.Time.values,
                    used_df[mass].values,
                    linewidth=2,
                    label=f"{mass}",
                    color=color_dict[mass]
                )

        except KeyError:
            print("Is there an entry on the color dict for each mass ?")
        except TypeError:
            for mass in used_df.columns[1:]:
                plt.plot(
                    used_df.Time.values,
                    used_df[mass].values,
                    linewidth=2,
                    label=f"{mass}",
                    color=color_dict[mass]
                )

        # Plot cursors
        try:
            for cursor_pos, cursor_colour in zip(
                cursor_positions,
                cursor_colours
            ):
                plt.axvline(
                    cursor_pos,
                    linestyle="--",
                    color=color_dict[cursor_colour],
                    linewidth=1.5,
                    label=cursor_colour
                )
        except (TypeError, KeyError):
            pass

        # Zoom
        try:
            plt.xlim(zoom[0], zoom[1])
            plt.ylim(zoom[2], zoom[3])
        except TypeError:
            pass

        # Vertical span color to show conditions
        # dict for y positions of text depending on the valve

        try:
            for x1, x2, cursor_colour in zip(
                cursor_positions[:-1],
                cursor_positions[1:],
                cursor_colours
            ):
                plt.axvspan(
                    x1,
                    x2,
                    alpha=0.2,
                    facecolor=color_dict[cursor_colour],
                )

                # plt.text(
                #     x1 + (x2-x1)/2,
                #     y=y_pos_text_valve[entry.lower()][j],
                #     s=cursor_colour,
                #     fontsize=fontsize+5
                # )
        except Exception as e:
            raise e

        plt.xlabel(x_label, fontsize=fontsize)
        plt.ylabel(f'Pressure ({y_units})', fontsize=fontsize)
        plt.legend(
            bbox_to_anchor=(0., -0.15, 1., .102),
            loc=3,
            ncol=5,
            mode="expand",
            borderaxespad=0.)
        plt.grid(
            b=True,
            which='major',
            linestyle='-')
        plt.grid(
            b=True,
            which='minor',
            linestyle='--')

        plt.tight_layout()
        if save:
            plt.savefig(f"{fig_name}.png")
            print(f"Saved as {fig_name} in png format.")

        plt.show()

    def plot_mass_spec_norm_leak(self,
                                 interpolated_data=True,
                                 leak_values=[1],
                                 leak_positions=[None, None],
                                 save=False,
                                 fig_name="rga_norm_leak",
                                 plotted_columns=None, zoom=[None, None, None, None], cursor_positions=[None], cursor_labels=[None]):
        """
        Plot RGA data normalized by the values given in leak_values on the intervals given by leak_positions
        e.g. leak_values = [1.3, 2] and leak_positions = [100, 200, 500] will result in a division by 1.3 between indices 100 and 200
        and by a division by 2 between indices 200 and 500.
        works on rga_df_interpolated
        if plotted_columns = None, it plots all the columns
        possible to use a zoom and vertical cursors (one or multiple) 
        """
        if interpolated_data:
            norm_df = self.rga_df_interpolated.copy()
        else:
            norm_df = self.rga_data.copy()

        # change to hours !!!!!!
        norm_df = norm_df/3600

        if len(leak_values) + 1 == len(leak_positions):

            plt.figure(figsize=(18, 13), dpi=80, facecolor='w', edgecolor='k')

            # for each column
            try:
                for element in plotted_columns:
                    normalized_values = norm_df[mass].values.copy()

                    # normalize between the leak positions
                    for j, value in enumerate(leak_values):

                        # print(f"We normalize between {leak_positions[j]} s and {leak_positions[j+1]} s by {value}")

                        normalized_values[leak_positions[j]:leak_positions[j+1]] = (
                            normalized_values[leak_positions[j]:leak_positions[j+1]] / value)

                    plt.plot(norm_df.Time.values, normalized_values,
                             linewidth=2, label=f"{element}", color=self.ammonia_reaction_colors[element])

            # if plotted_columns is None
            except:
                for element in norm_df.columns[1:]:
                    normalized_values = norm_df[mass].values.copy()

                    # normalize between the leak positions
                    for j, value in enumerate(leak_values):

                        # print(f"We normalize between {leak_positions[j]} s and {leak_positions[j+1]} s by {value}")

                        normalized_values[leak_positions[j]:leak_positions[j+1]] = (
                            normalized_values[leak_positions[j]:leak_positions[j+1]] / value)

                    plt.plot(norm_df.Time.values, normalized_values,
                             linewidth=2, label=f"{element}", color=self.ammonia_reaction_colors[element])

            plt.semilogy()

            # cursor
            try:
                # mcolors.CSS4_COLORS["teal"]
                for cursor_pos, cursor_lab in zip(cursor_positions, cursor_labels):
                    plt.axvline(cursor_pos,
                                linestyle="--", color=self.ammonia_conditions_colors[cursor_lab], linewidth=1.5, label=cursor_lab)

            except (TypeError, KeyError):
                print("No cursor")

            finally:
                plt.xlim(zoom[0], zoom[1])
                plt.ylim(zoom[2], zoom[3])

                plt.title(
                    f"Normalized by pressure in RGA (leak valve)", fontsize=24)

                plt.xlabel('Time (h)', fontsize=16)
                plt.ylabel('Pressure (mBar)', fontsize=16)
                plt.legend(bbox_to_anchor=(0., -0.15, 1., .102), loc=3,
                           ncol=5, mode="expand", borderaxespad=0.)
                plt.grid(b=True, which='major', color='b', linestyle='-')
                plt.grid(b=True, which='minor',
                         color=mcolors.CSS4_COLORS["teal"], linestyle='--')

                if save:
                    plt.tight_layout()
                    plt.savefig(f"{fig_name}.png")

                    print(f"Saved as {fig_name} in png format.")

                plt.show()

        else:
            print("Length of leak_positions should be one more than leak_values.")

    def plot_mass_spec_norm_carrier(self,
                                    interpolated_data=True,
                                    save=False,
                                    legend=True,
                                    fig_name="rga_norm_carrier",
                                    carrier_gaz="Ar",
                                    plotted_columns=None,
                                    zoom=[None, None, None, None],
                                    cursor_positions=[None],
                                    cursor_labels=[None],
                                    vlines=[None],
                                    title=False):
        """
        Plot RGA data normalized by one column (carrier_gaz)
        works on rga_df_interpolated
        if plotted_columns = None, it plots all the columns
        possible to use a zoom and vertical cursors (one or multiple) 
        """
        if interpolated_data:
            norm_df = self.rga_df_interpolated.copy()
        else:
            norm_df = self.rga_data.copy()

        # change to hours !!!!!!
        norm_df = norm_df/3600

        plt.figure(figsize=(17, 10), dpi=150, facecolor='w', edgecolor='k')
        plt.semilogy()

        try:
            if carrier_gaz in plotted_columns:
                plotted_columns.remove(carrier_gaz)

        except:
            # No plotted_columns given
            plotted_columns = list(norm_df.columns)
            plotted_columns.remove(carrier_gaz)

        for element in plotted_columns:
            norm_df[mass] = norm_df[mass] / norm_df[carrier_gaz]

            plt.plot(norm_df.Time.values, norm_df[mass].values,
                     linewidth=2, label=f"{element}", color=self.ammonia_reaction_colors[element])

        self.norm_carrier_gas_df = norm_df
        print("Saved df as self.norm_carrier_gas_df.")

        # Add vlines to help find good values
        try:
            for i in vlines:
                plt.axvline(i, color="r", linestyle="--", linewidth=1)
        except:
            pass

        # vertical span color to show conditions
        try:
            # mcolors.CSS4_COLORS["teal"]
            # range(len(cursor_positions)):#, cursor_lab in zip(cursor_positions, cursor_labels):
            for x1, x2, l in zip(cursor_positions[:-1], cursor_positions[1:], cursor_labels):
                plt.axvspan(x1, x2,
                            alpha=0.2,
                            facecolor=self.ammonia_conditions_colors[l],
                            # label = l
                            )

                plt.text(
                    x1 + (x2-x1)/2,
                    y=0.1,
                    s=l,
                    fontsize=25
                )

        except Exception as e:
            raise e
        # except (TypeError, KeyError):
        # 	print("No cursor")

        finally:
            plt.xlim(zoom[0], zoom[1])
            plt.ylim(zoom[2], zoom[3])

            if isinstance(title, str):
                plt.title(title, fontsize=26)

            plt.xlabel('Time (h)', fontsize=35)
            plt.ylabel('Pressure (a.u.)', fontsize=35)
            if legend:
                plt.legend(bbox_to_anchor=(1.15, 1), loc="upper right",
                           ncol=1, fontsize=20, fancybox=True)

            plt.xticks(fontsize=30)
            plt.yticks(fontsize=30)
            plt.grid(b=True, which='major', color='b', linestyle='-')
            plt.grid(b=True, which='minor',
                     color=mcolors.CSS4_COLORS["teal"], linestyle='--')

            if save:
                plt.tight_layout()
                plt.savefig(f"{fig_name}.png", bbox_inches='tight')

                print(f"Saved as {fig_name} in png format.")

            plt.show()

    def plot_mass_spec_norm_temp(self,
                                 start_heating,
                                 nb_points,
                                 amp_final, save=False,
                                 fig_name="rga_norm_temp",
                                 interpolated_data=True,
                                 delta_time=10,
                                 bins_nb=1000, binning=False,
                                 plotted_columns=None,
                                 zoom=[None, None, None, None],
                                 cursor_positions=[None],
                                 cursor_labels=[None]):
        """
        Plot RGA data normalized by the values given in intensity_values on the intervals given by leak_positions
        e.g. intensity_values = [1.3, 2] and leak_positions = [100, 200, 500] will result in a division by 1.3 between indices 100 and 200
        and by a division by 2 between indices 200 and 500.
        works on rga_df_interpolated
        if plotted_columns = None, it plots all the columns
        possible to use a zoom and vertical cursors (one or multiple) 
        """
        if interpolated_data:
            norm_df = self.rga_df_interpolated.copy()
        else:
            norm_df = self.rga_data.copy()

        # change to hours !!!!!!
        norm_df = norm_df/3600

        # recreate points that were used during data acquisition
        # amperage = np.round(np.concatenate((np.linspace(0, amp_final, nb_points*delta_time, endpoint=False), np.linspace(amp_final, 0, nb_points*delta_time))), 2)
        amperage = np.concatenate((np.linspace(0, amp_final, nb_points*delta_time,
                                  endpoint=False), np.linspace(amp_final, 0, nb_points*delta_time)))

        # recreate temperature from heater benchmarks
        temperature = self.heater_poly_fit(amperage)
        temperature_real = np.array([t if t > 25 else 25 for t in temperature])

        # find end time
        end_heating = start_heating + (2 * nb_points) * delta_time

        # timerange of heating
        time_column = norm_df["time"].values[start_heating:end_heating]

        # save in df
        self.rga_during_heating = pd.DataFrame({
            "time": time_column,
            "amperage": amperage,
            "temperature": temperature_real,
        })

        print("Results saved in self.rga_during_heating DataFrame")

        plt.figure(figsize=(18, 9))

        # for each column
        try:
            max_plot = 0
            for element in plotted_columns:
                data_column = norm_df[mass].values[start_heating:end_heating]
                self.rga_during_heating[element] = data_column

                if not binning:
                    plt.plot(temperature_real, data_column,
                             linewidth=2, linestyle="dashdot", label=f"{element}", color=self.ammonia_reaction_colors[element])
                    if max(data_column[element]) > max_plot:
                        max_plot = max(data_column[element])

        # if plotted_columns is None
        except:
            max_plot = 0
            for element in norm_df.columns[1:]:
                data_column = norm_df[mass].values[start_heating:end_heating]
                self.rga_during_heating[element] = data_column

                if not binning:
                    plt.plot(self.rga_during_heating.temperature,
                             self.rga_during_heating[element],
                             linewidth=2,
                             linestyle="dashdot",
                             label=f"{element}",
                             color=self.ammonia_reaction_colors[element])
                    # plt.plot(temperature_real, data_column, linewidth = 2, label = f"{element}")
                    if max(self.rga_during_heating[element]) > max_plot:
                        max_plot = max(self.rga_during_heating[element])

        # Bin after so that the DataFrame on heating timerange already exists
        if binning:

            # Bin the data frame by "time" with bins_nb bins...
            bins = np.linspace(self.rga_during_heating.time.min(
            ), self.rga_during_heating.time.max(), bins_nb)
            groups = self.rga_during_heating.groupby(
                np.digitize(self.rga_during_heating.time, bins))

            # Get the mean of each bin:
            self.binned_data = groups.mean()
            max_plot = 0
            for element in plotted_columns:
                plt.plot(self.binned_data.temperature,
                         self.binned_data[element],
                         linewidth=2,
                         #linestyle = "dashdot",
                         label=f"{element}",
                         color=self.ammonia_reaction_colors[element])
                if max(self.binned_data[element]) > max_plot:
                    max_plot = max(self.binned_data[element])
        plt.semilogy()

        # cursor
        try:
            # mcolors.CSS4_COLORS["teal"]
            for cursor_pos, cursor_lab in zip(cursor_positions, cursor_labels):
                plt.axvline(cursor_pos, linestyle="--",
                            color=self.ammonia_conditions_colors[cursor_lab], linewidth=1.5, label=cursor_lab)

        except (TypeError, KeyError):
            print("No cursor")

        finally:
            plt.xlim(zoom[0], zoom[1])
            plt.ylim(zoom[2], zoom[3])
            plt.text(x=0, y=1.4*max_plot, s="Ammonia oxidation on Pt nanoparticles",
                     fontsize=26, weight='bold', alpha=.75)
            if not binning:
                plt.text(x=0, y=1.25*max_plot,
                         s="""Pressure as a function of the temperature (from input current)""",
                         fontsize=19, alpha=.85)
            else:
                plt.text(x=0, y=1.25*max_plot,
                         s=f"Pressure as a function of the temperature (from input current), with {bins_nb} bins.",
                         fontsize=19, alpha=.85)

            # Ticks

            plt.xticks(np.arange(0, 700, 50), fontsize=20)
            #plt.yticks(np.arange(5, 22.5, 1.5), fontsize=20)

            plt.xlabel('Temperature (°C)', fontsize=25)
            plt.ylabel('Pressure (mBar)', fontsize=25)
            plt.legend(bbox_to_anchor=(0., -0.15, 1., .102), loc=3,
                       ncol=5, mode="expand", borderaxespad=0., fontsize=25)
            plt.grid(b=True, which='major', color='b', linestyle='-')
            plt.grid(b=True, which='minor',
                     color=mcolors.CSS4_COLORS["teal"], linestyle='--')

            if save:
                plt.tight_layout()
                plt.savefig(f"{fig_name}.png", bbox_inches='tight')

                print(f"Saved as {fig_name} in png format.")
            plt.show()

    def fit_error_function(self,
                           initial_guess,
                           new_amper_vect,
                           interpolated_data=True,
                           fitted_columns=None,
                           binning=False,
                           zoom=[None, None, None, None],
                           cursor_positions=[None],
                           cursor_labels=[None]):
        """fit pressure vs temperature dta with error function"""

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
                #axes.plot(xdata, func(xdata, *popt))
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
                #axes.plot(xdata, func(xdata, *popt))
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
