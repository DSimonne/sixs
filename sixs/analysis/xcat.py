import numpy as np
import pandas as pd
from datetime import datetime
import sixs
import inspect
import yaml
import os

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

    What's left to do?
    See if we can create a routine to normalize the data by the coefficient of
    the leak that is linked to the temperature. It can be extracted from the an
    experiment without any products but just a temperature change.
    """

    def __init__(self, configuration_file=False):
        """Initialize the module with a configuration file

        :param configuration_file: .yml file,
         stores metadata specific to the reaction
        """

        self.path_package = inspect.getfile(sixs).split("__")[0]

        # Load configuration file
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

        # Time for the gaz to travel from cell to detector
        try:
            print(
                f"Travel time from cell to detector fixed to {self.gaz_travel_time} seconds.")
            print(
                "\n#####################################################################\n")
        except AttributeError:
            self.gaz_travel_time = 12
            print(
                f"Travel time from cell to detector fixed to {self.gaz_travel_time} seconds.")
            print(
                "\n#####################################################################\n")

        # Time shift between computers
        try:
            print(
                f"Travel shift between computers fixed to {self.time_shift} seconds.")
            print(
                "\n#####################################################################\n")
        except AttributeError:
            self.time_shift = 502  # jan 2022
            # self.time_shift = 1287 # experiment in 2021
            print(
                f"Travel shift between computers fixed to {self.time_shift} seconds.")
            print(
                "\n#####################################################################\n")

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
        self.mass_flow_start_time_epoch = self.df.iloc[0, 0]
        self.mass_flow_start_time = datetime.fromtimestamp(
            self.df.iloc[0, 0]).strftime('%Y-%m-%d %H:%M:%S')
        self.mass_flow_end_time_epoch = self.df.iloc[0, 0]
        self.mass_flow_end_time = datetime.fromtimestamp(
            self.df.iloc[-1, 0]).strftime('%Y-%m-%d %H:%M:%S')

        print(
            f"Mass flow. starting time: {self.mass_flow_start_time_epoch} (unix epoch), {self.mass_flow_start_time}.")
        print(
            f"Mass flow. end time: {self.mass_flow_end_time_epoch} (unix epoch), {self.mass_flow_end_time}.")
        print("Careful, there are two hours added regarding utc time.")
        print("\n#####################################################################\n")

        # Show preview of DataFrame
        display(self.df.head())
        display(self.df.tail())

        # two times shunt valve ? TODO
        # INJ ?
        # OUT ?
        self.separate_mass_flow_dataframes()

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
                        self.mass_spec_start_time_epoch = int(datetime.strptime(
                            timeline[12:], "%b %d, %Y  %I:%M:%S  %p").timestamp())
                        self.mass_spec_start_time = datetime.fromtimestamp(
                            self.mass_spec_start_time_epoch).strftime('%Y-%m-%d %H:%M:%S')

                    # Get mass spectrometer channels
                    elif line[:3] in channel_index:
                        channels.append(line[25:35].replace(" ", ""))

            print(
                f"Mass spec. starting time: {self.mass_spec_start_time_epoch} (unix epoch), {self.mass_spec_start_time}.")

        except Exception as E:
            self.mass_spec_start_time_epoch = None
            self.mass_spec_start_time = None
            raise E  # TODO

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

            self.mass_spec_end_time_epoch = int(
                self.time_range + self.mass_spec_start_time_epoch)
            self.mass_spec_end_time = datetime.fromtimestamp(
                self.mass_spec_end_time_epoch).strftime('%Y-%m-%d %H:%M:%S')
            print(
                f"Mass spec. end time: {self.mass_spec_end_time_epoch} (unix epoch), {self.mass_spec_end_time}.")
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
            print("\n#######################################################")
            print("""
                \nCould not interpolate the rga data.
                \nTo interpolate also the RGA data, you need to load the mass spectrometer file first
                """)
            print("#######################################################\n")

        except ValueError:
            print("\n#######################################################")
            print("""
                \nCould not interpolate the rga data.
                \nPlay with the amount of columns names and the amount of rows skipped.")
                """)
            print("#######################################################\n")

        # Directly truncate df to avoid errors if forgotten
        self.truncate_mass_flow_df()

        self.interpolate_mass_flow_df()

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
                    temp_df[f"time_{mass}"] - self.mass_spec_start_time_epoch).abs().argsort()[0]

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

    def plot_mass_flow_entry(
        self,
        mass_list=["NO", "H2", "O2", "CO", "Ar", "shunt", "reactor", "drain"],
        df="interpolated",
        figsize=(10, 6),
        fontsize=15,
        zoom1=None,
        zoom2=None,
        color_dict="gas_colors",
        cursor_positions=[None],
        cursor_labels=[None],
        text_dict=None,
        hours=True,
        save=False,
        fig_name="mass_flow_entry",
    ):
        """Plot the evolution of the input of the reactor
        Each mass corresponds to one channel controlled by the mass flow
        controller.

        :param mass_list: list of mass to be plotted
        :param df: DataFrame from which the data will be plotted. Default is
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
         "o2": [5, 2], "h2": [5, 2]}
        :param hours: True to show x scale in hours instead of seconds
        :param save: True to save the plot:
        :param fig_name: figure name when saving
        """

        # Get coloring dictionnary
        try:
            color_dict = getattr(self, color_dict)
        except AttributeError:
            print("Wrong name for color dict.")

        mass_list = [g.lower() for g in mass_list]

        for entry in mass_list:
            print("#######################################################")
            print("Plotting for ", entry)
            print("#######################################################")
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
                label=f"valve_{entry}")

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
                    except KeyError:
                        print("Is there an entry on the color dict for each mass ?")
                    # except (TypeError, KeyError):
                    #     pass

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
            axes[1].set_ylabel("Valve position", fontsize=fontsize+2)
            axes[1].set_xlabel(x_label, fontsize=fontsize+2)

            axes[0].legend(ncol=len(mass_list), fontsize=fontsize)
            axes[1].legend(ncol=len(mass_list), fontsize=fontsize)

            plt.tight_layout()
            if save:
                plt.savefig(f"{fig_name}_{entry}.png")
                plt.savefig(f"{fig_name}_{entry}.pdf")
                print(
                    f"Saved as {fig_name}_{entry} in (png, pdf) formats.")

            plt.show()

    def plot_mass_flow_valves(
        self,
        df="interpolated",
        figsize=(10, 6),
        fontsize=15,
        zoom1=None,
        zoom2=None,
        color_dict="gas_colors",
        cursor_positions=[None],
        cursor_labels=[None],
        text_dict=None,
        hours=True,
        save=False,
        fig_name="mass_flow_valves",
    ):
        """Plot the evolution of the input of the reactor, for both the MIX and
         the MRS valve. The gases besided Argon (carrier gas) are mixed
         depending on the position of the MIX valve. At the MRS is then added
         Argon.

        :param df: DataFrame from which the data will be plotted. Default is
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
        :param save: True to save the plot
        :param fig_name: figure name when saving
        """

        try:
            color_dict = getattr(self, color_dict)
        except AttributeError:
            print("Wrong name for color dict.")

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

        plt.tight_layout()
        if save:
            plt.savefig(f"{fig_name}.png")
            plt.savefig(f"{fig_name}.pdf")
            print(f"Saved as {fig_name} in (png, pdf) formats.")

        plt.show()

    def plot_mass_spec(
        self,
        mass_list=None,
        df="interpolated",
        heater_file=None,
        figsize=(16, 9),
        fontsize=15,
        norm_ptot=False,
        ptot_mass_list=False,
        norm_carrier=False,
        pressure_carrier=False,
        zoom=None,
        color_dict="gas_colors",
        cursor_positions=[None],
        cursor_labels=[None],
        text_dict=None,
        hours=True,
        save=False,
        fig_name="rga_data",
    ):
        """Plot the evolution of the gas detected by the mass spectrometer.
        Each mass corresponds to one channel detected. Careful, some mass can
         overlap.

        :param mass_list: list of mass to be plotted (if set up prior to the
         experiment for detection).
        :param df: DataFrame from which the data will be plotted. Default is
         "interpolated" which corresponds to the data truncated on the mass
         spectrometer time range and in seconds.
        :param norm_ptot: False, normalize the data by total pressure if float.
        :param ptot_mass_list: False, list of mass to use for normalization, if
         False same as mass_list, used with norm_ptot
        :param norm_carrier: False, normalize the data by carrier gas
         (norm_carrier) pressure if str. Choose a gas in mass_list.
        :param pressure_carrier: False. Pressure of the carrier gas (bar) in the
         reactor (e.g. p_Argon = 0.5 bar). Allows to normalize the data by this
         value to find the pressures in the reactor cell. used with norm_carrier
        :param figsize: size of each figure, defaults to (16, 9)
        :param fontsize: size of labels, defaults to 15, title have +2.
        :param zoom: list of 4 integers to zoom
        :param color_dict: str, name of dict from the configuration file that
         will be used for the colors.
        :param cursor_positions: add cursors using these positions
        :param cursor_labels: colour of the cursors, same length as
         cursor_positions
        :param text_dict: Add text on plot, same length as
         cursor_positions, e.g. {"ar": [45, 3.45], "argon": [45, 3.45],
         "o2": [5, 2], "h2": [5, 2]}
        :param hours: True to show x scale in hours instead of seconds
        :param heater_file: path to heater data, in csv format, must contain
         a time and a temperature column.
        :param save: True to save the plot:
        :param fig_name: figure name when saving
        """

        # Get coloring dictionnary
        try:
            color_dict = getattr(self, color_dict)
        except AttributeError:
            print("Wrong name for color dict.")

        # Get dataframe
        if df == "interpolated":
            self.norm_df = self.rga_df_interpolated.copy()

        else:
            self.norm_df = self.rga_data.copy()

        # Use all comlumns if none specified
        if mass_list is None:
            mass_list = list(self.norm_df.columns[1:-1])
            print(mass_list)
            print("Defaulted mass_list to all columns")

        # Normalize data by total pressure
        if isinstance(norm_ptot, float):
            self.ptot = norm_ptot

            # Total pressure
            print(
                f"Using P={self.ptot} (bar)for total pressure in the reactor.")

            # What gases do we use for the normalization ?
            if isinstance(ptot_mass_list, list):  # list if specified
                self.ptot_mass_list = ptot_mass_list
                print("Using columns specified with ptot_mass_list for normalization.")

            elif isinstance(mass_list, list):  # otherwise list of mass to plot
                self.ptot_mass_list = mass_list
                print("""
                    \nUsing columns specified with mass_list for normalization. \
                    \nIf you want to use other gases, use ptot_mass_list.
                    """)

            # Get data
            used_arr = self.norm_df[self.ptot_mass_list].values

            # Normalize
            # Sum columns to have total pressure per second
            ptot_col = self.ptot/used_arr.sum(axis=1)
            ptot_arr = (np.ones(used_arr.shape).T * ptot_col).T

            # Careful units are of self.ptot
            norm_arr = used_arr*ptot_arr

            # Create new DataFrame
            norm_df_ptot = pd.DataFrame(norm_arr, columns=self.ptot_mass_list)
            norm_df_ptot["Time"] = self.norm_df["Time"]
            self.norm_df = norm_df_ptot  # to do better

            print("Normalized RGA pressure by total pressure.")
            display(self.norm_df.head())

            # Changed y scale
            y_units = "bar"

        # Normalize data by carrier gas
        if isinstance(norm_carrier, str):
            self.carrier_gaz = norm_carrier

            # Carrier gas
            print(
                f"Using {self.carrier_gaz} as carrier gas for normalization.")

            # Remove carrier gas from list to avoid division by 1
            try:
                mass_list.remove(self.carrier_gaz)
            except ValueError:
                # Not in the list
                pass

            # Normalize
            try:
                for mass in mass_list:
                    y = self.norm_df[mass] / self.norm_df[self.carrier_gaz]

                    # Multiply by carrier pressure if known
                    if isinstance(pressure_carrier, float):
                        self.norm_df[mass] = y * pressure_carrier
                    else:
                        self.norm_df[mass] = y

            except Exception as E:
                raise E

            # Now modify the carrier gas pressure
            if isinstance(pressure_carrier, float):
                self.norm_df[self.carrier_gaz] = np.ones(
                    len(self.norm_df)) * pressure_carrier
            else:
                self.norm_df[self.carrier_gaz] = np.ones(len(self.norm_df))

            # Put the carrier gas back in the list of mass to plot
            mass_list.append(self.carrier_gaz)

            print("Normalized RGA pressure by carrier gas pressure.")
            display(self.norm_df.head())

            # Change y scale
            y_units = "bar" if isinstance(pressure_carrier, float) else None

        # Normalize data by leak pressure

        # No normalisation
        else:
            y_units = "mbar"

        # Change to hours
        if hours:
            x_label = "Time (h)"
            self.norm_df["Time"] = self.norm_df["Time"].values/3600
        else:
            x_label = "Time (s)"

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.semilogy()
        ax.set_title(
            f"Pressure in XCAT reactor cell for each mass",
            fontsize=fontsize+5
        )

        # If only plotting a subset of the masses
        try:
            for mass in mass_list:
                ax.plot(
                    self.norm_df.Time.values,
                    self.norm_df[mass].values,
                    linewidth=2,
                    label=f"{mass}",
                    color=color_dict[mass]
                )

        except KeyError:
            print("Is there an entry on the color dict for each mass ?")

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
            cursor_labels
        ):
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
                    x1 + (x2-x1)/2,
                    y=text_dict[cursor_label],
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
            ncol=5,
            mode="expand",
            borderaxespad=0.,
            fontsize=fontsize-2
        )
        ax.grid(True, which='major', linestyle='-')
        ax.grid(True, which='minor', linestyle='--')

        fig.tight_layout()
        if save:
            plt.savefig(f"{fig_name}.png")
            plt.savefig(f"{fig_name}.pdf")
            print(f"Saved as {fig_name} in (png, pdf) formats.")

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
        Simple background reduction routine using np.polyfit()

        :param df: dataframe to use
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

        axs[0].plot(x_peak,
                    y_peak,
                    label="Peak"
                    )
        axs[0].set_ylabel(y_col)
        axs[0].set_xlabel(x_col)

        # No peak area in this range
        x_no_peak = np.concatenate([x[:peak_start], x[peak_end:]])
        y_no_peak = np.concatenate([y[:peak_start], y[peak_end:]])

        axs[0].plot(x_no_peak,
                    y_no_peak,
                    label="No peak"
                    )

        # Fit range without peak to get background
        poly = np.polyfit(x_no_peak, y_no_peak, degree)
        poly_fit = np.poly1d(poly)

        # Plot data on range of interest minus background
        axs[0].plot(x, poly_fit(x), label="Background fit")
        axs[0].legend(fontsize=15)

        # Save background
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
        """Add a temperature column to the DataFrame from a csv file
        The csv file must have a unix timestamp "time" column that will be used
        to add the temperature values at the right time in the DataFrame.

        :param df: DataFrame object to which we assign the new df,
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
