import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
import datetime
from scipy import interpolate
from scipy import special
from scipy.optimize import curve_fit

import matplotlib.colors as mcolors

plt.style.use('fivethirtyeight')

"""
To make the valves data and the rga data work together, we need to:
	Interpolate both datasets on integer seconds
	Cut the valves data depending on the timestamp of the rga data
"""


class XCAT():
	"""Transforms scat raw data to pandas.DataFrame object from path/to/data
	Then allows to compare with differnet rga datasets
	"""

	def __init__(self, log_file):
		"""init with raw valves data
		"""

		self.log_file = log_file
		self.pathtodata = self.log_file if self.log_file.endswith(".txt") else self.log_file + ".txt"

		self.MRS_pos = [
						"All", "Ar", "Ar+Shu", "All", "Rea+Ar", "Closed", "All", "Closed", "Ar", "Ar+Shu", "Shu",
						"Closed", "All", "Rea", "Rea", "Rea+Ar", "Rea+Ar", "Closed", "All", "Shu", "Rea+Shu",
						"Rea+Shu", "Shu", "Closed",
						]

		self.MIX_pos = [
						"NO", "All", "Closed", "H2+CO", "H2+O2+CO", "H2+O2", "H2", "All", "Closed", "NO+O2",
						"NO+O2+CO", "O2+CO", "O2", "All", "Closed", "H2+CO", "NO+H2+CO", "NO+CO", "CO", "All",
						"Closed", "NO+O2", "NO+H2+O2", "NO+H2"
						]

		self.gaz_travel_time = 12 #time for the gaz to travel from cell to detector
		print(f"Travel time from cell to detector fixed to {self.gaz_travel_time} seconds for now ...")

		self.time_shift = 1287
		print(f"Time shift fixed to {self.time_shift} seconds for now ...")

		self.ammonia_conditions_colors = {"Argon" : "#008fd5", "Ar" : "#008fd5", "A" : "#fc4f30", "B" : "#e5ae38", "C" : "#6d904f", 
		                   			"D" : "#8b8b8b", "E" : "#810f7c"}

		# self.diverging_colors_1 = {"Argon" : "#8c510a","Ar" : "#8c510a", "A" : "#d8b365",
		# 						"B" : "#f6e8c3", "C" : "#c7eae5","D" : "#5ab4ac", "E" : "#01665e"}

		self.ammonia_reaction_colors = {"Ar" : "#008fd5", "Argon" : "#008fd5", "NH3" : "#df65b0", "O2" : "#fc4f30",
									"H2O" : "#e5ae38", "NO" : "#6d904f", "N2O" : "#8b8b8b", "N2" : "#810f7c"}

		self.xcat_colors = {"H2" : "#008fd5", "O2" : "#008fd5", "Ar" : "#fc4f30"}

		# Create dataframe
		try:
			self.df = pd.read_csv(
			    self.pathtodata,
			    header = None,
			    delimiter = "\t",
			    skiprows = 1,
			    names = [
		            "time_no", "flow_no", "setpoint_no", "valve_no",
		            "time_h2", "flow_h2", "setpoint_h2", "valve_h2",
		            "time_o2", "flow_o2", "setpoint_o2", "valve_o2",
		            "time_co", "flow_co", "setpoint_co", "valve_co",
		            "time_ar", "flow_ar", "setpoint_ar", "valve_ar",
		            "time_shunt", "flow_shunt", "setpoint_shunt", "valve_shunt",
		            "time_reactor", "flow_reactor", "setpoint_reactor", "valve_reactor",
		            "time_drain", "flow_drain", "setpoint_drain", "valve_drain",
		            "time_valve", "valve_MRS", "valve_MIX"]
		            )

		except OSError:
			raise OSError
		
		# Change time to unix epoch
		for column_name in ["time_no" ,"time_h2" , "time_o2", "time_co", "time_ar" , "time_shunt", "time_reactor", "time_drain", "time_valve"]:
		    column = getattr(self.df, column_name)
		    
		    column -= 2082844800
		print("Changed time to unix epoch")

		display(self.df.head())

		display(self.df.tail())

		# define heater parameters
		self.heater_ramp = pd.DataFrame({
								    "Current" : [0.3 , 0.4 , 0.5 , 0.6 , 0.7 , 0.8 , 0.9 , 1.1 , 1.3 , 1.5 , 1.7],
								    "Temperature" : [T + 25 for T in [72 , 96 , 134 , 170 , 217 , 245 , 300, 355 ,433, 500, 571]]
								})
		
		self.heater_poly_fit_coef = np.polyfit(self.heater_ramp["Current"], self.heater_ramp["Temperature"], 1)
		self.heater_poly_fit = np.poly1d(self.heater_poly_fit_coef)

		print("Defined heater ramp as self.heater_ramp")

	
	# separate XCAT data
	def separate_xcat_dataframes(self, entry_list = False):
		"""Create new dataframes based on the gases that were analyzed"""
		if entry_list:
			self.entry_list = [g.lower() for g in entry_list]

		else:
			self.entry_list = ["no", "h2", "o2", "co", "ar", "shunt", "reactor", "drain", "valve"]

		try:
			for entry in self.entry_list:
				if entry == "no":
					self.no_df = self.df.iloc[:, 0:4].copy()
					print(f"New dataframe created starting  or NO.")

				if entry == "h2":
					self.h2_df = self.df.iloc[:, 4:8].copy()
					print(f"New dataframe created starting for H2.")

				if entry == "o2":
					self.o2_df = self.df.iloc[:, 8:12].copy()
					print(f"New dataframe created starting for O2.")

				if entry == "co":
					self.co_df = self.df.iloc[:, 12:16].copy()
					print(f"New dataframe created starting for CO.")

				if entry == "ar":
					self.ar_df = self.df.iloc[:, 16:20].copy()
					print(f"New dataframe created starting for Ar.")

				if entry == "shunt":
					self.shunt_df = self.df.iloc[:, 20:24].copy()
					print(f"New dataframe created starting for shunt.")

				if entry == "reactor":
					self.reactor_df = self.df.iloc[:, 24:28].copy()
					print(f"New dataframe created starting for reactor.")

				if entry == "drain":
					self.drain_df = self.df.iloc[:, 28:32].copy()
					print(f"New dataframe created starting for drain.")

				if entry == "valve":
					self.valve_df = self.df.iloc[:, 32:35].copy()
					print(f"New dataframe created starting for valve.")

		except Exception as e:
			raise e


	# load and interpolate rga data
	def load_rga(self, rga_file, skiprows = 33):
		"""Find timestamp of rga file and loads it as a DataFrame"""

		# Stupid trick to find channels automatically in good order
		names = ["time"]
		poss = [f"{i+1}  " for i in range(9)] + [f"{i+1} " for i in range(9, 20)]

		# Find timestamp
		with open(rga_file) as f:
			for line in f:
				if "Start time, " in line:
					timeline = line.rstrip() #rstrip remove the spaces at the end of the line
					self.time_stamp = int(datetime.datetime.strptime(timeline[12:], "%b %d, %Y  %I:%M:%S  %p").timestamp())
					self.start_time_utc = timeline[12:]
					print(f"Experiment starting time: {self.time_stamp} (unix epoch), {self.start_time_utc}.")


				if line[:3] in poss:
					names.append(line[25:35].replace(" ", ""))

		# Create dataframe
		try:
			self.rga_data = pd.read_csv(
			    rga_file,
			    delimiter = ',',
			    index_col = False,
			    names = names,
			    skiprows = skiprows)

		except OSError:
			raise OSError

		# Interpolate the valves data
		try:
			# get bad time axis 
			x = self.rga_data["time"]

			# get duration in seconds of the experiment linked to that dataset
			self.duration = int(float(x.values[-1]))
			h = self.duration//3600
			m = (self.duration - (h*3600))//60
			s = (self.duration - (h*3600) - (m*60))

			print(f"Experiment duration: {h}h:{m}m:{s}s")

			self.end_time = int(self.duration + self.time_stamp)
			self.end_timeutc = datetime.datetime.fromtimestamp(self.end_time).strftime("%b %d, %Y  %I:%M:%S  %p")
			print(f"Experiment end time: {self.end_time} (unix epoch), {self.end_timeutc}.")
			print("Careful, there are two hours added regarding utc time.")

			# create new time column in integer seconds
			new_time_column = np.round(np.linspace(0, self.duration, self.duration + 1), 0)

			# proceed to interpolation over the new time axis
			interpolated_df = pd.DataFrame({
			    f"time" : new_time_column
			    })

			# Iterate over all the columns
			for col in self.rga_data.columns[self.rga_data.columns != "time"]:

			    y = self.rga_data[col].values

			    # tck = interpolate.splrep(x, y, s = 0)
			    # y_new = interpolate.splev(new_time_column, tck)

			    # this way is better
			    f = interpolate.interp1d(x, y)
			    interpolated_df[col] = f(new_time_column)

			# make sure that the time is integer
			interpolated_df["time"] = interpolated_df["time"].astype(int)

			# save
			setattr(self, "rga_df_interpolated", interpolated_df)
			print(f"New dataframe created for rga, interpolated on its duration (in s).")

			display(self.rga_df_interpolated.head())
			display(self.rga_df_interpolated.tail())

		except NameError:
			print("To interpolate also the rga data, you need to run Valves.load_rga() first")

		except Exception as e:
			raise e
			print("Play with the amount of columns names and the amount of rows skipped.")


	# truncate data based on rga file
	def truncate_xcat_df(self):
		"""Truncate entry specific df based on timestamp and duration if given (otherwise from timestamp to end)"""

		if not self.entry_list:
			raise KeyError("Create rga dataframes first")

		try:
			for entry in self.entry_list:

				# find starting time for this df
				temp_df = getattr(self, f"{entry}_df").copy()
				index_start_time = (temp_df[f"time_{entry}"] - self.time_stamp).abs().argsort()[0]

				# create truncated df based on the duration of the rga data
				if self.duration:
					# reset time
					temp_df = temp_df.iloc[index_start_time:, :].reset_index(drop=True)
					temp_df[f"time_{entry}"] -= temp_df[f"time_{entry}"].values[0]

					# truncate based on duration (take one more sec to be sure to have enough data for interpolation)
					temp_df = temp_df[temp_df[f"time_{entry}"] < self.duration + 1]

					setattr(self, f"{entry}_df_truncated", temp_df)

					print(f"New truncated dataframe created starting from timestamp for {entry} and for {self.duration} seconds.")

				else:
					pass
					# temp_df = temp_df.iloc[index_start_time:, :].reset_index(drop=True)
					# temp_df[f"time_{entry}"] -= temp_df[f"time_{entry}"][0]

					# setattr(self, f"{entry}_df_truncated", temp_df)

					# print(f"New truncated dataframe created starting from timestamp for {entry}.")

		except Exception as e:
			raise e


	# only after truncation, otherwise too big
	def interpolate_xcat_df(self):
		"""Only possible if we already ran load_rga"""

		try:
			for entry in self.entry_list:

				# get dataframe
				temp_df = getattr(self, f"{entry}_df_truncated").copy()

				# get bad time axis 
				x = temp_df[f"time_{entry}"]

				x_0 = int(temp_df[f"time_{entry}"].values[0])
				x_1 = int(temp_df[f"time_{entry}"].values[-1])

				# # get duration in seconds of the experiment linked to that dataset from rga if exist, or just length oh dataset
				# try:
				# 	exp_duration = self.duration
				# except:
				# 	exp_duration = int(x.values[-1])

				# create new time column in integer seconds
				new_time_column = np.linspace(x_0, x_1, x_1 + 1)

				# proceed to interpolation over the new time axis
				interpolated_df = pd.DataFrame({
				    f"time_{entry}" : new_time_column
				    })

				# Iterate over all the columns
				for col in temp_df.columns[temp_df.columns != f"time_{entry}"]:

				    y = temp_df[col].values

				    tck = interpolate.splrep(x, y, s = 0)
				    y_new = interpolate.splev(new_time_column, tck)
				    interpolated_df[col] = y_new

				# make sure that the time is integer
				interpolated_df[f"time_{entry}"] = interpolated_df[f"time_{entry}"].astype(int)

				# save
				setattr(self, f"{entry}_df_interpolated", interpolated_df)
				print(f"New interpolated dataframe created for {entry}.")

		except NameError:
			print("You need to get the experiment duration first, run XCAT.load_rga()")

		except Exception as e:
			raise e


	#### plotting functions for xcat data ###

	def plot_xcat_entry(self,
						plot_entry_list,
						df = "interpolated",
						save = False,
						fig_name = "xcat_entry",
						zoom1 = [None, None, None, None],
						zoom2 = [None, None, None, None],
						cursor_positions = [None],
						cursor_labels = [None],
						y_pos_text_valve = {"ar" : [45, 3.45], "argon" : [45, 3.45], "o2" : [5, 2], "h2" : [5, 2]}):
		try:
			plot_entry_list = [g.lower() for g in plot_entry_list]

			for entry in plot_entry_list:
				plt.close()
				fig, axes = plt.subplots(2, 1, figsize = (16, 9))

				if df == "interpolated":
					plot_df = getattr(self, f"{entry}_df_interpolated").copy()

				elif df == "truncated":
					plot_df = getattr(self, f"{entry}_df_truncated").copy()

				else:
					raise NameError("Wrong df.")

				# change to hours !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
				plot_df[f"time_{entry}"] = plot_df[f"time_{entry}"].values/3600

				# No need to put colors bc each entry has its own plot
				axes[0].plot(plot_df[f"time_{entry}"], plot_df[f"flow_{entry}"], label = f"flow_{entry}")

				axes[0].plot(plot_df[f"time_{entry}"], plot_df[f"setpoint_{entry}"], "--", label = f"setpoint_{entry}")

				axes[1].plot(plot_df[f"time_{entry}"], plot_df[f"valve_{entry}"], "--", label = f"valve_{entry}")

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

				# cursor
				try:
					#mcolors.CSS4_COLORS["teal"]
					for cursor_pos, cursor_lab in zip(cursor_positions, cursor_labels):
						axes[0].axvline(cursor_pos, linestyle = "--", color = self.ammonia_conditions_colors[cursor_lab], linewidth = 1.5, label = cursor_lab)
						axes[1].axvline(cursor_pos, linestyle = "--", color = self.ammonia_conditions_colors[cursor_lab], linewidth = 1.5, label = cursor_lab)

				except (TypeError, KeyError):
					print("No cursor")

				# vertical span color to show conditions
				# dict for y positions of text depending on the valve
				try:
					#mcolors.CSS4_COLORS["teal"]
					for x1, x2, l in zip(cursor_positions[:-1], cursor_positions[1:], cursor_labels):# range(len(cursor_positions)):#, cursor_lab in zip(cursor_positions, cursor_labels):
						for j, ax in enumerate(axes):
							ax.axvspan(x1, x2,
								alpha = 0.2,
								facecolor = self.ammonia_conditions_colors[l],
								# label = l
								)

							ax.text(
								x1 + (x2-x1)/2,
								y = y_pos_text_valve[entry.lower()][j],
								s = l,
								fontsize = 25
								)


				finally:
					axes[0].set_ylabel("Flow",fontsize=16)
					axes[0].set_xlabel("Time (h)",fontsize=16)
					axes[1].set_ylabel("Valve position",fontsize=16)
					axes[1].set_xlabel("Time (h)",fontsize=16)

					axes[0].legend(ncol=len(plot_entry_list))
					axes[1].legend(ncol=len(plot_entry_list))

					if save == True:
						plt.tight_layout()
						plt.savefig(f"{fig_name}_{entry}.png")

						print(f"Saved as {fig_name}_{entry} in png formats.")

					plt.show()

		except KeyError as e:
			raise KeyError("Plot valves with Valves.plot_valves")


	def plot_xcat_valves(self,
						df = "interpolated",
						save = False,
						fig_name = "xcat_valves",
						zoom1 = [None, None, None, None],
						zoom2 = [None, None, None, None],
						cursor_positions = [None],
						cursor_labels = [None]):
		try:

			fig, axes = plt.subplots(2, 1, figsize = (16, 9))

			if df == "interpolated":
				plot_df = getattr(self, "valve_df_interpolated").copy()

			elif df == "truncated":
				plot_df = getattr(self, "valve_df_truncated").copy()

			else:
				raise NameError("Wrong df.")

			axes[0].plot(plot_df["time_valve"], plot_df["valve_MRS"], label = "valve_MRS")
			axes[1].plot(plot_df["time_valve"], plot_df["valve_MIX"], label = "valve_MIX")

			# cursor
			try:
				#mcolors.CSS4_COLORS["teal"]
				for cursor_pos, cursor_lab in zip(cursor_positions, cursor_labels):
					plt.axvline(cursor_pos, linestyle = "--", color = self.ammonia_conditions_colors[cursor_lab], linewidth = 1.5, label = cursor_lab)

			except (TypeError, KeyError):
				print("No cursor")

			finally:

				axes[0].set_xlim([zoom1[0], zoom1[1]])
				axes[0].set_ylim([zoom1[2], zoom1[3]])
				axes[1].set_xlim([zoom2[0], zoom2[1]])
				axes[1].set_ylim([zoom2[2], zoom2[3]])

				axes[0].set_ylabel("Valve position",fontsize=16)
				axes[0].set_xlabel("Time (h)",fontsize=16)
				axes[1].set_ylabel("Valve position",fontsize=16)
				axes[1].set_xlabel("Time (h)",fontsize=16)

				axes[0].legend()
				axes[1].legend()

				if save == True:
					plt.tight_layout()
					plt.savefig(f"{fig_name}_{entry}.png")

					print(f"Saved as {fig_name}_{entry} in png formats.")

				plt.show()

		except KeyError as e:
			raise KeyError("Are you sure you are trying to plot a gaz?")

		except AttributeError as e:
			raise e
			print("Interpolate data first")


	#### plotting functions for rga data ###

	def plot_rga(self,
				interpolated_data = True,
				save = False,
				fig_name = "rga",
				plotted_columns = None,
				zoom = [None, None, None, None],
				cursor_positions = [None],
				cursor_labels = [None]):
		"""Plot rga data
		if plotted_columns = None, it plots all the columns
		"""

		if interpolated_data==True:
			norm_df = self.rga_df_interpolated.copy()
		else:
			norm_df = self.rga_data.copy()

		# change to hours !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		norm_df = norm_df/3600

		plt.figure(figsize=(18, 13), dpi=80, facecolor='w', edgecolor='k')

		try:
			for element in plotted_columns:
				plt.plot(norm_df.time.values, norm_df[element].values,
					linewidth = 2, label = f"{element}", color = self.ammonia_reaction_colors[element])
		
		# if plotted_columns is None		
		except:
			for element in norm_df.columns[1:]:
				plt.plot(norm_df.time.values, norm_df[element].values,
					linewidth = 2, label = f"{element}", color = self.ammonia_reaction_colors[element])

		plt.semilogy()

		# cursor
		try:
			#mcolors.CSS4_COLORS["teal"]
			for cursor_pos, cursor_lab in zip(cursor_positions, cursor_labels):
				plt.axvline(cursor_pos,
					linestyle = "--", color = self.ammonia_conditions_colors[cursor_lab], linewidth = 1.5, label = cursor_lab)

		except (TypeError, KeyError):
			print("No cursor")

		finally:
			plt.xlim(zoom[0], zoom[1])
			plt.ylim(zoom[2], zoom[3])
			
			plt.title(f"Pressure for each element", fontsize=24)

			plt.xlabel('Time (h)',fontsize=16)
			plt.ylabel('Pressure (mBar)',fontsize=16)
			plt.legend(bbox_to_anchor=(0., -0.15, 1., .102), loc=3,
			       ncol=5, mode="expand", borderaxespad=0.)
			plt.grid(b=True, which='major', color='b', linestyle='-')
			plt.grid(b=True, which='minor', color=mcolors.CSS4_COLORS["teal"], linestyle='--')

			if save == True:
				plt.tight_layout()
				plt.savefig(f"{fig_name}.png")

				print(f"Saved as {fig_name} in png format.")

			plt.show()


	def plot_rga_norm_leak(self,
							interpolated_data = True,
							leak_values = [1],
							leak_positions = [None, None],
							save = False,
							fig_name = "rga_norm_leak",
							plotted_columns = None, zoom = [None, None, None, None], cursor_positions = [None], cursor_labels = [None]):
		"""
		Plot rga data normalized by the values given in leak_values on the intervals given by leak_positions
		e.g. leak_values = [1.3, 2] and leak_positions = [100, 200, 500] will result in a division by 1.3 between indices 100 and 200
		and by a division by 2 between indices 200 and 500.
		works on rga_df_interpolated
		if plotted_columns = None, it plots all the columns
		possible to use a zoom and vertical cursors (one or multiple) 
		"""
		if interpolated_data==True:
			norm_df = self.rga_df_interpolated.copy()
		else:
			norm_df = self.rga_data.copy()

		# change to hours !!!!!!
		norm_df = norm_df/3600

		if len(leak_values) + 1 == len(leak_positions):

			plt.figure(figsize = (18, 13), dpi = 80, facecolor = 'w', edgecolor = 'k')

			# for each column
			try:
				for element in plotted_columns:
					normalized_values = norm_df[element].values.copy()

					# normalize between the leak positions
					for j, value in enumerate(leak_values):

						# print(f"We normalize between {leak_positions[j]} s and {leak_positions[j+1]} s by {value}")

						normalized_values[leak_positions[j]:leak_positions[j+1]] = (normalized_values[leak_positions[j]:leak_positions[j+1]] / value)

					plt.plot(norm_df.time.values, normalized_values,
						linewidth = 2, label = f"{element}", color = self.ammonia_reaction_colors[element])
			
			# if plotted_columns is None
			except:
				for element in norm_df.columns[1:]:
					normalized_values = norm_df[element].values.copy()

					# normalize between the leak positions
					for j, value in enumerate(leak_values):

						# print(f"We normalize between {leak_positions[j]} s and {leak_positions[j+1]} s by {value}")

						normalized_values[leak_positions[j]:leak_positions[j+1]] = (normalized_values[leak_positions[j]:leak_positions[j+1]] / value)

					plt.plot(norm_df.time.values, normalized_values,
						linewidth = 2, label = f"{element}", color = self.ammonia_reaction_colors[element])
			
			plt.semilogy()

			# cursor
			try:
				#mcolors.CSS4_COLORS["teal"]
				for cursor_pos, cursor_lab in zip(cursor_positions, cursor_labels):
					plt.axvline(cursor_pos,
						linestyle = "--", color = self.ammonia_conditions_colors[cursor_lab], linewidth = 1.5, label = cursor_lab)

			except (TypeError, KeyError):
				print("No cursor")

			finally:
				plt.xlim(zoom[0], zoom[1])
				plt.ylim(zoom[2], zoom[3])

				plt.title(f"Normalized by pressure in rga (leak valve)", fontsize=24)

				plt.xlabel('Time (h)',fontsize=16)
				plt.ylabel('Pressure (mBar)',fontsize=16)
				plt.legend(bbox_to_anchor=(0., -0.15, 1., .102), loc=3,
				       ncol=5, mode="expand", borderaxespad=0.)
				plt.grid(b=True, which='major', color='b', linestyle='-')
				plt.grid(b=True, which='minor', color=mcolors.CSS4_COLORS["teal"], linestyle='--')

				if save == True:
					plt.tight_layout()
					plt.savefig(f"{fig_name}.png")

					print(f"Saved as {fig_name} in png format.")

				plt.show()

		else:
			print("Length of leak_positions should be one more than leak_values.")


	def plot_rga_norm_carrier(self,
							interpolated_data = True,
							save = False,
							legend = True,
							fig_name = "rga_norm_carrier",
							carrier_gaz = "Ar",
							plotted_columns = None,
							zoom = [None, None, None, None], 
							cursor_positions = [None], 
							cursor_labels = [None],
							vlines = [None],
							title = False):
		"""
		Plot rga data normalized by one column (carrier_gaz)
		works on rga_df_interpolated
		if plotted_columns = None, it plots all the columns
		possible to use a zoom and vertical cursors (one or multiple) 
		"""
		if interpolated_data==True:
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
			norm_df[element] = norm_df[element] / norm_df[carrier_gaz]

			plt.plot(norm_df.time.values, norm_df[element].values,
				linewidth = 2, label = f"{element}", color = self.ammonia_reaction_colors[element])

		self.norm_carrier_gas_df = norm_df
		print("Saved df as self.norm_carrier_gas_df.")
				
		# Add vlines to help find good values
		try:
			for i in vlines:
				plt.axvline(i, color = "r", linestyle= "--", linewidth = 1)
		except:
			pass

		# vertical span color to show conditions
		try:
			#mcolors.CSS4_COLORS["teal"]
			for x1, x2, l in zip(cursor_positions[:-1], cursor_positions[1:], cursor_labels):# range(len(cursor_positions)):#, cursor_lab in zip(cursor_positions, cursor_labels):
				plt.axvspan(x1, x2,
					alpha = 0.2,
					facecolor = self.ammonia_conditions_colors[l],
					# label = l
					)

				plt.text(
					x1 + (x2-x1)/2,
					y = 0.1,
					s = l,
					fontsize = 25
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

			plt.xlabel('Time (h)',fontsize = 35)
			plt.ylabel('Pressure (a.u.)',fontsize = 35)
			if legend:
				plt.legend(bbox_to_anchor=(1.15, 1), loc="upper right",
				       ncol=1, fontsize = 20, fancybox = True)

			plt.xticks(fontsize = 30)
			plt.yticks(fontsize = 30)
			plt.grid(b=True, which='major', color='b', linestyle='-')
			plt.grid(b=True, which='minor', color=mcolors.CSS4_COLORS["teal"], linestyle='--')

			if save == True:
				plt.tight_layout()
				plt.savefig(f"{fig_name}.png", bbox_inches='tight')

				print(f"Saved as {fig_name} in png format.")

			plt.show()


	def plot_rga_norm_ptot(self,
							interpolated_data = True,
							save = False,
							legend = True,
							fig_name = "rga_norm_ptot",
							plotted_columns = None,
							ptot = None,
							zoom = [None, None, None, None], cursor_positions = [None], cursor_labels = [None], title = False):
		"""
		Plot rga data normalized by the total pressure in the cell
		works on rga_df_interpolated
		if plotted_columns = None, it plots all the columns
		possible to use a zoom and vertical cursors (one or multiple) 
		"""

		if interpolated_data==True:
			norm_df = self.rga_df_interpolated.copy()

		else:
			norm_df = self.rga_data.copy()

		used_arr = norm_df.values
		ptot_row = used_arr[:,1:].sum(axis = 1) / ptot

		val = np.ones(used_arr.shape)
		val[:,1:] = (val[:,1:].T * ptot_row).T

		self.norm_df_ptot = pd.DataFrame(used_arr / val, columns = norm_df.columns)
		
		# change to hours !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
		self.norm_df_ptot.time = self.norm_df_ptot.time.values/3600

		print("Created a data frame normalized by total pressure, self.norm_df_ptot")
		display(self.norm_df_ptot.head())

		plt.figure(figsize=(17, 10), dpi=150, facecolor='w', edgecolor='k')
		plt.semilogy()

		try:
			for element in plotted_columns:
				plt.plot(self.norm_df_ptot.time.values, self.norm_df_ptot[element].values,
					linewidth = 2, label = f"{element}", color = self.ammonia_reaction_colors[element])

		# if plotted_columns is None
		except Exception as e:
			for element in self.norm_df_ptot.columns[1:]:
				plt.plot(self.norm_df_ptot.time.values,
					self.norm_df_ptot[element].values,
					linewidth = 2,
					label = f"{element}",
					color = self.ammonia_reaction_colors[element])


		# vertical span color to show conditions
		try:
			#mcolors.CSS4_COLORS["teal"]
			for x1, x2, l in zip(cursor_positions[:-1], cursor_positions[1:], cursor_labels):# range(len(cursor_positions)):#, cursor_lab in zip(cursor_positions, cursor_labels):
				plt.axvspan(x1, x2,
					alpha = 0.2,
					facecolor = self.ammonia_conditions_colors[l],
					# label = l
					)

				plt.text(
					x1 + (x2-x1)/2,
					y = 0.1,
					s = l,
					fontsize = 25
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

			plt.xlabel('Time (h)',fontsize = 35)
			plt.ylabel('Pressure (a.u.)',fontsize = 35)
			if legend:
				plt.legend(bbox_to_anchor=(1.15, 1), loc="upper right",
				       ncol=1, fontsize = 20, fancybox = True)

			plt.xticks(fontsize = 30)
			plt.yticks(fontsize = 30)
			plt.grid(b=True, which='major', color='b', linestyle='-')
			plt.grid(b=True, which='minor', color=mcolors.CSS4_COLORS["teal"], linestyle='--')

			if save == True:
				plt.tight_layout()
				plt.savefig(f"{fig_name}.png", bbox_inches='tight')

				print(f"Saved as {fig_name} in png format.")

			plt.show()


	def plot_rga_norm_temp(self, 
							start_heating, 
							nb_points, 
							amp_final, save = False, 
							fig_name = "rga_norm_temp", 
							interpolated_data = True, 
							delta_time = 10, 
							bins_nb = 1000, binning = False, 
							plotted_columns = None, 
							zoom = [None, None, None, None], 
							cursor_positions = [None], 
							cursor_labels = [None]):
		"""
		Plot rga data normalized by the values given in intensity_values on the intervals given by leak_positions
		e.g. intensity_values = [1.3, 2] and leak_positions = [100, 200, 500] will result in a division by 1.3 between indices 100 and 200
		and by a division by 2 between indices 200 and 500.
		works on rga_df_interpolated
		if plotted_columns = None, it plots all the columns
		possible to use a zoom and vertical cursors (one or multiple) 
		"""
		if interpolated_data==True:
			norm_df = self.rga_df_interpolated.copy()
		else:
			norm_df = self.rga_data.copy()

		# change to hours !!!!!!
		norm_df = norm_df/3600

		# recreate points that were used during data acquisition
		# amperage = np.round(np.concatenate((np.linspace(0, amp_final, nb_points*delta_time, endpoint=False), np.linspace(amp_final, 0, nb_points*delta_time))), 2)
		amperage = np.concatenate((np.linspace(0, amp_final, nb_points*delta_time, endpoint=False), np.linspace(amp_final, 0, nb_points*delta_time)))

		# recreate temperature from heater benchmarks
		temperature = self.heater_poly_fit(amperage)
		temperature_real = np.array([t if t >25 else 25 for t in temperature])

		# find end time
		end_heating = start_heating + (2 * nb_points) * delta_time

		# timerange of heating
		time_column = norm_df["time"].values[start_heating:end_heating]

		# save in df
		self.rga_during_heating = pd.DataFrame({
			"time" : time_column,
			"amperage" : amperage,
			"temperature" : temperature_real,
			})

		print("Results saved in self.rga_during_heating DataFrame")

		plt.figure(figsize = (18, 9))

		# for each column
		try:
			max_plot = 0
			for element in plotted_columns:
				data_column = norm_df[element].values[start_heating:end_heating]
				self.rga_during_heating[element] = data_column

				if not binning:
					plt.plot(temperature_real, data_column,
						linewidth = 2, linestyle = "dashdot", label = f"{element}", color = self.ammonia_reaction_colors[element])
					if max(data_column[element]) > max_plot:
						max_plot = max(data_column[element])
		
		# if plotted_columns is None
		except:
			max_plot = 0
			for element in norm_df.columns[1:]:
				data_column = norm_df[element].values[start_heating:end_heating]
				self.rga_during_heating[element] = data_column

				if not binning:
					plt.plot(self.rga_during_heating.temperature,
						self.rga_during_heating[element],
						linewidth = 2,
						linestyle = "dashdot",
						label = f"{element}", 
						color = self.ammonia_reaction_colors[element])
					# plt.plot(temperature_real, data_column, linewidth = 2, label = f"{element}")
					if max(self.rga_during_heating[element]) > max_plot:
						max_plot = max(self.rga_during_heating[element])

		# Bin after so that the dataframe on heating timerange already exists
		if binning:

			# Bin the data frame by "time" with bins_nb bins...
			bins = np.linspace(self.rga_during_heating.time.min(), self.rga_during_heating.time.max(), bins_nb)
			groups = self.rga_during_heating.groupby(np.digitize(self.rga_during_heating.time, bins))

			# Get the mean of each bin:
			self.binned_data = groups.mean()
			max_plot = 0
			for element in plotted_columns:
				plt.plot(self.binned_data.temperature,
					self.binned_data[element],
					linewidth = 2,
					#linestyle = "dashdot",
					label = f"{element}",
					color = self.ammonia_reaction_colors[element])
				if max(self.binned_data[element]) > max_plot:
					max_plot = max(self.binned_data[element])
		plt.semilogy()

		# cursor
		try:
			#mcolors.CSS4_COLORS["teal"]
			for cursor_pos, cursor_lab in zip(cursor_positions, cursor_labels):
				plt.axvline(cursor_pos, linestyle = "--", color = self.ammonia_conditions_colors[cursor_lab], linewidth = 1.5, label = cursor_lab)

		except (TypeError, KeyError):
			print("No cursor")

		finally:
			plt.xlim(zoom[0], zoom[1])
			plt.ylim(zoom[2], zoom[3])
			plt.text(x = 0, y = 1.4*max_plot, s = "Ammonia oxidation on Pt nanoparticles",
               fontsize = 26, weight = 'bold', alpha = .75)
			if not binning:
				plt.text(x = 0, y = 1.25*max_plot,
					s = """Pressure as a function of the temperature (from input current)""",
					fontsize = 19, alpha = .85)
			else:
				plt.text(x = 0, y = 1.25*max_plot,
					s = f"Pressure as a function of the temperature (from input current), with {bins_nb} bins.",
					fontsize = 19, alpha = .85)

			# Ticks

			plt.xticks(np.arange(0, 700, 50), fontsize=20)
			#plt.yticks(np.arange(5, 22.5, 1.5), fontsize=20)

			plt.xlabel('Temperature (°C)',fontsize = 25)
			plt.ylabel('Pressure (mBar)', fontsize = 25)
			plt.legend(bbox_to_anchor=(0., -0.15, 1., .102), loc=3,
			       ncol=5, mode="expand", borderaxespad=0., fontsize = 25)
			plt.grid(b=True, which='major', color='b', linestyle='-')
			plt.grid(b=True, which='minor', color=mcolors.CSS4_COLORS["teal"], linestyle='--')

			if save == True:
				plt.tight_layout()
				plt.savefig(f"{fig_name}.png", bbox_inches='tight')

				print(f"Saved as {fig_name} in png format.")
			plt.show()


	def fit_error_function(self, 
							initial_guess, 
							new_amper_vect, 
							interpolated_data = True, 
							fitted_columns = None, 
							binning = False, 
							zoom = [None, None, None, None], 
							cursor_positions = [None], 
							cursor_labels = [None]):
		"""fit pressure vs temperature dta with error function"""

		def error_function(z, a, b, c, d):
		    return a * (1 + special.erf((z - b) / c) ) + d

		longer_temp_vect = self.heater_poly_fit(new_amper_vect)

		if not binning:
			used_df = self.rga_during_heating.copy()
			xdata = used_df.temperature
		else:
			used_df = self.binned_data.copy()
			xdata = used_df.temperature

		fig, axes = plt.subplots(1, 1, figsize = (18, 13), dpi = 80, facecolor = 'w', edgecolor = 'k')

		try:
			for element in fitted_columns:
				ydata = used_df[element]
				popt, pcov = curve_fit(error_function, xdata, ydata, p0 = initial_guess)
				# axes.plot(xdata, func(xdata, *guessErf))
				axes.plot(xdata, ydata, linewidth = 2, linestyle = "dashdot", label = f"{element}")
				#axes.plot(xdata, func(xdata, *popt))
				axes.plot(longer_temp_vect, error_function(longer_temp_vect, *popt), linewidth = 2, linestyle = "dashdot", label = f"{element}")

		except:
			for element in used_df.columns[1:]:
				ydata = used_df[element]
				popt, pcov = curve_fit(error_function, xdata, ydata, p0 = initial_guess)
				# axes.plot(xdata, func(xdata, *guessErf))
				axes.plot(xdata, ydata, linewidth = 2, linestyle = "dashdot", label = f"{element}")
				#axes.plot(xdata, func(xdata, *popt))
				axes.plot(longer_temp_vect, error_function(longer_temp_vect, *popt), linewidth = 2, linestyle = "dashdot", label = f"{element}")

		# cursor
		try:
			#mcolors.CSS4_COLORS["teal"]
			for cursor_pos in cursor_positions:
				axes.axvline(x = cursor_pos, linestyle = "--", color = "#bb1e10")
		except TypeError:
			print("No cursor")

		finally:
			axes.set_xlim(zoom[0], zoom[1])
			axes.set_ylim(zoom[2], zoom[3])

			axes.set_xlabel('Temperature (°C)',fontsize=16)
			axes.set_ylabel('Pressure (mBar)',fontsize=16)
			axes.legend(bbox_to_anchor=(0., -0.1, 1., .102), loc=3,
			       ncol=5, mode="expand", borderaxespad=0.)
			axes.grid(b=True, which='major', color='b', linestyle='-')
			axes.grid(b=True, which='minor', color=mcolors.CSS4_COLORS["teal"], linestyle='--')

			plt.title("Temperature vs pressure graphs fitted with error functions")
			plt.show()