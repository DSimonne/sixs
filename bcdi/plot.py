import numpy as np
import tables as tb
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive, fixed
from IPython.display import display, Markdown, Latex, clear_output

import glob
import cmath

import warnings
warnings.filterwarnings("ignore")

# # Edit overall plot parameters
# # Font parameters
# # mpl.rcParams['font.family'] = 'Verdana'
# mpl.rcParams['font.size'] = 18

# # Edit axes parameters
# mpl.rcParams['axes.linewidth'] = 2

# # Tick properties
# mpl.rcParams['xtick.major.size'] = 10
# mpl.rcParams['xtick.major.width'] = 2
# mpl.rcParams['xtick.direction'] = 'out'
# mpl.rcParams['ytick.major.size'] = 10
# mpl.rcParams['ytick.major.width'] = 2
# mpl.rcParams['ytick.direction'] = 'out'


def plotting_npz(file, axplot, datapath):
	"""For .npz files"""

	rawdata = np.load(datapath)

	# Pick an array
	try:
	    data = rawdata[file]

	except Exception as E:
	    raise E

	# Take the shape of that array along 2 axis
	if axplot == "xy":
	    print(f"The shape of this projection is {np.shape(data[:, :, 0])}")

	    r = np.shape(data[0, 0, :])
	    print(f"The range in the last axis is {r[0]}")


	elif axplot == "yz":
	    print(f"The shape of this projection is {np.shape(data[0, :, :])}")

	    r = np.shape(data[:, 0, 0])
	    print(f"The range in the last axis is {r[0]}")

	elif axplot == "xz":
	    print(f"The shape of this projection is {np.shape(data[:, 0, :])}")

	    r = np.shape(data[0, :, 0])
	    print(f"The range in the last axis is {r[0]}")


	@interact(
		i = widgets.IntSlider(
		    min=0,
		    max=r[0]-1,
		    step=1,
		    description='Index along last axis:',
		    disabled=False,
		    orientation='horizontal',
		    continuous_update=False,
		    readout=True,
		    readout_format='d',
		    # style = {'description_width': 'initial'}
            ),
		PlottingOptions = widgets.ToggleButtons(
			options = [("2D plot", "2D"),
					("2D contour plot", "2DC"),
					# ("3D surface plot", "3D")
					],
		    value = "2D",
		    description='Plotting options',
		    disabled=False,
		    button_style='', # 'success', 'info', 'warning', 'danger' or ''
		    tooltip=['Plot only contour or not', "", ""],
		    #icon='check'
		    ),
		scale = widgets.Dropdown(
		    options = ["linear", "logarithmic"],
		    value = "linear",
		    description = 'Scale',
		    disabled=False,
		    style = {'description_width': 'initial'}),
	)
	def PickLastAxis(i, PlottingOptions, scale):
		if axplot == "xy":
		    dt = data[:, :, i]
		elif axplot == "yz":
		    dt = data[i, :, :]
		elif axplot == "xz":
		    dt = data[:, i, :]
		    
		else:
		    raise TypeError("Choose xy, yz or xz as axplot.")

		dmax = dt.max()
		dmin = dt.min()

		# Show image
		if PlottingOptions == "2D":
			plt.close()
			fig, ax = plt.subplots(figsize = (15, 15))
			img = ax.imshow(dt,
						norm= {"linear" : None, "logarithmic" : LogNorm()}[scale],
						cmap="cividis",
						)

			divider = make_axes_locatable(ax)
			cax = divider.append_axes('right', size='5%', pad=0.05)

			fig.colorbar(img, cax=cax, orientation='vertical')

		elif PlottingOptions == "2DC" :
			plt.close()
			# Show contour plot instead
			try:
				fig, ax = plt.subplots(figsize = (15,15))
				ticks = [dmin + n * (dmax-dmin)/10 for n in range(0, 11)] if scale == "linear" else [pow(10, x) for x in range (0, len(str(dmax)))]

				img = ax.contour(dt,
								ticks,
								norm= {"linear" : None, "logarithmic" : LogNorm()}[scale],
								cmap='cividis')

				divider = make_axes_locatable(ax)
				cax = divider.append_axes('right', size='5%', pad=0.05)

				fig.colorbar(img, cax=cax, orientation='vertical')

			except IndexError:
				plt.close()
				print("No contour levels were found within the data range. Meaning there is very little variation in the dat, change index")

		elif PlottingOptions == "3D" :
			plt.close()

			# Create figure and add axis
			fig = plt.figure(figsize=(15,15))
			ax = plt.subplot(111, projection='3d')

			# Create meshgrid

			X, Y = np.meshgrid(np.arange(0, dt.shape[0], 1), np.arange(0, dt.shape[1], 1))

			plot = ax.plot_surface(X=X, Y=Y, Z=dt, cmap='YlGnBu_r', vmin=dmin, vmax=dmax)

			# Adjust plot view
			ax.view_init(elev=50, azim=225)
			ax.dist=11

			# Add colorbar
			cbar = fig.colorbar(plot, ax=ax, shrink=0.6)

			# Edit colorbar ticks and labels
			ticks = [dmin + n * (dmax-dmin)/10 for n in range(0, 11)]
			tickslabel = [f"{t}" for t in ticks]

			cbar.set_ticks(ticks)
			cbar.set_ticklabels(tickslabel)

			# Set tick marks
			# ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
			# ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(25))

			# Set axis labels
			# ax.set_xlabel(r'$\mathregular{\mu}$m', labelpad=20)
			# ax.set_ylabel(r'$\mathregular{\mu}$m', labelpad=20)

			# Set z-limit
			# ax.set_zlim(0, 50);


def plotting_cxi(axplot, datapath, ComplexNumber):
	"""Interactive function to plot the cxi files, only open in read mode"""

	# Open the file
	try:
		with tb.open_file(datapath, "r") as f:
			# Since .cxi files follow a specific architectture, we know where our data is.
			data = f.root.entry_1.data_1.data[:]

	except Exception as E:
		raise NameError("Wrong path")

	# Decide what we want to plot
	if ComplexNumber == "Real":
		PlottedArrayType = np.real(data)
	elif ComplexNumber == "Imaginary":
		PlottedArrayType = np.imag(data)
	elif ComplexNumber == "Module":
		PlottedArrayType = np.abs(data)
	else:
		PlottedArrayType = np.angle(data)


	# Print the shape of that array along 2 axis, use the last dimension for plotting and Project along two axes
	if axplot == "xy":
		print(f"The shape of this projection is {np.shape(data[:, :, 0])}")

		r = np.shape(data[0, 0, :])
		print(f"Length of last axis: {r[0]}")

	elif axplot == "yz":
		print(f"The shape of this projection is {np.shape(data[0, :, :])}")

		r = np.shape(data[:, 0, 0])
		print(f"Length of last axis: {r[0]}")

	else:
		print(f"The shape of this projection is {np.shape(data[:, 0, :])}")

		r = np.shape(data[0, :, 0])
		print(f"Length of last axis: {r[0]}")


	@interact(
		i = widgets.IntSlider(
		    min=0,
		    max=r[0]-1,
		    step=1,
		    description='Index along last axis:',
		    disabled=False,
		    orientation='horizontal',
		    continuous_update=False,
		    readout=True,
		    readout_format='d',
		    style = {'description_width': 'initial'}),
		PlottingOptions = widgets.ToggleButtons(
			options = [("2D plot", "2D"),
						("2D contour plot", "2DC"),
						# ("3D surface plot", "3D")
						],
		    value = "2D",
		    description='Plotting options',
		    disabled=False,
		    button_style='', # 'success', 'info', 'warning', 'danger' or ''
		    tooltip=['Plot only contour or not', "", ""],
		    #icon='check'
		    ),
		scale = widgets.Dropdown(
		    options = ["linear", "logarithmic"],
		    value = "linear",
		    description = 'Scale',
		    disabled=False,
		    style = {'description_width': 'initial'}),
	)
	def PickLastAxis(i, PlottingOptions, scale):
		# Create a new figure
		plt.close()

		# Print the shape of that array along 2 axis, use the last dimension for plotting and Project along two axes
		if axplot == "xy":
			TwoDPlottedArray = PlottedArrayType[:, :, i]

		elif axplot == "yz":
			TwoDPlottedArray = PlottedArrayType[i, :, :]

		else:
			TwoDPlottedArray = PlottedArrayType[:, i, :]

		# Find max and min
		dmax = TwoDPlottedArray.max()
		dmin = TwoDPlottedArray.min()

		# Create figure and add axis
		fig = plt.figure(figsize=(20,10))
		ax = fig.add_subplot(111)

		# Remove x and y ticks
		ax.xaxis.set_tick_params(size=0)
		ax.yaxis.set_tick_params(size=0)
		ax.set_xticks([])
		ax.set_yticks([])

		# Show image
		if PlottingOptions == "2D":
			plt.close()
			fig, ax = plt.subplots(figsize = (15, 15))
			img = ax.imshow(TwoDPlottedArray,
						norm= {"linear" : None, "logarithmic" : LogNorm()}[scale],
						cmap="cividis",
						)

			divider = make_axes_locatable(ax)
			cax = divider.append_axes('right', size='5%', pad=0.05)

			fig.colorbar(img, cax=cax, orientation='vertical')

		elif PlottingOptions == "2DC" :
			plt.close()
			# Show contour plot instead
			try:
				fig, ax = plt.subplots(figsize = (15,15))
				ticks = [dmin + n * (dmax-dmin)/10 for n in range(0, 11)] if scale == "linear" else [pow(10, x) for x in range (0, len(str(dmax)))]

				img = ax.contour(TwoDPlottedArray,
								ticks,
								norm= {"linear" : None, "logarithmic" : LogNorm()}[scale],
								cmap='cividis')

				divider = make_axes_locatable(ax)
				cax = divider.append_axes('right', size='5%', pad=0.05)

				fig.colorbar(img, cax=cax, orientation='vertical')

			except IndexError:
				plt.close()
				print("No contour levels were found within the data range. Meaning there is very little variation in the dat, change index")

		elif PlottingOptions == "3D" :
			plt.close()

			# Create figure and add axis
			fig = plt.figure(figsize=(15,15))
			ax = plt.subplot(111, projection='3d')

			# Create meshgrid
			# print(TwoDPlottedArray.shape)
			# print(type(TwoDPlottedArray))
			# CAREFUL ALL SHAPE MUST BE THE SAME
			X, Y = np.meshgrid(np.arange(0, TwoDPlottedArray.shape[1], 1), np.arange(0, TwoDPlottedArray.shape[0], 1))
			# print(X.shape)
			# print(type(X))
			# print(Y.shape)
			# print(type(Y))


			plot = ax.plot_surface(X=X, Y=Y, Z=TwoDPlottedArray, cmap='YlGnBu_r', vmin=dmin, vmax=dmax)

			# Adjust plot view
			ax.view_init(elev=50, azim=225)
			ax.dist = 7

			# Add colorbar
			cbar = fig.colorbar(plot, ax=ax, shrink=0.6)

			# Edit colorbar ticks and labels
			ticks = [dmin + n * (dmax-dmin)/5 for n in range(0, 6)]
			tickslabel = [f"{t}" for t in ticks]

			cbar.set_ticks(ticks)
			cbar.set_ticklabels(tickslabel)

			# Set tick marks
			# ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(25))
			# ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(25))

			# Set axis labels
			# ax.set_xlabel(r'$\mathregular{\mu}$m', labelpad=20)
			# ax.set_ylabel(r'$\mathregular{\mu}$m', labelpad=20)

			# Set z-limit
			# ax.set_zlim(0, 50);


def plot(filename):
	# Path of file to be imported

	if filename.endswith(".cxi"):
		# Prints architecture of .cxi file
		try:
			# with tb.open_file(filename, "r") as f:
				# print("Data file architecture :\n")
				# print(f)

			WidgetPlottingCXI = interactive(plotting_cxi,
								axplot = widgets.Dropdown(
								    options = ["xy", "yz", "xz"],
								    value = "yz",
								    description = 'First 2 axes:',
								    disabled=False,
								    style = {'description_width': 'initial'}),
								ComplexNumber = widgets.ToggleButtons(
									options = ["Real", "Imaginary", "Module", "Phase"],
								    value = "Module",
								    description='Plotting options',
								    disabled=False,
								    button_style='', # 'success', 'info', 'warning', 'danger' or ''
								    tooltip=['Plot only contour or not', "", ""]),
								datapath =fixed(filename))

			display(WidgetPlottingCXI)

		except Exception as E:
			raise NameError("Wrong path")

	elif filename.endswith(".npz"):

		try:
			rawdata = np.load(filename)

			print(f"Stored arrays :{rawdata.files}")

			WidgetPlottingNPZ = interactive(plotting_npz,
								file = widgets.Dropdown(
								    options = rawdata.files,
								    value = rawdata.files[0],
								    description = 'Array used:',
								    disabled=False,
								    style = {'description_width': 'initial'}),
								axplot = widgets.Dropdown(
								    options = ["xy", "yz", "xz"],
								    value = "xy",
								    description = 'First 2 axes:',
								    disabled=False,
								    style = {'description_width': 'initial'}),
								datapath =fixed(filename))

			display(WidgetPlottingNPZ)


		except Exception as E:
			raise NameError("Wrong path")