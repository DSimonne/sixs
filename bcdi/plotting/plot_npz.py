import numpy as np
import tables as tb
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import ipywidgets as widgets
from ipywidgets import interact, Button, Layout, interactive, fixed
from IPython.display import display, Markdown, Latex, clear_output

import glob

p = input("Path to file from current folder: ")

try:
	rawdata = np.load(p)

	print(f"Stored arrays :{rawdata.files}")

except Exception as E:
	raise NameError("Wrong path")


def Plotting(file, axplot, datapath):
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
			options = [("2D plot", "2D"), ("2D contour plot", "2DC"),("3D surface plot", "3D")],
		    value = "2D",
		    description='Plotting options',
		    disabled=False,
		    button_style='', # 'success', 'info', 'warning', 'danger' or ''
		    tooltip=['Plot only contour or not', "", ""],
		    #icon='check'
		    ))
	def PickLastAxis(i, PlottingOptions):
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

		# print(f"Current index: {i}")
		# print(np.mean(dt))

		# Edit overall plot parameters# Font parameters
		# mpl.rcParams['font.family'] = 'Verdana'
		mpl.rcParams['font.size'] = 18

		# Edit axes parameters
		mpl.rcParams['axes.linewidth'] = 2

		# Tick properties
		mpl.rcParams['xtick.major.size'] = 10
		mpl.rcParams['xtick.major.width'] = 2
		mpl.rcParams['xtick.direction'] = 'out'
		mpl.rcParams['ytick.major.size'] = 10
		mpl.rcParams['ytick.major.width'] = 2
		mpl.rcParams['ytick.direction'] = 'out'

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
			fig, ax = plt.subplots(figsize = (15,15))
			img = ax.imshow(dt,
						origin='lower',
						cmap='YlGnBu_r',
						#extent=(0, 2, 0, 2),
						vmin = dmin,
						vmax = dmax)

			# # Create scale bar (only if we know the  image size)
			# ax.fill_between(x=[1.4, 1.9], y1=[0.1, 0.1], y2=[0.2, 0.2], color='white')

			# ax.text(x=1.65, y=0.25, s='500 nm', va='bottom', ha='center', color='white', size=20)

			# Create axis for colorbar
			cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='5%', pad=0.1)

			# Create colorbar
			cbar = fig.colorbar(mappable=img, cax=cbar_ax)

			# Edit colorbar ticks and labels
			ticks = [dmin + n * (dmax-dmin)/10 for n in range(0, 11)]
			tickslabel = [f"{t}" for t in ticks]

			cbar.set_ticks(ticks)
			cbar.set_ticklabels(tickslabel)


		elif PlottingOptions == "2DC" :
			plt.close()
			# Show contour plot instead
			try:
				fig, ax = plt.subplots(figsize = (15,15))
				ticks = [dmin + n * (dmax-dmin)/10 for n in range(0, 11)]

				img = ax.contour(dt,
								ticks,
							#extent=(0, 2, 0, 2),
							cmap='YlGnBu_r',
							vmin=dmin,
							vmax=dmax)

				# Create axis for colorbar
				cbar_ax = make_axes_locatable(ax).append_axes(position='right', size='5%', pad=0.1)

				# Create colorbar
				cbar = fig.colorbar(mappable=img, cax=cbar_ax)

				# Edit colorbar ticks and labels
				tickslabel = [f"{t}" for t in ticks]

				cbar.set_ticks(ticks)
				cbar.set_ticklabels(tickslabel)
				plt.show()

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


######################################################## END OF FUNCTION ##########################

WidgetPlotting = interactive(Plotting,
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
					datapath =fixed(p))

display(WidgetPlotting)