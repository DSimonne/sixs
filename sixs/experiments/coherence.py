import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm

import tables as tb
import numpy as np
import glob
import os
import h5py

from scipy import interpolate
from lmfit.models import GaussianModel
try:
    plt.switch_backend('Qt5Agg')
except Exception as E:
    pass
from sixs import ReadNxs4 as rn4


def find_focal_plane(
    file_list,
    scan_type,
    directory="./",
    omega=0,
    pos_hor_scans=None,
    ROI=(None, None, None, None),
    verbose=False,
    amplitude=0.005,
    center=0,
    sigma=0.002,
    baseline=0,
    scale='log',
):
    """
    Find the focal plane of the lenses by plotting the evolution 
    of the beam FWHM in the vertical position.

    It uses the x and y position of the goniometer, that should be attributes
    of rn.DataSet

    A gaussian curve is fitted to retrieve the FWHM.

    TODO: add derivative

    :param file_list: path to alignment files
    :param scan_type: 'basez' or 'gonio'
    :param pos_hor_scans: force horizontal position, necessary if scan_type == 'basez'
     container of length equal to length of file_list, for plotting
    :param omega: tilt angle between the vertical axis and x, in degress, must be constant
     , necessary if scan_type == 'gonio'
    :param ROI: container of length 4, determines the region of interest on the detector
     e.g. = (200, 650, 400, 600), 
    :param directory: e.g. = "./"
    :param verbose: e.g. False
    :param amplitude: init parameter for gaussian fit, e.g. 0.005
    :param center: init parameter for gaussian fit, e.g. = 0.36
    :param sigma: init parameter for gaussian fit, e.g. 0.002
    :param baseline: constant value to subtract to the data before fitting, e.g. = 14.75
    :param scale: log scale is applied if == 'log'
    """

    plt.figure(figsize=(15, 9))
    fwhm_scans = []

    if scan_type == 'basez':
        if len(pos_hor_scans) == len(file_list):
            print(
                "Using `pos_hor_scans` argument for horizontal positions."
                "\nUsing `basez attribute in files for vertical positions."
            )
            pos_hor_scans = pos_hor_scans
        else:
            return ("Provide a list of horizontal positions")

    elif scan_type == 'gonio':
        pos_hor_scans = []
        print(
            "Using `omega` argument to determine horizontal and vertical positions"
            "\n together with x and y attributes in files."
        )

    else:
        return ("`scan_type` argument must be 'gonio' or 'basez'")

    # Get matplotlib colors
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = colors[:len(file_list)]  # TODO IF TOO LONG

    for j, (file, color) in enumerate(zip(file_list, colors)):
        # Read data
        scan = rn4.DataSet(file, directory, verbose=verbose)

        # Find vertical and horixontal position from x, y, and omega
        if scan_type == 'basez':
            pos_ver = scan.basez
            pos_hor = pos_hor_scans[j]

        elif scan_type == 'gonio':
            pos_ver = np.cos(np.deg2rad(omega))*scan.x + \
                np.sin(np.deg2rad(omega))*scan.y
            pos_hor = np.round(
                np.mean(np.cos(np.deg2rad(omega))*scan.y - np.sin(np.deg2rad(omega))*scan.x), 0)
            pos_hor_scans.append(pos_hor)

        # Sum data in ROI
        sum_over_ROI = scan.cam2[:, ROI[0]:ROI[1],
                                 ROI[2]:ROI[3]].sum(axis=(1, 2))

        if scale == 'log':
            sum_over_ROI = np.log(sum_over_ROI)

        # Fit data
        gm = GaussianModel()
        result = gm.fit(
            data=sum_over_ROI-baseline,
            x=pos_ver,
            amplitude=amplitude,
            center=center,
            sigma=0.002
        )
        fwhm_scans.append(result.params["fwhm"].value)

        # Plot each scan
        plt.plot(
            pos_ver,
            sum_over_ROI-baseline,
            label=file + "_" + str(pos_hor),
            color=color,
        )

        plt.plot(
            pos_ver,
            result.best_fit,
            "--",
            color=color,
        )

    # Plot FWHM evolution
    plt.xlabel("Vertical position")
    plt.ylabel("Intensity")
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()

    plt.scatter(pos_hor_scans, fwhm_scans, color=colors)
    plt.xlabel("Horizontal position")
    plt.ylabel("FWHM")
    plt.show()


def show_map(
    file_name_list,
    directory="./",
    map_type="hexapod_scan",
    load_with_readnxs=False,
    scale="log",
    verbose=False,
    roi=None,
    x=None,
    y=None,
):
    """
    Quick solution to extract maps from a list of files

    The path to the X, Y and roi array are fixed

    Many bog possibilities still
    """
    # Check parameters
    if map_type not in ("hexapod_scan", "ascan_y"):
        return ("`map_type` parameter must be 'hexapod_scan' or 'ascan_y'")

    if scale not in ("log", "lin"):
        return ("`scale` parameter must be 'lin' or 'log'")

    if len(file_name_list) == 0:
        return ("`file_name_list` parameter must be of length at least 1")

    if not isinstance(verbose, bool):
        return ("`verbose` parameter must be a Boolean")

    if not isinstance(roi, str):
        if map_type == "hexapod_scan":
            roi = "roi4_merlin"
            print("Defaulted to roi4_merlin")
        elif map_type == "ascan_y":
            roi = "data_30"
            print("Defaulted to data_30")
    else:
        print("Using for roi:", roi)

    if not isinstance(x, str):
        if map_type == "hexapod_scan":
            x = "X"
            print("Defaulted to X")
        elif map_type == "ascan_y":
            x = "data_41"
            print("Defaulted to data_41")
    else:
        print("Using for x:", x)

    if not isinstance(y, str):
        if map_type == "hexapod_scan":
            y = "Y"
            print("Defaulted to Y")
        elif map_type == "ascan_y":
            y = "data_42"
            print("Defaulted to data_42")
    else:
        print("Using for y:", y)

    # Save file range index
    first_scan = file_name_list[0].split(".nxs")[0][-5:]
    last_scan = file_name_list[-1].split(".nxs")[0][-5:]

    # Get data
    X_lists, Y_lists, roi_sum_lists = [], [], []
    if load_with_readnxs:
        # Load datasets
        for file in file_name_list:
            dataset = rn4.DataSet(file, directory, verbose=verbose)
            if map_type == "hexapod_scan":
                X_lists.append(dataset.X)
                Y_lists.append(dataset.Y)
                roi_sum_lists.append(dataset.roi4_merlin)
            elif map_type == "ascan_y":
                X_lists.append(dataset.x)
                Y_lists.append(dataset.y)
                roi_sum_lists.append(dataset.roi4_merlin)

    else:
        # Load with tables
        for file in file_name_list:
            with h5py.File(directory + file) as f:
                X = (f["com"]["scan_data"][x][...])
                Y = (f["com"]["scan_data"][y][...])
                roi_sum = (f["com"]["scan_data"][roi][...])

            # Append to lists
            X_lists.append(X)
            Y_lists.append(Y)
            roi_sum_lists.append(roi_sum)
            if verbose:
                print(
                    f"Loading :{file}"
                    f"\n\tShape in X: {X.shape}"
                    f"\n\tShape in Y: {Y.shape}"
                    f"\n\tShape in roi_sum: {roi_sum.shape}\n"
                )

    # Create empty arrays
    arr_roi = np.zeros((len(file_name_list), len(Y_lists[0])))
    arr_y = np.zeros((len(file_name_list), len(Y_lists[0])))
    arr_x = np.zeros((len(file_name_list), len(Y_lists[0])))

    # Fill arrays
    for j, (X, Y, roi_sum) in enumerate(zip(X_lists, Y_lists, roi_sum_lists)):
        # Y is constant, X changes
        if verbose:
            print(f"Scan nb {j} for y=", np.round(np.mean(Y), 3))
        arr_roi[j] = roi_sum
        arr_y[j] = np.around(Y, 3)
        arr_x[j] = np.around(X, 3)

    if map_type == "ascan_y":
        # Flip because the scans go with increasing y every other scan
        # and otherwise decreasing y
        for i in range(len(arr_y)):
            if i % 2 == 1:
                arr_y[i] = np.flip(arr_y[i])
                arr_roi[i] = np.flip(arr_roi[i])

    # Apply scale
    plotted_array = arr_roi if scale == 'lin' else np.log10(arr_roi)

    # Plot data
    title = f"Scans {first_scan} ----> {last_scan}"
    plot_mesh(arr_x, arr_y, plotted_array, title)

    return ("Data successfully plotted.")


def plot_mesh(arr_x, arr_y, plotted_array, title=None):
    """
    """
    # Image
    plt.figure(figsize=(8, 8))
    plt.pcolormesh(
        arr_x,
        arr_y,
        plotted_array,
        cmap='jet'
    )
    plt.axis('square')
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')
    if isinstance(title, str):
        plt.title(title)
    plt.grid()
    plt.show()


def plot_2d_scan(
    file,
    colormap='jet',
    roi="roi4_merlin",
    x="X",
    y="Y",
):
    """
    Plot evolution of roi as a function of x and y

    Uses 
        - x = f.root.com.scan_data.X[...]
        - y = f.root.com.scan_data.Y[...]
        - roi_sum = f.root.com.scan_data.roi4_merlin[...]

    return: Error or success message
    """
    # Check arguments
    try:
        colormap = getattr(cm, colormap)
    except AttributeError:
        return ("Possible colormaps are 'viridis', 'jet', ...")

    # Load data
    try:
        with h5py.File(file) as f:
            x = f["com"]["scan_data"][x][...]
            y = f["com"]["scan_data"][y][...]
            roi_sum = f["com"]["scan_data"][roi][...]
    except Exception as E:
        return ("Data not supported")

    x_max = np.round(x[roi_sum.argmax()], 4)
    y_max = np.round(y[roi_sum.argmax()], 4)

    # Define colormap
    norm = matplotlib.colors.Normalize(vmin=min(roi_sum), vmax=max(roi_sum))
    colors = [colormap(norm(roi)) for roi in roi_sum]

    # Define plot
    left, width = 0.1, 0.6
    bottom, height = 0.1, 0.6
    h_spacing = 0.07
    v_spacing = 0.05
    smaller_width = 0.2
    smaller_height = 0.2

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + v_spacing, width, smaller_width]
    rect_histy = [left + width + h_spacing, bottom, smaller_height, height]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)
    ax.set_ylabel("Y", fontsize=15)
    ax.set_xlabel("X", fontsize=15)
    ax.grid()
    ax.tick_params(axis='both', labelsize=10)

    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histx.set_ylabel("Sum over ROI", fontsize=15)
    ax_histx.xaxis.set_label_position("top")
    ax_histx.set_xlabel(f"Max for x = {x_max}", fontsize=15)
    ax_histx.grid()
    ax_histx.tick_params(axis='both', labelsize=10)

    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    ax_histy.set_xlabel("Sum over ROI", fontsize=15)
    ax_histy.yaxis.set_label_position("right")
    ax_histy.set_ylabel(f"Max for y = {y_max}", fontsize=15)
    ax_histy.grid()
    ax_histy.tick_params(axis='both', labelsize=10)

    # Plot
    ax.scatter(x, y, color=colors)

    ax_histx.scatter(x, roi_sum, color=colors)

    ax_histy.scatter(roi_sum, y, color=colors)

    ax.axvline(x[roi_sum.argmax()], color=colors[roi_sum.argmax()], alpha=0.4)
    ax_histx.axvline(x[roi_sum.argmax()],
                     color=colors[roi_sum.argmax()], alpha=0.4)
    ax.axhline(y[roi_sum.argmax()], color=colors[roi_sum.argmax()], alpha=0.4)
    ax_histy.axhline(y[roi_sum.argmax()],
                     color=colors[roi_sum.argmax()], alpha=0.4)
    plt.show()
