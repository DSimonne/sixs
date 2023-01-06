"""
This module has functions to help with the alignment of the sample
during coherence experiments at SixS.

Functions to use in the ipy3 environment of srv4:
    get_file_range(): get a list of file in a directory, that is then
        used as entry in show_map()
    show_map(): show a map performed with the hexapod
    plot_xy_hexapod_scan(): plot a 2d scan, usually in x and y, and prints the command
        to align the sample
    plot_rocking_curve(): plot rocking curve in mu or omega and prints the command
        to align the sample
    plot_sample_position_evolution(): plot the evolution of the particle position
        taken from hexapod alignment scans

Other functions used in the module:
    load_nexus_attribute()
    find_FZP_focal_plane()
    plot_mesh()
    get_scan_number()
    
How to test these functions on srv4:
    test_dir = "/nfs/ruche-sixs/sixs-soleil/com-sixs/2022/Run3/Chatelier_20211332/B18S1P1_bis/"
    
    Position evolution:

        file_list_position_evolution = coh.get_file_range(directory=test_dir, start_number=1090,end_number=1160, pattern="*hexapod*.nxs")
        coh.plot_sample_position_evolution(directory=test_dir, file_list=file_list_position_evolution)
    
    Map:
        file_list_map = coh.get_file_range(directory=test_dir, start_number=470,end_number=1070)
        coh.show_map(file_list = file_list_map, directory=test_dir)
    
    2d scan:
        coh.plot_xy_hexapod_scan(file="B18S1P1_bis_hexapod_scan_00692.nxs", directory=test_dir)
"""

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

import tables as tb
import numpy as np
import glob
import os
import h5py

from scipy import interpolate
try:
    from lmfit.models import GaussianModel
    from sixs import ReadNxs4 as rn4
except ModuleNotFoundError:
    print(
        "Could not import `lmfit`, and `sixs`"
        "\nThis is normal on srv4"
    )

# Functions to load data from SixS NeXuS files


def load_nexus_attribute(key, file, directory=None):
    """Get attribute values from nexus file in f.com.scan_data"""
    if not isinstance(key, str):
        print("param 'key' must be a string")
        return None

    if isinstance(directory, str):
        path_to_nxs_data = os.path.join(directory, file)
    else:
        path_to_nxs_data = file

    if not os.path.isfile(path_to_nxs_data):
        print("param 'path_to_nxs_data' must be a valid path to a hdf5 file:",
              path_to_nxs_data)
        return None

    try:
        with h5py.File(path_to_nxs_data, "r") as f:
            return f[f"com/scan_data/{key}"][()]
    except OSError:
        print(
            f"path_to_nxs_data: {path_to_nxs_data} is not a valid NeXuS file")
        return None
    except KeyError:
        print(f"key: {key} was not found in f.com.scan_data")
        return None


def get_file_range(directory, start_number, end_number, pattern='*.nxs'):
    """
    Select a file range based on the scan number, selection is made using 
    the get_scan_number() function.

    :return: file list
    """
    print("Using as directory:", directory)
    all_file_list = sorted(
        glob.glob(directory + pattern),
        key=os.path.getmtime,
    )

    # Filter based on scan number
    file_list = []
    for file in all_file_list:
        file_number = get_scan_number(file)
        if start_number <= file_number <= end_number and os.path.isfile(file):
            file_list.append(os.path.basename(file))
    print(f"Found {len(file_list)} files.")
    return file_list


# Data visualisation for SixS
def find_FZP_focal_plane(
    file_list,
    scan_type,
    directory=None,
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

    # Show figure
    plt.tight_layout()
    plt.show(block=False)


def show_map(
    file_list,
    directory=None,
    map_type="hexapod_scan",
    load_with_readnxs=False,
    roi_key=None,
    x_key=None,
    y_key=None,
    verbose=False,
    scale="log",
    flip_axes=True,
    shading="nearest",
    cmap='turbo',
    square_aspect=True,
):
    """
    Extract map from a list of scans

    :param file_list: list of .nxs files in param directory
    :param directory: directory in which file_list are, default is None
    :param map_type: str in ("hexapod_scan", "ascan_y"),
     used to give default values for params roi_key, x_key and y_key
    :param roi_key: str, key used in f.root.com.scan_data for roi
     default value depends on 'map_type'
    :param x_key: str, key used in f.root.com.scan_data for x
     default value depends on 'map_type'
    :param y_key: str, key used in f.root.com.scan_data for y
     default value depends on 'map_type'
    :param verbose: True for more details
    :param scale: str in ("log", "lin")
    :param flip_axes: True to switch x and y axis
    :param shading: use 'nearest' for no interpolation and 'gouraud' otherwise
    :param cmap: colormap to use for the map
    :param square_aspect: True to force square aspect in the final plot

    return: Error or success message
    """
    # Check parameters
    if len(file_list) == 0:
        return ("`file_list` parameter must be of length at least 1")

    if map_type not in ("hexapod_scan", "ascan_y"):
        return ("`map_type` parameter must be 'hexapod_scan' or 'ascan_y'")

    if not isinstance(roi_key, str):
        if map_type == "hexapod_scan":
            roi_key = "roi1_merlin"
            print("Defaulted 'roi' to 'roi1_merlin'")
        elif map_type == "ascan_y":
            roi_key = "data_30"
            print("Defaulted 'roi' to 'data_30'")
    else:
        print("Using for roi:", roi_key)

    if not isinstance(x_key, str):
        if map_type == "hexapod_scan":
            x_key = "X"
            print("Defaulted 'x' to 'X'")
        elif map_type == "ascan_y":
            x_key = "data_41"
            print("Defaulted 'x' to 'data_41'")
    else:
        print("Using for x:", x_key)

    if not isinstance(y_key, str):
        if map_type == "hexapod_scan":
            y_key = "Y"
            print("Defaulted 'y' to 'Y'")
        elif map_type == "ascan_y":
            y_key = "data_42"
            print("Defaulted 'y' to 'data_42'")
    else:
        print("Using for y:", y_key)

    if not isinstance(verbose, bool):
        return ("`verbose` parameter must be a Boolean")

    if scale not in ("log", "lin"):
        return ("`scale` parameter must be 'lin' or 'log'")

    if not isinstance(flip_axes, bool):
        return ("`flip_axes` parameter must be a Boolean")

    # Save file range index, specific to SixS naming
    first_scan = get_scan_number(file_list[0])
    last_scan = get_scan_number(file_list[-1])

    # Get data
    X_list, Y_list, roi_sum_list = [], [], []

    # Load data
    for file in file_list:
        x = load_nexus_attribute(key=x_key, file=file, directory=directory)
        y = load_nexus_attribute(key=y_key, file=file, directory=directory)
        roi_sum = load_nexus_attribute(
            key=roi_key, file=file, directory=directory)

        # Append to lists
        X_list.append(x)
        Y_list.append(y)
        roi_sum_list.append(roi_sum)
        if verbose:
            print(
                f"Loading :{file}"
                f"\n\tShape in x: {x.shape}"
                f"\n\tShape in y: {y.shape}"
                f"\n\tShape in roi_sum: {roi_sum.shape}\n"
            )

    # Create empty arrays
    arr_roi = np.zeros((len(file_list), len(Y_list[0])))
    arr_y = np.zeros((len(file_list), len(Y_list[0])))
    arr_x = np.zeros((len(file_list), len(Y_list[0])))

    # Fill arrays
    for j, (x, y, roi_sum) in enumerate(zip(X_list, Y_list, roi_sum_list)):
        # y is constant, x changes
        if verbose:
            print(f"Scan nb {j} for y=", np.round(np.mean(y), 3))
        arr_roi[j] = roi_sum
        arr_y[j] = np.around(y, 3)
        arr_x[j] = np.around(x, 3)

    if map_type == "ascan_y":
        # Flip because the scans go with increasing y every other scan
        # and otherwise decreasing y
        for i in range(len(arr_y)):
            if i % 2 == 1:
                arr_y[i] = np.flip(arr_y[i])
                arr_roi[i] = np.flip(arr_roi[i])

    # Apply scale
    if scale == 'lin':
        plotted_array = arr_roi
    else:
        plotted_array = np.where(arr_roi > 0, np.log10(arr_roi), 0)

    # Plot data
    title = f"Scans {first_scan} ----> {last_scan}"
    plot_mesh(
        arr_x,
        arr_y,
        plotted_array,
        title,
        cmap=cmap,
        flip_axes=flip_axes,
        shading=shading,
    )

    return ("Data successfully plotted.")


def plot_mesh(
    arr_x,
    arr_y,
    plotted_array,
    title=None,
    cmap='jet',
    flip_axes=False,
    shading="nearest",
    square_aspect=True,
):
    """
    Plot mesh of values from x and y coordinates.
    Axis are forced to be square

    :param arr_x: positions in x, shape (X,)
    :param arr_y: positions in y, shape (Y,)
    :param plotted_array: values, 2D array of shape (X, Y)
    :param title: title (str) for the plot
    :param cmap: colormap used for mesh
    :param flip_axes: True to switch x and y
    :param shading: use 'nearest' for no interpolation and 'gouraud' otherwise
    :param square_aspect: True to force square aspect in the final plot
    """

    # Define figure
    fig, ax = plt.subplots(figsize=(8, 8))

    if flip_axes:
        im = ax.pcolormesh(
            arr_y,
            arr_x,
            plotted_array,
            cmap=cmap,
            shading=shading,
        )
    else:
        im = ax.pcolormesh(
            arr_x,
            arr_y,
            plotted_array,
            cmap=cmap,
            shading=shading,
        )

    if square_aspect:
        ax.axis('square')
    if flip_axes:
        ax.set_xlabel('y (mm)')
        ax.set_ylabel('x (mm)')
    else:
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')

    if isinstance(title, str):
        ax.set_title(title)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.grid()

    # Cursor format
    def format_coord(x, y):
        X, Y = np.meshgrid(arr_x, arr_y)
        xarr = X[0, :]
        yarr = Y[:, 0]
        if ((x > xarr.min()) & (x <= xarr.max()) &
                (y > yarr.min()) & (y <= yarr.max())):
            col = np.searchsorted(xarr, x)-1
            row = np.searchsorted(yarr, y)-1
            z = plotted_array[row, col]
            return f'x={x:1.4f}, y={y:1.4f}, z={z:1.4f}   [{row},{col}]'
        else:
            return f'x={x:1.4f}, y={y:1.4f}'
    ax.format_coord = format_coord

    # Show figure
    plt.tight_layout()
    plt.show(block=False)


def plot_xy_hexapod_scan(
    file,
    directory=None,
    colormap='jet',
    roi_key="roi1_merlin",
    x_key="X",
    y_key="Y",
):
    """
    Plot evolution of roi as a function of x_key and y_key

    :param file: absolute path to .nxs file
    :param directory: directory in which the file is, default is None
    :param colormap: colormap used to color roi, default
     is "jet", other possibilites are viridis, ...
    :param roi_key: str, key used in f.root.com.scan_data for roi
     default is roi1_merli"n"
    :param x_key: str, key used in f.root.com.scan_data for x
     default is "X"
    :param y_key: str, key used in f.root.com.scan_data for y
     default is "Y"

    return: Error or success message
    """
    # Check arguments
    try:
        colormap = getattr(cm, colormap)
    except AttributeError:
        return ("Possible colormaps are 'viridis', 'jet', ...")

    # Load data
    x = load_nexus_attribute(key=x_key, file=file, directory=directory)
    y = load_nexus_attribute(key=y_key, file=file, directory=directory)
    roi_sum = load_nexus_attribute(key=roi_key, file=file, directory=directory)

    # Print different command to move to max in x and y
    x_max = np.round(x[roi_sum.argmax()], 4)
    y_max = np.round(y[roi_sum.argmax()], 4)
    print(f"mv (x, {x_max}); mv (y, {y_max}); rct()")
    print(f"mv (x, {x_max}); rct()")
    print(f"mv (y, {y_max}); rct()")

    # Define colormap
    norm = matplotlib.colors.Normalize(
        vmin=min(roi_sum),
        vmax=max(roi_sum)
    )
    colors = [colormap(norm(roi)) for roi in roi_sum]

    # Define plot
    left, width = 0.13, 0.6
    bottom, height = 0.1, 0.63
    h_spacing = 0.02
    v_spacing = 0.02
    smaller_width = 0.2
    smaller_height = 0.2

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom + height + v_spacing, width, smaller_width]
    rect_histy = [left + width + h_spacing, bottom, smaller_height, height]

    fig = plt.figure(figsize=(8, 8))

    ax = fig.add_axes(rect_scatter)
    ax.set_ylabel("Y", fontsize=10)
    ax.set_xlabel("X", fontsize=10)
    ax.grid()
    ax.tick_params(axis='both', labelsize=10)

    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histx.set_ylabel("Sum over ROI", fontsize=10)
    ax_histx.xaxis.set_label_position("top")
    ax_histx.set_xlabel(f"Max for x = {x_max}", fontsize=10)
    ax_histx.grid()
    ax_histx.tick_params(axis='both', labelsize=10)
    ax_histx.xaxis.set_tick_params(labelbottom=False)

    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    ax_histy.set_xlabel("Sum over ROI", fontsize=10)
    ax_histy.yaxis.set_label_position("right")
    ax_histy.set_ylabel(f"Max for y = {y_max}", fontsize=10)
    ax_histy.grid()
    ax_histy.tick_params(axis='both', labelsize=10)
    ax_histy.yaxis.set_tick_params(labelleft=False)

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

    # Show figure
    fig.tight_layout()
    plt.show(block=False)


def plot_rocking_curve(
    file: str,
    directory: str = None,
    roi_key: str = "roi1_merlin",
    motor_key: str = "mu",
    method: str = "com",
    plot: bool = True,
    logscale: bool = True,
    figsize: tuple = (12, 6),
    color: str = "teal",
) -> None:
    """
    Find the rocking curve position of interest (maximum, or center of
    mass).

    :param file: path to data
    :param directory: directory in which file are, default is None
    :param roi_key: the roi key (str) in the nexus file. To avoid memory errors
        we load directly the sum over one of the ROI
    :param motor_key: the scanned motor key (str) in the nexus file.
    :param method: the method used to find the position of interest
        (str).
    :param plot: whether to plot the rocking curve (bool). Default: True
    :param logscale: whether to plot the rocking curve in logscale
        (bool). Default: True
    :param figsize: the matplotlib figure size (tuple). Default: (12, 6)
    :param color: the rocking curve color on the plot (str).
        Default: "teal"

    :return None:
    """
    # load the data, detector intensity and motor positions
    try:
        sum_intensity = load_nexus_attribute(
            key=roi_key, file=file, directory=directory)
        scanned_motor = load_nexus_attribute(
            key=motor_key, file=file, directory=directory)
    except Exception as e:
        print(e.__str__())
        return

    if method == "com":
        position = np.average(scanned_motor, weights=sum_intensity)
        title = f"Rocking curve center of mass, ({position:.4f})"
    elif method == "max":
        position = scanned_motor[np.argmax(sum_intensity)]
        title = f"Rocking curve maximum, ({position:.4f})"
    else:
        print(f"[INFO] Method not known ({method}), will use 'com'")
        position = np.average(scanned_motor, weights=sum_intensity)
        title = f"Rocking curve center of mass, ({position:.4f})"

    # print the command
    print(f"mv mu, {position:.4f}; rct();")

    if plot:
        # plot the rocking curve
        fig, ax = plt.subplots(1, figsize=figsize)
        if logscale:
            sum_intensity = np.log(sum_intensity)
        line = ax.plot(scanned_motor, sum_intensity, linewidth=2.5)

        # plot the center of mass vertical line
        ax.vlines(
            position,
            ymin=np.min(sum_intensity),
            ymax=np.max(sum_intensity),
            ls="--",
            linewidths=2,
        )

        # add grid and set background color
        ax.grid(which="both", ls=":")
        ax.patch.set_facecolor("grey")
        ax.patch.set_alpha(0.2)

        # handle title and labels
        ax.set_title(title, size=18)
        ax.set_xlabel("Scanned motor", size=16)
        if logscale:
            ylabel = "Summed intensity (logscale, a. u)"
        else:
            ylabel = "Summed intensity"
        ax.set_ylabel(ylabel, size=16)

        # Show figure
        plt.tight_layout()
        plt.show(block=False)


def plot_sample_position_evolution(
    file_list,
    directory=None,
    roi_key="roi1_merlin",
    x_key="X",
    y_key="Y",
    method="max",
    cmap="RdYlBu",
):
    """
    Plot the evolution of the position of the sample taken from the alignment 
    scans as a function of the scan number

    :param file_list: list of .nxs files in param directory
    :param directory: directory in which file are, default is None
    :param roi_key: str, key used in f.root.com.scan_data for roi
     default value depends on 'map_type'
    :param x_key: str, key used in f.root.com.scan_data for x
     default value depends on 'map_type'
    :param y_key: str, key used in f.root.com.scan_data for y
     default value depends on 'map_type'
    :param method: the method used to find the position of interest (str).
    :param cmap: cmap for color in plots
    """

    temp_colors = np.arange(0, len(file_list))
    x_max_list = []
    y_max_list = []

    for file in file_list:
        try:
            x = load_nexus_attribute(key=x_key, file=file, directory=directory)
            y = load_nexus_attribute(key=y_key, file=file, directory=directory)
            roi = load_nexus_attribute(
                key=roi_key, file=file, directory=directory)

            argmax_roi = roi.argmax()
            x_max_list.append(x[argmax_roi])
            y_max_list.append(y[argmax_roi])
        except AttributeError:
            print("AttributeError for ", file)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))

    sc = ax.scatter(
        x_max_list,
        y_max_list,
        s=50,
        c=temp_colors,
        cmap=cmap
    )
    ax.plot(x_max_list, y_max_list, alpha=0.5)

    ax.set_xlabel("x", size=15)
    ax.set_ylabel("y", size=15)
    ax.grid()
    ax.tick_params(labelsize=15)
    cbar = plt.colorbar(sc)
    cbar.ax.set_title('Last scan', size=15)
    cbar.set_ticks(np.arange(len(file_list)))
    cbar.set_ticklabels([get_scan_number(f) for f in file_list])
    plt.show()


def get_scan_number(file):
    "Return scan number of nexus file for SixS"
    file = os.path.basename(file)

    return int(os.path.splitext(file)[0].split("_")[-1])
