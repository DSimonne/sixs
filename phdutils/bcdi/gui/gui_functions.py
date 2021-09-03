# BCDI: tools for pre(post)-processing Bragg coherent X-ray diffraction imaging data
#   (c) 07/2017-06/2019 : CNRS UMR 7344 IM2NP
#   (c) 07/2019-present : DESY PHOTON SCIENCE
#       authors:
#         Jerome Carnis, carnis_jerome@yahoo.fr

# Preprocess
try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import xrayutilities as xu
import numpy as np
import matplotlib.pyplot as plt

import os
import scipy.signal  # for medfilt2d
from scipy.ndimage.measurements import center_of_mass
import sys
from scipy.io import savemat
import tkinter as tk
from tkinter import filedialog
import gc
import bcdi.graph.graph_utils as gu
from bcdi.experiment.detector import Detector
from bcdi.experiment.setup import Setup
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid

# Correct
try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import filedialog
import sys
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru
from bcdi.experiment.detector import Detector
from bcdi.experiment.setup import Setup
import bcdi.utils.utilities as util

# Strain
from collections.abc import Sequence
from datetime import datetime
from functools import reduce
import gc

try:
    import hdf5plugin  # for P10, should be imported before h5py or PyTables
except ModuleNotFoundError:
    pass
import h5py
from matplotlib import pyplot as plt
from numbers import Real
import numpy as np
import os
import pprint
import tkinter as tk
from tkinter import filedialog
import bcdi.graph.graph_utils as gu
from bcdi.experiment.detector import Detector
from bcdi.experiment.setup import Setup
import bcdi.postprocessing.postprocessing_utils as pu
import bcdi.preprocessing.preprocessing_utils as pru
import bcdi.simulation.simulation_utils as simu
import bcdi.utils.utilities as util
import bcdi.utils.validation as valid


# Functions used in the gui

def correct_angles_detector(
    filename,
    direct_inplane,
    direct_outofplane,
    get_temperature,
    reflection,
    reference_spacing,
    reference_temperature,
    high_threshold,
    save_dir,
    scan,
    root_folder,
    sample_name,
    filtered_data,
    peak_method,
    normalize_flux,
    debug,
    beamline,
    actuators,
    is_series,
    custom_scan,
    custom_images,
    custom_monitor,
    custom_motors,
    rocking_angle,
    specfile_name,
    detector,
    x_bragg,
    y_bragg,
    roi_detector,
    hotpixels_file,
    flatfield_file,
    template_imagefile,
    beam_direction,
    sample_offsets,
    directbeam_x,
    directbeam_y,
    sdd,
    energy,
    ):
    """
    Calculate exact inplane and out-of-plane detector angles from the direct beam and Bragg peak positions,
    based on the beamline geometry.

    Supported beamlines: ESRF ID01, PETRAIII P10, SOLEIL SIXS, SOLEIL CRISTAL.

    For Pt samples it gives also an estimation of the temperature based on the thermal expansion.

    Input: direct beam and Bragg peak position, sample to detector distance, energy
    Output: corrected inplane, out-of-plane detector angles for the Bragg peak.
    """

    #######################
    # Initialize detector #
    #######################
    detector = Detector(
        name=detector,
        template_imagefile=template_imagefile,
        roi=roi_detector,
        is_series=is_series,
    )

    ####################
    # Initialize setup #
    ####################
    setup = Setup(
        beamline=beamline,
        detector=detector,
        energy=energy,
        rocking_angle=rocking_angle,
        distance=sdd,
        beam_direction=beam_direction,
        custom_scan=custom_scan,
        custom_images=custom_images,
        custom_monitor=custom_monitor,
        custom_motors=custom_motors,
        sample_offsets=sample_offsets,
        actuators=actuators,
    )

    ########################################
    # Initialize the paths and the logfile #
    ########################################
    # initialize the paths
    setup.init_paths(
        sample_name=sample_name,
        scan_number=scan,
        root_folder=root_folder,
        save_dir=None,
        create_savedir=False,
        specfile_name=specfile_name,
        template_imagefile=template_imagefile,
        verbose=True,
    )

    logfile = setup.create_logfile(
        scan_number=scan, root_folder=root_folder, filename=detector.specfile
    )

    #################
    # load the data #
    #################
    flatfield = util.load_flatfield(flatfield_file)
    hotpix_array = util.load_hotpixels(hotpixels_file)

    if not filtered_data:
        data, _, monitor, frames_logical = pru.load_data(
            logfile=logfile,
            scan_number=scan,
            detector=detector,
            setup=setup,
            flatfield=flatfield,
            hotpixels=hotpix_array,
            normalize=normalize_flux,
            debugging=debug,
        )
        if normalize_flux == "skip":
            print("Skip intensity normalization")
        else:
            print("Intensity normalization using " + normalize_flux)
            data, monitor = pru.normalize_dataset(
                array=data,
                raw_monitor=monitor,
                frames_logical=frames_logical,
                norm_to_min=True,
                debugging=debug,
            )
    else:
        # root = tk.Tk()
        # root.withdraw()
        file_path = filedialog.askopenfilename(
            initialdir=detector.scandir + "pynxraw/",
            title="Select 3D data",
            filetypes=[("NPZ", "*.npz")],
        )
        data = np.load(file_path)["data"]
        data = data[detector.roi[0] : detector.roi[1], detector.roi[2] : detector.roi[3]]
        frames_logical = np.ones(data.shape[0]).astype(
            int
        )  # use all frames from the filtered data
    numz, numy, numx = data.shape
    print("Shape of dataset: ", numz, numy, numx)

    ##############################################
    # apply photon threshold to remove hotpixels #
    ##############################################
    if high_threshold != 0:
        nb_thresholded = (data > high_threshold).sum()
        data[data > high_threshold] = 0
        print(f'Applying photon threshold, {nb_thresholded} high intensity pixels masked')

    ###############################
    # load relevant motor values #
    ###############################
    (
        tilt_values,
        setup.grazing_angle,
        setup.inplane_angle,
        setup.outofplane_angle,
    ) = setup.diffractometer.goniometer_values(
        logfile=logfile, scan_number=scan, setup=setup, frames_logical=frames_logical
    )
    setup.tilt_angle = (tilt_values[1:] - tilt_values[0:-1]).mean()

    nb_frames = len(tilt_values)
    if numz != nb_frames:
        print("The loaded data has not the same shape as the raw data")
        sys.exit()

    #######################
    # Find the Bragg peak #
    #######################
    z0, y0, x0 = pru.find_bragg(data, peak_method=peak_method)
    z0 = np.rint(z0).astype(int)
    y0 = np.rint(y0).astype(int)
    x0 = np.rint(x0).astype(int)

    print(f"Bragg peak at (z, y, x): {z0}, {y0}, {x0}")
    print(
        f"Bragg peak (full detector) at (z, y, x): {z0},"
        f" {y0+detector.roi[0]}, {x0+detector.roi[2]}"
    )

    ######################################################
    # calculate rocking curve and fit it to get the FWHM #
    ######################################################
    rocking_curve = np.zeros(nb_frames)
    if filtered_data == 0:  # take a small ROI to avoid parasitic peaks
        for idx in range(nb_frames):
            rocking_curve[idx] = data[idx, y0 - 20:y0 + 20, x0 - 20:x0 + 20].sum()
        plot_title = "Rocking curve for a 40x40 pixels ROI"
    else:  # take the whole detector
        for idx in range(nb_frames):
            rocking_curve[idx] = data[idx, :, :].sum()
        plot_title = "Rocking curve (full detector)"
    z0 = np.unravel_index(rocking_curve.argmax(), rocking_curve.shape)[0]

    interpolation = interp1d(tilt_values, rocking_curve, kind='cubic')
    interp_points = 5*nb_frames
    interp_tilt = np.linspace(tilt_values.min(), tilt_values.max(), interp_points)
    interp_curve = interpolation(interp_tilt)
    interp_fwhm = (
        len(np.argwhere(interp_curve >= interp_curve.max() / 2))
        * (tilt_values.max() - tilt_values.min())
        / (interp_points - 1)
    )
    print('FWHM by interpolation', str('{:.3f}'.format(interp_fwhm)), 'deg')

    plt.close()
    fig, (ax0, ax1) = plt.subplots(2, 1, sharex='col', figsize=(10, 5))
    ax0.plot(tilt_values, rocking_curve, '.')
    ax0.plot(interp_tilt, interp_curve)
    ax0.axvline(tilt_values[z0], color='r', alpha = 0.7, linewidth = 1)
    ax0.set_ylabel('Integrated intensity')
    ax0.legend(('data', 'interpolation'))
    ax0.set_title(plot_title)
    ax1.plot(tilt_values, np.log10(rocking_curve), '.')
    ax1.plot(interp_tilt, np.log10(interp_curve))
    ax1.axvline(tilt_values[z0], color='r', alpha = 0.7, linewidth = 1)

    ax1.set_xlabel('Rocking angle (deg)')
    ax1.set_ylabel('Log(integrated intensity)')
    ax0.legend(('data', 'interpolation'))
    plt.savefig(save_dir + "rocking_curve.png")
    plt.show()

    ##############################
    # Calculate corrected angles #
    ##############################
    bragg_x = detector.roi[2] + x0  # convert it in full detector pixel
    bragg_y = detector.roi[0] + y0  # convert it in full detector pixel

    x_direct_0 = directbeam_x + setup.inplane_coeff * (
        direct_inplane * np.pi / 180 * sdd / detector.pixelsize_x
    )  # inplane_coeff is +1 or -1
    y_direct_0 = (
        directbeam_y
        - setup.outofplane_coeff
        * direct_outofplane
        * np.pi
        / 180
        * sdd
        / detector.pixelsize_y
    )  # outofplane_coeff is +1 or -1


    print(
        f"\nDirect beam at (gam={direct_inplane}, del={direct_outofplane}) (X, Y): {directbeam_x}, {directbeam_y}"
    )
    print(f"Direct beam at (gam=0, del=0) (X, Y): ({x_direct_0:.2f}, {y_direct_0:.2f})")
    print(
        f"\nBragg peak at (gam={setup.inplane_angle}, del={setup.outofplane_angle}) (X, Y): ({bragg_x:.2f}, {bragg_y:.2f})"
    )

    # add error on bragg peak position to computer errorbars
    # bragg_y+=5
    # bragg_x+=5

    bragg_inplane = setup.inplane_angle + setup.inplane_coeff * (
        detector.pixelsize_x * (bragg_x - x_direct_0) / sdd * 180 / np.pi
    )  # inplane_coeff is +1 or -1
    bragg_outofplane = (
        setup.outofplane_angle
        - setup.outofplane_coeff
        * detector.pixelsize_y
        * (bragg_y - y_direct_0)
        / sdd
        * 180
        / np.pi
    )  # outofplane_coeff is +1 or -1

    print(
        f"\nBragg angles before correction (gam, del): ({setup.inplane_angle:.4f}, {setup.outofplane_angle:.4f})"
    )
    print(
        f"Bragg angles after correction (gam, del): ({bragg_inplane:.4f}, {bragg_outofplane:.4f})"
    )

    # update setup with the corrected detector angles
    setup.inplane_angle = bragg_inplane
    setup.outofplane_angle = bragg_outofplane

    print(f"\nGrazing angle(s) = {setup.grazing_angle} deg")
    print(f"Rocking step = {setup.tilt_angle:.5f} deg")

    ####################################
    # wavevector transfer calculations #
    ####################################
    kin = (
        2 * np.pi / setup.wavelength * np.asarray(beam_direction)
    )  # in lab frame z downstream, y vertical, x outboard
    kout = setup.exit_wavevector  # in lab.frame z downstream, y vertical, x outboard
    q = (kout - kin) / 1e10  # convert from 1/m to 1/angstrom
    qnorm = np.linalg.norm(q)
    dist_plane = 2 * np.pi / qnorm
    print(f"\nWavevector transfer of Bragg peak: {q}, Qnorm={qnorm:.4f}")
    print(f"Interplanar distance: {dist_plane:.6f} angstroms")
    print(f"Wavelength = {setup.wavelength}")

    if get_temperature:
        print("\nEstimating the temperature:")
        temperature = pu.bragg_temperature(
            spacing=dist_plane,
            reflection=reflection,
            spacing_ref=reference_spacing,
            temperature_ref=reference_temperature,
            use_q=False,
            material="Pt",
        )

    else:
        temperature = None

    #########################
    # calculate voxel sizes #
    #########################
    #  update the detector angles in setup
    setup.inplane_angle = bragg_inplane
    setup.outofplane_angle = bragg_outofplane
    dz_realspace, dy_realspace, dx_realspace = setup.voxel_sizes(
        (nb_frames, numy, numx),
        tilt_angle=setup.tilt_angle,
        pixel_x=detector.pixelsize_x,
        pixel_y=detector.pixelsize_y,
        verbose=True,
    )

    #################################
    # plot image at Bragg condition #
    #################################
    plt.close()
    plt.imshow(np.log10(abs(data[int(round(z0)), :, :])), vmin=0, vmax=5)
    plt.title(f'Central slice at frame {int(np.rint(z0))}')
    plt.colorbar()

    plt.scatter(bragg_x, bragg_y, color='r', alpha = 0.7, linewidth = 1)
    plt.savefig(save_dir + "central_slice.png")
    plt.show()

    print("End of script \n")
    plt.close()

    # added script
    COM_rocking_curve = tilt_values[z0],
    detector_data_COM = abs(data[int(round(z0)), :, :]),

    metadata = {
        "tilt_values" : tilt_values,
        "rocking_curve" : rocking_curve,
        "interp_tilt" : interp_tilt,
        "interp_curve" : interp_curve,
        "COM_rocking_curve" : tilt_values[z0],
        "detector_data_COM" : abs(data[int(round(z0)), :, :]),
        "interp_fwhm" : interp_fwhm,
        "temperature" : temperature,
        "bragg_x" : bragg_x, 
        "bragg_y" : bragg_y,
        "q" : q, 
        "qnorm" : qnorm, 
        "dist_plane" : dist_plane, 
        "bragg_inplane" : bragg_inplane, 
        "bragg_outofplane" : bragg_outofplane,
    }

    return metadata


def preprocess_bcdi(
    scans,
    root_folder,
    save_dir,
    data_dirname, 
    sample_name,
    user_comment,
    debug, binning,
    flag_interact,
    background_plot,
    centering, 
    fix_bragg, 
    fix_size, 
    center_fft, 
    pad_size,
    normalize_flux, 
    mask_zero_event, 
    flag_medianfilter, 
    medfilt_order,
    reload_previous, 
    reload_orthogonal, 
    preprocessing_binning,
    save_rawdata, 
    save_to_npz, 
    save_to_mat, 
    save_to_vti, 
    save_asint,
    beamline, 
    actuators, 
    is_series, 
    custom_scan, 
    custom_images, 
    custom_monitor, 
    rocking_angle, 
    follow_bragg, 
    specfile_name,
    detector, 
    linearity_func, 
    x_bragg, 
    y_bragg, 
    roi_detector, 
    photon_threshold, 
    photon_filter, 
    background_file, 
    hotpixels_file, 
    flatfield_file, 
    template_imagefile, 
    nb_pixel_x, 
    nb_pixel_y,
    use_rawdata, 
    interp_method, 
    fill_value_mask, 
    beam_direction, 
    sample_offsets, 
    sdd, 
    energy, 
    custom_motors,
    align_q, 
    ref_axis_q, 
    outofplane_angle, 
    inplane_angle, 
    sample_inplane, 
    sample_outofplane, 
    offset_inplane, 
    cch1, 
    cch2, 
    detrot, 
    tiltazimuth, 
    tilt
    ):
    """
    Prepare experimental data for Bragg CDI phasing: crop/pad, center, mask, normalize and
    filter the data.

    Beamlines currently supported: ESRF ID01, SOLEIL CRISTAL, SOLEIL SIXS, PETRAIII P10 and
    APS 34ID-C.

    Output: data and mask as numpy .npz or Matlab .mat 3D arrays for phasing

    File structure should be (e.g. scan 1):
    specfile, hotpixels file and flatfield file in:    /rootdir/
    data in:                                           /rootdir/S1/data/

    output files saved in:   /rootdir/S1/pynxraw/ or /rootdir/S1/pynx/ depending on the
    'use_rawdata' option"""

    if flag_interact:
        plt.switch_backend(
            "Qt5Agg"
        )

    def close_event(event):
        """
        This function handles closing events on plots.

        :return: nothing
        """
        print(event, "Click on the figure instead of closing it!")
        sys.exit()


    def on_click(event):
        """
        Function to interact with a plot, return the position of clicked pixel.

        If flag_pause==1 or if the mouse is out of plot axes, it will not register the click

        :param event: mouse click event
        """
        global xy, flag_pause, previous_axis
        if not event.inaxes:
            return
        if not flag_pause:

            if (previous_axis == event.inaxes) or (previous_axis is None):  # collect points
                _x, _y = int(np.rint(event.xdata)), int(np.rint(event.ydata))
                xy.append([_x, _y])
                if previous_axis is None:
                    previous_axis = event.inaxes
            else:  # the click is not in the same subplot, restart collecting points
                print(
                    "Please select mask polygon vertices within "
                    "the same subplot: restart masking..."
                )
                xy = []
                previous_axis = None


    def press_key(event):
        """
        Interact with a plot for masking parasitic diffraction intensity or detector gaps

        :param event: button press event
        """
        global original_data, original_mask, updated_mask, data, mask, frame_index, width, flag_aliens, flag_mask, flag_pause, xy, fig_mask, max_colorbar, ax0, ax1, ax2, ax3, previous_axis, info_text, my_cmap

        try:
            if event.inaxes == ax0:
                dim = 0
                inaxes = True
            elif event.inaxes == ax1:
                dim = 1
                inaxes = True
            elif event.inaxes == ax2:
                dim = 2
                inaxes = True
            else:
                dim = -1
                inaxes = False

            if inaxes:
                if flag_aliens:
                    (
                        data,
                        mask,
                        width,
                        max_colorbar,
                        frame_index,
                        stop_masking,
                    ) = gu.update_aliens_combined(
                        key=event.key,
                        pix=int(np.rint(event.xdata)),
                        piy=int(np.rint(event.ydata)),
                        original_data=original_data,
                        original_mask=original_mask,
                        updated_data=data,
                        updated_mask=mask,
                        axes=(ax0, ax1, ax2, ax3),
                        width=width,
                        dim=dim,
                        frame_index=frame_index,
                        vmin=0,
                        vmax=max_colorbar,
                        cmap=my_cmap,
                        invert_yaxis=not use_rawdata,
                    )
                elif flag_mask:
                    if previous_axis == ax0:
                        click_dim = 0
                        x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                        points = np.stack((x.flatten(), y.flatten()), axis=0).T
                    elif previous_axis == ax1:
                        click_dim = 1
                        x, y = np.meshgrid(np.arange(nx), np.arange(nz))
                        points = np.stack((x.flatten(), y.flatten()), axis=0).T
                    elif previous_axis == ax2:
                        click_dim = 2
                        x, y = np.meshgrid(np.arange(ny), np.arange(nz))
                        points = np.stack((x.flatten(), y.flatten()), axis=0).T
                    else:
                        click_dim = None
                        points = None

                    (
                        data,
                        updated_mask,
                        flag_pause,
                        xy,
                        width,
                        max_colorbar,
                        click_dim,
                        stop_masking,
                        info_text,
                    ) = gu.update_mask_combined(
                        key=event.key,
                        pix=int(np.rint(event.xdata)),
                        piy=int(np.rint(event.ydata)),
                        original_data=original_data,
                        original_mask=mask,
                        updated_data=data,
                        updated_mask=updated_mask,
                        axes=(ax0, ax1, ax2, ax3),
                        flag_pause=flag_pause,
                        points=points,
                        xy=xy,
                        width=width,
                        dim=dim,
                        click_dim=click_dim,
                        info_text=info_text,
                        vmin=0,
                        vmax=max_colorbar,
                        cmap=my_cmap,
                        invert_yaxis=not use_rawdata,
                    )
                    if click_dim is None:
                        previous_axis = None
                else:
                    stop_masking = False

                if stop_masking:
                    plt.close("all")

        except AttributeError:  # mouse pointer out of axes
            pass


    #########################
    # check some parameters #
    #########################
    valid_name = "bcdi_preprocess_BCDI"
    if isinstance(scans, int):
        scans = (scans,)

    if len(scans) > 1:
        if center_fft not in ["crop_asymmetric_ZYX", "pad_Z", "pad_asymmetric_ZYX"]:
            center_fft = "skip"
            # avoid croping the detector plane XY while centering the Bragg peak
            # otherwise outputs may have a different size,
            # which will be problematic for combining or comparing them
    if len(fix_size) != 0:
        print('"fix_size" parameter provided, roi_detector will be set to []')
        roi_detector = []
        print("'fix_size' parameter provided, defaulting 'center_fft' to 'skip'")
        center_fft = "skip"

    if photon_filter == "loading":
        loading_threshold = photon_threshold
    else:
        loading_threshold = 0

    create_savedir = True

    valid.valid_container(user_comment, container_types=str, name=valid_name)
    if len(user_comment) != 0 and not user_comment.startswith("_"):
        user_comment = "_" + user_comment
    if reload_previous:
        user_comment += "_reloaded"
    else:
        preprocessing_binning = (1, 1, 1)
        reload_orthogonal = False

    if rocking_angle == "energy":
        use_rawdata = False  # you need to interpolate the data in QxQyQz for energy scans
        print(
            "Energy scan: defaulting use_rawdata to False,"
            " the data will be interpolated using xrayutilities"
        )

    if reload_orthogonal:
        use_rawdata = False

    if use_rawdata:
        save_dirname = "pynxraw"
        print("Output will be non orthogonal, in the detector frame")
        plot_title = ["YZ", "XZ", "XY"]
    else:
        if interp_method not in {"xrayutilities", "linearization"}:
            raise ValueError(
                "Incorrect value for interp_method,"
                ' allowed values are "xrayutilities" and "linearization"'
            )
        if rocking_angle == "energy":
            interp_method = "xrayutilities"
            print(f"Defaulting interp_method to {interp_method}")
        if not reload_orthogonal and preprocessing_binning[0] != 1:
            raise ValueError(
                "preprocessing_binning along axis 0 should be 1 when gridding reloaded data"
                " (angles won't match)"
            )
        save_dirname = "pynx"
        print(f"Output will be orthogonalized using {interp_method}")
        plot_title = ["QzQx", "QyQx", "QyQz"]

    if isinstance(sample_name, str):
        sample_name = [sample_name for idx in range(len(scans))]
    valid.valid_container(
        sample_name,
        container_types=(tuple, list),
        length=len(scans),
        item_types=str,
        name=valid_name,
    )

    if fill_value_mask not in {0, 1}:
        raise ValueError(f"fill_value_mask should be 0 or 1, got {fill_value_mask}")

    valid.valid_item(align_q, allowed_types=bool, name=valid_name)
    if align_q:
        user_comment += f"_align-q-{ref_axis_q}"
        if ref_axis_q not in {"x", "y", "z"}:
            raise ValueError("ref_axis_q should be either 'x', 'y' or 'z'")
    else:
        ref_axis_q = "y"  # ref_axis_q will not be used
    axis_to_array_xyz = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }  # in xyz order

    ###################
    # define colormap #
    ###################
    colormap = gu.Colormap()
    my_cmap = colormap.cmap
    plt.rcParams["keymap.fullscreen"] = [""]

    #######################
    # Initialize detector #
    #######################
    kwargs = {
        "is_series": is_series,
        "preprocessing_binning": preprocessing_binning,
        "nb_pixel_x": nb_pixel_x,  # fix to declare a known detector but with less pixels
        # (e.g. one tile HS)
        "nb_pixel_y": nb_pixel_y,  # fix to declare a known detector but with less pixels
        # (e.g. one tile HS)
        "linearity_func": linearity_func,
    }

    detector = Detector(
        name=detector,
        template_imagefile=template_imagefile,
        roi=roi_detector,
        binning=binning,
        **kwargs,
    )

    ####################
    # Initialize setup #
    ####################
    setup = Setup(
        beamline=beamline,
        detector=detector,
        energy=energy,
        rocking_angle=rocking_angle,
        distance=sdd,
        beam_direction=beam_direction,
        sample_inplane=sample_inplane,
        sample_outofplane=sample_outofplane,
        offset_inplane=offset_inplane,
        custom_scan=custom_scan,
        custom_images=custom_images,
        sample_offsets=sample_offsets,
        custom_monitor=custom_monitor,
        custom_motors=custom_motors,
        actuators=actuators,
    )

    ########################################
    # print the current setup and detector #
    ########################################
    print("\n##############\nSetup instance\n##############")
    print(setup)
    print("\n#################\nDetector instance\n#################")
    print(detector)

    ############################################
    # Initialize values for callback functions #
    ############################################
    flag_mask = False
    flag_aliens = False
    plt.rcParams["keymap.quit"] = [
        "ctrl+w",
        "cmd+w",
    ]  # this one to avoid that q closes window (matplotlib default)

    ############################
    # start looping over scans #
    ############################
    # root = tk.Tk()
    # root.withdraw()

    for scan_idx, scan_nb in enumerate(scans, start=1):

        comment = user_comment  # re-initialize comment
        tmp_str = f"Scan {scan_idx}/{len(scans)}: S{scan_nb}"
        print(f'\n{"#" * len(tmp_str)}\n' + tmp_str + "\n" + f'{"#" * len(tmp_str)}')

        # initialize the paths
        setup.init_paths(
            sample_name=sample_name[scan_idx - 1],
            scan_number=scan_nb,
            data_dirname=data_dirname,
            root_folder=root_folder,
            save_dir=save_dir,
            save_dirname=save_dirname,
            verbose=True,
            create_savedir=create_savedir,
            specfile_name=specfile_name,
            template_imagefile=template_imagefile,
        )

        logfile = setup.create_logfile(
            scan_number=scan_nb, root_folder=root_folder, filename=detector.specfile
        )

        if not use_rawdata:
            comment += "_ortho"
            if interp_method == "linearization":
                comment += "_lin"
                # load the goniometer positions needed in the calculation
                # of the transformation matrix
                (
                    tilt_angle,
                    setup.grazing_angle,
                    inplane,
                    outofplane,
                ) = setup.diffractometer.goniometer_values(
                    logfile=logfile,
                    scan_number=scan_nb,
                    setup=setup,
                    follow_bragg=follow_bragg,
                )
                setup.tilt_angle = (tilt_angle[1:] - tilt_angle[0:-1]).mean()
                # override detector motor positions if the corrected values
                # (taking into account the direct beam position)
                # are provided by the user
                setup.inplane_angle = (
                    inplane_angle if inplane_angle is not None else inplane
                )
                setup.outofplane_angle = (
                    outofplane_angle if outofplane_angle is not None else outofplane
                )
            else:  # 'xrayutilities'
                comment += "_xrutil"
        if normalize_flux:
            comment = comment + "_norm"

        #############
        # Load data #
        #############
        if reload_previous:  # resume previous masking
            print("Resuming previous masking")
            file_path = filedialog.askopenfilename(
                initialdir=detector.scandir,
                title="Select data file",
                filetypes=[("NPZ", "*.npz")],
            )
            data = np.load(file_path)
            npz_key = data.files
            data = data[npz_key[0]]
            nz, ny, nx = np.shape(data)

            # check that the ROI is correctly defined
            detector.roi = roi_detector or [0, ny, 0, nx]
            print("Detector ROI:", detector.roi)
            # update savedir to save the data in the same directory as the reloaded data
            if not save_dir:
                detector.savedir = os.path.dirname(file_path) + "/"
                print(f"Updated saving directory: {detector.savedir}")

            file_path = filedialog.askopenfilename(
                initialdir=os.path.dirname(file_path) + "/",
                title="Select mask file",
                filetypes=[("NPZ", "*.npz")],
            )
            mask = np.load(file_path)
            npz_key = mask.files
            mask = mask[npz_key[0]]

            if reload_orthogonal:  # the data is gridded in the orthonormal laboratory frame
                use_rawdata = False
                try:
                    file_path = filedialog.askopenfilename(
                        initialdir=detector.savedir,
                        title="Select q values",
                        filetypes=[("NPZ", "*.npz")],
                    )
                    reload_qvalues = np.load(file_path)
                    q_values = [
                        reload_qvalues["qx"],
                        reload_qvalues["qz"],
                        reload_qvalues["qy"],
                    ]
                except FileNotFoundError:
                    q_values = []

                normalize_flux = (
                    "skip"  # we assume that normalization was already performed
                )
                monitor = []  # we assume that normalization was already performed
                center_fft = (
                    "skip"  # we assume that crop/pad/centering was already performed
                )
                fix_size = []  # we assume that crop/pad/centering was already performed

                # bin data and mask if needed
                if (
                    (detector.binning[0] != 1)
                    or (detector.binning[1] != 1)
                    or (detector.binning[2] != 1)
                ):
                    print("Binning the reloaded orthogonal data by", detector.binning)
                    data = util.bin_data(data, binning=detector.binning, debugging=False)
                    mask = util.bin_data(mask, binning=detector.binning, debugging=False)
                    mask[np.nonzero(mask)] = 1
                    if len(q_values) != 0:
                        qx = q_values[0]
                        qz = q_values[1]
                        qy = q_values[2]
                        numz, numy, numx = len(qx), len(qz), len(qy)
                        qx = qx[
                            : numz - (numz % detector.binning[0]) : detector.binning[0]
                        ]  # along z downstream
                        qz = qz[
                            : numy - (numy % detector.binning[1]) : detector.binning[1]
                        ]  # along y vertical
                        qy = qy[
                            : numx - (numx % detector.binning[2]) : detector.binning[2]
                        ]  # along x outboard
                        del numz, numy, numx
            else:  # the data is in the detector frame
                data, mask, frames_logical, monitor = pru.reload_bcdi_data(
                    logfile=logfile,
                    scan_number=scan_nb,
                    data=data,
                    mask=mask,
                    detector=detector,
                    setup=setup,
                    debugging=debug,
                    normalize=normalize_flux,
                    photon_threshold=loading_threshold,
                )

        else:  # new masking process
            reload_orthogonal = False  # the data is in the detector plane
            flatfield = util.load_flatfield(flatfield_file)
            hotpix_array = util.load_hotpixels(hotpixels_file)
            background = util.load_background(background_file)

            data, mask, frames_logical, monitor = pru.load_bcdi_data(
                logfile=logfile,
                scan_number=scan_nb,
                detector=detector,
                setup=setup,
                flatfield=flatfield,
                hotpixels=hotpix_array,
                background=background,
                normalize=normalize_flux,
                debugging=debug,
                photon_threshold=loading_threshold,
            )

        nz, ny, nx = np.shape(data)
        print("\nInput data shape:", nz, ny, nx)

        binning_comment = (
            f"_{detector.preprocessing_binning[0]*detector.binning[0]}"
            f"_{detector.preprocessing_binning[1]*detector.binning[1]}"
            f"_{detector.preprocessing_binning[2]*detector.binning[2]}"
        )

        if not reload_orthogonal:
            if save_rawdata:
                np.savez_compressed(
                    detector.savedir + "S" + str(scan_nb) + "_data_before_masking_stack",
                    data=data,
                )
                if save_to_mat:
                    # save to .mat, the new order is x y z
                    # (outboard, vertical up, downstream)
                    savemat(
                        detector.savedir
                        + "S"
                        + str(scan_nb)
                        + "_data_before_masking_stack.mat",
                        {"data": np.moveaxis(data, [0, 1, 2], [-1, -2, -3])},
                    )

            if use_rawdata:
                q_values = []
                # binning along axis 0 is done after masking
                data[np.nonzero(mask)] = 0
            else:
                tmp_data = np.copy(
                    data
                )  # do not modify the raw data before the interpolation
                tmp_data[mask == 1] = 0
                fig, _, _ = gu.multislices_plot(
                    tmp_data,
                    sum_frames=True,
                    scale="log",
                    plot_colorbar=True,
                    vmin=0,
                    title="Data before gridding\n",
                    is_orthogonal=False,
                    reciprocal_space=True,
                )
                plt.savefig(
                    detector.savedir
                    + f"data_before_gridding_S{scan_nb}_{nz}_{ny}_{nx}"
                    + binning_comment
                    + ".png"
                )
                plt.close(fig)
                del tmp_data

                if interp_method == "xrayutilities":
                    qconv, offsets = pru.init_qconversion(setup)
                    detector.offsets = offsets
                    hxrd = xu.experiment.HXRD(
                        sample_inplane, sample_outofplane, en=energy, qconv=qconv
                    )
                    # the first 2 arguments in HXRD are the inplane reference direction
                    # along the beam and surface normal of the sample

                    # Update the direct beam vertical position,
                    # take into account the roi and binning
                    cch1 = (cch1 - detector.roi[0]) / (
                        detector.preprocessing_binning[1] * detector.binning[1]
                    )
                    # Update the direct beam horizontal position,
                    # take into account the roi and binning
                    cch2 = (cch2 - detector.roi[2]) / (
                        detector.preprocessing_binning[2] * detector.binning[2]
                    )
                    # number of pixels after taking into account the roi and binning
                    nch1 = (detector.roi[1] - detector.roi[0]) // (
                        detector.preprocessing_binning[1] * detector.binning[1]
                    ) + (detector.roi[1] - detector.roi[0]) % (
                        detector.preprocessing_binning[1] * detector.binning[1]
                    )
                    nch2 = (detector.roi[3] - detector.roi[2]) // (
                        detector.preprocessing_binning[2] * detector.binning[2]
                    ) + (detector.roi[3] - detector.roi[2]) % (
                        detector.preprocessing_binning[2] * detector.binning[2]
                    )
                    # detector init_area method, pixel sizes are the binned ones
                    hxrd.Ang2Q.init_area(
                        setup.detector_ver,
                        setup.detector_hor,
                        cch1=cch1,
                        cch2=cch2,
                        Nch1=nch1,
                        Nch2=nch2,
                        pwidth1=detector.pixelsize_y,
                        pwidth2=detector.pixelsize_x,
                        distance=setup.distance,
                        detrot=detrot,
                        tiltazimuth=tiltazimuth,
                        tilt=tilt,
                    )
                    # first two arguments in init_area are the direction of the detector,
                    # checked for ID01 and SIXS

                    data, mask, q_values, frames_logical = pru.grid_bcdi_xrayutil(
                        data=data,
                        mask=mask,
                        scan_number=scan_nb,
                        logfile=logfile,
                        detector=detector,
                        setup=setup,
                        frames_logical=frames_logical,
                        hxrd=hxrd,
                        follow_bragg=follow_bragg,
                        debugging=debug,
                    )
                else:  # 'linearization'
                    # for q values, the frame used is
                    # (qx downstream, qy outboard, qz vertical up)
                    # for reference_axis, the frame is z downstream, y vertical up,
                    # x outboard but the order must be x,y,z
                    data, mask, q_values = pru.grid_bcdi_labframe(
                        data=data,
                        mask=mask,
                        detector=detector,
                        setup=setup,
                        align_q=align_q,
                        reference_axis=axis_to_array_xyz[ref_axis_q],
                        debugging=debug,
                        follow_bragg=follow_bragg,
                        fill_value=(0, fill_value_mask),
                    )
                nz, ny, nx = data.shape
                print(
                    "\nData size after interpolation into an orthonormal frame:", nz, ny, nx
                )

                # plot normalization by incident monitor for the gridded data
                if normalize_flux:
                    tmp_data = np.copy(
                        data
                    )  # do not modify the raw data before the interpolation
                    tmp_data[tmp_data < 5] = 0  # threshold the background
                    tmp_data[mask == 1] = 0
                    fig = gu.combined_plots(
                        tuple_array=(monitor, tmp_data),
                        tuple_sum_frames=(False, True),
                        tuple_sum_axis=(0, 1),
                        tuple_width_v=None,
                        tuple_width_h=None,
                        tuple_colorbar=(False, False),
                        tuple_vmin=(np.nan, 0),
                        tuple_vmax=(np.nan, np.nan),
                        tuple_title=(
                            "monitor.min() / monitor",
                            "Gridded normed data (threshold 5)\n",
                        ),
                        tuple_scale=("linear", "log"),
                        xlabel=("Frame number", "Q$_y$"),
                        ylabel=("Counts (a.u.)", "Q$_x$"),
                        position=(323, 122),
                        is_orthogonal=not use_rawdata,
                        reciprocal_space=True,
                    )

                    fig.savefig(
                        detector.savedir
                        + f"monitor_gridded_S{scan_nb}_{nz}_{ny}_{nx}"
                        + binning_comment
                        + ".png"
                    )
                    if flag_interact:
                        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
                        cid = plt.connect("close_event", close_event)
                        fig.waitforbuttonpress()
                        plt.disconnect(cid)
                    plt.close(fig)
                    del tmp_data

        ########################
        # crop/pad/center data #
        ########################
        data, mask, pad_width, q_values, frames_logical = pru.center_fft(
            data=data,
            mask=mask,
            detector=detector,
            frames_logical=frames_logical,
            centering=centering,
            fft_option=center_fft,
            pad_size=pad_size,
            fix_bragg=fix_bragg,
            fix_size=fix_size,
            q_values=q_values,
        )

        starting_frame = [
            pad_width[0],
            pad_width[2],
            pad_width[4],
        ]  # no need to check padded frames
        print("\nPad width:", pad_width)
        nz, ny, nx = data.shape
        print("\nData size after cropping / padding:", nz, ny, nx)

        ##########################################
        # optional masking of zero photon events #
        ##########################################
        if mask_zero_event:
            # mask points when there is no intensity along the whole rocking curve
            # probably dead pixels
            temp_mask = np.zeros((ny, nx))
            temp_mask[np.sum(data, axis=0) == 0] = 1
            mask[np.repeat(temp_mask[np.newaxis, :, :], repeats=nz, axis=0) == 1] = 1
            del temp_mask

        ###########################################
        # save data and mask before alien removal #
        ###########################################
        fig, _, _ = gu.multislices_plot(
            data,
            sum_frames=True,
            scale="log",
            plot_colorbar=True,
            vmin=0,
            title="Data before aliens removal\n",
            is_orthogonal=not use_rawdata,
            reciprocal_space=True,
        )
        if debug:
            plt.savefig(
                detector.savedir + f"data_before_masking_sum_S{scan_nb}_{nz}_{ny}_{nx}_"
                f"{detector.binning[0]}_"
                f"{detector.binning[1]}_{detector.binning[2]}.png"
            )
        if flag_interact:
            fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
            cid = plt.connect("close_event", close_event)
            fig.waitforbuttonpress()
            plt.disconnect(cid)
        plt.close(fig)

        piz, piy, pix = np.unravel_index(data.argmax(), data.shape)
        fig = gu.combined_plots(
            (data[piz, :, :], data[:, piy, :], data[:, :, pix]),
            tuple_sum_frames=False,
            tuple_sum_axis=0,
            tuple_width_v=None,
            tuple_width_h=None,
            tuple_colorbar=True,
            tuple_vmin=0,
            tuple_vmax=np.nan,
            tuple_scale="log",
            tuple_title=("data at max in xy", "data at max in xz", "data at max in yz"),
            is_orthogonal=not use_rawdata,
            reciprocal_space=False,
        )
        if debug:
            plt.savefig(
                detector.savedir
                + f"data_before_masking_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}_"
                f"{detector.binning[1]}_{detector.binning[2]}.png"
            )
        if flag_interact:
            fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
            cid = plt.connect("close_event", close_event)
            fig.waitforbuttonpress()
            plt.disconnect(cid)
        plt.close(fig)

        fig, _, _ = gu.multislices_plot(
            mask,
            sum_frames=True,
            scale="linear",
            plot_colorbar=True,
            vmin=0,
            vmax=(nz, ny, nx),
            title="Mask before aliens removal\n",
            is_orthogonal=not use_rawdata,
            reciprocal_space=True,
        )
        if debug:
            plt.savefig(
                detector.savedir
                + f"mask_before_masking_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}_"
                f"{detector.binning[1]}_{detector.binning[2]}.png"
            )

        if flag_interact:
            fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
            cid = plt.connect("close_event", close_event)
            fig.waitforbuttonpress()
            plt.disconnect(cid)
        plt.close(fig)

        ###############################################
        # save the orthogonalized diffraction pattern #
        ###############################################
        if not use_rawdata and len(q_values) != 0:
            qx = q_values[0]
            qz = q_values[1]
            qy = q_values[2]

            if save_to_vti:
                # save diffraction pattern to vti
                (
                    nqx,
                    nqz,
                    nqy,
                ) = (
                    data.shape
                )  # in nexus z downstream, y vertical / in q z vertical, x downstream
                print("\ndqx, dqy, dqz = ", qx[1] - qx[0], qy[1] - qy[0], qz[1] - qz[0])
                # in nexus z downstream, y vertical / in q z vertical, x downstream
                qx0 = qx.min()
                dqx = (qx.max() - qx0) / nqx
                qy0 = qy.min()
                dqy = (qy.max() - qy0) / nqy
                qz0 = qz.min()
                dqz = (qz.max() - qz0) / nqz

                gu.save_to_vti(
                    filename=os.path.join(
                        detector.savedir, f"S{scan_nb}_ortho_int" + comment + ".vti"
                    ),
                    voxel_size=(dqx, dqz, dqy),
                    tuple_array=data,
                    tuple_fieldnames="int",
                    origin=(qx0, qz0, qy0),
                )

        if flag_interact:
            #############################################
            # remove aliens
            #############################################
            nz, ny, nx = np.shape(data)
            width = 5
            max_colorbar = 5
            flag_mask = False
            flag_aliens = True

            fig_mask, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
                nrows=2, ncols=2, figsize=(12, 6)
            )
            fig_mask.canvas.mpl_disconnect(fig_mask.canvas.manager.key_press_handler_id)
            original_data = np.copy(data)
            original_mask = np.copy(mask)
            frame_index = starting_frame
            ax0.imshow(data[frame_index[0], :, :], vmin=0, vmax=max_colorbar, cmap=my_cmap)
            ax1.imshow(data[:, frame_index[1], :], vmin=0, vmax=max_colorbar, cmap=my_cmap)
            ax2.imshow(data[:, :, frame_index[2]], vmin=0, vmax=max_colorbar, cmap=my_cmap)
            ax3.set_visible(False)
            ax0.axis("scaled")
            ax1.axis("scaled")
            ax2.axis("scaled")
            if not use_rawdata:
                ax0.invert_yaxis()  # detector Y is vertical down
            ax0.set_title(f"XY - Frame {frame_index[0] + 1} / {nz}")
            ax1.set_title(f"XZ - Frame {frame_index[1] + 1} / {ny}")
            ax2.set_title(f"YZ - Frame {frame_index[2] + 1} / {nx}")
            fig_mask.text(
                0.60, 0.30, "m mask ; b unmask ; u next frame ; d previous frame", size=12
            )
            fig_mask.text(
                0.60,
                0.25,
                "up larger ; down smaller ; right darker ; left brighter",
                size=12,
            )
            fig_mask.text(0.60, 0.20, "p plot full image ; q quit", size=12)
            plt.tight_layout()
            plt.connect("key_press_event", press_key)
            fig_mask.set_facecolor(background_plot)
            plt.show()
            del fig_mask, original_data, original_mask

            mask[np.nonzero(mask)] = 1

            fig, _, _ = gu.multislices_plot(
                data,
                sum_frames=True,
                scale="log",
                plot_colorbar=True,
                vmin=0,
                title="Data after aliens removal\n",
                is_orthogonal=not use_rawdata,
                reciprocal_space=True,
            )

            fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
            cid = plt.connect("close_event", close_event)
            fig.waitforbuttonpress()
            plt.disconnect(cid)
            plt.close(fig)

            fig, _, _ = gu.multislices_plot(
                mask,
                sum_frames=True,
                scale="linear",
                plot_colorbar=True,
                vmin=0,
                vmax=(nz, ny, nx),
                title="Mask after aliens removal\n",
                is_orthogonal=not use_rawdata,
                reciprocal_space=True,
            )

            fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
            cid = plt.connect("close_event", close_event)
            fig.waitforbuttonpress()
            plt.disconnect(cid)
            plt.close(fig)

            #############################################
            # define mask
            #############################################
            width = 0
            max_colorbar = 5
            flag_aliens = False
            flag_mask = True
            flag_pause = False  # press x to pause for pan/zoom
            previous_axis = None
            xy = []  # list of points for mask

            fig_mask, ((ax0, ax1), (ax2, ax3)) = plt.subplots(
                nrows=2, ncols=2, figsize=(12, 6)
            )
            fig_mask.canvas.mpl_disconnect(fig_mask.canvas.manager.key_press_handler_id)
            original_data = np.copy(data)
            updated_mask = np.zeros((nz, ny, nx))
            data[mask == 1] = 0  # will appear as grey in the log plot (nan)
            ax0.imshow(
                np.log10(abs(data).sum(axis=0)), vmin=0, vmax=max_colorbar, cmap=my_cmap
            )
            ax1.imshow(
                np.log10(abs(data).sum(axis=1)), vmin=0, vmax=max_colorbar, cmap=my_cmap
            )
            ax2.imshow(
                np.log10(abs(data).sum(axis=2)), vmin=0, vmax=max_colorbar, cmap=my_cmap
            )
            ax3.set_visible(False)
            ax0.axis("scaled")
            ax1.axis("scaled")
            ax2.axis("scaled")
            if not use_rawdata:
                ax0.invert_yaxis()  # detector Y is vertical down
            ax0.set_title("XY")
            ax1.set_title("XZ")
            ax2.set_title("YZ")
            fig_mask.text(
                0.60, 0.45, "click to select the vertices of a polygon mask", size=12
            )
            fig_mask.text(
                0.60, 0.40, "x to pause/resume polygon masking for pan/zoom", size=12
            )
            fig_mask.text(0.60, 0.35, "p plot mask ; r reset current points", size=12)
            fig_mask.text(
                0.60,
                0.30,
                "m square mask ; b unmask ; right darker ; left brighter",
                size=12,
            )
            fig_mask.text(
                0.60, 0.25, "up larger masking box ; down smaller masking box", size=12
            )
            fig_mask.text(0.60, 0.20, "a restart ; q quit", size=12)
            info_text = fig_mask.text(0.60, 0.05, "masking enabled", size=16)
            plt.tight_layout()
            plt.connect("key_press_event", press_key)
            plt.connect("button_press_event", on_click)
            fig_mask.set_facecolor(background_plot)
            plt.show()

            mask[np.nonzero(updated_mask)] = 1
            data = original_data

            del fig_mask, flag_pause, flag_mask, original_data, updated_mask

        mask[np.nonzero(mask)] = 1
        data[mask == 1] = 0

        #############################################
        # mask or median filter isolated empty pixels
        #############################################
        if flag_medianfilter in {"mask_isolated", "interp_isolated"}:
            print("\nFiltering isolated pixels")
            nb_pix = 0
            for idx in range(
                pad_width[0], nz - pad_width[1]
            ):  # filter only frames whith data (not padded)
                data[idx, :, :], processed_pix, mask[idx, :, :] = pru.mean_filter(
                    data=data[idx, :, :],
                    nb_neighbours=medfilt_order,
                    mask=mask[idx, :, :],
                    interpolate=flag_medianfilter,
                    min_count=3,
                    debugging=debug,
                )
                nb_pix += processed_pix
                sys.stdout.write(
                    f"\rImage {idx}, number of filtered pixels: {processed_pix}"
                )
                sys.stdout.flush()
            print("\nTotal number of filtered pixels: ", nb_pix)
        elif flag_medianfilter == "median":  # apply median filter
            print("\nApplying median filtering")
            for idx in range(
                pad_width[0], nz - pad_width[1]
            ):  # filter only frames whith data (not padded)
                data[idx, :, :] = scipy.signal.medfilt2d(data[idx, :, :], [3, 3])
        else:
            print("\nSkipping median filtering")

        ##########################
        # apply photon threshold #
        ##########################
        if photon_threshold != 0:
            mask[data < photon_threshold] = 1
            data[data < photon_threshold] = 0
            print("\nApplying photon threshold < ", photon_threshold)

        ################################################
        # check for nans and infs in the data and mask #
        ################################################
        nz, ny, nx = data.shape
        print("\nData size after masking:", nz, ny, nx)

        data, mask = util.remove_nan(data=data, mask=mask)

        data[mask == 1] = 0

        ####################
        # debugging plots  #
        ####################
        if debug:
            z0, y0, x0 = center_of_mass(data)
            fig, _, _ = gu.multislices_plot(
                data,
                sum_frames=False,
                scale="log",
                plot_colorbar=True,
                vmin=0,
                title="Masked data",
                slice_position=[int(z0), int(y0), int(x0)],
                is_orthogonal=not use_rawdata,
                reciprocal_space=True,
            )
            plt.savefig(
                detector.savedir
                + f"middle_frame_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}_"
                f"{detector.binning[1]}_{detector.binning[2]}" + comment + ".png"
            )
            if not flag_interact:
                plt.close(fig)

            fig, _, _ = gu.multislices_plot(
                data,
                sum_frames=True,
                scale="log",
                plot_colorbar=True,
                vmin=0,
                title="Masked data",
                is_orthogonal=not use_rawdata,
                reciprocal_space=True,
            )
            plt.savefig(
                detector.savedir + f"sum_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}_"
                f"{detector.binning[1]}_{detector.binning[2]}" + comment + ".png"
            )
            if not flag_interact:
                plt.close(fig)

            fig, _, _ = gu.multislices_plot(
                mask,
                sum_frames=True,
                scale="linear",
                plot_colorbar=True,
                vmin=0,
                vmax=(nz, ny, nx),
                title="Mask",
                is_orthogonal=not use_rawdata,
                reciprocal_space=True,
            )
            plt.savefig(
                detector.savedir + f"mask_S{scan_nb}_{nz}_{ny}_{nx}_{detector.binning[0]}_"
                f"{detector.binning[1]}_{detector.binning[2]}" + comment + ".png"
            )
            if not flag_interact:
                plt.close(fig)

        ##################################################
        # bin the stacking axis if needed, the detector  #
        # plane was already binned when loading the data #
        ##################################################
        if (
            detector.binning[0] != 1 and not reload_orthogonal
        ):  # data was already binned for reload_orthogonal
            data = util.bin_data(data, (detector.binning[0], 1, 1), debugging=False)
            mask = util.bin_data(mask, (detector.binning[0], 1, 1), debugging=False)
            mask[np.nonzero(mask)] = 1
            if not use_rawdata and len(q_values) != 0:
                numz = len(qx)
                qx = qx[
                    : numz - (numz % detector.binning[0]) : detector.binning[0]
                ]  # along Z
                del numz
        print("\nData size after binning the stacking dimension:", data.shape)

        ##################################################################
        # final check of the shape to comply with FFT shape requirements #
        ##################################################################
        final_shape = util.smaller_primes(data.shape, maxprime=7, required_dividers=(2,))
        com = tuple(map(lambda x: int(np.rint(x)), center_of_mass(data)))
        crop_center = pu.find_crop_center(
            array_shape=data.shape, crop_shape=final_shape, pivot=com
        )
        data = util.crop_pad(data, output_shape=final_shape, crop_center=crop_center)
        mask = util.crop_pad(mask, output_shape=final_shape, crop_center=crop_center)
        print("\nData size after considering FFT shape requirements:", data.shape)
        nz, ny, nx = data.shape
        comment = f"{comment}_{nz}_{ny}_{nx}" + binning_comment

        ############################
        # save final data and mask #
        ############################
        print("\nSaving directory:", detector.savedir)
        if save_asint:
            data = data.astype(int)
        print("Data type before saving:", data.dtype)
        mask[np.nonzero(mask)] = 1
        mask = mask.astype(int)
        print("Mask type before saving:", mask.dtype)
        if not use_rawdata and len(q_values) != 0:
            if save_to_npz:
                np.savez_compressed(
                    detector.savedir + f"QxQzQy_S{scan_nb}" + comment, qx=qx, qz=qz, qy=qy
                )
            if save_to_mat:
                savemat(detector.savedir + f"S{scan_nb}_qx.mat", {"qx": qx})
                savemat(detector.savedir + f"S{scan_nb}_qz.mat", {"qz": qz})
                savemat(detector.savedir + f"S{scan_nb}_qy.mat", {"qy": qy})
            max_z = data.sum(axis=0).max()
            fig, _, _ = gu.contour_slices(
                data,
                (qx, qz, qy),
                sum_frames=True,
                title="Final data",
                plot_colorbar=True,
                scale="log",
                is_orthogonal=True,
                levels=np.linspace(0, np.ceil(np.log10(max_z)), 150, endpoint=False),
                reciprocal_space=True,
            )
            fig.savefig(
                detector.savedir + f"final_reciprocal_space_S{scan_nb}" + comment + ".png"
            )
            plt.close(fig)

        if save_to_npz:
            np.savez_compressed(
                detector.savedir + f"S{scan_nb}_pynx" + comment, data=data
            )
            np.savez_compressed(
                detector.savedir + f"S{scan_nb}_maskpynx" + comment, mask=mask
            )

        if save_to_mat:
            # save to .mat, the new order is x y z (outboard, vertical up, downstream)
            savemat(
                detector.savedir + f"S{scan_nb}_data.mat",
                {"data": np.moveaxis(data.astype(np.float32), [0, 1, 2], [-1, -2, -3])},
            )
            savemat(
                detector.savedir + f"S{scan_nb}_mask.mat",
                {"data": np.moveaxis(mask.astype(np.int8), [0, 1, 2], [-1, -2, -3])},
            )

        ############################
        # plot final data and mask #
        ############################
        data[np.nonzero(mask)] = 0
        fig, _, _ = gu.multislices_plot(
            data,
            sum_frames=True,
            scale="log",
            plot_colorbar=True,
            vmin=0,
            title="Final data",
            is_orthogonal=not use_rawdata,
            reciprocal_space=True,
        )
        plt.savefig(detector.savedir + f"finalsum_S{scan_nb}" + comment + ".png")
        if not flag_interact:
            plt.close(fig)

        fig, _, _ = gu.multislices_plot(
            mask,
            sum_frames=True,
            scale="linear",
            plot_colorbar=True,
            vmin=0,
            vmax=(nz, ny, nx),
            title="Final mask",
            is_orthogonal=not use_rawdata,
            reciprocal_space=True,
        )
        plt.savefig(detector.savedir + f"finalmask_S{scan_nb}" + comment + ".png")
        if not flag_interact:
            plt.close(fig)

        del data, mask

        if len(scans) > 1:
            plt.close("all")


        # Modify file for phase retrieval
        print("Saving in pynx run ...")

        with open(f"{detector.savedir}pynx_run.txt", "r") as f:
            text_file = f.readlines()
            
            text_file[1] = f"data = \"{detector.savedir}S{scan_nb}_pynx{comment}.npz\"\n"
            text_file[2] = f"mask = \"{detector.savedir}S{scan_nb}_maskpynx{comment}.npz\"\n"

            with open(f"{save_dir}pynx_run.txt", "w") as v:
                new_file_contents = "".join(text_file)
                v.write(new_file_contents)

    print("\nEnd of script")

    plt.switch_backend(
        'module://ipykernel.pylab.backend_inline'
    )


def strain_bcdi(
    scan, 
    root_folder,
    save_dir,
    data_dirname,
    sample_name, 
    comment, 
    sort_method, 
    correlation_threshold,
    original_size, 
    phasing_binning, 
    preprocessing_binning, 
    output_size, 
    keep_size, 
    fix_voxel,
    data_frame,
    ref_axis_q,
    save_frame,
    isosurface_strain,
    strain_method,
    phase_offset,
    phase_offset_origin,
    offset_method,
    centering_method,
    beamline,
    actuators,
    rocking_angle,
    sdd,
    energy,
    beam_direction,
    outofplane_angle,
    inplane_angle,
    tilt_angle,
    sample_offsets,
    specfile_name,
    custom_scan,
    custom_motors,
    detector,
    nb_pixel_x,
    nb_pixel_y,
    pixel_size,
    template_imagefile,
    correct_refraction,
    optical_path_method,
    dispersion,
    absorption,
    threshold_unwrap_refraction,
    simu_flag,
    invert_phase,
    flip_reconstruction,
    phase_ramp_removal,
    threshold_gradient,
    save_raw,
    save_support,
    save,
    debug,
    roll_modes,
    align_axis,
    ref_axis,
    axis_to_align,
    strain_range,
    phase_range,
    grey_background,
    tick_spacing,
    tick_direction,
    tick_length,
    tick_width,
    get_temperature,
    reflection,
    reference_spacing,
    reference_temperature,
    avg_method,
    avg_threshold,
    hwidth,
    apodize_flag,
    apodize_window,
    mu,
    sigma,
    alpha,
    h5_data
    ):
    """
    Interpolate the output of the phase retrieval into an orthonormal frame,
    and calculate the strain component along the direction of the experimental diffusion
    vector q.

    Input: complex amplitude array, output from a phase retrieval program.
    Output: data in an orthonormal frame (laboratory or crystal frame), amp_disp_strain
    array.The disp array should be divided by q to get the displacement (disp = -1*phase
    here).

    Laboratory frame: z downstream, y vertical, x outboard (CXI convention)
    Crystal reciprocal frame: qx downstream, qz vertical, qy outboard
    Detector convention: when out_of_plane angle=0   Y=-y , when in_plane angle=0   X=x

    In arrays, when plotting the first parameter is the row (vertical axis), and the
    second the column (horizontal axis). Therefore the data structure is data[qx, qz,
    qy] for reciprocal space, or data[z, y, x] for real space


    Remember to delete the waitforbuttonpress, root tk and file path
    """

    # Temporary solution

    file_path = h5_data,

    print(file_path)
    ####################
    # Check parameters #
    ####################
    valid_name = "bcdi_strain"
    if simu_flag:
        invert_phase = False
        correct_absorption = 0
        correct_refraction = 0

    if invert_phase:
        phase_fieldname = "disp"
    else:
        phase_fieldname = "phase"

    if fix_voxel:
        if isinstance(fix_voxel, Real):
            fix_voxel = (fix_voxel, fix_voxel, fix_voxel)
        if not isinstance(fix_voxel, Sequence):
            raise TypeError("fix_voxel should be a sequence of three positive numbers")
        if any(val <= 0 for val in fix_voxel):
            raise ValueError(
                "fix_voxel should be a positive number or "
                "a sequence of three positive numbers"
            )

    if actuators is not None and not isinstance(actuators, dict):
        raise TypeError("actuators should be a dictionnary of actuator fieldnames")

    if data_frame not in {"detector", "crystal", "laboratory"}:
        raise ValueError('Uncorrect setting for "data_frame" parameter')
    if data_frame == "detector":
        is_orthogonal = False
    else:
        is_orthogonal = True

    if ref_axis_q not in {"x", "y", "z"}:
        raise ValueError("ref_axis_q should be either 'x', 'y', 'z'")

    if ref_axis not in {"x", "y", "z"}:
        raise ValueError("ref_axis should be either 'x', 'y', 'z'")

    if save_frame not in {"crystal", "laboratory", "lab_flat_sample"}:
        raise ValueError(
            "save_frame should be either 'crystal', 'laboratory' or 'lab_flat_sample'"
        )

    if data_frame == "crystal" and save_frame != "crystal":
        print(
            "data already in the crystal frame before phase retrieval,"
            " it is impossible to come back to the laboratory "
            "frame, parameter 'save_frame' defaulted to 'crystal'"
        )
        save_frame = "crystal"

    if isinstance(output_size, Real):
        output_size = (output_size,) * 3
    valid.valid_container(
        output_size,
        container_types=(tuple, list, np.ndarray),
        length=3,
        allow_none=True,
        item_types=int,
        name=valid_name,
    )
    axis_to_array_xyz = {
        "x": np.array([1, 0, 0]),
        "y": np.array([0, 1, 0]),
        "z": np.array([0, 0, 1]),
    }  # in xyz order

    if isinstance(save_dir, str) and not save_dir.endswith("/"):
        save_dir += "/"

    if len(comment) != 0 and not comment.startswith("_"):
        comment = "_" + comment

    ##################################################
    # parameters that will be saved with the results #
    ##################################################
    params = {
        "isosurface_threshold": isosurface_strain,
        "strain_method": strain_method,
        "phase_offset": phase_offset,
        "phase_offset_origin": phase_offset_origin,
        "centering_method": centering_method,
        "data_frame": data_frame,
        "ref_axis_q": ref_axis_q,
        "save_frame": save_frame,
        "fix_voxel": fix_voxel,
        "original_size": original_size,
        "sample": f"{sample_name}+{scan}",
        "correct_refraction": correct_refraction,
        "optical_path_method": optical_path_method,
        "dispersion": dispersion,
        "time": f"{datetime.now()}",
        "threshold_unwrap_refraction": threshold_unwrap_refraction,
        "invert_phase": invert_phase,
        "phase_ramp_removal": phase_ramp_removal,
        "threshold_gradient": threshold_gradient,
        "tick_spacing_nm": tick_spacing,
        "hwidth": hwidth,
        "apodize_flag": apodize_flag,
        "apodize_window": apodize_window,
        "apod_mu": mu,
        "apod_sigma": sigma,
        "apod_alpha": alpha,
    }
    pretty = pprint.PrettyPrinter(indent=4)

    ###################
    # define colormap #
    ###################
    if grey_background:
        bad_color = "0.7"
    else:
        bad_color = "1.0"  # white background
    colormap = gu.Colormap(bad_color=bad_color)
    my_cmap = colormap.cmap

    #######################
    # Initialize detector #
    #######################
    kwargs = {
        "preprocessing_binning": preprocessing_binning,
        "nb_pixel_x": nb_pixel_x,  # fix to declare a known detector but with less pixels
        # (e.g. one tile HS)
        "nb_pixel_y": nb_pixel_y,  # fix to declare a known detector but with less pixels
        # (e.g. one tile HS)
        "pixel_size": pixel_size,  # to declare the pixel size of the "Dummy" detector
    }

    detector = Detector(
        name=detector,
        template_imagefile=template_imagefile,
        binning=phasing_binning,
        **kwargs,
    )

    ####################################
    # define the experimental geometry #
    ####################################
    # correct the tilt_angle for binning
    tilt_angle = tilt_angle * preprocessing_binning[0] * phasing_binning[0]
    setup = Setup(
        beamline=beamline,
        detector=detector,
        energy=energy,
        outofplane_angle=outofplane_angle,
        inplane_angle=inplane_angle,
        tilt_angle=tilt_angle,
        rocking_angle=rocking_angle,
        distance=sdd,
        sample_offsets=sample_offsets,
        actuators=actuators,
        custom_scan=custom_scan,
        custom_motors=custom_motors,
    )

    ########################################
    # Initialize the paths and the logfile #
    ########################################
    setup.init_paths(
        sample_name=sample_name,
        scan_number=scan,
        root_folder=root_folder,
        save_dir=save_dir,
        specfile_name=specfile_name,
        template_imagefile=template_imagefile,
        create_savedir=True,
    )

    logfile = setup.create_logfile(
        scan_number=scan, root_folder=root_folder, filename=detector.specfile
    )

    #########################################################
    # get the motor position of goniometer circles which    #
    # are below the rocking angle (e.g., chi for eta/omega) #
    #########################################################
    _, setup.grazing_angle, _, _ = setup.diffractometer.goniometer_values(
        logfile=logfile, scan_number=scan, setup=setup
    )

    ###################
    # print instances #
    ###################
    print(f'{"#"*(5+len(str(scan)))}\nScan {scan}\n{"#"*(5+len(str(scan)))}')
    print("\n##############\nSetup instance\n##############")
    pretty.pprint(setup.params)
    print("\n#################\nDetector instance\n#################")
    pretty.pprint(detector.params)

    ################
    # preload data #
    ################
    # root = tk.Tk()
    # root.withdraw()
    # file_path = filedialog.askopenfilenames(
    #     initialdir=detector.scandir,
    #     filetypes=[("HDF5", "*.h5"), ("NPZ", "*.npz"), ("NPY", "*.npy"), ("CXI", "*.cxi")],
    # )
    nbfiles = len(file_path)

    obj, extension = util.load_file(file_path[0])
    if extension == ".h5":
        comment = comment + "_mode"

    print("\n###############\nProcessing data\n###############")
    nz, ny, nx = obj.shape
    print("Initial data size: (", nz, ",", ny, ",", nx, ")")
    if len(original_size) == 0:
        original_size = obj.shape
    print("FFT size before accounting for phasing_binning", original_size)
    original_size = tuple(
        [
            original_size[index] // phasing_binning[index]
            for index in range(len(phasing_binning))
        ]
    )
    print("Binning used during phasing:", detector.binning)
    print("Padding back to original FFT size", original_size)
    obj = util.crop_pad(array=obj, output_shape=original_size)
    nz, ny, nx = obj.shape

    ###########################################################################
    # define range for orthogonalization and plotting - speed up calculations #
    ###########################################################################
    zrange, yrange, xrange = pu.find_datarange(
        array=obj, amplitude_threshold=0.05, keep_size=keep_size
    )

    numz = zrange * 2
    numy = yrange * 2
    numx = xrange * 2
    print(f"Data shape used for orthogonalization and plotting: ({numz}, {numy}, {numx})")

    ####################################################################################
    # find the best reconstruction from the list, based on mean amplitude and variance #
    ####################################################################################
    if nbfiles > 1:
        print("\nTrying to find the best reconstruction\nSorting by ", sort_method)
        sorted_obj = pu.sort_reconstruction(
            file_path=file_path,
            amplitude_threshold=isosurface_strain,
            data_range=(zrange, yrange, xrange),
            sort_method="variance/mean",
        )
    else:
        sorted_obj = [0]

    #######################################
    # load reconstructions and average it #
    #######################################
    avg_obj = np.zeros((numz, numy, numx))
    ref_obj = np.zeros((numz, numy, numx))
    avg_counter = 1
    print("\nAveraging using", nbfiles, "candidate reconstructions")
    for counter, value in enumerate(sorted_obj):
        obj, extension = util.load_file(file_path[value])
        print("\nOpening ", file_path[value])
        params[f"from_file_{counter}"] = file_path[value]

        if flip_reconstruction:
            obj = pu.flip_reconstruction(obj, debugging=True)

        if extension == ".h5":
            centering_method = "do_nothing"  # do not center, data is already cropped
            # just on support for mode decomposition
            # correct a roll after the decomposition into modes in PyNX
            obj = np.roll(obj, roll_modes, axis=(0, 1, 2))
            fig, _, _ = gu.multislices_plot(
                abs(obj),
                sum_frames=True,
                plot_colorbar=True,
                title="1st mode after centering",
            )
            # fig.waitforbuttonpress()
            plt.close(fig)
        # use the range of interest defined above
        obj = util.crop_pad(obj, [2 * zrange, 2 * yrange, 2 * xrange], debugging=False)

        # align with average reconstruction
        if counter == 0:  # the fist array loaded will serve as reference object
            print("This reconstruction will be used as reference.")
            ref_obj = obj

        avg_obj, flag_avg = pu.average_obj(
            avg_obj=avg_obj,
            ref_obj=ref_obj,
            obj=obj,
            support_threshold=0.25,
            correlation_threshold=avg_threshold,
            aligning_option="dft",
            method=avg_method,
            reciprocal_space=False,
            is_orthogonal=is_orthogonal,
            debugging=debug,
        )
        avg_counter = avg_counter + flag_avg

    avg_obj = avg_obj / avg_counter
    if avg_counter > 1:
        print("\nAverage performed over ", avg_counter, "reconstructions\n")
    del obj, ref_obj

    ################
    # unwrap phase #
    ################
    phase, extent_phase = pu.unwrap(
        avg_obj,
        support_threshold=threshold_unwrap_refraction,
        debugging=debug,
        reciprocal_space=False,
        is_orthogonal=is_orthogonal,
    )

    print(
        "Extent of the phase over an extended support (ceil(phase range)) ~ ",
        int(extent_phase),
        "(rad)",
    )
    phase = pru.wrap(phase, start_angle=-extent_phase / 2, range_angle=extent_phase)
    if debug:
        gu.multislices_plot(
            phase,
            width_z=2 * zrange,
            width_y=2 * yrange,
            width_x=2 * xrange,
            plot_colorbar=True,
            title="Phase after unwrap + wrap",
            reciprocal_space=False,
            is_orthogonal=is_orthogonal,
        )

    #############################################
    # phase ramp removal before phase filtering #
    #############################################
    amp, phase, rampz, rampy, rampx = pu.remove_ramp(
        amp=abs(avg_obj),
        phase=phase,
        initial_shape=original_size,
        method="gradient",
        amplitude_threshold=isosurface_strain,
        gradient_threshold=threshold_gradient,
    )
    del avg_obj

    if debug:
        gu.multislices_plot(
            phase,
            width_z=2 * zrange,
            width_y=2 * yrange,
            width_x=2 * xrange,
            plot_colorbar=True,
            title="Phase after ramp removal",
            reciprocal_space=False,
            is_orthogonal=is_orthogonal,
        )

    ########################
    # phase offset removal #
    ########################
    support = np.zeros(amp.shape)
    support[amp > isosurface_strain * amp.max()] = 1
    phase = pu.remove_offset(
        array=phase,
        support=support,
        offset_method=offset_method,
        user_offset=phase_offset,
        offset_origin=phase_offset_origin,
        title="Phase",
        debugging=debug,
    )
    del support

    phase = pru.wrap(obj=phase, start_angle=-extent_phase / 2, range_angle=extent_phase)

    ##############################################################################
    # average the phase over a window or apodize to reduce noise in strain plots #
    ##############################################################################
    if hwidth != 0:
        bulk = pu.find_bulk(
            amp=amp, support_threshold=isosurface_strain, method="threshold"
        )
        # the phase should be averaged only in the support defined by the isosurface
        phase = pu.mean_filter(array=phase, support=bulk, half_width=hwidth)
        del bulk

    if hwidth != 0:
        comment = comment + "_avg" + str(2 * hwidth + 1)

    gridz, gridy, gridx = np.meshgrid(
        np.arange(0, numz, 1), np.arange(0, numy, 1), np.arange(0, numx, 1), indexing="ij"
    )

    phase = (
        phase + gridz * rampz + gridy * rampy + gridx * rampx
    )  # put back the phase ramp otherwise the diffraction
    # pattern will be shifted and the prtf messed up

    if apodize_flag:
        amp, phase = pu.apodize(
            amp=amp,
            phase=phase,
            initial_shape=original_size,
            window_type=apodize_window,
            sigma=sigma,
            mu=mu,
            alpha=alpha,
            is_orthogonal=is_orthogonal,
            debugging=True,
        )
        comment = comment + "_apodize_" + apodize_window

    ################################################################
    # save the phase with the ramp for PRTF calculations,          #
    # otherwise the object will be misaligned with the measurement #
    ################################################################
    np.savez_compressed(
        detector.savedir + "S" + str(scan) + "_avg_obj_prtf" + comment,
        obj=amp * np.exp(1j * phase),
    )

    ####################################################
    # remove again phase ramp before orthogonalization #
    ####################################################
    phase = phase - gridz * rampz - gridy * rampy - gridx * rampx

    avg_obj = amp * np.exp(1j * phase)  # here the phase is again wrapped in [-pi pi[

    del amp, phase, gridz, gridy, gridx, rampz, rampy, rampx

    ######################
    # centering of array #
    ######################
    if centering_method == "max":
        avg_obj = pu.center_max(avg_obj)
        # shift based on max value,
        # required if it spans across the edge of the array before COM
    elif centering_method == "com":
        avg_obj = pu.center_com(avg_obj)
    elif centering_method == "max_com":
        avg_obj = pu.center_max(avg_obj)
        avg_obj = pu.center_com(avg_obj)

    #######################
    #  save support & vti #
    #######################
    if (
        save_support
    ):  # to be used as starting support in phasing, hence still in the detector frame
        support = np.zeros((numz, numy, numx))
        support[abs(avg_obj) / abs(avg_obj).max() > 0.01] = 1
        # low threshold because support will be cropped by shrinkwrap during phasing
        np.savez_compressed(
            detector.savedir + "S" + str(scan) + "_support" + comment, obj=support
        )
        del support

    if save_raw:
        np.savez_compressed(
            detector.savedir + "S" + str(scan) + "_raw_amp-phase" + comment,
            amp=abs(avg_obj),
            phase=np.angle(avg_obj),
        )

        # voxel sizes in the detector frame
        voxel_z, voxel_y, voxel_x = setup.voxel_sizes_detector(
            array_shape=original_size,
            tilt_angle=tilt_angle,
            pixel_x=detector.pixelsize_x,
            pixel_y=detector.pixelsize_y,
            verbose=True,
        )
        # save raw amp & phase to VTK
        # in VTK, x is downstream, y vertical, z inboard, thus need to flip the last axis
        gu.save_to_vti(
            filename=os.path.join(
                detector.savedir, "S" + str(scan) + "_raw_amp-phase" + comment + ".vti"
            ),
            voxel_size=(voxel_z, voxel_y, voxel_x),
            tuple_array=(abs(avg_obj), np.angle(avg_obj)),
            tuple_fieldnames=("amp", "phase"),
            amplitude_threshold=0.01,
        )

    #########################################################
    # calculate q of the Bragg peak in the laboratory frame #
    #########################################################
    q_lab = (
        setup.q_laboratory
    )  # (1/A), in the laboratory frame z downstream, y vertical, x outboard
    qnorm = np.linalg.norm(q_lab)
    q_lab = q_lab / qnorm

    angle = simu.angle_vectors(
        ref_vector=[q_lab[2], q_lab[1], q_lab[0]], test_vector=axis_to_array_xyz[ref_axis_q]
    )
    print(
        f"\nNormalized diffusion vector in the laboratory frame (z*, y*, x*): "
        f"({q_lab[0]:.4f} 1/A, {q_lab[1]:.4f} 1/A, {q_lab[2]:.4f} 1/A)"
    )

    planar_dist = 2 * np.pi / qnorm  # qnorm should be in angstroms
    print(f"Wavevector transfer: {qnorm:.4f} 1/A")
    print(f"Atomic planar distance: {planar_dist:.4f} A")
    print(f"\nAngle between q_lab and {ref_axis_q} = {angle:.2f} deg")
    if debug:
        print(
            f"Angle with y in zy plane = {np.arctan(q_lab[0]/q_lab[1])*180/np.pi:.2f} deg"
        )
        print(
            f"Angle with y in xy plane = {np.arctan(-q_lab[2]/q_lab[1])*180/np.pi:.2f} deg"
        )
        print(
            f"Angle with z in xz plane = {180+np.arctan(q_lab[2]/q_lab[0])*180/np.pi:.2f} "
            "deg\n"
        )

    planar_dist = planar_dist / 10  # switch to nm

    #######################
    #  orthogonalize data #
    #######################
    print("\nShape before orthogonalization", avg_obj.shape)
    if data_frame == "detector":
        if debug:
            phase, _ = pu.unwrap(
                avg_obj,
                support_threshold=threshold_unwrap_refraction,
                debugging=True,
                reciprocal_space=False,
                is_orthogonal=False,
            )
            gu.multislices_plot(
                phase,
                width_z=2 * zrange,
                width_y=2 * yrange,
                width_x=2 * xrange,
                sum_frames=False,
                plot_colorbar=True,
                reciprocal_space=False,
                is_orthogonal=False,
                title="unwrapped phase before orthogonalization",
            )
            del phase

        obj_ortho, voxel_size = setup.ortho_directspace(
            arrays=avg_obj,
            q_com=np.array([q_lab[2], q_lab[1], q_lab[0]]),
            initial_shape=original_size,
            voxel_size=fix_voxel,
            reference_axis=axis_to_array_xyz[ref_axis_q],
            fill_value=0,
            debugging=True,
            title="amplitude",
        )

    else:  # data already orthogonalized using xrayutilities
        # or the linearized transformation matrix
        obj_ortho = avg_obj
        try:
            print("Select the file containing QxQzQy")
            file_path = filedialog.askopenfilename(
                title="Select the file containing QxQzQy",
                initialdir=detector.savedir,
                filetypes=[("NPZ", "*.npz")],
            )
            npzfile = np.load(file_path)
            qx = npzfile["qx"]
            qy = npzfile["qy"]
            qz = npzfile["qz"]
        except FileNotFoundError:
            raise FileNotFoundError(
                "q values not provided, the voxel size cannot be calculated"
            )
        dy_real = (
            2 * np.pi / abs(qz.max() - qz.min()) / 10
        )  # in nm qz=y in nexus convention
        dx_real = (
            2 * np.pi / abs(qy.max() - qy.min()) / 10
        )  # in nm qy=x in nexus convention
        dz_real = (
            2 * np.pi / abs(qx.max() - qx.min()) / 10
        )  # in nm qx=z in nexus convention
        print(
            f"direct space voxel size from q values: ({dz_real:.2f} nm,"
            f" {dy_real:.2f} nm, {dx_real:.2f} nm)"
        )
        if fix_voxel:
            voxel_size = fix_voxel
            print(f"Direct space pixel size for the interpolation: {voxel_size} (nm)")
            print("Interpolating...\n")
            obj_ortho = pu.regrid(
                array=obj_ortho,
                old_voxelsize=(dz_real, dy_real, dx_real),
                new_voxelsize=voxel_size,
            )
        else:
            # no need to interpolate
            voxel_size = dz_real, dy_real, dx_real  # in nm

        if data_frame == "laboratory":  # the object must be rotated into the crystal frame
            # before the strain calculation
            print("Rotating the object in the crystal frame for the strain calculation")

            amp, phase = util.rotate_crystal(
                arrays=(abs(obj_ortho), np.angle(obj_ortho)),
                is_orthogonal=True,
                reciprocal_space=False,
                voxel_size=voxel_size,
                debugging=(True, False),
                axis_to_align=q_lab[::-1],
                reference_axis=axis_to_array_xyz[ref_axis_q],
                title=("amp", "phase"),
            )

            obj_ortho = amp * np.exp(
                1j * phase
            )  # here the phase is again wrapped in [-pi pi[
            del amp, phase

    del avg_obj

    ######################################################
    # center the object (centering based on the modulus) #
    ######################################################
    print("\nCentering the crystal")
    obj_ortho = pu.center_com(obj_ortho)

    ####################
    # Phase unwrapping #
    ####################
    print("\nPhase unwrapping")
    phase, extent_phase = pu.unwrap(
        obj_ortho,
        support_threshold=threshold_unwrap_refraction,
        debugging=True,
        reciprocal_space=False,
        is_orthogonal=True,
    )
    amp = abs(obj_ortho)
    del obj_ortho

    #############################################
    # invert phase: -1*phase = displacement * q #
    #############################################
    if invert_phase:
        phase = -1 * phase

    ########################################
    # refraction and absorption correction #
    ########################################
    if correct_refraction:  # or correct_absorption:
        bulk = pu.find_bulk(
            amp=amp,
            support_threshold=threshold_unwrap_refraction,
            method=optical_path_method,
            debugging=debug,
        )

        kin = setup.incident_wavevector
        kout = setup.exit_wavevector
        # kin and kout were calculated in the laboratory frame,
        # but after the geometric transformation of the crystal, this
        # latter is always in the crystal frame (for simpler strain calculation).
        # We need to transform kin and kout back
        # into the crystal frame (also, xrayutilities output is in crystal frame)
        kin = util.rotate_vector(
            vectors=[kin[2], kin[1], kin[0]],
            axis_to_align=axis_to_array_xyz[ref_axis_q],
            reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
        )
        kout = util.rotate_vector(
            vectors=[kout[2], kout[1], kout[0]],
            axis_to_align=axis_to_array_xyz[ref_axis_q],
            reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
        )

        # calculate the optical path of the incoming wavevector
        path_in = pu.get_opticalpath(
            support=bulk, direction="in", k=kin, debugging=debug
        )  # path_in already in nm

        # calculate the optical path of the outgoing wavevector
        path_out = pu.get_opticalpath(
            support=bulk, direction="out", k=kout, debugging=debug
        )  # path_our already in nm

        optical_path = path_in + path_out
        del path_in, path_out

        if correct_refraction:
            phase_correction = (
                2 * np.pi / (1e9 * setup.wavelength) * dispersion * optical_path
            )
            phase = phase + phase_correction

            gu.multislices_plot(
                np.multiply(phase_correction, bulk),
                width_z=2 * zrange,
                width_y=2 * yrange,
                width_x=2 * xrange,
                sum_frames=False,
                plot_colorbar=True,
                vmin=0,
                vmax=np.nan,
                title="Refraction correction on the support",
                is_orthogonal=True,
                reciprocal_space=False,
            )
        correct_absorption = False
        if correct_absorption:
            # TODO: it is correct to compensate also
            #  the X-ray absorption in the reconstructed modulus?
            amp_correction = np.exp(
                2 * np.pi / (1e9 * setup.wavelength) * absorption * optical_path
            )
            amp = amp * amp_correction

            gu.multislices_plot(
                np.multiply(amp_correction, bulk),
                width_z=2 * zrange,
                width_y=2 * yrange,
                width_x=2 * xrange,
                sum_frames=False,
                plot_colorbar=True,
                vmin=1,
                vmax=1.1,
                title="Absorption correction on the support",
                is_orthogonal=True,
                reciprocal_space=False,
            )

        del bulk, optical_path

    ##############################################
    # phase ramp and offset removal (mean value) #
    ##############################################
    print("\nPhase ramp removal")
    amp, phase, _, _, _ = pu.remove_ramp(
        amp=amp,
        phase=phase,
        initial_shape=original_size,
        method=phase_ramp_removal,
        amplitude_threshold=isosurface_strain,
        gradient_threshold=threshold_gradient,
        debugging=debug,
    )

    ########################
    # phase offset removal #
    ########################
    print("\nPhase offset removal")
    support = np.zeros(amp.shape)
    support[amp > isosurface_strain * amp.max()] = 1
    phase = pu.remove_offset(
        array=phase,
        support=support,
        offset_method=offset_method,
        user_offset=phase_offset,
        offset_origin=phase_offset_origin,
        title="Orthogonal phase",
        debugging=debug,
        reciprocal_space=False,
        is_orthogonal=True,
    )
    del support

    # Wrap the phase around 0 (no more offset)
    phase = pru.wrap(obj=phase, start_angle=-extent_phase / 2, range_angle=extent_phase)

    ################################################################
    # calculate the strain depending on which axis q is aligned on #
    ################################################################
    print(f"\nCalculation of the strain along {ref_axis_q}")
    strain = pu.get_strain(
        phase=phase,
        planar_distance=planar_dist,
        voxel_size=voxel_size,
        reference_axis=ref_axis_q,
        extent_phase=extent_phase,
        method=strain_method,
        debugging=debug,
    )

    ################################################
    # optionally rotates back the crystal into the #
    # laboratory frame (for debugging purpose)     #
    ################################################
    q_final = None
    if save_frame in {"laboratory", "lab_flat_sample"}:
        comment = comment + "_labframe"
        print("\nRotating back the crystal in laboratory frame")
        amp, phase, strain = util.rotate_crystal(
            arrays=(amp, phase, strain),
            axis_to_align=axis_to_array_xyz[ref_axis_q],
            voxel_size=voxel_size,
            is_orthogonal=True,
            reciprocal_space=False,
            reference_axis=[q_lab[2], q_lab[1], q_lab[0]],
            debugging=(True, False, False),
            title=("amp", "phase", "strain"),
        )
        # q_lab is already in the laboratory frame
        q_final = q_lab

    if save_frame == "lab_flat_sample":
        comment = comment + "_flat"
        print("\nSending sample stage circles to 0")
        sample_angles = setup.diffractometer.goniometer_values(
            logfile=logfile, scan_number=scan, setup=setup, stage_name="sample"
        )
        (amp, phase, strain), q_final = setup.diffractometer.flatten_sample(
            arrays=(amp, phase, strain),
            voxel_size=voxel_size,
            angles=sample_angles,
            q_com=q_lab[::-1],  # q_com needs to be in xyz order
            is_orthogonal=True,
            reciprocal_space=False,
            rocking_angle=rocking_angle,
            debugging=(True, False, False),
            title=("amp", "phase", "strain"),
        )
    if save_frame == "crystal":
        # rotate also q_lab to have it along ref_axis_q,
        # as a cross-checkm, vectors needs to be in xyz order
        comment = comment + "_crystalframe"
        q_final = util.rotate_vector(
            vectors=q_lab[::-1],
            axis_to_align=axis_to_array_xyz[ref_axis_q],
            reference_axis=q_lab[::-1],
        )

    ###############################################
    # rotates the crystal e.g. for easier slicing #
    # of the result along a particular direction  #
    ###############################################
    # typically this is an inplane rotation, q should stay aligned with the axis
    # along which the strain was calculated
    if align_axis:
        print("\nRotating arrays for visualization")
        amp, phase, strain = util.rotate_crystal(
            arrays=(amp, phase, strain),
            reference_axis=axis_to_array_xyz[ref_axis],
            axis_to_align=axis_to_align,
            voxel_size=voxel_size,
            debugging=(True, False, False),
            is_orthogonal=True,
            reciprocal_space=False,
            title=("amp", "phase", "strain"),
        )
        # rotate q accordingly, vectors needs to be in xyz order
        q_final = util.rotate_vector(
            vectors=q_final[::-1],
            axis_to_align=axis_to_array_xyz[ref_axis],
            reference_axis=axis_to_align,
        )

    print(f"\nq_final = ({q_final[0]:.4f} 1/A, {q_final[1]:.4f} 1/A, {q_final[2]:.4f} 1/A)")

    ##############################################
    # pad array to fit the output_size parameter #
    ##############################################
    if output_size is not None:
        amp = util.crop_pad(array=amp, output_shape=output_size)
        phase = util.crop_pad(array=phase, output_shape=output_size)
        strain = util.crop_pad(array=strain, output_shape=output_size)
    print(f"\nFinal data shape: {amp.shape}")

    ######################
    # save result to vtk #
    ######################
    print(
        f"\nVoxel size: ({voxel_size[0]:.2f} nm, {voxel_size[1]:.2f} nm,"
        f" {voxel_size[2]:.2f} nm)"
    )
    bulk = pu.find_bulk(amp=amp, support_threshold=isosurface_strain, method="threshold")
    if save:
        params["comment"] = comment
        np.savez_compressed(
            f"{detector.savedir}S{scan}_amp{phase_fieldname}strain{comment}",
            amp=amp,
            phase=phase,
            bulk=bulk,
            strain=strain,
            q_com=q_final,
            voxel_sizes=voxel_size,
            detector=detector.params,
            setup=setup.params,
            params=params,
        )

        # save results in hdf5 file
        with h5py.File(
            f"{detector.savedir}S{scan}_amp{phase_fieldname}strain{comment}.h5", "w"
        ) as hf:
            out = hf.create_group("output")
            par = hf.create_group("params")
            out.create_dataset("amp", data=amp)
            out.create_dataset("bulk", data=bulk)
            out.create_dataset("phase", data=phase)
            out.create_dataset("strain", data=strain)
            out.create_dataset("q_com", data=q_final)
            out.create_dataset("voxel_sizes", data=voxel_size)
            par.create_dataset("detector", data=str(detector.params))
            par.create_dataset("setup", data=str(setup.params))
            par.create_dataset("parameters", data=str(params))

        # save amp & phase to VTK
        # in VTK, x is downstream, y vertical, z inboard, thus need to flip the last axis
        gu.save_to_vti(
            filename=os.path.join(
                detector.savedir,
                "S" + str(scan) + "_amp-" + phase_fieldname + "-strain" + comment + ".vti",
            ),
            voxel_size=voxel_size,
            tuple_array=(amp, bulk, phase, strain),
            tuple_fieldnames=("amp", "bulk", phase_fieldname, "strain"),
            amplitude_threshold=0.01,
        )


    ######################################
    # estimate the volume of the crystal #
    ######################################
    amp = amp / amp.max()
    temp_amp = np.copy(amp)
    temp_amp[amp < isosurface_strain] = 0
    temp_amp[np.nonzero(temp_amp)] = 1
    volume = temp_amp.sum() * reduce(lambda x, y: x * y, voxel_size)  # in nm3
    del temp_amp

    ##############################
    # plot slices of the results #
    ##############################
    pixel_spacing = [tick_spacing / vox for vox in voxel_size]
    print(
        "\nPhase extent without / with thresholding the modulus "
        f"(threshold={isosurface_strain}): {phase.max()-phase.min():.2f} rad, "
        f"{phase[np.nonzero(bulk)].max()-phase[np.nonzero(bulk)].min():.2f} rad"
    )
    piz, piy, pix = np.unravel_index(phase.argmax(), phase.shape)
    print(
        f"phase.max() = {phase[np.nonzero(bulk)].max():.2f} at voxel ({piz}, {piy}, {pix})"
    )
    strain[bulk == 0] = np.nan
    phase[bulk == 0] = np.nan

    # plot the slice at the maximum phase
    gu.combined_plots(
        (phase[piz, :, :], phase[:, piy, :], phase[:, :, pix]),
        tuple_sum_frames=False,
        tuple_sum_axis=0,
        tuple_width_v=None,
        tuple_width_h=None,
        tuple_colorbar=True,
        tuple_vmin=np.nan,
        tuple_vmax=np.nan,
        tuple_title=("phase at max in xy", "phase at max in xz", "phase at max in yz"),
        tuple_scale="linear",
        cmap=my_cmap,
        is_orthogonal=True,
        reciprocal_space=False,
    )

    # bulk support
    fig, _, _ = gu.multislices_plot(
        bulk,
        sum_frames=False,
        title="Orthogonal bulk",
        vmin=0,
        vmax=1,
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.text(0.60, 0.45, "Scan " + str(scan), size=20)
    fig.text(
        0.60, 0.40, "Bulk - isosurface=" + str("{:.2f}".format(isosurface_strain)), size=20
    )
    plt.pause(0.1)
    if save:
        plt.savefig(detector.savedir + "S" + str(scan) + "_bulk" + comment + ".png")

    # amplitude
    fig, _, _ = gu.multislices_plot(
        amp,
        sum_frames=False,
        title="Normalized orthogonal amp",
        vmin=0,
        vmax=1,
        tick_direction=tick_direction,
        tick_width=tick_width,
        tick_length=tick_length,
        pixel_spacing=pixel_spacing,
        plot_colorbar=True,
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.text(0.60, 0.45, f"Scan {scan}", size=20)
    fig.text(
        0.60,
        0.40,
        f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)",
        size=20,
    )
    fig.text(0.60, 0.35, f"Ticks spacing={tick_spacing} nm", size=20)
    fig.text(0.60, 0.30, f"Volume={int(volume)} nm3", size=20)
    fig.text(0.60, 0.25, "Sorted by " + sort_method, size=20)
    fig.text(0.60, 0.20, f"correlation threshold={correlation_threshold}", size=20)
    fig.text(0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20)
    fig.text(0.60, 0.10, f"Planar distance={planar_dist:.5f} nm", size=20)
    if get_temperature:
        temperature = pu.bragg_temperature(
            spacing=planar_dist * 10,
            reflection=reflection,
            spacing_ref=reference_spacing,
            temperature_ref=reference_temperature,
            use_q=False,
            material="Pt",
        )
        fig.text(0.60, 0.05, f"Estimated T={temperature} C", size=20)
    if save:
        plt.savefig(detector.savedir + f"S{scan}_amp" + comment + ".png")

    # amplitude histogram
    fig, ax = plt.subplots(1, 1)
    ax.hist(amp[amp > 0.05 * amp.max()].flatten(), bins=250)
    ax.set_ylim(bottom=1)
    ax.tick_params(
        labelbottom=True,
        labelleft=True,
        direction="out",
        length=tick_length,
        width=tick_width,
    )
    ax.spines["right"].set_linewidth(1.5)
    ax.spines["left"].set_linewidth(1.5)
    ax.spines["top"].set_linewidth(1.5)
    ax.spines["bottom"].set_linewidth(1.5)
    fig.savefig(detector.savedir + f"S{scan}_histo_amp" + comment + ".png")

    # phase
    fig, _, _ = gu.multislices_plot(
        phase,
        sum_frames=False,
        title="Orthogonal displacement",
        vmin=-phase_range,
        vmax=phase_range,
        tick_direction=tick_direction,
        cmap=my_cmap,
        tick_width=tick_width,
        tick_length=tick_length,
        pixel_spacing=pixel_spacing,
        plot_colorbar=True,
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.text(0.60, 0.30, f"Scan {scan}", size=20)
    fig.text(
        0.60,
        0.25,
        f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)",
        size=20,
    )
    fig.text(0.60, 0.20, f"Ticks spacing={tick_spacing} nm", size=20)
    fig.text(0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20)
    if hwidth > 0:
        fig.text(0.60, 0.10, f"Averaging over {2*hwidth+1} pixels", size=20)
    else:
        fig.text(0.60, 0.10, "No phase averaging", size=20)
    if save:
        plt.savefig(detector.savedir + f"S{scan}_displacement" + comment + ".png")

    # strain
    fig, _, _ = gu.multislices_plot(
        strain,
        sum_frames=False,
        title="Orthogonal strain",
        vmin=-strain_range,
        vmax=strain_range,
        tick_direction=tick_direction,
        tick_width=tick_width,
        tick_length=tick_length,
        plot_colorbar=True,
        cmap=my_cmap,
        pixel_spacing=pixel_spacing,
        is_orthogonal=True,
        reciprocal_space=False,
    )
    fig.text(0.60, 0.30, f"Scan {scan}", size=20)
    fig.text(
        0.60,
        0.25,
        f"Voxel size=({voxel_size[0]:.1f}, {voxel_size[1]:.1f}, {voxel_size[2]:.1f}) (nm)",
        size=20,
    )
    fig.text(0.60, 0.20, f"Ticks spacing={tick_spacing} nm", size=20)
    fig.text(0.60, 0.15, f"average over {avg_counter} reconstruction(s)", size=20)
    if hwidth > 0:
        fig.text(0.60, 0.10, f"Averaging over {2*hwidth+1} pixels", size=20)
    else:
        fig.text(0.60, 0.10, "No phase averaging", size=20)
    if save:
        plt.savefig(detector.savedir + f"S{scan}_strain" + comment + ".png")

    print("\nEnd of script")
    plt.show()
