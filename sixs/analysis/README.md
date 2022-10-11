# Each module here targets specific data acquired at SixS

## reciprocal_space
Analysis of Crystal Truncation Rods (CTR) and in-plane maps that have been
created with binoculars.

### Plotting a map:

```python
from sixs.analysis import reciprocal_space as rs

map_450 = rs.Map("T450C/Map_hkl_surf_or_2905-2953.hdf5")
map_450.project_data(projection_axis="L")

circles_450_third_cycle_G = [
    (0.675, -0.955, 0.02, "m", 0.8),
    (0.640, -0.920, 0.02, "m", 0.8),
    (0.68, -1.031, 0.02, "m", 0.8),

    (0.955, -0.675, 0.02, "m", 0.8),
    (0.920, -0.640, 0.02, "m", 0.8),
    (1.031, -0.68, 0.02, "m", 0.8),

    (0.36, -0.92, 0.02, "g", 0.8),
    (0.325, -0.955, 0.02, "g", 0.8),
    (0.325, -1.032, 0.02, "g", 0.8),

    (0.92, -0.36, 0.02, "g", 0.8),
    (0.955, -0.325, 0.02, "g", 0.8),
    (1.032, -0.325, 0.02, "g", 0.8),

    (0.32, -0.042, 0.02, "b", 0.8),
    (0.32, 0.035, 0.02, "b", 0.8),

    (0.042, -0.32, 0.02, "b", 0.8),
    (-0.035, -0.32, 0.02, "b", 0.8),

    (1.31, 0.035, 0.02, "r", 0.8),
    (1.31, -0.04, 0.02, "r", 0.8),

    (-0.035, -1.31, 0.02, "r", 0.8),
    (0.04, -1.31, 0.02, "r", 0.8),
]

lines = [
    (0, 0, 2, -2, "w", 0.8),
]

map_450.plot_map(
    vmin=1,
    vmax=200,
    grid=True,
    lines=lines,
    circles=circles_450_third_cycle_G,
)
```

```
>>> ############################################################
    Data shape: (27, 595, 519)
    	HKL data: True
    	QxQy data: False
    	QparQper data: False
    	Qphi data: False
    	Qindex: False
    ###########################################################
```
![image](https://user-images.githubusercontent.com/51970962/195026398-76fa7a75-285f-4306-b49e-e29cecd98833.png)


### Integrating a CTR directly on an hdf5 file
```python
test_ctr = rs.CTR()

test_ctr.integrate_CTR(
    folder="T450C/",
    scan_indices=[1409],
    save_name="test_integration",
    glob_string_match="*res.hdf5",
    interpol_step=False,
    CTR_width_H=0.02,
    CTR_width_K=0.02,
    background_width_H=0.01,
    background_width_K=0.01,
    HK_peak=[2.01,1.99],
    center_background=[2.01,1.99],
    verbose=True,
)
```

```
>>>
    ###########################################################
    Could not find configuration file. Defaulted to ammonia configuration.
    Loaded configuration file.
    ###########################################################

    ###########################################################
    Working on the following files:
    	 scan_1377-1409_low_res.hdf5
    ###########################################################

    ###########################################################
    Range in H: [1.99 : 2.03]
    Range in K: [1.97 : 2.01]
    ###########################################################

    ###########################################################
    Background range in H: [1.98 : 2.04]
    Background range in K: [1.96 : 2.02]
    ###########################################################

    ###########################################################
    Finding smallest common range in L
    Depends on the config file in binoculars-process.
    ###########################################################

    ###########################################################
    Opening file scan_1377-1409_low_res.hdf5 ...
    	Axis number, range and stepsize in H: [0.000: 1.835: 2.135]
    	Axis number, range and stepsize in K: [1.000: 1.830: 2.165]
    	Axis number, range and stepsize in L: [2.000: 0.000: 4.350]
    ###########################################################

    ###########################################################
    Smallest common range in L is [0.0 : 4.35]
    ###########################################################

    ###########################################################
    Opening file scan_1377-1409_low_res.hdf5 ...
    Data ROI (H, K): [1.99, 2.03, 1.97, 2.01] ; [31, 39, 28, 36]
    Background ROI (H, K): [1.98, 2.04, 1.96, 2.02] ; [29, 41, 26, 38]
    ###########################################################
```
![image](https://user-images.githubusercontent.com/51970962/195029370-9eaabf2c-78fd-4d5e-a0be-1baf0238aff0.png)

### Loading a CTR integrated via binoculars-fitaid

```python
scan_indices_2_0 = [
    1409,
    1670,
    2023,
    2318,
    2605,
    2802,
    2988
]
label_2_0 = {
    1409: "Argon",
    1670: "O2",
    # Sputter annealing here
    2023: "CondE",
    2318: "CondB",
    2605: "CondA",
    2802: "Argon",
    2988: "CondG",
}

save_name_2_0 = "T450_ctr_evolution_2_0.npy"

ctr_2_0 = rs.CTR()
ctr_2_0.load_fitaid_data(
    folder=folder,
    scan_indices=scan_indices_2_0,
    save_name=save_name_2_0,
    verbose=False,
)
```
### Plotting different CTR together
```python
ctr_2_0.plot_CTR(
    numpy_array=folder + save_name_2_0,
    scan_indices=scan_indices_2_0,
    line_plot=False,
    marker="x",
    labels=label_2_0,
    title="CTR perpendicular to [2, 0] node"
)
```
![image](https://user-images.githubusercontent.com/51970962/195031221-686fb1b2-4522-44c4-b27c-3c3b0ad555b2.png)

## reflectivity
Analysis of reflectivity data, also includes fitting routines.

## xcat
Analysis of gas products / reactants evolution, controlled by the mass flow controller and the mass spetrometer at the beamline.

![mass_flow](https://user-images.githubusercontent.com/51970962/150782601-01500902-614c-4bd3-bfed-7ea41dfe1cc8.png)