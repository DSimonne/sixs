```bash
    data=data.npy: name of the data file including the 3D observed intensity.
                   recognized formats include .npy, .npz (if several arrays are included iobs, 
                   should be named 'data' or 'iobs'), .tif or .tiff 
                   (assumes a multiframe tiff image), or .cxi (hdf5).
                   [mandatory unless another beamline-specific method to import data is used]
                   
    detector_distance=0.7: detector distance in meters
    
    pixel_size_data=55e-6: deprecated, use pixel_size_detector instead

    pixel_size_detector=55e-6: pixel size of the supplied data (detector pixel size)
    
    wavelength=1.5e-10: wavelength in meters
    
    verbose=20: the run will print and optionally plot every 'verbose' cycles
    
    live_plot: if used as keyword (or live_plot=True in a parameters file), a live plot 
               will be shown  every 'verbose' cycle. If an integer number N is given, 
               display every N cycle.
    
    gpu=Titan: name of the gpu to use [optional, by default the fastest available will be used]
    
    auto_center_resize: if used (command-line keyword) or =True, the input data will be centered 
                        and cropped  so that the size of the array is compatible with the (GPU) 
                        FFT library used. If 'roi' is used, centering is based on ROI. 
                        [default=False]
    
    roi=0,120,0,235,0,270: set the region-of-interest to be used, with min,max given along each 
                           dimension in python fashion (min included, max excluded), for the 2 or 3
                           axis. ROI coordinates should be indicated before any rebin is done.
                           Note that if 'auto_center_resize' is used, the ROI may still be shrunk
                           to obtain an array size compatible with the FFT library used. Similarly 
                           it will be shrunk if 'max_size' is used but ROI size is larger.
                           Other example: roi=0,-1,300-356,300+256,500-256,500+256
                           [default=None]
                        
    nb_run=1: number of times to run the optimization [default=1]
    
    nb_run_keep: number of best run results to keep, according to likelihood statistics. This is only useful
                 associated with nb_run [default: keep all run results]
    
    data2cxi: if used as keyword (or data2cxi=True in a parameters file), convert the original 
              data to CXI(HDF5)  format. Will be saved to file 'data.cxi', or if a data file
              has been supplied (e.g. data57.npz), to the same file with extension .cxi.
              
    output_format='cxi': choose the output format for the final object and support.
                         Other possible choice: 'npz', 'none'
                         [Default='cxi']
    
    note='This dataset was measure... Citation: Journal of coherent imaging (2018), 729...':
         Optional text note which will be saved as a note in the output CXI file 
         (and also for data2cxi).
         
    instrument='ESRF-idxx': the name of the beamline/instrument used for data collection
                            [default: depends on the script actually called]
                            
    sample_name='GaN nanowire': optional name for the sample
                         
    mask=zero: mask for the diffraction data. If 'zero', all pixels with iobs <= 0 will be masked.
              If 'negative', all pixels with iobs < 0 will be masked. 
              If 'maxipix', the maxipix gaps will be masked.
              Other possibilities: give a filename for the mask file and  import mask from .npy, .npz, .edf, .mat.
              (the first available array will be used if multiple are present) file.
              Pixels = 0 are valid, > 0 are masked. If the mask is 2D
              and the data 3D, the mask is repeated for all frames along the first dimension (depth).
              [default=None, no mask]
    
    iobs_saturation=1e6: saturation value for the observed intensity. Any pixel above this intensity will be masked
                         [default: no saturation value]
    
    zero_mask: by default masked pixels are free and keep the calculated intensities during HIO, RAAR, ER and CF cycles.
               Setting this flag will force all masked pixels to zero intensity. This can be more stable with a large 
               number of masked pixels and low intensity diffraction data.
               If a value is supplied the following options can be used:
               zero_mask=0: masked pixels are free and keep the calculated complex amplitudes
               zero_mask=1: masked pixels are set to zero
               zero_mask=auto: this is only meaningful when using a 'standard' algorithm below. The masked pixels will
                               be set to zero during the first 60% of the HIO/RAAR cycles, and will be free during the 
                               last 40% and ER, ML ones.

    object=obj.npy: starting object. Import object from .npy, .npz, .mat (the first available array 
          will  be used if multiple are present), or CXI file.
          [default=None, object will be defined as random values inside the support area]
    
    support=sup.npy: starting support. Import support from .npy, .npz, .edf, .mat (the first 
              available array will be used if multiple are present) file.  Pixels > 0 are in
              the support, 0 outside. if 'auto', support will be estimated using the intensity
              auto-correlation. If 'circle' or 'square', the suppport will be initialized to a 
              circle (sphere in 3d), or a square (cube).
              [default='auto', support will be defined otherwise]
    
    support_size=50: size (radius or half-size) for the initial support, to be used in 
                     combination with 'support_type'. The size is given in pixel units.
                     Alternatively one value can be given for each dimension, i.e. 
                     support_size=20,40 for 2D data, and support_size=20,40,60 for 3D data. 
                     This will result in an initial support which is a rectangle/parallelepiped
                     or ellipsis/ellipsoid. 
                     [if not given, this will trigger the use of auto-correlation 
                      to estimate the initial support]

    support_autocorrelation_threshold=0.1: if no support is given, it will be estimated 
                                           from the intensity autocorrelation, with this relative 
                                           threshold.
                                           [default value: 0.1]

    support_threshold=0.25: threshold for the support update. Alternatively two values can be given, and the threshold
                            will be randomly chosen in the interval given by two values: support_threshold=0.20,0.28.
                            This is mostly useful in combination with nb_run.
                            [default=0.25]
    
    support_threshold_method=max: method used to determine the absolute threshold for the 
                                  support update. Either:'max' or 'average' (the default) values,
                                  taken over the support area/volume, after smoothing

    support_only_shrink: if set or support_only_shrink=True, the support will only shrink 
                         (default: the support can grow again)
                         
    support_smooth_width_begin=2
    support_smooth_width_end=0.25: during support update, the object amplitude is convoluted by a
                                   gaussian with a size
                                   (sigma) exponentially decreasing from support_smooth_width_begin
                                   to support_smooth_width_end from the first to the last RAAR or 
                                   HIO cycle.
                                   [default values: 2 and 0.5]
    
    support_smooth_width_relax_n: the number of cycles over which the support smooth width will
                                  exponentially decrease from support_smooth_width_begin to 
                                  support_smooth_width_end, and then stay constant. 
                                  This is ignored if nb_hio, nb_raar, nb_er are used, 
                                  and the number of cycles used
                                  is the total number of HIO+RAAR cycles [default:500]
    
    support_post_expand=1: after the support has been updated using a threshold,  it can be shrunk 
                           or expanded by a few pixels, either one or multiple times, e.g. in order
                           to 'clean' the support:
                           - support_post_expand=1 will expand the support by 1 pixel
                           - support_post_expand=-1 will shrink the support by 1 pixel
                           - support_post_expand=-1,1 will shrink and then expand the support 
                             by 1 pixel
                           - support_post_expand=-2,3 will shrink and then expand the support 
                             by 2 and 3 pixels
                           - support_post_expand=2,-4,2 will expand/shrink/expand the support 
                             by 2, 4 and 2 pixels
                           - etc..
                           [default=None, no shrinking or expansion]

    support_update_border_n: if > 0, the only pixels affected by the support updated lie within +/- N pixels around the
                             outer border of the support.

    positivity: if set or positivity=True, the algorithms will be biased towards a real, positive
                object. Object is still complex-valued, but random start will begin with real 
                values. [default=False]
    
    beta=0.9: beta value for the HIO/RAAR algorithm [default=0.9]
    
    crop_output=0: if 1 (the default), the output data will be cropped around the final
                   support plus 'crop_output' pixels. If 0, no cropping is performed.
    
    rebin=2: the experimental data can be rebinned (i.e. a group of n x n (x n) pixels is
             replaced by a single one whose intensity is equal to the sum of all the pixels).
             Both iobs and mask (if any) will be rebinned, but the support (if any) should
             correspond to the new size. The supplied pixel_size_detector should correspond
             to the original size. The rebin factor can also be supplied as one value per
             dimension, e.g. "rebin=4,1,2".
             [default: no rebin]
    
    max_size=256: maximum size for the array used for analysis, along all dimensions. The data
                  will be cropped to this value after centering. [default: no maximum size]
                  
    user_config*=*: this can be used to store a custom configuration parameter which will be ignored by the 
                    algorithm, but will be stored among configuration parameters in the CXI file (data and output).
                    e.g.: user_config_temperature=268K  user_config_comment="Vibrations during measurement" etc...

    ############# ALGORITHMS: standard version, using RAAR, then HIO, then ER and ML

    nb_raar=600: number of relaxed averaged alternating reflections cycles, which the 
                 algorithm will use first. During RAAR and HIO, the support is updated regularly

    nb_hio=0: number of hybrid input/output cycles, which the algorithm will use after RAAR. 
                During RAAR and HIO, the support is updated regularly

    nb_er=200: number of error reduction cycles, performed after HIO, without support update

    nb_ml=20: number of maximum-likelihood conjugate gradient to perform after ER

    detwin: if set (command-line) or if detwin=True (parameters file), 10 cycles will be performed
            at 25% of the total number of RAAR or HIO cycles, with a support cut in half to bias
            towards one twin image

    support_update_period=50: during RAAR/HIO, update support every N cycles.
                              If 0, support is never updated.



    ############# ALGORITHMS: customized version 
    
    algorithm="ER**50,(Sup*ER**5*HIO**50)**10": give a specific sequence of algorithms and/or 
              parameters to be 
              used for the optimisation (note: this string is case-insensitive).
              Important: 
              1) when supplied from the command line, there should be NO SPACE in the expression !
              And if there are parenthesis in the expression, quotes are required around the 
              algorithm string
              2) the string and operators are applied from right to left

              Valid changes of individual parameters include (see detailed description above):
                positivity = 0 or 1
                support_only_shrink = 0 or 1
                beta = 0.7
                live_plot = 0 (no display) or an integer number N to trigger plotting every N cycle
                support_update_period = 0 (no update) or a positivie integer number
                support_smooth_width_begin = 2.0
                support_smooth_width_end = 0.5
                support_smooth_width_relax_n = 500
                support_threshold = 0.25
                support_threshold_method=max or average
                support_post_expand=-1#2 (in this case the commas are replaced by # for parsing)
                zero_mask = 0 or 1
                verbose=20
                fig_num=1: change the figure number for plotting
                
              Valid basic operators include:
                ER: Error Reduction
                HIO: Hybrid Input/Output
                RAAR: Relaxed Averaged Alternating Reflections
                DetwinHIO: HIO with a half-support (along first dimension)
                DetwinHIO1: HIO with a half-support (along second dimension)
                DetwinHIO2: HIO with a half-support (along third dimension)
                DetwinRAAR: RAAR with a half-support (along first dimension)
                DetwinRAAR1: RAAR with a half-support (along second dimension)
                DetwinRAAR2: RAAR with a half-support (along third dimension)
                CF: Charge Flipping
                ML: Maximum Likelihood conjugate gradient (incompatible with partial coherence PSF)
                PSF or EstimatePSF: calculate partial coherence point-spread function 
                                    with 50 cycles of Richardson-Lucy
                Sup or SupportUpdate: update the support according to the support_* parameters
                ShowCDI: display of the object and calculated/observed intensity. This can be used
                         to trigger this plot at specific steps, instead of regularly using 
                         live_plot=N. This is thus best used using live_plot=0
              
              Examples of algorithm strings, where steps are separated with commas (and NO SPACE!),
              and are applied from right to left. Operations in a given step will be applied
              mathematically, also from right to left, and **N means repeating N tymes (N cycles) 
              the  operation on the left of the exponent:
                algorithm=HIO : single HIO cycle
                
                algorithm=ER**100 : 100 cycles of HIO
                
                algorithm=ER**50,HIO**100 : 100 cycles of HIO, followed by 50 cycles of ER
                
                algorithm=ER**50*HIO**100 : 100 cycles of HIO, followed by 50 cycles of ER
                
                algorithm="ER**50,(Sup*ER**5*HIO**50)**10" : 
                    10 times [50 HIO + 5 ER + Support update], followed by 50 ER
                
                algorithm="ER**50,verbose=1,(Sup*ER**5*HIO**50)**10,verbose=100,HIO**100":
                    change the periodicity of verbose output
                
                algorithm="ER**50,(Sup*ER**5*HIO**50)**10,support_post_expand=1,
                           (Sup*ER**5*HIO**50)**10,support_post_expand=-1#2,HIO**100"
                    same but change the post-expand (wrap) method
                
                algorithm="ER**50,(Sup*PSF*ER**5*HIO**50)**5,(Sup*ER**5*HIO**50)**10,HIO**100"
                    activate partial correlation after a first series of algorithms
                
                algorithm="ER**50,(Sup*PSF*HIO**50)**4,(Sup*HIO**50)**8"
                    typical algorithm steps with partial coherence
                
                algorithm="ER**50,(Sup*HIO**50)**4,(Sup*HIO**50)**4,positivity=0,
                          (Sup*HIO**50)**8,positivity=1"
                    same as previous but starting with positivity constraint, removed at the end.

                

            [default: use nb_raar, nb_hio, nb_er and nb_ml to perform the sequence of algorithms]     

    save=all: either 'final' or 'all' this keyword will activate saving after each optimisation 
              step (comma-separated) of the algorithm in any given run [default=final]

Script to perform a CDI reconstruction of data from id01@ESRF.
command-line/file parameters arguments: (all keywords are case-insensitive):

    specfile=/some/dir/to/specfile.spec: path to specfile [mandatory, unless data= is used instead]

    scan=56: scan number in specfile [mandatory].
             Alternatively a list or range of scans can be given:
                scan=12,23,45 or scan="range(12,25)" (note the quotes)

    imgcounter=mpx4inr: spec counter name for image number
                        [default='auto', will use either 'mpx4inr' or 'ei2mint']

    imgname=/dir/to/images/prefix%05d.edf.gz: images location with prefix 
            [default: will be extracted from the ULIMA_mpx4 entry in the spec scan header]
            
    Specific defaults for this script:
        auto_center_resize = True
        detwin = True
        nb_raar = 600
        nb_hio = 0
        nb_er = 200
        nb_ml = 0
        support_size = None
        zero_mask = auto
```