# A Beamforming Approach to RM Determination

## Basic Idea 

These scripts are provided to enable rapid RM characterization 
using beamformed visibilities. For a given list of sky positions, 
we can apply the appropriate phase offsets and extract visibility 
averaged intensities per polarization, frequency channel, and time 
sample. By then averaging in time we get the full frequency resolution 
spectra in Stokes I, Q, U, and V.  Using the full resolution Q and U 
spectra we can then run RM synthesis to measure the Faraday spectrum 
and from this the RM and peak polarized intensity.

Since beamforming is just multiplying by a phase offset and summing, 
it is much faster than standard interferometric imaging.  This speedup 
is substantial, especially for wide fields of view and large numbers 
of channels.  However, these are not mathematically equivalent, and 
Q U channel imaging (and CLEANing) will always give the more precise 
measurement of polarized flux and RM.  The beamforming approach is 
equivalent to making dirty images of each channel and then extracting 
data from pixels centered on each target position.  While this will 
not be an exact representation of the polarized sky, it is sufficient 
for quickly identifying interesting polarized sources and measuring 
their RMs.

In the following, we will give a recipe for running the scripts for 
beamforming and RM synthesis to go from a CASA measurement set to 
a catalog of RMs.


## Installation and Requirements

For the beamforming, the main requirement is `casatools`, which 
is needed to read the visibilities in CASA measurement set format. 
For the RM synthesis we are using 
[RM-Tools](https://github.com/CIRADA-Tools/RM-Tools), which needs 
to be installed independently.  I will figure out installation 
instructions later, but for now you just need to have RM-Tools 
installed and I will provide a singularity image that has 
`casatools`.


## Example End-to-End Processing

Let's now go through an example to see how things work.  

### Getting Ready

To get started, we will need:

 - A calibrated data set in CASA MS format
 - A numpy array of source positions

The MS data can be split up so that each spectral window 
has its own measurement set (ie, MMGPS style) or can be 
one MS containing many spectral windows (ie, normal style). 
Right now (08 Jul 2026), this code only works with the MMGPS 
style, but the other style will be supported very soon.

The numpy array of source positions give the sky coordinates 
at which beams will be formed and data extracted.  The positions 
will come from some existing catalog (e.g., from source finding) 
or from a known target of interest.  I can add a script for 
converting a FITS catalog (or csv or whatever) to the proper 
numpy format, but it is very simple.  The numpy array will 
have the form:

```
np.array([['J2000', '17:42:08.57714464', '-28.55.36.81117535'],
          ['J2000', '17:42:07.88886029', '-29.18.45.58701964'],
          ['J2000', '17:42:06.58545272', '-29.31.47.76699699']], dtype='<U18')
```

where each row contains three elements: `'J2000'`, the Right Asencsion string, 
and the Declination string (in CASA coordinate format).

Save the array to a `*.npy` file with `np.save()`.

For this example, I will use a list of 8 positions corresponding to 
point sources in our field.  I have save these positions as a numpy 
array in the file `example_beams.npy`.  The MS data are spread over 
16 files corresponding to 16 spectral windows and I have put them in 
a folder called `selfcal_data`:

```
> ls selfcal_data/
MSGPS_S_3021_spw000.ms  MSGPS_S_3021_spw004.ms  MSGPS_S_3021_spw008.ms  MSGPS_S_3021_spw012.ms
MSGPS_S_3021_spw001.ms  MSGPS_S_3021_spw005.ms  MSGPS_S_3021_spw009.ms  MSGPS_S_3021_spw013.ms
MSGPS_S_3021_spw002.ms  MSGPS_S_3021_spw006.ms  MSGPS_S_3021_spw010.ms  MSGPS_S_3021_spw014.ms
MSGPS_S_3021_spw003.ms  MSGPS_S_3021_spw007.ms  MSGPS_S_3021_spw011.ms  MSGPS_S_3021_spw015.ms
```

### Beamforming

With our beam list and data set, we are now ready to run the beamforming code that 
will extract the IQUV data from the measurement set(s). The script we will need is 
`run_beamform.py`, which is in the `proc` folder.  If we run with `-h` we can see 
the usage:

```
> python ~/src/ms_beamform/proc/run_beamform.py -h
usage: run_beamform.py [-h] --beamfile BEAMFILE --outbase OUTBASE 
                       [--outdir OUTDIR] [--uv_taper_lam UV_TAPER_LAM]
                       [--field FIELD] [--nproc NPROC] [--no_weights] 
                       [--no_flags] [--use_data_col]
                       ms_files [ms_files ...]

Extract beamformed IQUV dynamic spectra from MS data

positional arguments:
  ms_files              Path to data Measurement Set(s). If each spw has its own MS, 
                        then each should end in 'spw[num].ms'

optional arguments:
  -h, --help            show this help message and exit
  --beamfile BEAMFILE   Path to *npy beam positions file
  --outbase OUTBASE     Base name for output files
  --outdir OUTDIR       Output directory for beam data (def: cwd)
  --uv_taper_lam UV_TAPER_LAM
                        Lower baseline taper in wavelengths (default: 0, no taper)
  --field FIELD         CASA field number of target (def=0)
  --nproc NPROC         Number of parallel processes
  --no_weights          Ignore visibility weights (def: False)
  --no_flags            Ignore MS data flags (def: False)
  --use_data_col        Use DATA column (def: use CORRECTED column)
```

The `beamfile` is the `*.npy` file of our desired beam positions, `outbase` is the desired base name for all output, `outdir` is the directory where you want the beam data to be written (each beam will get its own directory under this), `uv_taper_lam` sets an inner cutoff (in wavelengths) that is a cheap way of downweighting the shorter baselines, `field` is the field ID of the target in the measuremen set, `nproc` is the number of processes to run simultaneously, `no_weights` ignores the visibility weights in the measurement set, `no_flags` ignores the flags in the measurement set, and `use_data_col` says that we should use the `DATA` column and not the default `CORRECTED_DATA` column.

One note of caution is that we are using the python package `multiprocessing` to run the parallel processing specified by `nproc`.  This works very well, but if you are running a machine with OpenMP, you should set the following environment variable:

```
export OMP_NUM_THREADS=1
```

so that each process just goes to one thread.  Otherwise you will see a significant slowdown.

OK, now we are ready to run. I will make a directory called `beams` (but you can call it whatever you want) as the output directory for the beam data.  I will also set a `uv_taper_lam` of 5500 wavelengths.  We then run this as: 

```
> python ~/src/ms_beamform/proc/run_beamform.py --beamfile example_beams.npy --outbase test --outdir beams --nproc 10 --uv_taper_lam 5500 selfcal_data/MSGPS_S_3021_spw0*ms
```

When it starts it should give you some info about the inputs:

```
Found 16 MS
selfcal_data/MSGPS_S_3021_spw000.ms
selfcal_data/MSGPS_S_3021_spw001.ms
selfcal_data/MSGPS_S_3021_spw002.ms
selfcal_data/MSGPS_S_3021_spw003.ms
selfcal_data/MSGPS_S_3021_spw004.ms
selfcal_data/MSGPS_S_3021_spw005.ms
selfcal_data/MSGPS_S_3021_spw006.ms
selfcal_data/MSGPS_S_3021_spw007.ms
selfcal_data/MSGPS_S_3021_spw008.ms
selfcal_data/MSGPS_S_3021_spw009.ms
selfcal_data/MSGPS_S_3021_spw010.ms
selfcal_data/MSGPS_S_3021_spw011.ms
selfcal_data/MSGPS_S_3021_spw012.ms
selfcal_data/MSGPS_S_3021_spw013.ms
selfcal_data/MSGPS_S_3021_spw014.ms
selfcal_data/MSGPS_S_3021_spw015.ms

Found 16 spws
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

Found beamfile: example_beams.npy
  with 8 beams
```

and then will print some debugging info as it's working.  If you look in the ouput directory you will see

```
> ls beams/
beam00000  beam00001  beam00002  beam00003  beam00004  beam00005  
beam00006  beam00007  spw00_tt_step000.npy  test_20260708T135254.log
```

that there are now folders for each beam (containing the data), a numpy file recording sample times, and a log file.  Looking inside one of the beam folders, you will see that the spectral windows are each being processed separately. 

```
> ls beam00000
spw00_beam00000_step000.npy  spw01_beam00000_step000.npy  
spw02_beam00000_step000.npy  spw03_beam00000_step000.npy
```

These are intermediate data products and will be combined into one file at the end.  All beams are being processed at the same time and each spw is processed sequentially.  For me, each spw took about 40 seconds each, and total processing was about 10 minutes.  More beams will take longer, of course, but not by a ton.  The slowest part of the process is just reading the measurement set files and all the beams are read at the same time.

Now that processing is done, let's take a look to see what was produced.  In the output directory we now have:

```
> ls
beam00000  beam00001  beam00002  beam00003  beam00004  beam00005  
beam00006  beam00007  test_20260708T135254.log  test_freqs.npy  times
```

We get a log file, the beam data folders, and a `npy` file containing each of the channel frequencies of the full band (ie, combining all the spectral windows) in Hz.  The `times` folder contains the time stamps for each time sample, which we won't really need. 

Each of the beam folders now just contains one spw-combined data file.  For example in beam0 we find the file `test_beam00000_full.npy`, where `test` is the output base we gave.  The data files are just numpy arrays with shape `(4, Nt, Nf)` where 4 is the number of stokes parameters (I, Q, U, V), `Nt` is the number of time samples, and `Nf` is the number of frequency channels.


### Plotting Beam Data

It can often be useful to look at the data for each of these beams.  
To do this we have a script called `plot_beam.py` that will make plots 
of the time series of IQUV and the spectra of IQUV.  Running `-h` gives 
the usage for this script:

```
> python ~/src/ms_beamform/proc/plot_beam.py -h
usage: plot_beam.py [-h] [--outdir OUTDIR] --freqfile FREQFILE 
                    [--tsamp TSAMP] beam_files [beam_files ...]

Make time and spectrum plots for IQUV data in beams

positional arguments:
  beam_files           Beam data file(s) for plotting

options:
  -h, --help           show this help message and exit
  --outdir OUTDIR      Output directory for plots (def: cwd)
  --freqfile FREQFILE  npy array of channel frequencies
  --tsamp TSAMP        Visibility time resolution (s)
```

The `tsamp` value here is the integration time of each visibility sample.  For most of our MeerKAT observations it is about 8 seconds.  This is just to convert sample number to time (in sec) in the plots.  If you don't know it, you can just leave it blank and interpret the time axis as simply sample number.

I will now make a directory called `plots` and run:

```
> $ python ~/src/ms_beamform/proc/plot_beam.py --outdir plots --freqfile beams/test_freqs.npy --tsamp 8 beams/beam0000*/*npy
```


![alt text](https://github.com/rwharton/ms_beamform/tree/main/example/test_spec.png?raw=true)
