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
