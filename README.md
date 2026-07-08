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
To get started, we will need:

 - A calibrated data set in CASA MS format
 - A numpy array of source positions

test
