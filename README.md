# A Beamforming Approach to RM Determination

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


