import numpy as np
from astropy.io import fits
import glob

def get_freqs(fitsfile):
    hdulist = fits.open(fitsfile)
    hdu1, hdu2 = hdulist
    freqs = hdu2.data[0]['dat_freq']
    hdulist.close()
    return freqs


def update_freqs(fitsfile, freq_lo, df, nchan):
    print("FILE:  %s" %fitsfile)
    hdulist = fits.open(fitsfile, mode='update')
    hdu1, hdu2 = hdulist

    old_freqs = hdu2.data[0]['dat_freq']
    new_freqs = freq_lo + np.arange(nchan, dtype=old_freqs.dtype) * df
    
    Nsubint = hdu2.data.shape[0]
    freq_arr = np.vstack([new_freqs] * Nsubint)

    #print hdu2.data[:]['dat_freq'].shape
    #print freq_arr.shape
    
    hdu2.data[:]['dat_freq'] = freq_arr
    
    hdulist.flush()
    hdulist.close()
    return new_freqs


def update_freqs_files(fitslist, freq_lo, df, nchan):
    for fitsfile in fitslist:
        new_freqs = update_freqs(fitsfile, freq_lo, df, nchan)
    return new_freqs


def update_freqs_dir(fitsdir, freq_lo, df, nchan):
    glob_str = "%s/*fits" %fitsdir
    fitsfiles = glob.glob(glob_str)
    fitsfiles.sort()
    new_freqs = update_freqs_files(fitsfiles, freq_lo, df, nchan)
    return new_freqs
