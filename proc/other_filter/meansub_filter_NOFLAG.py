import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sys
import os
import glob
import multiprocessing
from contextlib import closing


## GET / WRITE DATA ##
def get_data(infile):
    hdulist = fits.open(infile)
    hdu = hdulist[1]
    freqs = hdu.data[0]['dat_freq']
    dat = hdu.data[:]['data']
    dd = np.reshape(dat, (-1, len(freqs)) )
    hdulist.close()    
    return freqs, dd


def apply_changes(infile, outfile, dd):
    hdulist = fits.open(infile)
    hdu = hdulist[1]
    freqs = hdu.data[0]['dat_freq']
    dat = hdu.data[:]['data']

    dd_out = np.reshape(dd, dat.shape)
    hdu.data[:]['data'] = dd_out[:]

    hdulist.writeto(outfile)
    hdulist.close()
    return


## RUNNING AVERAGE ##
def runavg_chan(ddc, win=100):
    N = len(ddc)
    for ii in xrange(len(ddc)):
        xmin = max(0, ii - win/2)
        xmax = min(N, ii + win/2)
        xslice = slice(xmin, xmax)
        xm = np.sum(ddc[xslice]) / float(xmax-xmin)
        ddc[ii] -= xm
    return 


def runavg_arr(dd, win=100):
    N = len(dd)
    for ii in xrange(len(dd)):
        xmin = max(0, ii - win/2)
        xmax = min(N, ii + win/2)
        xslice = slice(xmin, xmax)
        xm = np.sum(dd[xslice], axis=0) / float(xmax-xmin)
        dd[ii] -= xm
    return dd


def runmed_arr(dd, win=100):
    N = len(dd)
    for ii in xrange(len(dd)):
        xmin = max(0, ii - win/2)
        xmax = min(N, ii + win/2)
        xslice = slice(xmin, xmax)
        xm = np.median(dd[xslice], axis=0)
        dd[ii] -= xm
    return dd
    

## MISC SELECTIONS ##
def remove_zeros(dd):
    xx = np.where( np.abs(dd) > 0 )[0]
    return dd[xx]


def get_edge_channels(total_chan, spw_chan, nedge):
    nspw = total_chan / spw_chan
    one_spw = np.hstack( (np.arange(0, nedge), np.arange(spw_chan-nedge, spw_chan)) )
    all_spw = (one_spw + spw_chan *  np.arange(nspw).reshape(nspw, 1)).ravel()
    return all_spw
    

def get_mask_chans():
    """
    Mask out known bad channels
    """
    # Mask edge channels 
    xx_edge = get_edge_channels(512, 32, 2)

    # mask edge of spw 1 
    xx_man1 = np.arange(51, 62)

    # Mask spw 2
    xx_man2 = np.arange(32 * 2, 32 * 3)

    # Mask first and last 5 channels
    xx_man3 = np.hstack( (np.arange(0, 5), np.arange(512-5, 512)) )

    # Mask middle 10 channels
    xx_man4 = np.arange(256-5, 256+5)

    # Add all the manual flags together
    xx_man = np.hstack( (xx_man1, xx_man2, xx_man3, xx_man4) )
    
    xx_mask = np.unique( np.hstack( (xx_edge, xx_man) ) )

    return xx_mask


def apply_mask(dd, xx_mask):
    if len(xx_mask):
        for ii in xx_mask:
            dd[:, ii] = 0
    else:
        pass

    return dd


def filter_FITS(infile, outfile, window=100, debug=False):
    print("INFILE: %s" %(infile))
    print("OUTFILE: %s" %(outfile))

    if os.path.isfile(outfile):
        print("OUTFILE ALREADY EXISTS: %s" %(outfile))
        return
    else: pass

    if not debug:
        try:
            freqs, dd = get_data(infile)
            dd = runavg_arr(dd, win=window)
            xx_mask = get_mask_chans()
            dd = apply_mask(dd, xx_mask)
            apply_changes(infile, outfile, dd)

        except RuntimeError:
            print("FAILED:  %s" %(infile))
    else:
        pass

    return


def filter_many_FITS_single(indir, outdir, window=100, debug=False):
    infiles = glob.glob("%s/*fits" %indir)
    fnames = [ ff.split("/")[-1] for ff in infiles ]
    outfiles = [ "%s/%s" %(outdir, ff) for ff in fnames ]
    
    for ii in xrange(len(infiles)):
        print("%d / %d" %(ii, len(infiles)))
        infile  = infiles[ii]
        outfile = outfiles[ii]

        print("IN:  %s" %infile)
        print("OUT: %s" %outfile)
        print("\n")

        if not debug:
            filter_FITS(infile, outfile, window=window)
        else:
            pass
    return


def multi_filter_FITS(nproc, indir, outdir, window=100, debug=False):
    infiles = glob.glob("%s/*fits" %indir)
    fnames = [ ff.split("/")[-1] for ff in infiles ]
    outfiles = [ "%s/%s" %(outdir, ff) for ff in fnames ]
    N = len(infiles)

    # Check that output directory exists
    if not os.path.isdir(outdir):
        print("NO DIRECTORY:  %s" %(outdir))
        return
    else: pass

    with closing(multiprocessing.Pool(processes=nproc)) as pool:
        results = [ pool.apply_async(filter_FITS,
                    args=(infiles[ii], outfiles[ii], window, debug)) \
                    for ii in xrange(N) ]
        all_files = [p.get() for p in results]

    return


