import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import sys
import os
import glob
import multiprocessing
from contextlib import closing
import time
from scipy.signal import fftconvolve

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


## GET STATS + FLAGGING ##
def get_global_stats(dd, frac=0.80):
    """
    Get the mean and standard deviation 
    from dd using all data that is non-zero 
    (ie, not flagged) and falls within the 
    middle frac (80%) of the distribution

    The idea is to remove any extreme outliers 
    for calculation of the mean and standard 
    deviation (mostly for the latter)
    """
    ddr = dd.ravel()
    ddr = ddr[ np.abs(ddr) > 0 ] 
    ddr_sort = np.sort(ddr)
    N = len(ddr_sort)

    f_lo = 0.5 * (1.0 - frac)
    f_hi = 0.5 * (1.0 + frac)
    fslice = slice( int(f_lo * N), int(f_hi * N) )
    fmean = np.mean(ddr_sort[fslice])
    fstd  = np.std(ddr_sort[fslice])

    return fmean, fstd


def flag_drops(dd, dmean, dsig, nsig=8):
    xx = np.where( dd < -1.0 * nsig * dsig)
    dd[xx] = 0
    return dd


def chan_chunk_std(ddc, n):
    """
    Calculate standard deviation of n-sample chunks 
    of the data in a single frequency channel
    """
    # Total number of samples
    N  = len(ddc)

    # Number of n-sample chunks
    nchunks = int(N / n)
    if nchunks * n < N:
        nchunks += 1

    chunk_sigs = np.zeros(nchunks)

    for ii in xrange(nchunks):
        ddc_i = ddc[ ii * n : (ii+1) * n ]
        xx = np.where( np.abs(ddc_i) > 0 )[0]

        sig_i = np.std( ddc_i[xx] )

        chunk_sigs[ii] = sig_i

    return chunk_sigs


def chan_chunk_var(ddc, n):
    """
    Calculate standard deviation of n-sample chunks 
    of the data in a single frequency channel
    """
    # Total number of samples
    N  = len(ddc)

    # Number of n-sample chunks
    nchunks = int(N / n)
    if nchunks * n < N:
        nchunks += 1

    chunk_vars = np.zeros(nchunks)

    for ii in xrange(nchunks):
        ddc_i = ddc[ ii * n : (ii+1) * n ]
        xx = np.where( np.abs(ddc_i) > 0 )[0]
        ni = len(xx)

        # Calc sample variance 
        # ddof=1 makes denom N-1
        var_i = np.var( ddc_i[xx], ddof=1 )

        chunk_vars[ii] = var_i

    return chunk_vars


def chan_chunk_var_mask(ddc, n, maxvar, cfrac=0.5):
    """
    Calculate the sample variance of n-sample chunks 
    of data channel ddc and flag chunks that have a 
    variance above maxvar

    Also, if >cfrac of chunk is already flagged, flag
    whole thing
    """
    # Total number of samples
    N  = len(ddc)

    # Number of n-sample chunks
    nchunks = int(N / n)
    if nchunks * n < N:
        nchunks += 1

    ni_min = max( 1, (1-cfrac) * n )

    for ii in xrange(nchunks):
        ddc_i = ddc[ ii * n : (ii+1) * n ]
        xx = np.where( np.abs(ddc_i) > 0 )[0]
        ni = len(xx)

        if ni <= ni_min:
            ddc[ ii * n : (ii+1) * n] = 0
            continue
        else: 
            pass

        # Calc sample variance 
        # ddof=1 makes denom N-1
        var_i = np.var( ddc_i[xx], ddof=1 )

        # if var_i > maxvar, flag values
        if var_i > maxvar:
            ddc[ ii * n : (ii+1) * n ] = 0
        else:
            pass

    return ddc


def flag_chunk_var(dd, n, fmean, fstd, nsig=10, cfrac=0.5):
    """
    Use global mean / std (using only mid-80% vals) 
    to flag chunks with sample variances that deviate 
    from global expectation by nsig standard deviations
    """
    # Correct std dev b/c only used inner 80% for fstd
    gstd = fstd / 0.661 

    # Calculate mean sample variance for a chunk
    s2_avg = gstd**2.0 

    # Calculate std dev of sample variance for chunks
    s2_std = np.sqrt( 2.0 / (n - 1.0) ) * gstd**2.0 

    # Set max sample variance
    maxvar = s2_avg + nsig * s2_std

    # Loop over channels and flag
    Nchan = dd.shape[1]
    for ii in xrange(Nchan):
        dd[:, ii] = chan_chunk_var_mask(dd[:, ii], n, maxvar, cfrac=cfrac)

    return dd


def extend_flags(dd, freq_frac=0.3, time_frac=0.5):
    """
    Extend flags -- 

      If f > freq_frac of frequency channels are flagged 
       in a given time sample, flag all the channels

      if f > time_frac of time samples are flagged in a 
       channel, flag all the time samples 

    """
    Nt, Nf = dd.shape

    # First loop over time samples to check channel fracs
    Nt_min = int(time_frac * Nt)
    
    zero_chans = np.sum( dd == 0, axis=0 )
    xxf = np.where( zero_chans >= Nt_min )[0]
    print(len(xxf))

    # Next loop over frequency samples to check time fracs
    Nf_min = int(freq_frac * Nf)
    
    zero_samps = np.sum( dd == 0, axis=1 )
    xxt = np.where( zero_samps >= Nf_min )[0]
    print(len(xxt))
    
    # Apply flags

    dd[:, xxf] = 0
    dd[xxt, :] = 0

    return dd


def rfi_flag(dd, n, nsig_drop=8, nsig_var=10, freq_frac=0.3, 
             time_frac=0.5, chunk_frac=0.5):
    # Get global stats
    fmean, fstd = get_global_stats(dd, frac=0.80)
    print("  %f %f" %(fmean, fstd))

    N = float( dd.size )

    # Flag drop-outs
    if nsig_drop > 0:
        dd = flag_drops(dd, fmean, fstd, nsig=nsig_drop)
        xx = np.where(dd == 0)[0]
        print("  -After drops:  %f" %(len(xx) / N))

    # Flag variances in chunks of n samples 
    if nsig_var > 0:
        dd = flag_chunk_var(dd, n, fmean, fstd, nsig=nsig_var, 
                            cfrac=chunk_frac)
        xx = np.where(dd == 0)[0]
        print("  -After var:  %f" %(len(xx) / N))

    # Extend flags
    if freq_frac < 1 or time_frac < 1:
        dd = extend_flags(dd, freq_frac=freq_frac, time_frac=time_frac)
        xx = np.where(dd == 0)[0]
        print("  -After ext:  %f" %(len(xx) / N))

    return dd
        


## RUNNING AVERAGE ##
def runavg(ddc, win=100):
    N = len(ddc)
    mavg = np.zeros(N)
    for ii in xrange(N):
        xmin = max(0, ii - win/2)
        xmax = min(N, ii + win/2)
        xslice = slice(xmin, xmax)
        xm = np.sum(ddc[xslice]) / float(xmax-xmin)
        mavg[ii] = xm
    return mavg

def runavg_chan(ddc, win=100):
    N = len(ddc)
    for ii in xrange(len(ddc)):
        xmin = max(0, ii - win/2)
        xmax = min(N, ii + win/2)
        xslice = slice(xmin, xmax)
        xm = np.sum(ddc[xslice]) / float(xmax-xmin)
        ddc[ii] -= xm
    return 


def runavg_chan_fft(ddc, win=100):
    eps = 1e-15
    window = np.ones(win) 
    ddsum = fftconvolve(ddc, window, mode='same')
    ddn = fftconvolve( np.abs(ddc) > eps, window, mode='same').astype('int')
    ddn[ddn == 0] = 1
    mm = ddsum / ddn.astype('float')
    return ddc - mm


def runavg_arr_orig(dd, win=100):
    N = len(dd)
    for ii in xrange(len(dd)):
        xmin = max(0, ii - win/2)
        xmax = min(N, ii + win/2)
        xslice = slice(xmin, xmax)
        xm = np.sum(dd[xslice], axis=0) / float(xmax-xmin)
        dd[ii] -= xm
    return dd


def runavg_arr_zero(dd, win=100):
    N = len(dd)
    for ii in xrange(len(dd)):
        xmin = max(0, ii - win/2)
        xmax = min(N, ii + win/2)
        xslice = slice(xmin, xmax)
        dds = dd[xslice]
        nn = np.sum( np.abs(dds) > 0, axis=0 ).astype('float')
        nn[ nn == 0 ] = 1.0
        xm = np.sum(dds, axis=0) / nn
        xx = np.where( np.abs(dd[ii]) == 0 )[0]
        dd[ii] -= xm
        dd[ii, xx] = 0
    return dd


def runavg_arr(dd, win=100):
    Nf = dd.shape[1]
    xx = np.where(dd == 0)
    for jj in xrange(Nf):
        dd[:, jj] = runavg_chan_fft(dd[:, jj], win=win)
    dd[xx] = 0
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
    # Mask first 10 channels
    xx_start = np.arange(0, 10)

    # Mask edge channels 
    xx_edge = get_edge_channels(512, 32, 2)

    # mask edge of spw 1 
    xx_man1 = np.arange(51, 62)

    # Mask spw 2
    xx_man2 = np.arange(32 * 2, 32 * 3)

    # Mask first and last 5 channels
    xx_man3 = np.hstack( (np.arange(0, 5), np.arange(512-5, 512)) )

    # Mask middle 15 channels
    xx_man4 = np.arange(256-5, 256+10)

    # Add all the manual flags together
    xx_man = np.hstack( (xx_man1, xx_man2, xx_man3, xx_man4) )
    
    xx_mask = np.unique( np.hstack( (xx_start, xx_edge, xx_man) ) )

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


def filter_many_FITS(nproc, indir, outdir, window=100, debug=False):
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


def filter_flag_FITS(infile, outfile, window=100, nt=500, nsig_drop=8, 
                     nsig_var=10, freq_frac=0.3, time_frac=0.5, chunk_frac=0.5, 
                     debug=False):
    #rfi_flag(dd, n, nsig_drop=8, nsig_var=10, freq_frac=0.3, time_frac=0.5)
    print("INFILE: %s" %(infile))
    print("OUTFILE: %s" %(outfile))

    if os.path.isfile(outfile):
        print("OUTFILE ALREADY EXISTS: %s" %(outfile))
        return
    else: pass

    if not debug:
        try:
            t0 = time.time()
            freqs, dd_orig = get_data(infile)
            t1 = time.time()
            print("READ DATA:  %d s" %(t1 - t0))
            dd = np.copy(dd_orig)
            t2 = time.time()
            print("COPY DATA:  %d s" %(t2 - t1))
            dd = runavg_arr(dd, win=window)
            t3 = time.time()
            print("AVG1:  %d s" %(t3 - t2))

            xx_mask = get_mask_chans()
            dd = apply_mask(dd, xx_mask)

            dd = rfi_flag(dd, nt, nsig_drop=nsig_drop, nsig_var=nsig_var, 
                          freq_frac=freq_frac, time_frac=time_frac, 
                          chunk_frac=chunk_frac)

            xx = np.where(dd == 0)
            dd_orig[xx] = 0
           
            t4 = time.time()
            print("RFI:  %d s" %(t4 - t3))
           
            dd = runavg_arr(dd_orig, win=window)

            t5 = time.time()
            print("AVG2:  %d s" %(t5 - t4))
            
            apply_changes(infile, outfile, dd)
            
            t6 = time.time()
            print("APPLY:  %d s" %(t6 - t5))

        except RuntimeError:
            print("FAILED:  %s" %(infile))

        except:
            print("HELLO??")
    else:
        pass

    return


def filter_flag_many_FITS(nproc, indir, outdir, window=100, nt=500, nsig_drop=8, 
                     nsig_var=10, freq_frac=0.3, time_frac=0.5, chunk_frac=0.5, 
                     debug=False):
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
        results = [ pool.apply_async(filter_flag_FITS,
                    args=(infiles[ii], outfiles[ii], window, nt, nsig_drop, nsig_var, 
                          freq_frac, time_frac, chunk_frac, debug)) \
                    for ii in xrange(N) ]
        all_files = [p.get() for p in results]

    return
