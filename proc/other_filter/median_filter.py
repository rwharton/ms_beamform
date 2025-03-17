import numpy as np
from scipy.special import erf
from astropy.io import fits
import scipy.optimize as op
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


def apply_changes(infile, outfile, dd, avgs=None):
    hdulist = fits.open(infile)
    hdu = hdulist[1]
    freqs = hdu.data[0]['dat_freq']
    dat = hdu.data[:]['data']

    dd_out = np.reshape(dd, dat.shape)
    hdu.data[:]['data'] = dd_out[:]

    if avgs is not None:
        if len(avgs) != len(freqs):
            print("Avgs are wrong shape!")
            print("len(avgs) = %d" %(len(avgs)))
            print("len(freqs) = %d" %(len(freqs)))
        else: pass
        offs = np.outer( np.ones(dd_out.shape[0]), avgs )
        hdu.data[:]['dat_offs'] = offs[:]
    else: 
        pass

    hdulist.writeto(outfile)
    hdulist.close()
    return
    

## MISC SELECTIONS ##
def remove_zeros(dd):
    xx = np.where( np.abs(dd) > 0 )[0]
    return dd[xx]


def equiv_sig_normal(prob):
    x = np.linspace(0, 20, 10000)
    pp = 0.5 * (1 - erf(x / np.sqrt(2)))
    sig = x[np.argmin( np.abs(pp - prob) )]
    return sig


## SPECIAL EXP FUNCTIONS ##
def log_sum_exp(loga, logb):
    mval = np.maximum(loga, logb)
    return mval + np.log( np.exp(loga-mval) + np.exp(logb-mval) )


def log1mexp(x):
    """
    Calculate log(1-exp(-x)) real good
    """
    x0 = np.log(2)

    if np.any( np.abs(x) == 0):
        x[ np.abs(x) == 0 ] += 1e-10
    
    f_lta = np.log( -1.0 * np.expm1(-1.0 * x) )
    f_gta = np.log1p( -1.0 * np.exp(-1.0 * x) )

    f_out = f_lta[:]
    f_out[x > x0] = f_gta[x > x0]

    return f_out    
    
    
## Likelihoods ##
def ln_normal(x, mu, sig):
    return -0.5 * ( ((x-mu)/sig)**2.0 + 2.0 * np.log(sig) + np.log(2 * np.pi) )
    

def ln_exp(x, mu):
    if x < 0:
        return -np.inf
    else:
        return -np.log(mu) - x / mu


def lnprior(theta):
    av, mu, sig = theta

    p_av   = ln_normal(av, 0.5, 0.5)
    p_mu   = ln_normal(mu, 0.0, 5.0)
    p_sig  = ln_exp(sig, 0.2)

    return p_av + p_mu + p_sig 


def lnlike(theta, x, y, yerr):
    av, mu, sig = theta
    a  = 1e3 * av
    
    ymod = a * np.exp(-(x-mu)**2.0 / (2 * sig**2.0))
    R = np.abs(y - ymod) / yerr

    if np.any( R == 0 ):
        R[R == 0] += 1e-5
        #print("ZERO")
    
    #arg1 = np.log( (1 - np.exp(-R**2.0 / 2.0)) )
    exp_val = R**2.0 / 2.0
    
    arg1 = log1mexp(exp_val)
    arg2 = -2.0 * np.log( np.abs(R) )

    return np.sum( arg1 + arg2 ) 


def lnprob(theta, x, y, yerr):
    av, mu, sig = theta
    a  = 1e3 * av 

    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf 

    return lp + lnlike(theta, x, y, yerr)


def minus_lnprob(theta, x, y, yerr):
    return -1.0 * lnprob(theta, x, y, yerr)


def minus_lnlike(theta, x, y, yerr):
    return -1.0 * lnlike(theta, x, y, yerr)


## FIT ALL CHANNELS ##
def fit_all_chans_med(dd, minfrac=0.5):
    avgs = []
    sigs = []
    likes = []

    nchans = dd.shape[-1]
    nsamps = dd.shape[0]

    for ii in xrange(nchans):
        #print("%d / %d" %(ii, nchans))
        ddi = dd[:, ii]
        ddi = remove_zeros(ddi)

        bb = np.linspace(-10, 10, 10**4)
        y, be = np.histogram(ddi, bins=bb)
        yerr = np.sqrt(y)
        x = bb[1:] - 0.5 * (bb[1] - bb[0])
        xx = np.where( (y > 10) )[0]
       
        good_frac = np.sum(y) / float(nsamps)
        #print(good_frac)
       
        if good_frac < minfrac:
            avg  = 1e10
            sig  = 1.0
            like = -1e10

        else:
            x = x[xx]
            y = y[xx]
            yerr = yerr[xx]

            imed = np.argmin( np.abs(x - np.median(x) ))
            xmed = x[imed]
            ymed = y[imed]

            theta0 = np.array([ ymed / 1e3, xmed, 0.2 ])

            res = op.minimize(minus_lnlike, theta0, 
                              args=(x, y, yerr), 
                              options={'disp' : False}, 
                              bounds=[(0, None), 
                                      (None, None), 
                                      (0.01, 1.0)])

            aval, avg, sig = res['x']
            like = lnlike(res['x'], x, y, yerr)

        avgs.append( avg )
        sigs.append( sig )
        likes.append( like )

    avgs = np.array( avgs )
    sigs = np.array( sigs )
    likes = np.array(likes)

    return avgs, sigs, likes


def filter_one_chan(dd1, avg1, sig1, sig_max):
    xx = np.where( np.abs(dd1) > 0 )[0]
    dd1[xx] -= avg1
    
    #yy = np.where( np.abs(dd1) / sig1 > sig_max )[0]
    # Only remove negatives
    yy = np.where( dd1 / sig1 < -1.0 * sig_max )[0]
    if len(yy):
        dd1[yy] = 0
    else:
        pass
    
    return dd1


def filter_chans(dd, avgs, sigs, sig_max=5):
    nchans = dd.shape[-1]
    
    for ii in xrange(nchans):
        ddi = dd[:, ii]
        ddo = filter_one_chan(ddi, avgs[ii], sigs[ii], sig_max)
        dd[:, ii] = ddo[:]

    return dd
        
        
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


def filter_FITS(infile, outfile, minfrac=0.5, debug=False):
    print("INFILE: %s" %(infile))
    print("OUTFILE: %s" %(outfile))

    if os.path.isfile(outfile):
        print("OUTFILE ALREADY EXISTS: %s" %(outfile))
        return
    else: pass
    
    if not debug:
        try:
            freqs, dd = get_data(infile)
            avgs, sigs, likes = fit_all_chans_med(dd, minfrac=minfrac)
            dd = filter_chans(dd, avgs, sigs, sig_max=5)
            xx_mask = get_mask_chans()
            dd = apply_mask(dd, xx_mask)
            avgs_masked = avgs[:]
            avgs_masked[xx_mask] = 0
            apply_changes(infile, outfile, dd, avgs=avgs_masked)

        except RuntimeError:
            print("FAILED:  %s" %(infile))
    else: 
        pass

    return


def multi_filter_FITS(nproc, indir, outdir, minfrac=0.5, debug=False):
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
                    args=(infiles[ii], outfiles[ii], minfrac, debug)) \
                    for ii in xrange(N) ]
        all_files = [p.get() for p in results]
    
    return


