import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import os
from collections import namedtuple
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
import simple_dm as dedisp


def get_data(infile, tstart, tdur):
    hdulist = fits.open(infile)
    hdu1, hdu2 = hdulist

    tstop = tstart + tdur 
    
    dt = hdu2.header['tbin']
    freqs = hdu2.data[0]['dat_freq']
    df = freqs[1] - freqs[0]
    nchan = len(freqs)
    tsub = hdu2.data[0]['TSUBINT']
    nsblk = hdu2.header['nsblk']

    row_start = max(int(tstart / tsub), 0)
    row_stop  = max(int(np.ceil(tstop / tsub)), row_start + 1)
    trow_start = row_start * tsub

    #print tstart, row_start, row_stop
    
    dd = hdu2.data[row_start : row_stop]['data']
    #print dd.shape
    dd = np.reshape(dd, ( (row_stop - row_start) * nsblk, nchan ))
    
    idx = max(int( (tstart - trow_start) / dt ), 0)
    ntx = int(tdur / dt)

    #print(idx, ntx)
    dd = dd[idx : idx + ntx, :]

    tt = np.arange(dd.shape[0]) * dt + trow_start + idx * dt

    hdulist.close()
    
    return tt, freqs, dd


def avg_chan(dd, nchan=4):
    nsteps = int(dd.shape[1] / nchan)
    for nn in xrange(nsteps):
        dd[:,nn*nchan:(nn+1)*nchan] = \
            np.outer(np.mean(dd[:, nn*nchan:(nn+1)*nchan], axis=1), np.ones(nchan))
    return dd


def get_dedispersed_timeseries(tt, freqs, dat, dm):
    dt = tt[1] - tt[0]
    df = freqs[1] - freqs[0]
    ddm = dedisp.dedisperse_one(dat.T, dm, dt, df, freqs[0])
    return ddm


def get_ddm_spec_mod(tt, freqs, dat, dm, chan_weights=None):
    dt = tt[1] - tt[0]
    df = freqs[1] - freqs[0]
    dd_dm = dedisp.dedisperse_dspec(dat.T, dm, dt, df, freqs[0]).T
    
    if chan_weights is None:
        ddm = np.mean(dd_dm, axis=1)
        xpk = np.argmax(ddm)
        spec = dd_dm[xpk]
        tpk = tt[xpk]
        mod_idx = np.std(dd_dm, axis=1) / np.mean(dd_dm, axis=1)

    else:
        ddm = np.sum(dd_dm * chan_weights, axis=1) / np.sum(chan_weights)
        xpk = np.argmax(ddm)
        spec = dd_dm[xpk] * chan_weights
        tpk = tt[xpk]
        mom2 = np.sum(dd_dm**2.0 * chan_weights, axis=1) / np.sum(chan_weights)
        mom1 = np.sum(dd_dm * chan_weights, axis=1) / np.sum(chan_weights)
        mod_idx = (mom2 - mom1**2.0) / mom1**2.0

    return tpk, ddm, spec, mod_idx


def make_plot(infile, tmid, tsearch, tshow, DM, beamnum=-999, 
              chan_weights=None, outfile=None, nchan_avg=1):
    tstart = tmid - tsearch * 0.5
    tt, freqs, dd = get_data(infile, tstart, tsearch)

    if nchan_avg > 1:
        dd = avg_chan(dd, nchan=nchan_avg)
    else: pass

    tpk, ddm, spec, midx = get_ddm_spec_mod(tt, freqs, dd, DM, chan_weights)
    tpk0, ddm0, spec0, midx0 = get_ddm_spec_mod(tt, freqs, dd, 0.0)

    if outfile is not None:
        plt.ioff()
    
    # Make Figure
    fig = plt.figure(figsize=(10,8))

    # Some params 
    dspec_snr = 20 * np.sqrt(nchan_avg)
    #cmap = plt.cm.coolwarm
    cmap = plt.cm.coolwarm

    # Dynamic Spectrum Axis
    bpad = 0.1
    wpad = hpad = 0.01
    wds = hds = 0.55
    xds = bpad 
    yds = bpad
    
    # Time Series Axis
    hts = 1.0 - 2 * bpad - hpad - hds
    wts = wds
    xts = xds
    yts = yds + hds + hpad 
    
    # Spectrum Axis
    hs = hds 
    ws = 1.0 - 2 * bpad - wpad - wds 
    xs = xds + wds + wpad 
    ys = yds 

    # Modulation Index / SNR Axis
    mod_pad = 0.035
    hm = hts - mod_pad
    wm = ws  - mod_pad 
    xm = xs  + mod_pad
    ym = yts + mod_pad

    ## Some Useful Limits ##
    df = freqs[1] - freqs[0]
    dt = tt[1] - tt[0]

    tlim = (- 0.4 * tshow,  0.6 * tshow)
    flim = (freqs[0] - 0.5 * df, freqs[-1] + 0.5 * df)

    ###  Make Dynamic Spectrum Plot ###
    ax_ds = fig.add_axes([xds, yds, wds, hds])
    dd_sig = np.std(dd)

    #im_ext = [tt[0] - 0.5 * dt, tt[-1] + 0.5 * dt,
    #          freqs[0] - 0.5 * df, freqs[-1] + 0.5 * df]

    im_ext = [tt[0] - tpk, tt[-1] - tpk, freqs[0], freqs[-1]]

    ax_ds.imshow(dd.T / dd_sig, interpolation='nearest', origin='lower', 
                 aspect='auto', extent = im_ext, cmap=cmap, 
                 vmin= -dspec_snr * dd_sig, vmax = dspec_snr * dd_sig)

    # Add Dispersion Sweep
    offset = dt 
    dm_curve = dedisp.dm_delay(freqs, freqs[-1], DM)
    ax_ds.plot(dm_curve - offset, freqs, lw=2, c='k', alpha=0.2)
    ax_ds.plot(dm_curve + offset, freqs, lw=2, c='k', alpha=0.2)

    # Set limits
    ax_ds.set_xlim(tlim)
    ax_ds.set_ylim(flim)

    # Add labels
    ax_ds.set_ylabel('Frequency (MHz)', fontsize=14, labelpad=10)
    ax_ds.set_xlabel('Time Offset (s)', fontsize=14, labelpad=10)

    ### Make Time Series Plot ###
    ax_ts = fig.add_axes([xts, yts, wts, hts])
    ddm_sig = np.std(ddm)

    ax_ts.plot(tt - tpk, ddm / ddm_sig, c='k')
    #ax_ts.plot(tt - tpk, ddm0 / ddm_sig, c='LimeGreen') 
    ax_ts.axvline(0, lw=3, c='r', alpha=0.2)
    ax_ts.axhline(y=0, lw=3, c='k', alpha=0.2)

    ax_ts.set_xlim(tlim)
    ax_ts.set_xticklabels([])
    ax_ts.set_ylabel('SNR', fontsize=14)
    ax_ts.set_title('De-dispersed Time Series', fontsize=14)


    ### Make Spectrum Plot ###
    ax_s = fig.add_axes([xs, ys, ws, hs])    
    ax_s.plot(spec / dd_sig, freqs, c='k')
    #ax_s.plot(spec0 / dd_sig, freqs, c='LimeGreen')
    ax_s.axvline(x=0, lw=3, c='k', alpha=0.2)
    
    ax_s.set_ylim(flim)
    ax_s.set_yticklabels([])
    ax_s.set_xlabel('SNR', fontsize=14)
    ax_s.set_ylabel('De-dispersed Spectrum', fontsize=14, 
                    rotation=-90, labelpad=20)
    ax_s.yaxis.set_label_position("right")

    ### Make Mod Index Plot ###
    ax_m = fig.add_axes([xm, ym, wm, hm])
    snrs = ddm / ddm_sig 
    xpk = np.where(tt == tpk)[0]

    ax_m.plot(midx, snrs, ls='', marker='o', c='k', alpha=0.5)
    #ax_m.plot(midx0, ddm0/ddm_sig, ls='', marker='o', c='LimeGreen', alpha=0.5)
    ax_m.plot(midx[xpk], snrs[xpk], ls='', marker='o', c='r')

    if chan_weights is None:
        snr_cut = np.sqrt(len(freqs) - 1)
    else:
        snr_cut = np.sqrt(np.sum(chan_weights) - 1)

    ax_m.text(0.95, 0.95, r"$m_{\rm I} = %.2f$" %midx[xpk],
              ha='right', va='top', transform=ax_m.transAxes)

    ax_m.axvline(x=snr_cut, c='b', lw=3, alpha=0.2)
    
    #ax_m.set_xscale('log')
    #ax_m.set_yscale('log')

    #ax_m.set_xlim(0.3, 3000)
    #ax_m.set_ylim(0.003, 30)
    ax_m.set_xlim(0, np.sqrt(len(freqs) -1) * 2 )
    ax_m.set_ylim(0, 10)

    ax_m.set_ylabel('SNR', rotation=-90, labelpad=20, fontsize=14)
    ax_m.set_xlabel("Mod Index", fontsize=14)

    ax_m.yaxis.set_label_position("right")
    ax_m.xaxis.set_label_position("top")

    # Set the figure title
    title_str = r"$t_{\rm pk} = %.3f\, {\rm s}$" %(tpk) + "     " +\
                r"${\rm DM} = %.1f\, {\rm pc\, cm}^{-3}$" %(DM) + "     " +\
                r"${\rm Beam} = %d$" %(beamnum)
    fig.suptitle(title_str, fontsize=16)
    
    # Add a footer with the file name
    fig.text(0.02, 0.02, infile.split('/')[-1], 
             va='bottom', ha='left', fontsize=10)
    
    if outfile is not None:
        plt.savefig(outfile, dpi=100, bbox_inches='tight')
        plt.close()
        plt.ion()

    else:
        plt.show()
    return    


def dspec_stats(infile, tmid, tsearch, tshow, DM, chan_weights=None):
    tstart = tmid - tsearch * 0.5
    tt, freqs, dd = get_data(infile, tstart, tsearch)
    tpk, ddm, spec, midx = get_ddm_spec_mod(tt, freqs, dd, DM, chan_weights)
    tpk0, ddm0, spec0, midx0 = get_ddm_spec_mod(tt, freqs, dd, 0.0)

    dd_sig = np.std(dd)
    ddm_sig = np.std(ddm)
    snrs = ddm / ddm_sig 
    xpk = np.where(tt == tpk)[0][0]
    
    return tpk, snrs[xpk], midx[xpk]
    

def get_file_name(indir, basename, beamnum, nzeros=4):
    return "{}/{}_beam{:0{}}.fits".format(indir, basename, beamnum, nzeros)


def plots_from_candlist(candfile, fitsdir, basename, tsearch, tshow, 
                        nzeros=4, chan_weights=None, fixdm=None, nchan_avg=1):
    dat = np.loadtxt(candfile)
    times = dat[:, 0]
    beams = dat[:, 1].astype('int')
    dms   = dat[:, 2]
    snrs  = dat[:, 3]
    N = len(dat)

    if fixdm is not None:
        dms = dms * 0 + fixdm

    for ii in xrange(N):
        print("%d / %d" %(ii, N))
        infile = get_file_name(fitsdir, basename, beams[ii], nzeros=nzeros)
        outfile = "%s_beam%04d_T%08.3f_dspec.png" %( \
            basename, beams[ii], times[ii])
        make_plot(infile, times[ii], tsearch, tshow, dms[ii], beamnum=beams[ii],
                  chan_weights=chan_weights, outfile=outfile, nchan_avg=nchan_avg)    
    return


def stats_from_candlist(candfile, fitsdir, basename, tsearch, tshow, 
                        nzeros=4, chan_weights=None, fixdm=None):
    dat = np.loadtxt(candfile)
    times = dat[:, 0]
    beams = dat[:, 1].astype('int')
    dms   = dat[:, 2]
    snrs  = dat[:, 3]
    N = len(dat)

    Cand = namedtuple('Cand', ['tt', 'beam', 'dm', 'snr', 'fit_tt', 
                               'fit_snr', 'fit_midx'])
    candlist = []

    if fixdm is not None:
        dms = dms * 0 + fixdm

    for ii in xrange(N):
        print("%d / %d" %(ii, N))
        infile = get_file_name(fitsdir, basename, beams[ii], nzeros=nzeros)
        ftt, fss, fmm = dspec_stats(infile, times[ii], tsearch, tshow, dms[ii])
        cc = Cand(tt=times[ii], beam=beams[ii], dm=dms[ii], snr=snrs[ii], 
                  fit_tt = ftt, fit_snr=fss, fit_midx=fmm)
        candlist.append(cc)
    return candlist



def cands_from_candlist(candfile, fixdm=None):
    dat = np.loadtxt(candfile)
    times = dat[:, 0]
    beams = dat[:, 1].astype('int')
    dms   = dat[:, 2]
    snrs  = dat[:, 3]
    N = len(dat)

    if dat.shape[-1] == 6:
        fit_snr = dat[:, 4]
        fit_midx = dat[:, 5]
    else:
        fit_snr = np.zeros(len(snrs))
        fit_midx = np.zeros(len(snrs))

    Cand = namedtuple('Cand', ['tt', 'beam', 'dm', 'snr', 'fit_snr', 'fit_midx'])
    candlist = []

    if fixdm is not None:
        dms = dms * 0 + fixdm

    for ii in xrange(N):
        cc = Cand(tt=times[ii], beam=beams[ii], dm=dms[ii], snr=snrs[ii], 
                  fit_snr=fit_snr[ii], fit_midx=fit_midx[ii])
        candlist.append(cc)
    return candlist



def adjacent_beam_timeseries(cand, coords, maxsep, fitsdir, basename, 
                             tsearch, tshow, nzeros=4, chan_weights=None, fixdm=None):
    tstart   = cand.tt - tsearch * 0.5
    beam_num = cand.beam
    DM       = cand.dm if fixdm is None else fixdm

    infile = get_file_name(fitsdir, basename, beam_num, nzeros=nzeros)
    beams, offsets = get_nearby_beams(coords, beam_num, maxsep)
    
    tt_list  = []
    ddm_list = []
    
    for bb in beams:
        infile = get_file_name(fitsdir, basename, bb, nzeros=nzeros)
        tti, ffi, ddi = get_data(infile, tstart, tsearch)
        tpki, ddmi, speci, midxi = get_ddm_spec_mod(tti, ffi, ddi, DM, chan_weights)

        tt_list.append(tti)
        ddm_list.append(ddmi)

    tts = np.array(tt_list)
    ddms = np.array(ddm_list)

    return tts, ddms, beams


def get_skycoords(beamfile):
    bb = np.load(beamfile)
    beams = np.arange(len(bb), dtype='int')
    ra_strs  = bb[:, 1]
    dec_strs = bb[:, 2]
    dec_strs = np.array([ dd.replace('.', ':', 2) for dd in dec_strs ])
    coords = SkyCoord(ra=ra_strs, dec=dec_strs, unit=(u.hour, u.deg))
    return coords


def get_nearby_beams(coords, bnum, maxsep):
    """
    Read in a beamlist and find all beams within
    'maxsep' arcseconds of beam number 'bnum'.
    Returns the beam numbers
    """
    pos0 = coords[bnum]
    dd = pos0.separation(coords).arcsec
    xx = np.where(dd <= maxsep)[0]

    yy = np.argsort(dd[xx])
    xx = xx[yy]
    dd = dd[xx]

    return xx, dd


def get_beam_offsets(coords, beams, fwhm):
    cc = coords[beams]
    ra_offset = ((cc.ra - cc[0].ra) * np.cos(cc[0].dec.to('radian'))).arcsec
    dec_offset = (cc.dec - cc[0].dec).arcsec
    return ra_offset, dec_offset


def make_beam_plot(coords, beams, fwhm, showbeam=None, add_label=False):
    ra_offs, dec_offs = get_beam_offsets(coords, beams, fwhm)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for ii, bb in enumerate(beams):
        cc = (ra_offs[ii], dec_offs[ii])
        if showbeam is not None and showbeam == bb:
            bcirc = Circle(cc, 0.5 * fwhm, fill=True, lw=0, color='r')
        else:
            bcirc = Circle(cc, 0.5 * fwhm, fill=True, lw=0, color='b', alpha=0.3)
        ax.add_artist(bcirc)
        if add_label:
            ax.text(cc[0], cc[1], "%d" %(bb), va='center', ha='center', fontsize=12)
    
    ax.set_aspect('equal')
    pad = 0.6 * fwhm
    ax.set_xlim( np.min(ra_offs) - pad, np.max(ra_offs) + pad )
    ax.set_ylim( np.min(dec_offs) - pad, np.max(dec_offs) + pad )
    ax.set_xlabel("RA Offset (arcsec)")
    ax.set_ylabel("Dec Offset (arcsec)")
    
    plt.show()
    return    


def get_beam_subplot(ax, coords, beams, fwhm, showbeam=None, add_label=False):
    ra_offs, dec_offs = get_beam_offsets(coords, beams, fwhm)

    for ii, bb in enumerate(beams):
        cc = (ra_offs[ii], dec_offs[ii])
        if showbeam is not None and showbeam == bb:
            bcirc = Circle(cc, 0.5 * fwhm, fill=True, lw=0, color='r')
        else:
            bcirc = Circle(cc, 0.5 * fwhm, fill=True, lw=0, color='b', alpha=0.3)
        ax.add_artist(bcirc)
        if add_label:
            ax.text(cc[0], cc[1], "%d" %(bb), va='center', ha='center', fontsize=16)
    
    ax.set_aspect('equal')
    pad = 0.6 * fwhm
    ax.set_xlim( np.min(ra_offs) - pad, np.max(ra_offs) + pad )
    ax.set_ylim( np.min(dec_offs) - pad, np.max(dec_offs) + pad )
    ax.set_xlabel("RA Offset (arcsec)")
    ax.set_ylabel("Dec Offset (arcsec)")
    
    return ax


def make_snr_beam_plot(cand, coords, maxsep, fwhm, fitsdir, basename,
                       tsearch, tshow, nzeros=4, chan_weights=None, 
                       fixdm=None, add_label=True, outfile=None):

    tts, ddms, beams = adjacent_beam_timeseries(cand, coords, maxsep, fitsdir, basename,
                                                tsearch, tshow, nzeros=nzeros, 
                                                chan_weights=chan_weights, fixdm=fixdm)
    xpk = np.argmax(ddms[0])
    sigmas = np.std(ddms, axis=1)
    snrs = ddms[:, xpk] / sigmas

    DM = cand.DM if fixdm is None else fixdm
    
    ra_offs, dec_offs = get_beam_offsets(coords, beams, fwhm)

    if outfile is not None:
        plt.ioff()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)

    cmap = plt.cm.coolwarm
    norm = Normalize(vmin=-6, vmax=6)

    for ii, bb in enumerate(beams):
        cc = (ra_offs[ii], dec_offs[ii])
        bcirc = Circle(cc, 0.5 * fwhm, fill=False, color='k', lw=3)
        ax.add_artist(bcirc)

    for ii, bb in enumerate(beams):
        cc = (ra_offs[ii], dec_offs[ii])
        bcolor = cmap(norm(snrs[ii]))
        bcirc = Circle(cc, 0.5 * fwhm, fill=True, color=bcolor, alpha=0.9, lw=0)
        ax.add_artist(bcirc)
        if add_label:
            ax.text(cc[0], cc[1], "%d" %(bb), va='center', ha='center', fontsize=12)

    cbax = fig.add_axes([0.88, 0.15, 0.05, 0.7])
    cb1 = ColorbarBase(cbax, cmap=cmap, norm=norm)
    
    ax.set_aspect('equal')
    pad = 0.6 * fwhm
    #ax.set_xlim( np.min(ra_offs) - pad, np.max(ra_offs) + pad )
    #ax.set_ylim( np.min(dec_offs) - pad, np.max(dec_offs) + pad )
    ax.set_xlim( -maxsep - pad, maxsep + pad)
    ax.set_ylim( -maxsep - pad, maxsep + pad)
    ax.set_xlabel("RA Offset (arcsec)")
    ax.set_ylabel("Dec Offset (arcsec)")
    ax.set_title(r"$\rm Beam %04d, $" %beams[0] + "  " + \
                 r"$\rm Time = %.3f\, s, $" %tts[0][xpk] + "  " + \
                 r"$\rm DM = %.1f\, {pc\,cm}^{-3}$" %DM)                     
    
    if outfile is not None:
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()
        plt.ion()
    else:
        plt.show()
    return


def snr_beam_plot_from_candlist(candlist, coords, maxsep, fwhm, fitsdir, basename,
                                tsearch, tshow, nzeros=4, chan_weights=None,
                                fixdm=None, add_label=True, outbase="beamplot"):

    for ii, cand in enumerate(candlist):
        print("Cand %d / %d" %(ii, len(candlist)))
        outfile = "%s_beam%04d_T%08.3f_bplot.png" %(outbase, cand.beam, cand.tt)
        make_snr_beam_plot(cand, coords, maxsep, fwhm, fitsdir, basename,
                           tsearch, tshow, nzeros=nzeros, chan_weights=chan_weights,
                           fixdm=fixdm, add_label=add_label, outfile=outfile)

    return
    
    
        
def array_attr(clist, attr):
    if hasattr(clist[0], attr):
        return np.array([ getattr(cc, attr) for cc in clist ])
    else:
        print "Object has no attribute: %s" %attr
        return


def write_select_cands(cands, outfile):
    fout = open(outfile, 'w')
    hdr = "#{:<12}{:<10}{:<10}{:<10}{:<10}{:<10}".format(\
        "Time", "Beam", "DM", "SNR", "New SNR", "Mod Index")
    fout.write(hdr + "\n")
        
    for cc in cands:
        outstr = "{:<12.3f}{:<10d}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}".format(\
            cc.tt, cc.beam, cc.dm, cc.snr, cc.fit_snr, cc.fit_midx)
        fout.write(outstr + "\n")
    fout.close()
    return


# MAIN
plt.rc('font', family='serif')

beamfile = '/lustre/aoc/projects/16A-459/beamforming/tiling/beamlist_3GHz_3arcmin.npy'
beam_coords = get_skycoords(beamfile)
beam_coords = beam_coords[:2000]

fitsdir = '/lustre/aoc/projects/16A-459/beamforming/57511/part1/psrfits'
candfile = '/lustre/aoc/projects/16A-459/beamforming/57511/plots/mjd57511_part1.txt'
basename = 'mjd57511_part1'

#candlist = stats_from_candlist(candfile, fitsdir, basename, 3.0, 
#                               0.45, fixdm=557.0)

cutfile = 'mjd57511_part1_top.txt'
candlist = cands_from_candlist(cutfile, fixdm=557.0)

plots_from_candlist(cutfile, fitsdir, basename, 4.0, 0.45, fixdm=557.0, nchan_avg=8)

snr_beam_plot_from_candlist(candlist, beam_coords, 15.0, 3.0, fitsdir, basename,
                            4.0, 0.45, fixdm=557.0, outbase=basename)


