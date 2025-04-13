import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.coordinates import Angle
import astropy.coordinates as acoord 

def get_tiles_test(x0, y0, Nr, theta, align='x'):
    jj = np.arange(2 * Nr + 1) - Nr
    nb = 2 * Nr - np.abs(jj) + 1

    if align == 'x':
        yvals = y0 + jj * theta * np.sqrt(3) / 2.0
        xx = [ x0 + (np.arange(njj) - 0.5 * (njj-1.0)) * theta for njj in nb]
        yy = [ np.repeat(yvals[ii], njj) for ii, njj in enumerate(nb) ]
    else:
        xvals = x0 + jj * theta * np.sqrt(3) / 2.0
        yy = [ y0 + (np.arange(njj) - 0.5 * (njj-1.0)) * theta for njj in nb]
        xx = [ np.repeat(xvals[ii], njj) for ii, njj in enumerate(nb) ]
    
    xx = np.hstack(xx)
    yy = np.hstack(yy)

    return xx, yy


def tile_circle_test(dcirc, dbeam, x0, y0, align='x'):
    Nr = int(np.ceil( (0.5 * dcirc / dbeam) * (2.0/np.sqrt(3.0))))
    xx, yy = get_tiles_test(x0, y0, Nr, dbeam, align=align)
    return xx, yy


def filter_beams_test(xx, yy, x0, y0, rmin=-1, rmax=1e10):
    dd = np.sqrt( (xx - x0)**2.0 + (yy - y0)**2.0 )
    xii = np.where( (dd < rmax) & (dd > rmin) )[0]
    return xx[xii], yy[xii]


def get_tiles(pos0, Nr, theta, align='x'):
    jj = np.arange(2 * Nr + 1) - Nr
    nb = 2 * Nr - np.abs(jj) + 1

    dec_step = theta * u.arcsec
    ra_step = theta * u.arcsec / np.cos(pos0.dec.to('radian'))
    
    if align == 'x':
        dec_vals = pos0.dec + jj * dec_step * np.sqrt(3) / 2.0
        ras  = [ pos0.ra + (np.arange(njj) - 0.5 * (njj-1.0)) * ra_step for njj in nb]
        decs = [ np.repeat(dec_vals[ii], njj) for ii, njj in enumerate(nb) ]
    else:
        ra_vals = pos0.ra + jj * ra_step * np.sqrt(3) / 2.0
        decs = [ pos0.dec + (np.arange(njj) - 0.5 * (njj-1.0)) * dec_step for njj in nb]
        ras  = [ np.repeat(ra_vals[ii], njj) for ii, njj in enumerate(nb) ]
        
    ras_out = []
    decs_out = []
    for ii in range(len(ras)):
        ras_out.extend(ras[ii])
        decs_out.extend(decs[ii])

    cc = SkyCoord(ra=ras_out, dec=decs_out)

    return ras_out, decs_out, cc


def tile_circle(dcirc, dbeam, pos0, align='x'):
    Nr = int(np.ceil( (0.5 * dcirc / dbeam) * (2.0/np.sqrt(3.0))))
    xx, yy, cc = get_tiles(pos0, Nr, dbeam, align=align)
    return cc


def filter_beams(coords, pos0, rmin=-1, rmax=1e10, dsort=True):
    dd = pos0.separation(coords).arcsec
    xii = np.where( (dd < rmax) & (dd > rmin) )[0]
    if dsort:
        yii = np.argsort(dd[xii])
        zii = xii[yii]
    else:
        zii = xii
    return coords[zii]


def get_and_filter_beams(pos0, dbeam, align='x', 
                         rmin=-1, rmax=1e10, dsort=True):
    cc = tile_circle(rmax * 2, dbeam, pos0, align=align)
    cc_out = filter_beams(cc, pos0, rmin=rmin, rmax=rmax, dsort=dsort)
    return cc_out


def multi_loc_beams(pos_list, dbeam, align='x',
                    rmin=-1, rmax=1e10, dsort=True):
    beam_groups = []
    for pos in pos_list:
        cc_tmp = get_and_filter_beams(pos, dbeam, align=align,
                                      rmin=rmin, rmax=rmax, dsort=dsort)
        beam_groups.append( cc_tmp )
    cc_out = acoord.concatenate( beam_groups )
    return cc_out


def group_beams(coords, pos0, nbeams):
    dd = pos0.separation(coords).arcsec
    ddi = np.argsort(dd)
    nidx = np.arange(0, int(len(dd) / nbeams) + 1)
    groups = []
    for nn in nidx:
        groups.append(coords[ddi[nn * nbeams : (nn+1) * nbeams]])
    return groups


def make_plot(xx, yy, theta):
    coords = zip(xx, yy)
    fig = plt.figure()
    ax = fig.add_subplot(111) 
    for cc in coords:
        beam_circ = Circle(cc, theta/2.0, color='0.8', fill=False, lw=1)
        ax.add_artist(beam_circ)
    ax.set_aspect('equal')
    pad = 2 * theta
    ax.set_xlim(np.min(xx) - pad, np.max(xx) + pad)
    ax.set_ylim(np.min(yy) - pad, np.max(yy) + pad)
    plt.show()
    return


def make_plot_layers(xxlist, yylist, theta):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    clist = ['b', 'r', 'LimeGreen', 'Orange', 'Purple']
    for ii in range(len(xxlist)):
        xx = xxlist[ii]
        yy = yylist[ii]
        coords = zip(xx, yy)
        color = clist[ii % 5]
        for cc in coords:
            beam_circ = Circle(cc, theta/2.0, color=color, fill=False, lw=1)
            ax.add_artist(beam_circ)
    ax.set_aspect('equal')
    pad = 2 * theta
    ax.set_xlim(np.min(xx) - pad, np.max(xx) + pad)
    ax.set_ylim(np.min(yy) - pad, np.max(yy) + pad)
    plt.show()
    return


def write_reg_file(outfile, coords, radius, color=None, hdrfile=None, add_nums=True):
    """
    coords : list of SkyCoord objects
    radius : radius of circle in arcsec 
    outfile: output file name
    """
    fout = open(outfile, 'w')
    if hdrfile is not None:
        fin = open(hdrfile, 'r')
        lines = fin.readlines()
        for line in lines:
            fout.write(line)
        fin.close()

    for cc in coords:
        if color is not None:
            outstr = "circle(%s, %s, %.2f\") #color=%s\n" %(cc.ra.to_string(decimal=True, precision=8),
                                                            cc.dec.to_string(decimal=True, precision=8),
                                                            radius, color)
        else:
            outstr = "circle(%s, %s, %.2f\")\n" %(cc.ra.to_string(decimal=True, precision=8),
                                                  cc.dec.to_string(decimal=True, precision=8),
                                                  radius)
    
        fout.write(outstr)

    if add_nums:
        for ii, cc in enumerate(coords):
            outstr_num = "text %s %s {%d}\n" %(cc.ra.to_string(decimal=True, precision=8),
                                             cc.dec.to_string(decimal=True, precision=8),
                                             ii)
            fout.write(outstr_num)

    fout.close()
    return


def make_beam_list(coords):
    blist = []
    for cc in coords:
        ra_str  = cc.ra.to_string(u.hour, sep=':', pad=True)
        dec_str = cc.dec.to_string(u.deg, sep='.', pad=True, 
                                   alwayssign=True)
        blist.append( ["J2000", ra_str, dec_str] )
    blist = np.array(blist, dtype=str)
    return blist


def convert_ra(instr):
    parts = instr.split(":")
    return "%sh%sm%ss" %(parts[0], parts[1], parts[2])


def convert_dec(instr):
    parts = instr.split(".", 2)
    return "%sd%sm%ss" %(parts[0], parts[1], parts[2])


def blist_to_cc(blist):
    sra  = blist[:, 1]
    sdec = blist[:, 2]

    gsra  = np.array([ convert_ra(srai) for srai in sra ])
    gsdec = np.array([ convert_dec(sdeci) for sdeci in sdec ])

    #coords = np.array([ gsra[ii] + ' ' + gsdec[ii] for ii in range(len(gsra)) ])

    cc = SkyCoord(ra=gsra, dec=gsdec)

    #cc = [ SkyCoord(cci) for cci in coords ]
    return cc


def gc_hex_tile(pos0, hexlist, hdrfile):
    theta = 3.0 
    Nr = 20
    Nrow0 = 2 * Nr + 1
    # Tiling hex # 
    # Makes 37 x 1261 beams
    theta_tile = (Nrow0 * theta) * (np.sqrt(3)/2.)
    rr_t, dd_t, cc_t = get_tiles(pos0, 3, theta_tile, align='y')
    # Sort by dist from pos0 
    cc_t = filter_beams(cc_t, pos0, rmax=10**5)

    nc = len(hexlist)
    
    blist = []
    for ii, pp_ii in enumerate(cc_t):
        print(ii)
        ras, decs, cc = get_tiles(pp_ii, Nr, theta)
        cc = filter_beams(cc, pp_ii, rmax=10**5)
        
        outname = "gc_hex_%02d.reg" %ii
        write_reg_file(outname, cc, theta/2.0, hdrfile=hdrfile, 
                       color=hexlist[ii%nc], add_nums=False)
        
        blist_ii = make_beam_list(cc)
        blist_ii = blist_ii.tolist()
        
        blist += blist_ii 

    return blist







# MAIN

#hexlist = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3"]
hexlist = ["#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3"]
hdrfile = "/Users/wharton/src/ms_beamform/tiling/header.txt"
#pos0 = SkyCoord("12h24m22.03080s -64d07m53.2091s")  #bright source
#pos0 = SkyCoord("12h21m50.736s -64d07m42.416")      # pulsar 
pos0 = SkyCoord("12h23m54.487s -64d17m29.36")      # transient


cc = get_and_filter_beams(pos0, 2.0, rmax=2.0)
blist = make_beam_list(cc)
#np.save(blist, 
print(blist)

