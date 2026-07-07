import numpy as np
import multiprocessing
import time, sys, glob
from functools import partial
from contextlib import closing
import gc
import casatools

import beam_extract_old as be
import beam_combine as bc

ms = casatools.ms()
msmd = casatools.msmetadata()
qa = casatools.quanta()
me = casatools.measures()

#sys.exit()

if __name__ == "__main__":
    print("START")

    do_avg_vis = 1
    do_get_tf  = 1
    do_get_mjd = 0

    datdir = '/data/GLOSTAR/SGRA/ms_data'
    inbase = 'SGRA_scan1'
    basename = "sgra_test"
    src_base = "sgra_test"
    logfile = "%s.log" %basename

    beamlist_file = '/data/GLOSTAR/SGRA/beams/sgra_beams.npy'
    beam_list = np.load(beamlist_file)
    beam_nums = np.arange(len(beam_list), dtype=int)

    beam_locs = beam_list[:]
    beam_nums = beam_nums[:]

    spw_list = range(16)

    proc_kwargs = {'tstep' : 300,
                   'nproc' : 20,
                   'write_tstep' : 5,
                   'datacolumn' : 'data',
                   'basename' : basename,
                   'target_id' : 0, 'phase_id' : 3, 'flux_id' : 1,
                   'Nbl_min' : 0,
                   'Nskip' : 0,
                   'use_flags' : True, 
                   'use_weights' : True, 
                   'uv_lam_taper' : 0.0, #2e3, 
                   'uvlim' : [1.0, 1e10]}

    # set ms file
    msfile = "%s/%s.ms" %(datdir, inbase)

    # get phase center
    pos0 = be.get_field_phasecenter(msfile, proc_kwargs['target_id'])
    ell_ems = np.vstack( [be.get_ell_m(pp, pos0) for pp in beam_locs] )

    if do_avg_vis:
        full_freqs = []
        tstart = time.time()

        mdats, tt, freqs, proc_times = \
                be.average_visibilities(msfile, ell_ems, beam_nums,
                                        spws=spw_list, logfile=logfile,
                                        **proc_kwargs)

        dt = time.time() - tstart
        print("TOTAL TIME = %.2f min" %(dt / 60.0))

    if do_get_tf:
        be.get_and_save_spw_freqs(msfile, spw_list, basename=basename)

    if do_get_mjd:
        tfiles = glob.glob("%s_tt_step*.npy" %(basename))
        tfiles.sort()
        tt = np.load(tfiles[0])
        tdict = be.get_time_info(tt[0])
        np.save("%s_mjd_start.npy" %(basename), [tdict['mjd']])
        
    print("STOP")
