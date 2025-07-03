import numpy as np
import multiprocessing
import time, sys, glob
from functools import partial
from contextlib import closing
import gc
import casatools

import beam_extract as be
import beam_combine as bc

ms = casatools.ms()
msmd = casatools.msmetadata()
qa = casatools.quanta()
me = casatools.measures()


if __name__ == "__main__":
    print("START")

    do_avg_vis = 1
    do_get_tf  = 1
    do_get_mjd = 0

    datdir = '/data/S_0433/selfcal_data'
    inbase = 'MSGPS_S_0433'
    basename = 's0433'
    src_base = "S0433"
    logfile = "%s.log" %basename

    beamlist_file = '/data/S_0433/new_beams/new_beams.npy'
    beam_list = np.load(beamlist_file)
    beam_nums = np.arange(len(beam_list), dtype=int)

    beam_locs = beam_list[:]
    beam_nums = beam_nums[:]

    spw_list = range(16)

    proc_kwargs = {'tstep' : 9999.0,
                   'nproc' : 10,
                   'write_tstep' : 99999,
                   'datacolumn' : 'corrected',
                   'basename' : basename,
                   'target_id' : 0, 'phase_id' : 3, 'flux_id' : 1,
                   'Nbl_min' : 0,
                   'Nskip' : 0,
                   'use_flags' : True, 
                   'use_weights' : True, 
                   'uv_lam_taper' : 5.5e3, 
                   'uvlim' : [1.0, 1e10]}

    # open first spw to phase center
    msfile = "%s/%s_spw%03d.ms" %(datdir, inbase, 0)
    pos0 = be.get_field_phasecenter(msfile, proc_kwargs['target_id'])
    ell_ems = np.vstack( [be.get_ell_m(pp, pos0) for pp in beam_locs] )

    if do_avg_vis:
        full_freqs = []
        for spw_ii in spw_list:
            tstart = time.time()
            proc_kwargs['basename'] = "spw%02d" %spw_ii
            inms = "%s/%s_spw%03d.ms" %(datdir, inbase, spw_ii)
            mdats, tt, freqs, proc_times = be.average_visibilities(inms, ell_ems, beam_nums,
                                                                   spws=[0], logfile=logfile,
                                                                   **proc_kwargs)
            full_freqs.append( freqs )

        # Write freqs
        full_freqs = np.hstack( full_freqs )
        np.save(f"{basename}_freqs.npy", full_freqs)
            
        # Combine spw
        be.multibeam_combine('.', len(beam_list))

        dt = time.time() - tstart
        print("TOTAL TIME = %.2f min" %(dt / 60.0))

    #if do_get_tf:
    #    be.get_and_save_spw_freqs(infile, spw_list, basename=basename)

    if do_get_mjd:
        tfiles = glob.glob("%s_tt_step*.npy" %(basename))
        tfiles.sort()
        tt = np.load(tfiles[0])
        tdict = be.get_time_info(tt[0])
        np.save("%s_mjd_start.npy" %(basename), [tdict['mjd']])
        
    print("STOP")
