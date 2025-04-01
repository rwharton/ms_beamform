import numpy as np
import multiprocessing
import time, sys, glob
from functools import partial
from contextlib import closing
import gc
import casatools

import beam_extract_wt as be
import beam_combine as bc

ms = casatools.ms()
msmd = casatools.msmetadata()
qa = casatools.quanta()
me = casatools.measures()


if __name__ == "__main__":
    print("START")

    do_avg_vis = 1
    do_get_tf  = 1
    do_get_mjd = 1
    do_fits    = 0

    datdir = '/data/S_1102/selfcal_data'
    inbase = 'MSGPS_S_1102'
    basename = 's1102'
    src_base = "S1102"
    logfile = "%s.log" %basename

    beamlist_file = '/data/S_1102/beams/S1102_full.npy'
    beam_list = np.load(beamlist_file)
    beam_list = beam_list[:6]
    beam_nums = np.arange(len(beam_list), dtype=int)

    beam_locs = beam_list[:]
    beam_nums = beam_nums[:]

    pos0 = ["J2000", "13:46:39.26", "-63.03.37.9"]
    ell_ems = np.vstack( (be.get_ell_m(pp, pos0) for pp in beam_locs) )

    spw_list = range(16)

    proc_kwargs = {'tstep' : 9999.0,
                   'nproc' : 10,
                   'write_tstep' : 99999,
                   'datacolumn' : 'corrected',
                   'basename' : basename,
                   'target_id' : 0, 'phase_id' : 3, 'flux_id' : 1,
                   'Nbl_min' : 0,
                   'Nskip' : 0,
                   'use_flags' : True}

    if do_avg_vis:
        for spw_ii in spw_list:
            tstart = time.time()
            proc_kwargs['basename'] = "spw%02d" %spw_ii
            inms = "%s/%s_spw%03d.ms" %(datdir, inbase, spw_ii)
            mdats, tt, freqs, proc_times = be.average_visibilities(inms, ell_ems, beam_nums,
                                                                   spws=[0], logfile=logfile,
                                                                   **proc_kwargs)
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
        
    
    if do_fits:
        tt, freqs = bc.get_freqs_and_times(basename)
        tmjd = np.load("%s_mjd_start.npy" %basename)
        mjd_start = tmjd[0]
    
        nchan = 512
        chan_weights = np.ones(512)
        #chan_weights[128:-128] = 0

        # SKIP SEC
        
        obs_params = {"mjd_start": mjd_start,
                      "dt"       : 0.010,
                      "freq_lo"  : 1977.0,
                      "chan_df"  : 4.0,
                      "beam_info": np.array([3.0, 3.0, 0.0]),
                      'chan_weights': chan_weights,
                      'tgrow_zeros' : 3,
                      'skip_start_zeros' : True}

        bc.combine_beams_to_psrfits(basename, src_base, beam_nums, beam_list,
                                    out_type=np.float32, keep_npy=False,
                                    obs_params=obs_params, skip_sec=skip_sec)


    print("STOP")
