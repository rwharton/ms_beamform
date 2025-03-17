import numpy as np
import multiprocessing
import pwkit.environments.casa.util as casautil
import time, sys, glob
from functools import partial
from contextlib import closing
import gc

import beam_extract as be
import beam_combine as bc

ms = casautil.tools.ms()
msmd = casautil.tools.msmetadata()
qa = casautil.tools.quanta()
me = casautil.tools.measures()


if __name__ == "__main__":
    print("START")

    do_avg_vis = 1
    do_get_tf  = 1
    do_get_mjd = 1
    do_fits    = 1

    datdir = '/hercules/results/rwharton/fastvis_gc/ms_data'
    part_num = int(sys.argv[-1])
    infile = '%s/57519_part%d.ms' %(datdir, part_num)
    basename = 'mjd57519_part%d' %(part_num)
    src_base = "MJD57519"
    logfile = "%s.log" %basename

    beamlist_file = '/hercules/results/rwharton/fastvis_gc/proc/cannon_beams.npy'
    beam_list = np.load(beamlist_file)
    beam_nums = np.arange(len(beam_list), dtype='int')

    beam_locs = beam_list[:]
    beam_nums = beam_nums[:]

    pos0 = ["J2000", "17:45:40.03", "-29.00.28.1"]
    ell_ems = np.vstack( (be.get_ell_m(pp, pos0) for pp in beam_locs) )

    spw_list = range(16)

    proc_kwargs = {'tstep' : 1.0,
                   'nproc' : 12,
                   'nchunks' : 12,
                   'write_tstep' : 30,
                   'datacolumn' : 'corrected',
                   'basename' : basename,
                   'target_id' : 2, 'phase_id' : 3, 'flux_id' : 0,
                   'Nbl_min' : 0,
                   'Nskip' : 0,
                   'use_flags' : False}

    if do_avg_vis:
        tstart = time.time()
        mdats, tt, freqs, proc_times = be.average_visibilities(infile, ell_ems, beam_nums,
                                                               spws=spw_list, logfile=logfile,
                                                               **proc_kwargs)
        dt = time.time() - tstart
        print("TOTAL TIME = %.2f min" %(dt / 60.0))

    if do_get_tf:
        be.get_and_save_spw_freqs(infile, spw_list, basename=basename)

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
        if part_num == 1:
            skip_sec = 500.0
        else:
            skip_sec = 0.0
        
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
