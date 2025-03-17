import numpy as np
import multiprocessing
import pwkit.environments.casa.util as casautil
import time, sys, glob, os
from functools import partial
from contextlib import closing
import gc

ms = casautil.tools.ms()
msmd = casautil.tools.msmetadata()
qa = casautil.tools.quanta()
me = casautil.tools.measures()

def avg_vis_at_newdir(ell_em, dat, ulams, vlams):
    """
    Calculate the average visibilities after shifting to a new
    direction given by ell, m
    """
    #dt_start = time.time()
    ell, m = ell_em
    dt_phase0 = time.time()
    phase_term = np.exp(-2.0j * np.pi * (ulams * ell + vlams * m) )
    #phase_term = np.exp(-2.0j * np.pi * (ulams[:,:,0] * ell + vlams[:,:,0] * m) )
    dt_phase = time.time() - dt_phase0

    dt_mdat0 = time.time()
    #mdat = np.sum(dat * phase_term, axis=1)
    mdat = np.sum( (dat.T * phase_term.T).T, axis=1)
    dt_mdat = time.time() - dt_mdat0
            
    #dt = time.time() - dt_start
    #print("SINGLE VIS = %.2f sec" %dt)
    #print("SINGLE VIS = (%.3f / %.3f)" %(dt_phase, dt_mdat))
    return mdat


def calc_beam(pnum):
    global ell_ems, dat, ulams, vlams
    partial_avg_vis = partial(avg_vis_at_newdir, dat=dat, 
                              ulams=ulams, vlams=vlams)
    one_beam = partial_avg_vis(ell_ems[pnum])
    return (pnum, one_beam)


def initialize_calc_beam(_ell_ems, _dat, _ulams, _vlams):
    global ell_ems, dat, ulams, vlams
    ell_ems = _ell_ems
    dat     = _dat
    ulams   = _ulams
    vlams   = _vlams


def calc_all_beams(ell_ems, dat, ulams, vlams, nproc=4):
    with closing(multiprocessing.Pool(processes=nproc, initializer = initialize_calc_beam, 
                                      initargs = (ell_ems, dat, ulams, vlams))) as pool:
        results = [pool.apply_async(calc_beam, args=(ii,)) for ii in xrange(len(ell_ems))]
        print(len(results))
        all_beams = [p.get() for p in results]
    return all_beams


def get_start_stop_chunk(Nt, Nchunk):
    step = int(Nt / Nchunk) + 1
    lovals = []
    hivals = []
    loval = 0
    hival = 0
    for ii in xrange(Nchunk):
        hival = loval + step
        if hival > Nt:
            hival = Nt
            lovals.append(loval)
            hivals.append(hival)
            break
        else:
            lovals.append(loval)
            hivals.append(hival)
        loval = hival
    return zip(lovals, hivals)


def check_make_dir(dirname):
    """
    Makes new directory called "dirname" if it does't exist.
    If it does exist, does nothing.

    try/except removes race condition (apparently)
    """
    try:
        os.mkdir(dirname)
    except OSError:
        if not os.path.isdir(dirname):
            raise
    return


def save_beam_files(basename, write_step_num, tt, mdats_all, bnums):
    print("    Writing Step %d" %write_step_num)
    sys.stdout.flush()

    tt_file = "%s_tt_step%03d.npy" %(basename, write_step_num)
    np.save(tt_file, tt)
    for ii, mdat in enumerate(mdats_all):
        bdir = "beam%05d" %(bnums[ii])
        check_make_dir(bdir)
        beam_file = "%s_beam%05d_step%03d.npy" %(basename, bnums[ii], write_step_num)
        beam_path = "%s/%s" %(bdir, beam_file)
        np.save(beam_path, mdat)
    return        


def divide_by_baselines(dd, flags_int, Nbl_min):
    """
    Divide data by number of unflagged baselines 
    per channel 
    """
    Ns = np.sum(flags_int, axis=2, dtype=np.float64)
    #xx = np.where(Ns > 0)
    xx = np.where(Ns > Nbl_min)
    Ns_inv = np.zeros(shape=Ns.shape, dtype=np.float64)
    if np.any(xx):
        Ns_inv[xx] = 1.0 / Ns[xx]
    else: pass
    # Reshape so we can mutliply by data array
    #Ns_inv.shape = (1, Ns.shape[0], 1, Ns.shape[1])
    Ns_inv.shape = (Ns.shape[0], Ns.shape[1], 1, Ns.shape[2])
    Ns_inv = np.concatenate( [Ns_inv] * dd.shape[2], axis=2 )
    #Ns_inv = np.concatenate( [Ns_inv] * dd.shape[0], axis=0 )
    dd *= Ns_inv
    return dd

            

def average_visibilities(infile, ell_ems, beam_nums, spws=[0], tstep=1.0, nproc=4, nchunks=4, 
                         write_tstep=0, basename="", target_id=2, phase_id=1, flux_id=0, 
                         Nbl_min=0, datacolumn='corrected', Nskip=0, use_flags=True, logfile=None):
    # Get channel frequencies using the ms metadata tool
    msmd.open(infile)
    freqs = np.array([])
    for spw in spws:
        freqs = np.hstack( (freqs, msmd.chanfreqs(spw)) )
    all_spw = msmd.spwsforfield(target_id)
    msmd.close()

    speed_of_light = 299792458.0 # m/s
    waves = speed_of_light / freqs
    inv_waves = freqs / speed_of_light

    inv_waves_mat = np.vstack( (inv_waves, inv_waves) )

    Nchan = len(freqs)
    Nbeams = len(ell_ems)
    Nspw = len(all_spw)

    # Open MS and average visibilities
    ms.open(infile)

    Tsec = np.diff(ms.range('time')['time'])[0]
    Niter = int(Tsec / tstep) + 1
    #Niter = 60
    print Niter, Tsec

    # Default columns : array_id, field_id, data_desc_id, time
    ms.selectinit(datadescid=0, reset=True)
    selection = {'field_id' : [target_id, phase_id], 'uvdist' : [1., 1e10]}
    ms.select(items = selection)

    log_output("Selected target_id = %d" %target_id, logfile)
    log_output("Selected phase_id  = %d" %phase_id, logfile)
    log_output("Ignoring flux_id  = %d" %flux_id, logfile)

    ms.iterinit(["TIME"], tstep, adddefaultsortcolumns=False)
    ms.iterorigin()

    if datacolumn == 'data':
        data_col = 'data'
    else:
        data_col = 'corrected_data'

    # Output arrays
    loop_time_start = time.time()
    read_times  = []
    avg_times   = []
    ns_times    = []
    write_times = []
    
    mdats_all = []
    tt_all = np.array([])

    write_step_num = 0

    for jj in xrange(Niter):
        log_output("Time Step %d / %d" %(jj, Niter), logfile)
        tloop = time.time()

        # Check to see if we are skipping this time step
        if jj < Nskip:
            log_output("  SKIPPING STEP %d / %d" %(jj, Nskip), logfile)

            if ms.iternext():
                continue
            else:
                break
        else:
            pass

        if write_tstep > 0:
            write_step_num = int(jj / write_tstep)
        else:
            pass

        read_time_start = time.time()
        rec = ms.getdata([data_col, "flag", "time", "u", "v", "field_id", "data_desc_id"], ifraxis=True)
        read_times.append(time.time() - read_time_start)

        # Read in the times
        tt = np.reshape(rec["time"], (-1, Nspw))[:, 0]
        Nt = len(tt)   

        # Read in the array of field_ids
        field_ids = np.reshape(rec['field_id'], (-1, Nspw))[:, 0]

        # Calculate ulam and vlam
        # Need to do this b/c default unit is meters
        # Result should have shape (2, Nchan, Nbl, Nt)
        ulams = np.multiply.outer(inv_waves_mat, np.reshape(rec["u"], (-1, Nt, Nspw))[:, :, 0])
        vlams = np.multiply.outer(inv_waves_mat, np.reshape(rec["v"], (-1, Nt, Nspw))[:, :, 0])

        # Loop over spectral windows, apply flags, divide by baselines 
        ns_time_start = time.time()
        for kk, spw in enumerate(spws):
            # Get indices of current spw
            xx = np.where(rec['data_desc_id'] == spw)[0]

            # Convert flags from boolean values (True = FLAG, False = No Flag)
            # to integers (0 = Flag, 1 = No Flag) and apply to data
            flags = rec['flag'][:, :, :, xx]
            if use_flags:
                flags_int = ((flags * 1) + 1) % 2
            else: 
                flags_int = np.ones( flags.shape )

            ddk = rec[data_col][:, :, :, xx] * flags_int
            
            # Divide data by number of unflagged baselines per channel
            ddk = divide_by_baselines(ddk, flags_int, Nbl_min)

            # Extend data along frequency channel axis
            if kk == 0:
                dd = ddk
            else:
                dd = np.concatenate((dd, ddk), axis=1)
        ns_times.append(time.time() - ns_time_start)

        # Sum over polarization
        dd = np.sum(dd, axis=0)
        ulams = np.mean(ulams[0], axis=-1)
        vlams = np.mean(vlams[0], axis=-1)
        
        # Get time slices in prep for averaging
        log_output("  Nt = %d, len(tt) = %d, dd.shape[-1] = %d" %(Nt, len(tt), dd.shape[-1]), logfile)
        #print(dd.shape)
        #print(ulams.shape)

        #start_stops = get_start_stop_chunk(Nt, nchunks)
        #tslices = [ slice(*ss) for ss in start_stops ]

        # Stack the new times
        tt_all = np.hstack( (tt_all, tt) )

        # Calculate mdats if field_id == target_id, otherwise just dump zeros
        avg_time_start = time.time()
        if phase_id in field_ids:
            log_output("  **Blanking out phase-cal field**", logfile)
            sys.stdout.flush()
            #mdats_beams = [ np.zeros( (Nt, Nchan), dtype=dd.dtype ) ] * Nbeams
            mdats_beams = [ np.zeros( (Nt, Nchan), dtype=np.float32 ) ] * Nbeams
        elif flux_id in field_ids:
            log_output("  **Blanking out flux-cal field**", logfile)
            sys.stdout.flush()
            #mdats_beams = [ np.zeros( (Nt, Nchan), dtype=dd.dtype ) ] * Nbeams
            mdats_beams = [ np.zeros( (Nt, Nchan), dtype=np.float32 ) ] * Nbeams
        else:
            mdats = calc_all_beams(ell_ems, dd, ulams, vlams, nproc=nproc)
            mdats.sort()
            #mdats_beams = [mm[1].T for mm in mdats]
            mdats_beams = [np.real(mm[1].T).astype(np.float32) for mm in mdats]
        avg_times.append(time.time() - avg_time_start)

        # If mdats_all is empty (ie, this is step 0 or after a write), then 
        # fill it directly with the beams
        if len(mdats_all) == 0:
            mdats_all = [ mdats_beams[ii] for ii in xrange(Nbeams) ]
        # Otherwise stack on to previous data
        else:
            mdats_all = [ np.vstack((mdats_all[ii], mdats_beams[ii])) for ii in xrange(Nbeams) ]

        # If this is a write step, write out numpy arrays
        if write_tstep > 0:
            if jj == ((write_step_num + 1) * write_tstep - 1):
                write_time_start = time.time()
                save_beam_files(basename, write_step_num, tt_all, mdats_all, beam_nums)
                mdats_all = []
                tt_all = np.array([])
                write_times.append( time.time() - write_time_start )
            
        dtloop = time.time() - tloop
        log_output("   Step %d proc time: %.2f min" %(jj, dtloop / 60.0), logfile)
        # Go on to next iteration if it's there
        if ms.iternext():
            pass
        else:
            break
    # If we are writing data to file and there is something left, 
    # write that now.
    if len(mdats_all) and write_tstep > 0:
        write_time_start = time.time()
        save_beam_files(basename, write_step_num, tt_all, mdats_all, beam_nums)
        mdats_all = []
        tt_all = np.array([])
        write_times.append( time.time() - write_time_start )

    loop_time = time.time() - loop_time_start

    ms.iterend()
    ms.close()

    # Cut out the dummy row for data arrays
    #mdats_all = [ mm[1:] for mm in mdats_all ]

    log_output("LOOP TIME   = %.2f minutes" %(loop_time / 60.0), logfile)
    log_output("READ TIME   = %.2f minutes" %(np.sum(read_times) / 60.0), logfile)
    log_output("WRITE TIME  = %.2f minutes" %(np.sum(write_times) / 60.0), logfile)
    log_output("VISAVG TIME = %.2f minutes" %(np.sum(avg_times) / 60.0), logfile)
    log_output("DENOM TIME  = %.2f minutes" %(np.sum(ns_times) / 60.0), logfile)

    proc_times = [read_times, write_times, avg_times, ns_times]

    return mdats_all, tt_all, freqs, proc_times



def data_cross_check(t0, freqs0, dat0, t1, freqs1, dat1):
    if len(t0) != len(t1):
        print("Time mismatch!")
        return 0
    elif len(set(np.diff(freqs0))) > 1  or len(set(np.diff(freqs1))) > 1:
        print("Multiple channel widths!")
        return 0
    elif np.max( np.abs( t1 - t0 ) ) > 0:
        print("Time Offset!")
        return 0
    else:
        return 1


def get_ell_m(pos1, pos0):
    """
    Input positions are expected to be in the form
    [ref-frame (string), RA (string), DEC (string)]
    where each is in a valid casa format.

    Example:

    ["J2000", "03:58:53.716501", "+54.13.13.72701"]
    
    """
    
    dir0 = me.direction(pos0[0], pos0[1], pos0[2])
    dir1 = me.direction(pos1[0], pos1[1], pos1[2])

    dir1_ra_fix  = me.direction(pos1[0], pos1[1], pos0[2])
    dir1_dec_fix = me.direction(pos1[0], pos0[1], pos1[2])

    ra_sep  = me.separation(dir1_ra_fix, dir0)
    dec_sep = me.separation(dir1_dec_fix, dir0)

    sgn_ell = +1.0 if qa.gt(dir1['m0'], dir0['m0']) else -1.0
    sgn_m   = +1.0 if qa.gt(dir1['m1'], dir0['m1']) else -1.0

    ell = sgn_ell * qa.getvalue(qa.sin(ra_sep))[0]
    m   = sgn_m  * qa.getvalue(qa.sin(dec_sep))[0]

    return ell, m

def get_time_info(tt):
    """
    tt : Time in MJD seconds
    """
    tt_tu = {'value' : tt, 'unit' : 's'}
    tt_dict = qa.splitdate(tt_tu)
    mjd = tt_dict['mjd']
    stt_imjd = int( mjd )
    stt_smjd = int((mjd - int(mjd)) * 24 * 3600.0)
    stt_offs = (mjd - int(mjd)) * 24 * 3600.0 - stt_smjd
    fits_date = qa.time(tt_tu, form='fits')
    out_dict = {'obs_date' : fits_date, 
                'stt_imjd' : stt_imjd, 
                'stt_smjd' : stt_smjd,
                'stt_offs' : stt_offs, 
                'mjd'      : mjd}
    return out_dict


def get_and_save_spw_freqs(infile, spw_list, basename=""):
    """
    Save freqs in MHz
    """
    for spw in spw_list:
        msmd.open(infile)
        freqs = msmd.chanfreqs(spw)
        msmd.close()

        outfile = "%s_freqs_spw%02d.npy" %(basename, spw)
        np.save(outfile, freqs / 1e6)
    return


def double_check_freqs(infile, spw_list, tstep=0.01, field=2):
    freqlist_msmd = []
    freqlist_dat  = []
    
    for spw in spw_list:    
        # Get channel frequencies using the ms metadata tool
        msmd.open(infile)
        freqs_msmd = msmd.chanfreqs(spw) / 1e6
        msmd.close()
        freqlist_msmd.append(freqs_msmd)
        
        # Get frequencies from ms
        ms.open(infile)
        ms.selectinit(datadescid=spw)
        ms.select({'field_id' : field})
        ms.iterinit(["TIME"], tstep, adddefaultsortcolumns=False)
        ms.iterorigin()
        data_col = 'corrected_data'
        rec = ms.getdata([data_col, "axis_info"], ifraxis=True)
        freqs_dat = rec["axis_info"]["freq_axis"]["chan_freq"].ravel() / 1e6
        freqlist_dat.append(freqs_dat)
        
    return freqlist_msmd, freqlist_dat        


def log_output(text, logfile):
    if logfile is None:
        print(text)
        sys.stdout.flush()
    else:
        fout = open(logfile, 'a+')
        fout.write("%s\n" %text)
        fout.close()
    return


