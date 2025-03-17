import numpy as np
import sys, os

def get_avgdata_flags(infile, tdur=1.0, target_id=0):
    # Get spw info 
    msmd.open(infile)
    spws = msmd.spwsforfield(target_id)
    msmd.close()
    nspws = len(spws)

    # Open file and get flags
    ms.open(infile)
    Tsec = np.diff(ms.range('time')['time'])[0]
    Niter = int(Tsec / tdur) + 1
    Niter = 1

    ms.selectinit(datadescid=0, reset=True)
    ms.select({'field_id' : target_id, 'uvdist' : [1., 1e10]})
    ms.iterinit(["TIME"], tdur, adddefaultsortcolumns=False)
    ms.iterorigin()

    tt = []
    flags = []

    loop_time_start = time.time()
    for ii in xrange(Niter):
        rec = ms.getdata(["time", "flag", "field_id", "data_desc_id"], ifraxis=True)
        tt.append(np.reshape(rec['time'], (-1, nspws))[:, 0])
        npol, nchan, nbl, nt_nspw = rec['flag'].shape
        flags.append(np.reshape(rec['flag'], (npol, nchan, nbl, -1, nspws)))
        print(rec['field_id'])
        print(rec['flag'].shape)
        if ms.iternext():
            pass
        else:
            break
    ms.iterend()
    ms.close()
    loop_time = time.time() - loop_time_start
    
    for ff in flags:
        print ff.shape

    tt    = np.concatenate(tt, axis=1)
    flags = np.concatenate(flags, axis=3)

    print "LOOP TIME = %.2f minutes" %(loop_time / 60.0)

    return tt, flags


def get_and_save_avgdata_flags(basename, infile, tdur=1.0, target_id=0, ret_all=False):
    tt, flags = get_avgdata_flags(infile, tdur=tdur, target_id=target_id)
    time_file = "%s_times.npy" %(basename)
    flag_file = "%s_flags.npy" %(basename)
    np.save(time_file, tt) 
    np.save(flag_file, flags)
    if ret_all:
        return tt, flags
    else:
        return


def get_and_save_all_avgdata_flags(basename, infile, spw_list=[0], tdur=1.0, target_id=0):
    for spw in spw_list:
        print("Working on spw = %d" %spw)
        get_and_save_avgdata_flags(basename, infile, spw=spw, tdur=tdur, 
                                   target_id=target_id, ret_all=False)
    return


def expand_flags(avg_flags, N):
    #npol, nchan, nifr = avg_flags.shape
    #tmp_flags = np.reshape(avg_flags, (npol, nchan, nifr, 1))
    #out_flags = np.concatenate( [tmp_flags.T] * N, axis=0).T
    out_flags = np.repeat(avg_flags, N, axis=-1)
    return out_flags


def get_closest_time(tt, tt_avg, dt_min=1.0):
    idx_min = np.argmin( np.abs(tt_avg - tt) )
    dt = tt - tt_avg[idx_min]
    if np.abs(dt) > dt_min:
        idx_min = -1
        print(dt)
    else:
        pass
    return idx_min, dt


def check_then_OR_flags(dat_flags, Navg_flags):
    # First make sure flag shapes are the same
    if dat_flags.shape == Navg_flags.shape:
        pass
    else:
        print("Shape mismatch (dat/avg):")
        print("dat", dat_flags.shape)
        print("avg", Navg_flags.shape)
        return -1

    out_flags = np.logical_or(dat_flags, Navg_flags)
    return out_flags


def apply_flags_from_avg(infile, tvals, flag_vals, tdur=1.0, no_write=True):
    if no_write:
        print("no_write = True ... Will NOT write flags.")
        sys.stdout.flush()
    else: pass
    
    ms.open(infile, nomodify=no_write)

    Tsec = np.diff(ms.range('time')['time'])[0]
    Niter = int(Tsec / tdur) + 1
    #Niter = 10
    nspws = flag_vals.shape[-1]

    ms.selectinit(datadescid=0, reset=True)
    ms.select({'uvdist' : [1., 1e10]})
    ms.iterinit(["TIME"], tdur, adddefaultsortcolumns=False)
    #ms.iterinit(["TIME"], tdur)
    ms.iterorigin()

    tmids = []
    dts = []
    
    loop_time_start = time.time()
    for ii in xrange(Niter):
        print(ii)
        sys.stdout.flush()
        rec = ms.getdata(["time", "flag", "field_id", "data_desc_id"], ifraxis=True)
        #rec = ms.getdata(["time", "flag", "field_id"], ifraxis=True)

        tt_full = np.reshape(rec['time'], (-1, nspws))[:, 0]
        Nt = len(tt_full)
        print(Nt)

        if Nt == 0:
            break
        else:
            pass

        tt_mid = np.mean(tt_full)
        tmids.append(tt_mid - int(tvals[0]))
        
        idx, dt = get_closest_time(tt_mid, tvals, dt_min=tdur)
        dts.append(dt)

        dat_flags = rec['flag']
        print(set(rec['field_id']))

        if idx < 0:
            out_flags = dat_flags | True
            print("No match")
            sys.stdout.flush()
        else:
            avg_flags = flag_vals[:, :, :, idx, :]
            Navg_flags = expand_flags(avg_flags, Nt)
            out_flags = check_then_OR_flags(dat_flags, Navg_flags)

        rec_new = {'flag' : out_flags}

        if no_write:
            pass
        else:
            ms.putdata(rec_new)

        if ms.iternext():
            pass
        else:
            break

    loop_time = time.time() - loop_time_start

    tmids = np.array(tmids)
    dts = np.array(dts)
    ms.close()

    print "LOOP TIME = %.2f minutes" %(loop_time / 60.0)

    return tmids, dts


def apply_flags_from_avg_multi(infile, npy_dir, basename, spw_list=[0], 
                               tdur=1.0, no_write=True):
    if no_write:
        print("no_write = True ... Will NOT write flags.")
        sys.stdout.flush()
    else: pass
    
    ms.open(infile, nomodify=no_write)

    Tsec = np.diff(ms.range('time')['time'])[0]
    Niter = int(Tsec / tdur) + 1
    #Niter = 10

    for spw in spw_list:
        print("Working on spw = %d" %spw)
        tvals = np.load("%s/%s_spw%d_times.npy" %(npy_dir, basename, spw))
        flag_vals = np.load("%s/%s_spw%d_flags.npy" %(npy_dir, basename, spw))

        ms.selectinit(datadescid=spw)
        staql = {'baseline' : '*&*'}
        ms.msselect(staql)
        ms.iterinit(["TIME"], tdur)
        ms.iterorigin()
    
        tmids = []
        dts = []
    
        loop_time_start = time.time()
        for ii in xrange(Niter):
            print(ii)
            sys.stdout.flush()
            rec = ms.getdata(["time", "flag", "field_id"], ifraxis=True)

            tt_full = rec['time']
            Nt = len(tt_full)

            if Nt == 0:
                break
            else:
                pass

            tt_mid = np.mean(tt_full)
            tmids.append(tt_mid - int(tvals[0]))
        
            idx, dt = get_closest_time(tt_mid, tvals, dt_min=tdur)
            dts.append(dt)

            dat_flags = rec['flag']
            print(set(rec['field_id']))

            if idx < 0:
                out_flags = dat_flags | True
                print("No match")
                sys.stdout.flush()
            else:
                avg_flags = flag_vals[:, :, :, idx]
                Navg_flags = expand_flags(avg_flags, Nt)
                out_flags = check_then_OR_flags(dat_flags, Navg_flags)

            rec_new = {'flag' : out_flags}

            if no_write:
                pass
            else:
                ms.putdata(rec_new)

            if ms.iternext():
                pass
            else:
                break

        loop_time = time.time() - loop_time_start

        tmids = np.array(tmids)
        dts = np.array(dts)
        print "SPW %d LOOP TIME = %.2f minutes" %(spw, loop_time / 60.0)


    ms.close()

    return tmids, dts


def apply_all_flags_from_avg(infile, npy_dir, basename, spw_list=[0], 
                             tdur=1.0, no_write=True):
    for spw in spw_list:
        tvals = np.load("%s/%s_spw%d_times.npy" %(npy_dir, basename, spw))
        fvals = np.load("%s/%s_spw%d_flags.npy" %(npy_dir, basename, spw))
        
        tmids, dt = apply_flags_from_avg(infile, tvals, fvals, spw=spw, 
                                         tdur=tdur, no_write=no_write)
    return tmids, dt

##############
# PROCESSING #
##############

do_get_flags = True

work_dir = '/hercules/results/rwharton/fastvis_gc'

avg_dir  = '%s/pipeline/57519' %(work_dir)
avg_file = '%s/16A-329_sb31934939_1.57519.32433290509.avg.ms' %avg_dir

dat_dir  = '%s/ms_data' %(work_dir)
ms_name  = '57519_part2.ms'
dat_file = '%s/%s' %(dat_dir, ms_name)

npy_dir  = '%s/proc/57519/avg_flags' %(work_dir)

basename = 'mjd57519_full'
target = 2

print("Check:  %s" %(avg_file))
print("   %r" %(os.path.exists(avg_file)))
print("Check:  %s" %(dat_file))
print("   %r" %(os.path.exists(dat_file)))
print("Check:  %s" %(npy_dir))
print("   %r" %(os.path.exists(npy_dir)))

# The following gets the flags from the avg data set and saves them
#spw_list = range(8)

if do_get_flags:
    get_and_save_avgdata_flags(basename, dat_file, tdur=30.0, target_id=target)

