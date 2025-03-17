import numpy as np
import sys

def unflag_all_flags(infile, tdur=1.0, no_write=True):
    if no_write:
        print("no_write = True ... Will NOT write flags.")
        sys.stdout.flush()
    else: pass
    
    ms.open(infile, nomodify=no_write)

    Tsec = np.diff(ms.range('time')['time'])[0]
    Niter = int(Tsec / tdur) + 1
    #Niter = 10
    ms.selectinit(datadescid=0, reset=True)
    ms.select({'uvdist' : [1., 1e10]})
    ms.iterinit(["TIME"], tdur, adddefaultsortcolumns=False)
    ms.iterorigin()

    loop_time_start = time.time()
    for ii in xrange(Niter):
        print("%d / %d" %(ii, Niter))
        sys.stdout.flush()
        rec = ms.getdata(["flag", "data_desc_id"], ifraxis=True)

        # REMOVE ALL FLAGS 
        out_flags = rec['flag'] * False
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
    ms.close()

    print "LOOP TIME = %.2f minutes" %(loop_time / 60.0)

    return loop_time


##############
# PROCESSING #
##############

dat_dir  = '/hercules/results/rwharton/fastvis_gc/ms_data/model_sub'
ms_name  = "part2_test.ms"
dat_file = '%s/%s' %(dat_dir, ms_name)

undo_flags = True
no_write   = False  # ie, do NOT write if True

print("Check:  %s" %(dat_file))
print("   %r" %(os.path.exists(dat_file)))

# The following reads and applies the flags
if undo_flags:
    tstart = time.time()
    loop_time = unflag_all_flags(dat_file, tdur=3.0, no_write=no_write)
    dt = time.time() - tstart

    print("Flagging time = %.2f min" %(dt / 60.0))

