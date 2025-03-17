import sys

################################
# TEST OF BEAMFORMING PIPELINE #
################################

def check_caltables(caltables):
    for caltable in caltables:
        cal_exist = os.path.exists(caltable)
        if not cal_exist:
            print("Missing caltable: %s" %caltable)
        else:
            print("Found caltable: %s" %caltable)
    return


# Here we want to test out the beamforming pipeline on 
# one scan of the FRB field from the MJD 57511 data set
dat_name = sys.argv[-1]
#dat_name = '57519_part3.ms'
dat_dir = '/hercules/results/rwharton/fastvis_gc/ms_data'
datms = "%s/%s" %(dat_dir, dat_name)

avgms_name = '16A-329_sb31934939_1.57519.32433290509.avg.ms'
cal_dir = '/hercules/results/rwharton/fastvis_gc/pipeline/57519'

cal_tables = ['%s/%s.hifv_priorcals.s5_3.gc.tbl' %(cal_dir, avgms_name), 
              '%s/%s.hifv_priorcals.s5_4.opac.tbl' %(cal_dir, avgms_name),
              '%s/%s.hifv_priorcals.s5_5.rq.tbl' %(cal_dir, avgms_name),
              '%s/%s.finaldelay.k' %(cal_dir, avgms_name),
              '%s/%s.finalBPcal.b' %(cal_dir, avgms_name),
              '%s/%s.averagephasegain.g'  %(cal_dir, avgms_name),
              '%s/%s.finalampgaincal.g'   %(cal_dir, avgms_name),
              '%s/%s.finalphasegaincal.g' %(cal_dir, avgms_name) ]


# Version of myimportdata.py used to generate this 
# MS is saved in the notes folder (with this file)

print("VIS = %s\n" %datms)

check_caltables(cal_tables)

#sys.exit(0)

############################
# APPLY CALIBRATION TABLES #
############################

applycal(vis=datms,
        field='2',
        spw='',
        selectdata=True,
        timerange="",uvrange="",antenna="*&*",scan="",observation="",msselect="",
        docallib=False,callib="",
        gaintable=cal_tables, 
        gainfield=['', '', '', '', '', '', '', ''],
        interp=['', '', '', '', '', '', '', ''],
        spwmap=[[], [], [], [], [], [], [], []],
        calwt=[False, False, False, False, False, False, False, False],
        parang=False,
        applymode="calflagstrict",
        flagbackup=True)
