import glob
import sys
import xml.etree.ElementTree
import os.path
import numpy as np

##################
### parameters ### 
##################

sdmdir  = '/hercules/results/rwharton/fastvis_gc/sdm'
datadir = '/hercules/results/rwharton/fastvis_gc/ms_data'
sdmfile = '16A-329_sb31934939_1.57519.32433290509'


################################
# Argparse to get part number  #
################################

print("\n\n")
script_name = '/u/rwharton/src/beamforming/prelim/myimportdata.py'
script_idx = np.where( np.array(sys.argv) == script_name )[0]
if len(sys.argv) == script_idx + 2:
  part = int( sys.argv[-1] )
  print("Working on Part %d" %part)
else:
  print("Wrong number of arguments")
  print len(sys.argv)
  print(sys.argv)
  sys.exit(0)


outms   = '57519_part%d.ms' %(part)
do_import  = True


############################
### parse for good scans ###
############################

#os.chdir(sdmdir)

mainroot = xml.etree.ElementTree.parse("%s/%s/Main.xml" %(sdmdir, sdmfile)).getroot()
scanlist  = []
fieldlist = []

select_field = ['Field_2', 'Field_3']  #'Field_2'

for scan in mainroot.findall('row'):
  scanNumber = scan.find('scanNumber').text
  dataSize = scan.find('dataSize').text
  fieldId = scan.find('fieldId').text

  if dataSize != '0':
    #scans = scans + scanNumber + ',' 
    if select_field is not None:
      if fieldId in select_field:
        fieldlist.append(fieldId)
        scanlist.append(scanNumber)
    else:
      fieldlist.append(fieldId)
      scanlist.append(scanNumber)
      

#scans = scans[0:len(scans)-1]
#fields = fields[0:len(fields)-1]
print("Good scans: %s\n" %(','.join(scanlist)))
#print("Good fields: %s\n" %fields)


################
# SELECT SCAN  #
################

# Cut out last scan bc calibrator
scanlist  = scanlist[:-1]
fieldlist = fieldlist[:-1]

# Get groups of three
scans = ','.join(scanlist[ (part-1) * 3 : part * 3])
fields = ','.join(fieldlist[ (part-1) * 3 : part * 3 ])

#scans = '5,6,7,8,9,10,11,12,13'
#scans = '14,15,16,17,18,19' 
#scans = '20,21,22,23,24'

print("Using scans = %s" %scans)
print("fields: %s" %fields)

#####################
### creation time ###
#####################

asdmroot = xml.etree.ElementTree.parse("%s/%s/ASDM.xml" %(sdmdir, sdmfile)).getroot()
toc = asdmroot.findall('TimeOfCreation')[0].text


############################
### create new directory ###
############################

#os.chdir(datadir)

#if not(os.path.exists(toc)):
#  os.mkdir(toc)

#os.chdir(toc)


###################
### import data ###
###################

msfile = "%s/%s" %(datadir, outms)
print("MSFILE = %s\n" %msfile)
print("\n\n")

asdm_params = {'asdm' : "%s/%s" %(sdmdir, sdmfile),
               'vis'  : msfile, 
               'corr_mode' :"co",
               'srt' : "all",
               'time_sampling' : "all",
               'ocorr_mode' : "co",
               'compression' : False,
               'lazy' : True,
               'asis' : "",
               'wvr_corrected_data' : "no",
               'scans' : scans,
               'ignore_time' : False,
               'process_syspower' : True,
               'process_caldevice' : True,
               'process_pointing' : True,
               'process_flags' : True,
               'tbuff' : 0.0,
               'applyflags' : False,
               'savecmds' : False,
               'outfile' : "",
               'flagbackup' : True,
               'verbose' : True,
               'overwrite' : False,
               'showversion' : False,
               'useversion' : "v3",
               'bdfflags' : False,
               'with_pointing_correction' : False,
               'remove_ref_undef' : False }

print asdm_params 
print("\n")

if do_import: 
  importasdm(**asdm_params)


