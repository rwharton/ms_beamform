import glob
import sys
import xml.etree.ElementTree
import os.path


##################
### parameters ### 
##################

sdmdir  = '/hercules/results/rwharton/fastvis_gc/sdm'
datadir = '/hercules/results/rwharton/fastvis_gc/ms_data'
sdmfile = '16A-329_sb31934939_1.57519.32433290509.avg'

do_import  = True


############################
### parse for good scans ###
############################

#os.chdir(sdmdir)

mainroot = xml.etree.ElementTree.parse("%s/%s/Main.xml" %(sdmdir, sdmfile)).getroot()
scans  = ""
fields = ""

select_field = ['Field_2', 'Field_3']  #'Field_2'

for scan in mainroot.findall('row'):
  scanNumber = scan.find('scanNumber').text
  dataSize = scan.find('dataSize').text
  fieldId = scan.find('fieldId').text

  if dataSize != '0':
    #scans = scans + scanNumber + ',' 
    if select_field is not None:
      if fieldId in select_field:
        fields = fields + fieldId + ','
        scans = scans + scanNumber + ','
    else:
      fields = fields + fieldId + ','
      scans = scans + scanNumber + ','
      

scans = scans[0:len(scans)-1]
fields = fields[0:len(fields)-1]
print("Good scans: %s\n" %scans)
#print("Good fields: %s\n" %fields)


################
# SELECT SCAN  #
################

#scans = '4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23'
#scans = '25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44'
#scans = '5'
print("Using scans = %s" %scans)
print("fields: %s" %fields)

#####################
### creation time ###
#####################

asdmroot = xml.etree.ElementTree.parse("%s/%s/ASDM.xml" %(sdmdir, sdmfile)).getroot()
toc = asdmroot.findall('TimeOfCreation')[0].text
print(toc)
print("\n")


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

msfile = "%s/%s.ms" %(datadir, sdmfile)
print("MSFILE = %s\n" %msfile)

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


