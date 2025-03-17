################################
# TEST OF BEAMFORMING PIPELINE #
################################

dat_name = '57519_part2.ms'
dat_dir = '/hercules/results/rwharton/fastvis_gc/ms_data'
datms = "%s/%s" %(dat_dir, dat_name)

outms = "%s/part2_avg.ms" %(dat_dir)

# Version of myimportdata.py used to generate this 
# MS is saved in the notes folder (with this file)

print("VIS = %s\n" %datms)

############################
# AVG DATA DOWN TO 3s      #
############################

mstransform(vis=datms, outputvis=outms, timeaverage=True, 
            timebin='3s', datacolumn='all')

