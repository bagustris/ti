#!/usr/bin/env python3
# bagus@ep.its.ac.id, 
# changelog:
# 2019-04-16: init code from avec
# 2019-07-02: modify to extract 10039 iemocap data
# 2021-05-10: modify to extract JTES dataset

import numpy as np
import os
import glob
import time
# import pickle

feature_type = 'emo_large'
exe_opensmile = '~/opensmile-3.0-linux-x64/bin/SMILExtract'  
path_config   = '~/opensmile-3.0-linux-x64/config/'
data_path = '/data/jtes_v1.1/'

if feature_type=='mfcc':
    folder_output = '../audio_features_mfcc/'  # output folder
    conf_smileconf = path_config + 'MFCC12_0_D_A.conf'  # MFCCs 0-12 with delta and acceleration coefficients
    opensmile_options = '-configfile ' + conf_smileconf + ' -appendcsv 0 -timestampcsv 1 -headercsv 1'  # options from standard_data_output_lldonly.conf.inc
    outputoption = '-csvoutput'  # options from standard_data_output_lldonly.conf.inc
elif feature_type=='egemaps':
    folder_output = './audio_features_egemaps_10039/'  # output folder
    conf_smileconf = path_config + 'gemaps/eGeMAPSv01a.conf'  # eGeMAPS feature set
    opensmile_options = '-configfile ' + conf_smileconf + ' -appendcsvlld 0 -timestampcsvlld 1 -headercsvlld 1'  # options from standard_data_output.conf.inc
    outputoption = '-lldcsvoutput'  # options from standard_data_output.conf.inc
elif feature_type=='emo_large':
    folder_output = './emo_large_hsf/'
    conf_smileconf = path_config + 'misc/emo_large.conf'
    opensmile_options = '-configfile ' \
                        + conf_smileconf 
    outputoption = '-lldcsvoutput'  # options from standard_data_output.conf.in
else:
    print('Error: Feature type ' + feature_type + ' unknown!')
    
if not os.path.exists(folder_output):
    os.mkdir(folder_output)

files = glob.glob(os.path.join(data_path + 'wav/*/*/', '*.wav'))
files.sort()

for filename in files:
    instname = os.path.basename(filename)[:-4]
    outfilename = folder_output + instname + '.csv'
    opensmile_call = exe_opensmile + ' ' + opensmile_options + ' -inputfile ' + filename + ' ' + ' -O ' + outfilename
    os.system(opensmile_call)
    time.sleep(0.01)

os.remove('smile.log')