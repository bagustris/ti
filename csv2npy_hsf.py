# read opesmile csv file, convert to npy

import os
import glob

import numpy as np
import opensmile
import pandas as pd

# define jtes path
data_path = '/home/bagus/research/2021/jtes_base/emo_large/emo_large_hsf/'
files = glob.glob(os.path.join(data_path, '*.csv'))
files.sort()

feat_train = []
feat_test = []

for file in files:
    # processing file
    print("Processing... ", file)
    feat = np.loadtxt(file, 
                      skiprows=6558, 
                      delimiter=',', 
                      usecols=range(1, 6553))
    if int(os.path.basename(file)[1:3]) in range(1, 46):
        if int(os.path.basename(file)[8:10]) in range(1, 41):
            # train = train + 1   
            feat_train.append(feat)
    elif int(os.path.basename(file)[1:3]) in range(46, 51): # = else:
        if int(os.path.basename(file)[8:10]) in range(41, 51):
            # test = test + 1
            feat_test.append(feat)

np.save('./hsf/emo_large_train.npy', feat_train)
np.save('./hsf/emo_large_test.npy', feat_test)