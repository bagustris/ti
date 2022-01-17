# read opesmile csv file, convert to npy
# 20210531: configured for speaker dependent

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
lab_train_sd = []

emo = {'ang':0, 'joy':1, 'neu':2, 'sad':3} 

for file in files:
    # processing file
    print("Processing... ", file)
    feat = np.loadtxt(file, 
                      skiprows=6558, 
                      delimiter=',', 
                      usecols=range(1, 6553))
    # cond_a = int(os.path.basename(file)[1:3]) not in range(46, 51)
    # cond_b = int(os.path.basename(file)[8:10]) not in range(41, 51)
    lab_str = os.path.basename(file)[4:7]
    lab_int = emo[lab_str]
    # if  cond_a and cond_b:
    feat_train.append(feat)
    lab_train_sd.append(lab_int)

np.save('./data/feat_sd.npy', feat_train)
np.save('./data/lab_sd.npy', lab_train_sd)