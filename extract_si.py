import os
import glob

import numpy as np
import opensmile

# define emolarge csv path
data_path = '/home/bagus/research/2021/jtes_base/emo_large/emo_large_hsf/'
files = glob.glob(os.path.join(data_path, '*.csv'))
files.sort()

hsf_train = []
hsf_test = []

y_train = []
y_test = []

# label dictionary
emo = {'ang':0, 'joy':1, 'neu':2, 'sad':3} 

for file in files:
    # processing file
    print("Processing... ", file)
    lab_str = os.path.basename(file)[4:7]
    lab_int = emo[lab_str]
    feat = np.loadtxt(file, 
                    skiprows=6558, 
                    delimiter=',', 
                    usecols=range(1, 6553))    
    if int(os.path.basename(file)[1:3])==50:
        # print("test")
        hsf_test.append(feat)
        y_test.append(lab_int)
    else:
        # print("training")
        hsf_train.append(feat)
        y_train.append(lab_int)

np.save('data/x_train_si.npy', hsf_train)
np.save('data/x_test_si.npy', hsf_test)

np.save('data/y_train_si.npy', y_train)
np.save('data/y_test_si.npy', y_test)