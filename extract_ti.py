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
    if int(os.path.basename(file)[-6:-4])==50:
        # print("test")
        hsf_test.append(feat)
        y_test.append(lab_int)
    else:
        # print("training")
        hsf_train.append(feat)
        y_train.append(lab_int)

np.save('data/x_train_ti.npy', hsf_train)
np.save('data/x_test_ti.npy', hsf_test)

np.save('data/y_train_ti.npy', y_train)
np.save('data/y_test_ti.npy', y_test)

# np.save('x_val_ti1.npy', hsf_train)
# np.save('x_test_ti1.npy', hsf_val)
# np.save('x_train_ti1.npy', hsf_test)

# np.save('y_val_ti1.npy', y_train)
# np.save('y_test_ti1.npy', y_val)
# np.save('y_train_ti1.npy', y_test)