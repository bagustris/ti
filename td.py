import os
import glob

import numpy as np
import opensmile

# define jtes path
data_path = '/data/jtes_v1.1/'
files = glob.glob(os.path.join(data_path + 'wav/*/*/', '*.wav'))
files.sort()

hsf_train = []
hsf_val = []
hsf_test = []

y_train = []
y_val = []
y_test = []

# label dictionary
emo = {'ang':0, 'joy':1, 'neu':2, 'sad':3} 


smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.ComParE_2016,
    feature_level=opensmile.FeatureLevel.Functionals,
)

for file in files:
    # processing file
    print("Processing... ", file)
    y = smile.process_file(file)
    lab_str = os.path.basename(file)[4:7]
    lab_int = emo[lab_str]
    if int(os.path.basename(file)[-6:-4]) in range(31, 41):
        # print("val")
        hsf_val.append(y.to_numpy().flatten())
        y_val.append(lab_int)
    elif int(os.path.basename(file)[-6:-4]) in range(41, 51):
        # print("test")
        hsf_test.append(y.to_numpy().flatten())
        y_test.append(lab_int)
    else:
        # print("training")
        hsf_train.append(y.to_numpy().flatten())
        y_train.append(lab_int)

np.save('data_ti/x_train_ti1.npy', hsf_train)
np.save('data_ti/x_val_ti1.npy', hsf_val)
np.save('data_ti/x_test_ti1.npy', hsf_test)

np.save('data_ti/y_train_ti1.npy', y_train)
np.save('data_ti/y_val_ti1.npy', y_val)
np.save('data_ti/y_test_ti1.npy', y_test)

# np.save('x_val_ti1.npy', hsf_train)
# np.save('x_test_ti1.npy', hsf_val)
# np.save('x_train_ti1.npy', hsf_test)

# np.save('y_val_ti1.npy', y_train)
# np.save('y_test_ti1.npy', y_val)
# np.save('y_train_ti1.npy', y_test)