#!/usr/bin/env python

# classify.py: make classification of emotion given input features (in npy files)
# dataset: JSET v1.1
# author: Bagus Tris Atmaja (b-atmaja@aist.go.jp)
# Changelong 
# 20210420: initial commit 
# 20210428: change to dense
# 20210531: modified to use emo_large feature
# 20210602: modified for CV (speaker-dependent)

import numpy as np
import tensorflow as tf

# import feature and labels
# data_path = '/home/bagus/research/2021/jtes_base/'

feat = np.load('../data/feat_sd.npy')
label = np.load('../data/lab_sd.npy')

# change test set to 200 last female, and 200 last male
# feat_ and label_ are for cross validation
x_train = feat[:14400]
x_test = feat[19600:]
feat_train = x_train.reshape(14400, 6552)
x_test = x_test.reshape(400, 6552)
label_train = label[:19600]
y_test = label[19600:]

def lstm_model():
    inputs = tf.keras.Input(shape=(6552,))
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='jtes_dense')
    return model


# model compilation  
model = lstm_model()
print(model.summary())

# callbacks
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

# define 5 folds
# define fold, 5 folds
from sklearn.model_selection import KFold
kf = KFold(n_splits=5)
kf.get_n_splits(feat_train)

acc_avg = []
# for SD with emolarge, smaller batch is better, i.e. 8
# for i in range(30):
for train_index, val_index in kf.split(feat_train):
    x_train, x_val = feat_train[train_index], feat_train[val_index]
    y_train, y_val = label_train[train_index], label_train[val_index]
    history = model.fit(x_train, 
                        y_train, 
                        batch_size=1024, 
                        epochs=25, 
                        # callbacks=[callback],
                        validation_data = (x_val, y_val))


    # test the model
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    # print("Test loss:", test_scores[0])   
    # print("Test accuracy:", test_scores[1])
    acc_avg.append(test_scores[1])

print("Test accuracy: ", np.mean(acc_avg), "std: ", np.std(acc_avg))