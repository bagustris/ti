#!/usr/bin/env python

# classify.py: make classification of emotion given input features (in npy files)
# dataset: JSET v1.1
# author: Bagus Tris Atmaja (b-atmaja@aist.go.jp)
# Changelong 
# 20210420: initial commit 
# 20210428: change to dense
# 20210531: modified to use emo_large feature

import numpy as np
import tensorflow as tf

# import feature and labels
# data_path = '/home/bagus/research/2021/jtes_base/'

feat = np.load('./data/feat_sd.npy')
label = np.load('./data/lab_sd.npy')

# change test set to 200 last female, and 200 last male
x_train = feat[:19600]
x_test = feat[19600:]
x_train = x_train.reshape(19600, 6552)
x_test = x_test.reshape(400, 6552)
y_train = label[:19600]
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

acc_avg = []
# for SD with emolarge, smaller batch is better, i.e. 8
for i in range(30):
    history = model.fit(x_train, 
                        y_train, 
                        batch_size=1024, 
                        epochs=25, 
                        # callbacks=[callback],
                        validation_split=0.2)


# test the model
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    # print("Test loss:", test_scores[0])
# print("Test accuracy:", test_scores[1])
    acc_avg.append(test_scores[1])

print("Test accuracy: ", np.mean(acc_avg), " +/- ", np.std(acc_avg))