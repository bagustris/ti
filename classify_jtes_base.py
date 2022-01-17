#!/usr/bin/env python

# classify.py: make classification of emotion given input features (in npy files)
# dataset: JSET v1.1
# author: Bagus Tris Atmaja (b-atmaja@aist.go.jp)
# Changelong
# 20210420: initial commit

import numpy as np
import tensorflow as tf

# import feature and labels
data_path = '/home/bagus/research/2021/jtes_base/'

feat = np.load(data_path + 'jtes_hsf.npy')
label = np.load(data_path + 'jtes_label.npy')

x_train = feat[:18000]
x_test = feat[18000:]
x_train = x_train.reshape(18000, 1, 68)
x_test = x_test.reshape(2000, 1, 68)
y_train = label[:18000]
y_test = label[18000:]


def lstm_model():
    inputs = tf.keras.Input(shape=(1, 68))
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)(x)
    # x = tf.keras.layers.LSTM(256, return_sequences=True, return_state=True)(x)
    x = tf.keras.layers.LSTM(256, return_sequences=True, return_state=False)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='jtes_lstm')
    return model


# model compilation
model = lstm_model()
print(model.summary())

# callbacks
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train,
                    y_train,
                    batch_size=64, 
                    epochs=100, 
                    callbacks=[callback],
                    validation_split=0.2)


# test the model
test_scores = model.evaluate(x_test, y_test, batch_size=8, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
