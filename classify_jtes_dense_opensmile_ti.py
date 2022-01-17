#!/usr/bin/env python

# classify.py: make classification of emotion given input features (in npy files)
# dataset: JSET v1.1
# author: Bagus Tris Atmaja (b-atmaja@aist.go.jp)
# Changelong 
# 20210420: initial commit 
# 20210428: change to dense
# 20210430: use the real text-independent (TI) split

import numpy as np
import tensorflow as tf
# tf.random.set_seed(221)

# import feature and labels
path_base = '/home/bagus/research/2021/jtes_base/'
feat = np.load(path_base +'jtes_compare16_hsf.npy')
label = np.load(path_base + 'jtes_label.npy')

x_train = np.load(path_base + './data_ti/x_train_ti1.npy')
x_val = np.load(path_base + './data_ti/x_val_ti1.npy')
x_test = np.load(path_base + './data_ti/x_test_ti1.npy')
y_train = np.load(path_base + './data_ti/y_train_ti1.npy')
y_val = np.load(path_base + './data_ti/y_val_ti1.npy')
y_test = np.load(path_base + './data_ti/y_test_ti1.npy')

def lstm_model():
    inputs = tf.keras.Input(shape=(6373,))
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dropout(0.3)(x)
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
    optimizer=tf.keras.optimizers.RMSprop(),
    metrics=["accuracy"],
)

history = model.fit(x_train, 
                    y_train, 
                    batch_size=1024, 
                    epochs=1000, 
                    callbacks=[callback],
                    validation_data=(x_val, y_val))


# test the model
test_scores = model.evaluate(x_test, y_test, batch_size=1024, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
