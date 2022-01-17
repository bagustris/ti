#!/usr/bin/env python

# classify.py: make classification of emotion given input features (in npy files)
# dataset: JSET v1.1
# author: Bagus Tris Atmaja (b-atmaja@aist.go.jp)
# Changelong 
# 20210420: initial commit 
# 2021042512: change to dense
# 20210430: use the real text-independent (TI) split

import numpy as np
import tensorflow as tf

tf.random.set_seed(221)

import random
random.seed(221)
np.random.seed(221)

# import feature and labels
path_base = '/home/bagus/research/2021/jtes_base/emo_large/'
x_train = np.load(path_base + 'hsf/emo_large_train.npy', allow_pickle=True)
x_test = np.load(path_base + 'hsf/emo_large_test.npy', allow_pickle=True)

x_train = np.vstack(x_train).astype(np.float)
x_test = np.vstack(x_test).astype(np.float)

# label
y_train = np.load(path_base + '../data_sti/y_train_sti1.npy')
y_test = np.load(path_base + '../data_sti/y_test_sti1.npy')

def dense_model():
    inputs = tf.keras.Input(shape=(6552,))
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(4, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


# model compilation  
model = dense_model()
print(model.summary())

# callbacks, for dense better not to use best weights
callback = tf.keras.callbacks.EarlyStopping(monitor='loss', 
                                            # restore_best_weights=True,
                                            patience=10)

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
)

acc_avg = []

for i in range(0, 30):
    history = model.fit(x_train, 
                        y_train, 
                        batch_size=1024, 
                        epochs=25,
                        # callbacks=[callback],
                        validation_split=0.2)
    # test the model
    test_scores = model.evaluate(x_test, y_test, verbose=2)
    acc_avg.append(test_scores[1])

# print("Test accuracy:", test_scores[1])
print("Test accuracy:", np.mean(acc_avg), ' +/- ', np.std(acc_avg))
