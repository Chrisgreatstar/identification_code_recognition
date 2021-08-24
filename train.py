import tensorflow as tf
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import cv2


data_path = "codes/processed_codes/"
data_labels_path = "codes/processed_codes/code_labels.csv"

data_labels = pd.read_csv(data_labels_path, header=None, names=['index', 'number'])


x_train = []
y_train = []

x_test = []
y_test = []
for f in listdir(data_path):
    if isfile(join(data_path, f)):
        # pass code_labels.csv
        if f.startswith('code'): continue
        code_file_path = join(data_path, f)
        img = cv2.imread(code_file_path, cv2.IMREAD_GRAYSCALE)
        index_i = int(f.replace('.jpg', ''))

        if index_i < 350:
            x_train.append(img)
            y_train.append(int(data_labels['number'][index_i + 1]))
        else:
            x_test.append(img)
            y_test.append(int(data_labels['number'][index_i + 1]))


x_train = np.array(x_train)
y_train = np.array(y_train)

x_test = np.array(x_test)
y_test = np.array(y_test)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(14, 10)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1000)
model.evaluate(x_test,  y_test, verbose=2)

# 保存权重
model.save_weights('saved_checkpoint')



