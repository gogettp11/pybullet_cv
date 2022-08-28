import tensorflow as tf
import cv2 as cv
from models.model import AlexNet, TestModel
from random import randint
import pandas as pd
import numpy as np
from datetime import datetime
from const import *
import os

# https://www.inference.org.uk/itprnn/book.pdf

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ensuring it is running on cpu
tf.keras.backend.set_floatx('float64')
batch_size = 16
model = AlexNet(14)
train_source = DATA_TRAIN
test_source = DATA_TEST
output_data = pd.read_csv(f"{train_source}/joints.csv").to_numpy()
output_data_test = pd.read_csv(f"{test_source}/joints.csv").to_numpy()
loss_fun = tf.keras.losses.MeanSquaredError()
opt = tf.optimizers.Adam(learning_rate=0.0001)
log_dir = f"./logs/{datetime.now().strftime('%H%M%S')}"
writer = tf.summary.create_file_writer(log_dir)
writer.set_as_default()

data_indicies = range(len(output_data))
test_indicies = range(len(output_data_test))

for step in range(50000):
    # train
    indices = np.random.choice(data_indicies, size=batch_size)
    batch_input = tf.constant([np.load(f"{train_source}/images/{i}", allow_pickle=True) for i in indices], dtype=tf.float64)
    batch_output = tf.constant([output_data[i] for i in indices], dtype=tf.float64)

    with tf.GradientTape() as tape:
        out, _ = model(batch_input)
        loss = loss_fun(batch_output, out)

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    tf.summary.scalar(f'loss {train_source}', loss, step=step)

    # test
    indices = np.random.choice(test_indicies, size=4)
    batch_input = tf.constant([np.load(f"{test_source}/images/{i}", allow_pickle=True) for i in indices], dtype=tf.float64)
    batch_output = tf.constant([output_data_test[i] for i in indices],
                            dtype=tf.float64)

    out, images = model(batch_input)
    loss = loss_fun(batch_output, out)
    tf.summary.scalar(f'loss {test_source}', loss, step=step)
    tf.summary.image(f'original {test_source}', batch_input, step=step)
    imgs = tf.split(images, num_or_size_splits=5, axis=3)
    for i, img in enumerate(imgs):
        tf.summary.image(f'processed {i}', img, step=step)
