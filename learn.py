import tensorflow as tf
import cv2 as cv
from models.model import TestModel
from random import random
import pandas as pd
import numpy as np
from datetime import datetime

tf.keras.backend.set_floatx('float64')
batch_size = 16
model = TestModel(14)
output_data = pd.read_csv("joints.csv").to_numpy()
loss_fun = tf.keras.losses.MeanSquaredError()
opt = tf.optimizers.Adam()
log_dir = f"./logs/{datetime.now().strftime('%H%M%S')}"
writer = tf.summary.create_file_writer(log_dir)
writer.set_as_default()

for step in range(1000):
    indices = np.random.choice(range(len(output_data)), size=batch_size)
    batch_input = tf.constant([cv.imread(f"images/{i}.jpg", cv.IMREAD_UNCHANGED) for i in indices], dtype=tf.float64)
    batch_output = tf.constant([output_data[i] for i in indices], dtype=tf.float64)

    with tf.GradientTape() as tape:
        out = model(batch_input)
        loss = loss_fun(batch_output, out)
    
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    tf.summary.scalar('loss', loss, step=step)