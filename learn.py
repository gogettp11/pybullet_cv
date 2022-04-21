import tensorflow as tf
# from models.model import AlexNet
import pandas as pd
import numpy as np
from datetime import datetime
from const import *
import clearml as cl
import random

# https://www.inference.org.uk/itprnn/book.pdf


class AlexNet(tf.keras.Model):

    def __init__(self, outputs):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(11, 96, input_shape=(
            224, 224, 2), strides=4, activation=tf.nn.tanh)
        self.pool1 = tf.keras.layers.MaxPool2D(3, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(5, 4, activation=tf.nn.tanh)
        self.pool2 = tf.keras.layers.MaxPool2D(3)
        self.conv3 = tf.keras.layers.Conv2D(3, 1, activation=tf.nn.tanh)
        self.conv4 = tf.keras.layers.Conv2D(3, 1, activation=tf.nn.tanh)
        self.conv5 = tf.keras.layers.Conv2D(3, 1, activation=tf.nn.tanh)
        self.pool3 = tf.keras.layers.MaxPool2D(3)
        self.flatten = tf.keras.layers.Flatten()
        self.dense2 = tf.keras.layers.Dense(500, activation=tf.nn.tanh)
        self.dense4 = tf.keras.layers.Dense(outputs)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        images = self.conv2(x)
        x = self.pool2(images)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.dense2(x)
        x = self.dense4(x)
        return x, images


# data = cl.Dataset.get(
#     dataset_id='803ac666d2ad47d897f3c5b4a1e3bd2b', only_published=True)
data_path = './train_data' # data.get_local_copy()
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ensuring it is running on cpu
tf.keras.backend.set_floatx('float64')
batch_size = 64
model = AlexNet(14)
train_source = data_path
test_source = data_path
output_data = pd.read_csv(f"{train_source}/joints.csv").to_numpy()
output_data_test = pd.read_csv(f"{test_source}/joints.csv").to_numpy()
loss_fun = tf.keras.losses.MeanSquaredError()
opt = tf.optimizers.Adam(learning_rate=0.0001)
log_dir = f"./logs/{datetime.now().strftime('%H%M%S')}"
writer = tf.summary.create_file_writer(log_dir)
writer.set_as_default()

test_indicies = random.sample(range(len(output_data_test)), 300)
data_indicies = [x for x in range(len(output_data)) if x not in test_indicies]

print("start training!")

for step in range(50000):
    # train
    indices = np.random.choice(data_indicies, size=batch_size)
    batch_input = tf.constant([np.load(
        f"{train_source}/images/{i}", allow_pickle=True) for i in indices], dtype=tf.float64)
    batch_output = tf.constant([output_data[i]
                                for i in indices], dtype=tf.float64)

    with tf.GradientTape() as tape:
        out, _ = model(batch_input)
        loss = loss_fun(batch_output, out)

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    tf.summary.scalar(f'loss train', loss, step=step)

    # test
    indices = np.random.choice(test_indicies, size=8)
    batch_input = tf.constant([np.load(
        f"{test_source}/images/{i}", allow_pickle=True) for i in indices], dtype=tf.float64)
    batch_output = tf.constant([output_data_test[i] for i in indices],
                               dtype=tf.float64)

    out, images = model(batch_input)
    loss = loss_fun(batch_output, out)
    tf.summary.scalar(f'loss test', loss, step=step)
    tf.summary.image(f'original test', batch_input, step=step)
    imgs = tf.split(images, num_or_size_splits=5, axis=3)
    for i, img in enumerate(imgs):
        tf.summary.image(f'processed {i}', img, step=step)
