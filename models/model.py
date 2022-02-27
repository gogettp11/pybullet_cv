import tensorflow as tf

class TestModel(tf.keras.Model):

  def __init__(self, outputs):
    super().__init__()
    self.conv1 = tf.keras.layers.Conv2D(5, (3,3), input_shape=(224,224,4))
    self.conv2 = tf.keras.layers.Conv2D(5, (5,5))
    self.flatten = tf.keras.layers.Flatten()
    self.dense2 = tf.keras.layers.Dense(250, activation=tf.nn.relu)
    self.dense4 = tf.keras.layers.Dense(outputs)

  def call(self, inputs):
    x = self.conv1(inputs)
    x = self.conv2(x)
    x = self.flatten(x)
    x = self.dense2(x)
    x = self.dense4(x)
    return x