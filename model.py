import tensorflow as tf
from tensorflow.keras import layers

class EncoderNet(tf.keras.Model):
    def __init__(self, dims):
        super(EncoderNet, self).__init__()
        self.layers1 = layers.Dense(dims[1], activation=None)
        self.layers2 = layers.Dense(dims[1], activation=None)

    def call(self, x, training=None):
        out1 = self.layers1(x)
        out2 = self.layers2(x)

        out1 = tf.nn.l2_normalize(out1, axis=1)
        out2 = tf.nn.l2_normalize(out2, axis=1)

        return out1, out2
