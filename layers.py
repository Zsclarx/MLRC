import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import glorot_uniform

class GraphConvolution(layers.Layer):
    def __init__(self, in_features, out_features, dropout=0., activation=tf.nn.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.activation = activation
        self.weight = self.add_weight("weight",
                                      shape=(in_features, out_features),
                                      initializer=glorot_uniform(),
                                      trainable=True)

    def call(self, inputs, adj, training=None):
        inputs = tf.nn.dropout(inputs, rate=self.dropout)
        support = tf.matmul(inputs, self.weight)
        output = tf.sparse.sparse_dense_matmul(adj, support)
        output = self.activation(output)
        return output

class SampleDecoder(layers.Layer):
    def __init__(self, activation=tf.nn.sigmoid):
        super(SampleDecoder, self).__init__()
        self.activation = activation

    def call(self, zx, zy):
        sim = tf.reduce_sum(tf.multiply(zx, zy), axis=1)
        sim = self.activation(sim)
        return sim
