import numpy as np
import tensorflow as tf


class Model:
    def __init__(self, input, n_layers=20, k=3, n_filters=64):
        self.input = input
        self.n_layers = n_layers
        self.k = k
        self.n_filters = n_filters
        self.weights = []
        self.biases = []
        self.output = self.input
        
        for i in range(self.n_layers):
            if i == 0:
                in_shape = 1
            else:
                in_shape = self.n_filters

            if i == self.n_layers - 1:
                out_shape = 1
            else:
                out_shape = self.n_filters

            weight = tf.Variable(tf.random_normal([self.k, self.k, in_shape, out_shape],
                                                  stddev=np.sqrt(2 / (k ** 2 * in_shape))))
            bias = tf.Variable(tf.zeros([out_shape]))

            self.weights.append(weight)
            self.biases.append(bias)
            self.output = tf.nn.bias_add(tf.nn.conv2d(self.output, weight, strides=[1, 1, 1, 1], padding='SAME'), bias)

            if i < self.n_layers - 1:
                self.output = tf.nn.relu(self.output)

        self.residual = self.output
        self.output = tf.add(self.output, self.input)
