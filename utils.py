import numpy as np
import tensorflow as tf


def psnr(x, y, maximum=1.0):
    return 20 * np.log10(maximum) - 10 * tf.log(tf.maximum(tf.reduce_mean(tf.pow(x - y, 2)), 1e-20)) / np.log(10)
