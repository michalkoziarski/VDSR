import numpy as np
import tensorflow as tf


def psnr(x, y, maximum=1.0):
    return 20 * tf.log(maximum / tf.sqrt(tf.reduce_mean(tf.pow(x - y, 2)))) / np.log(10)
