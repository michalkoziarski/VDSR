import model
import os
import json
import argparse
import tensorflow as tf


def predict(images):
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'model')

    assert os.path.exists(checkpoint_path)

    with open(os.path.join(os.path.dirname(__file__), 'params.json')) as f:
        params = json.load(f)

    input = tf.placeholder(tf.float32)
    network = model.Model(input, params['n_layers'], params['kernel_size'], params['n_filters'])

    with tf.Session() as sess:
        checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint.model_checkpoint_path)

        predictions = []

        for image in images:
            predictions.append(network.output.eval(feed_dict={input: image}))

    return predictions
