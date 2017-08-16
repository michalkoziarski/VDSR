import model
import os
import json
import argparse
import numpy as np
import tensorflow as tf

from scipy import misc


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
            predictions.append(np.clip(network.output.eval(feed_dict={input: image}), 0.0, 1.0))

    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', help='a path of the input image', required=True)
    parser.add_argument('-output', help='a path of the output image', required=True)
    args = vars(parser.parse_args())

    image = np.expand_dims((misc.imread(args['input']).astype(np.float) / 255), axis=2)
    prediction = predict([np.array([image])])[0][0][:, :, 0]
    misc.imsave(args['output'], prediction)
