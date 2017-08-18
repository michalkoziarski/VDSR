import model
import os
import json
import argparse
import numpy as np
import tensorflow as tf

from scipy import misc
from skimage import color


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
            if len(image.shape) == 3:
                image_ycbcr = color.rgb2ycbcr(image)
                image_y = image_ycbcr[:, :, 0]
            else:
                image_y = image.copy()

            image_y = image_y.astype(np.float) / 255
            reshaped_image_y = np.array([np.expand_dims(image_y, axis=2)])
            prediction = network.output.eval(feed_dict={input: reshaped_image_y})[0]
            prediction = (np.clip(prediction, 0.0, 1.0) * 255).astype(np.uint)

            if len(image.shape) == 3:
                prediction = color.ycbcr2rgb(np.concatenate((prediction, image_ycbcr[:, :, 1:3]), axis=2))
            else:
                prediction = prediction[:, :, 0]

            predictions.append(prediction)

    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', help='a path of the input image', required=True)
    parser.add_argument('-output', help='a path of the output image', required=True)
    args = vars(parser.parse_args())

    image = misc.imread(args['input'])
    prediction = predict([image])[0]
    misc.imsave(args['output'], prediction)
