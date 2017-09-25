import model
import utils
import os
import json
import argparse
import numpy as np
import tensorflow as tf

from scipy import misc
from skimage import color


def load_model(session):
    checkpoint_path = os.path.join(os.path.dirname(__file__), 'model')

    assert os.path.exists(checkpoint_path)

    with open(os.path.join(os.path.dirname(__file__), 'params.json')) as f:
        params = json.load(f)

    input = tf.placeholder(tf.float32)
    network = model.Model(input, params['n_layers'], params['kernel_size'], params['n_filters'])
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)
    saver = tf.train.Saver()
    saver.restore(session, checkpoint.model_checkpoint_path)

    return network


def predict(images, session=None, network=None, targets=None, border=0):
    session_passed = session is not None

    if not session_passed:
        session = tf.Session()

    if network is None:
        network = load_model(session)

    predictions = []

    if targets is not None:
        psnr = []

    for i in range(len(images)):
        image = images[i]

        if len(image.shape) == 3:
            image_ycbcr = color.rgb2ycbcr(image)
            image_y = image_ycbcr[:, :, 0]
        else:
            image_y = image.copy()

        image_y = image_y.astype(np.float) / 255
        reshaped_image_y = np.array([np.expand_dims(image_y, axis=2)])
        prediction = network.output.eval(feed_dict={network.input: reshaped_image_y}, session=session)[0]
        prediction *= 255

        if targets is not None:
            if len(targets[i].shape) == 3:
                target_y = color.rgb2ycbcr(targets[i])[:, :, 0]
            else:
                target_y = targets[i].copy()

            psnr.append(utils.psnr(prediction[border:-border, border:-border, 0],
                                   target_y[border:-border, border:-border], maximum=255.0))

        if len(image.shape) == 3:
            prediction = color.ycbcr2rgb(np.concatenate((prediction, image_ycbcr[:, :, 1:3]), axis=2)) * 255
        else:
            prediction = prediction[:, :, 0]

        prediction = np.clip(prediction, 0, 255).astype(np.uint8)
        predictions.append(prediction)

    if not session_passed:
        session.close()

    if targets is not None:
        return predictions, psnr
    else:
        return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-in', help='a path of the input image or a directory of the input images', required=True)
    parser.add_argument('-out', help='a path for the output image or a directory for the output images', required=True)
    args = vars(parser.parse_args())

    if os.path.isfile(args['in']):
        image = misc.imread(args['in'])
        prediction = predict([image])[0]
        misc.imsave(args['out'], prediction)
    elif os.path.isdir(args['in']):
        images = []
        file_names = []

        for file_name in os.listdir(args['in']):
            images.append(misc.imread(os.path.join(args['in'], file_name)))
            file_names.append(file_name)

        predictions = predict(images)

        if not os.path.exists(args['out']):
            os.mkdir(args['out'])

        for file_name, prediction in zip(file_names, predictions):
            misc.imsave(os.path.join(args['out'], file_name), prediction)
    else:
        raise ValueError('Incorrect input path.')
