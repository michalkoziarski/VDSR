import data
import model
import utils
import os
import json
import numpy as np
import tensorflow as tf


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

    for set_name in ['Set5', 'Set14', 'B100', 'Urban100']:
        for scaling_factor in [2, 3, 4]:
            dataset = data.load('test', set_name, scaling_factors=[scaling_factor], verbose=False)

            scores = []

            while True:
                fetched = dataset.fetch()

                if fetched is None:
                    break

                image, target = fetched
                prediction = network.output.eval(feed_dict={input: image})
                scores.append(utils.psnr(target, prediction).eval())

            print('Dataset "%s", scaling factor = %d. Mean PSNR = %.2f.' % (set_name, scaling_factor, np.mean(scores)))
