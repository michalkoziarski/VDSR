import data
import predict
import numpy as np
import tensorflow as tf


with tf.Session() as session:
    network = predict.load_model(session)

    for set_name in ['Set5', 'Set14', 'B100', 'Urban100']:
        for scaling_factor in [2, 3, 4]:
            dataset = data.TestSet(set_name, scaling_factors=[scaling_factor])
            predictions, psnr = predict.predict(dataset.images, session, network, targets=dataset.targets,
                                                border=scaling_factor)

            print('Dataset "%s", scaling factor = %d. Mean PSNR = %.2f.' % (set_name, scaling_factor, np.mean(psnr)))
