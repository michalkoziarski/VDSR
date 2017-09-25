import data
import model
import utils
import predict
import os
import json
import numpy as np
import tensorflow as tf


with open(os.path.join(os.path.dirname(__file__), 'params.json')) as f:
    params = json.load(f)

train_set = data.TrainSet(params['train_set'], params['batch_size'], params['patch_size'])
validation_set = data.TestSet(params['validation_set'])

input = tf.placeholder(tf.float32)
ground_truth = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32, shape=[])
global_step = tf.Variable(0, trainable=False, name='global_step')
network = model.Model(input, params['n_layers'], params['kernel_size'], params['n_filters'])
base_loss = tf.reduce_sum(tf.nn.l2_loss(tf.subtract(network.output, ground_truth)))
weight_loss = params['weight_decay'] * tf.reduce_sum(tf.stack([tf.nn.l2_loss(weight) for weight in network.weights]))
loss = base_loss + weight_loss
validation_score = tf.placeholder(tf.float32)

tf.summary.scalar('base loss', base_loss)
tf.summary.scalar('weight loss', weight_loss)
tf.summary.scalar('total loss', loss)
tf.summary.scalar('learning rate', learning_rate)
tf.summary.scalar('validation score', validation_score)
tf.summary.image('input', input)
tf.summary.image('output', network.output)
tf.summary.image('ground truth', ground_truth)
tf.summary.image('residual', network.residual)

for i in range(len(network.weights)):
    tf.summary.histogram('weights/layer #%d' % (i + 1), network.weights[i])
    tf.summary.histogram('biases/layer #%d' % (i + 1), network.biases[i])

summary_step = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=0)

optimizer = tf.train.AdamOptimizer(learning_rate)
gradients = optimizer.compute_gradients(loss)
clip_value = params['gradient_clipping'] / learning_rate
capped_gradients = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in gradients]
train_step = optimizer.apply_gradients(capped_gradients, global_step=global_step)

checkpoint_path = os.path.join(os.path.dirname(__file__), 'model')
model_path = os.path.join(checkpoint_path, 'model.ckpt')
log_path = os.path.join(os.path.dirname(__file__), 'log')

summary_writer = tf.summary.FileWriter(log_path)

for path in [checkpoint_path, log_path]:
    if not os.path.exists(path):
        os.mkdir(path)

with tf.Session() as session:
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

    if checkpoint and checkpoint.model_checkpoint_path:
        print('Restoring model...')

        session.run(tf.global_variables_initializer())
        saver.restore(session, checkpoint.model_checkpoint_path)

        print('Restoration complete.')
    else:
        print('Initializing new model...')

        session.run(tf.global_variables_initializer())

        print('Initialization complete.')

    print('Training model...')

    while tf.train.global_step(session, global_step) * params['batch_size'] < train_set.length * params['epochs']:
        batch = tf.train.global_step(session, global_step)
        epoch = batch * params['batch_size'] / train_set.length

        x, y = train_set.batch()

        current_learning_rate = params['learning_rate'] * params['learning_rate_decay'] ** (epoch // params['learning_rate_decay_step'])

        feed_dict = {input: x, ground_truth: y, learning_rate: current_learning_rate}

        if batch * params['batch_size'] % train_set.length == 0:
            print('Processing epoch #%d...' % (epoch + 1))

            predictions, psnr = predict.predict(validation_set.images, session, network, targets=validation_set.targets)
            feed_dict[validation_score] = np.mean(psnr)

            _, summary = session.run([train_step, summary_step], feed_dict=feed_dict)
            saver.save(session, model_path)
            summary_writer.add_summary(summary, epoch)
        else:
            session.run([train_step], feed_dict=feed_dict)

    print('Training complete.')
