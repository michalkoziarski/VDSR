import data
import model
import os
import json
import tensorflow as tf


with open(os.path.join(os.path.dirname(__file__), 'params.json')) as f:
    params = json.load(f)

data_set = data.load('train', params['benchmark'], params['batch_size'])

input = tf.placeholder(tf.float32, shape=(params['batch_size'], params['patch_size'], params['patch_size'], 1))
ground_truth = tf.placeholder(tf.float32, shape=(params['batch_size'], params['patch_size'], params['patch_size'], 1))
learning_rate = tf.placeholder(tf.float32, shape=[])
global_step = tf.Variable(0, trainable=False, name='global_step')
network = model.Model(input, params['n_layers'], params['kernel_size'], params['n_filters'])
base_loss = tf.reduce_mean(tf.square(tf.subtract(ground_truth, network.output)))
weight_loss = params['weight_decay'] * tf.reduce_sum(tf.stack([tf.nn.l2_loss(weight) for weight in network.weights]))
loss = base_loss + weight_loss

tf.summary.scalar('base loss', base_loss)
tf.summary.scalar('weight loss', weight_loss)
tf.summary.scalar('total loss', loss)
tf.summary.scalar('learning rate', learning_rate)
tf.summary.image('input', input)
tf.summary.image('output', network.output)
tf.summary.image('ground truth', ground_truth)
tf.summary.image('residual', network.residual)

for i in range(len(network.weights)):
    tf.summary.histogram('weights/layer #%d' % (i + 1), network.weights[i])
    tf.summary.histogram('biases/layer #%d' % (i + 1), network.biases[i])

summary_step = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=0)

optimizer = tf.train.MomentumOptimizer(learning_rate, params['momentum'])
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

with tf.Session() as sess:
    checkpoint = tf.train.get_checkpoint_state(checkpoint_path)

    if checkpoint and checkpoint.model_checkpoint_path:
        print('Restoring model...')

        saver.restore(sess, checkpoint.model_checkpoint_path)

        print('Restoration complete.')
    else:
        print('Initializing new model...')

        sess.run(tf.global_variables_initializer())

        print('Initialization complete.')

    print('Training model...')

    while tf.train.global_step(sess, global_step) * params['batch_size'] < data_set.length * params['epochs']:
        batch = tf.train.global_step(sess, global_step)
        epoch = batch * params['batch_size'] / data_set.length

        x, y = data_set.batch()

        current_learning_rate = params['learning_rate'] * params['learning_rate_decay'] ** (epoch // params['learning_rate_decay_step'])

        feed_dict = {input: x, ground_truth: y, learning_rate: current_learning_rate}

        if batch * params['batch_size'] % data_set.length == 0:
            print('Processing epoch #%d...' % (epoch + 1))

            _, summary = sess.run([train_step, summary_step], feed_dict=feed_dict)
            saver.save(sess, model_path)
            summary_writer.add_summary(summary, epoch)
        else:
            sess.run([train_step], feed_dict=feed_dict)

    print('Training complete.')
