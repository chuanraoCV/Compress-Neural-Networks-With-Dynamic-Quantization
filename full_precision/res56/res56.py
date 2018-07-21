# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
import math
import numpy as np
from datetime import datetime
import time
import sys
import os

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('residual_net_n', 9, '')
tf.app.flags.DEFINE_string('train_tf_path', 'F:/dynamic quantization/data/train.tf', '')
tf.app.flags.DEFINE_string('val_tf_path', 'F:/dynamic quantization/data/test.tf', '')
tf.app.flags.DEFINE_integer('train_batch_size', 128, '')
tf.app.flags.DEFINE_integer('val_batch_size', 100, '')
tf.app.flags.DEFINE_float('weight_decay', 1e-3, 'Weight decay')
tf.app.flags.DEFINE_integer('summary_interval', 100, 'Interval for summary.')
tf.app.flags.DEFINE_integer('val_interval', 1000, 'Interval for evaluation.')
tf.app.flags.DEFINE_integer('max_steps', 80000, 'Maximum number of iterations.')
tf.app.flags.DEFINE_integer('save_interval', 5000, '')

def one_hot_embedding(label, n_classes):
  embedding_params = np.eye(n_classes, dtype=np.float32)
  with tf.device('/cpu:0'):
    params = tf.constant(embedding_params)
    embedding = tf.gather(params, label)
  return embedding

def conv2d(x, n_in, n_out, k, s, p='SAME', bias=False, scope='conv'):
  with tf.variable_scope(scope):
    kernel = tf.get_variable(name='weight', shape=[k, k, n_in, n_out],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    tf.add_to_collection('weights', kernel)
    conv = tf.nn.conv2d(x, kernel, [1,s,s,1], padding=p)
    if bias:
      bias = tf.get_variable('bias', [n_out], initializer=tf.constant_initializer(0.0))
      tf.add_to_collection('biases', bias)
      conv = tf.nn.bias_add(conv, bias)
  return conv

def batch_norm(x, n_out, phase_train, scope='bn', affine=True):
  with tf.variable_scope(scope):
    # beta = tf.Variable(tf.constant(0.0, shape=[n_out]),name='beta', trainable=True)
    beta = tf.get_variable(name='beta',shape=[n_out],initializer=tf.constant_initializer(0.0))
    # gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),name='gamma', trainable=affine)
    gamma = tf.get_variable(name='gamma',shape=[n_out],initializer=tf.constant_initializer(1.0))
    tf.add_to_collection('biases', beta)
    tf.add_to_collection('weights', gamma)

    batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
    ema = tf.train.ExponentialMovingAverage(decay=0.99)
    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = control_flow_ops.cond(phase_train,mean_var_with_update,lambda: (ema.average(batch_mean), ema.average(batch_var)))

    normed = tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, 1e-3, affine)
  return normed

def residual_block(x, n_in, n_out, subsample, phase_train, scope='res_block'):
  with tf.variable_scope(scope):
    if subsample:
      y = conv2d(x, n_in, n_out, 3, 2, 'SAME', False, scope='conv_1')
      shortcut = conv2d(x, n_in, n_out, 3, 2, 'SAME',
                False, scope='shortcut')
    else:
      y = conv2d(x, n_in, n_out, 3, 1, 'SAME', False, scope='conv_1')
      shortcut = tf.identity(x, name='shortcut')
    y = batch_norm(y, n_out, phase_train, scope='bn_1')
    y = tf.nn.relu(y, name='relu_1')
    y = conv2d(y, n_out, n_out, 3, 1, 'SAME', True, scope='conv_2')
    y = batch_norm(y, n_out, phase_train, scope='bn_2')
    y = y + shortcut
    y = tf.nn.relu(y, name='relu_2')
  return y

def residual_group(x, n_in, n_out, n, first_subsample, phase_train, scope='res_group'):
  with tf.variable_scope(scope):
    y = residual_block(x, n_in, n_out, first_subsample, phase_train, scope='block_1')
    for i in range(n - 1):
      y = residual_block(y, n_out, n_out, False, phase_train, scope='block_%d' % (i + 2))
  return y

def residual_net(x, n, n_classes, phase_train, scope='res_net'):
  with tf.variable_scope(scope):
    y = conv2d(x, 3, 16, 3, 1, 'SAME', False, scope='conv_init')
    y = batch_norm(y, 16, phase_train, scope='bn_init')
    y = tf.nn.relu(y, name='relu_init')
    y = residual_group(y, 16, 16, n, False, phase_train, scope='group_1')
    y = residual_group(y, 16, 32, n, True, phase_train, scope='group_2')
    y = residual_group(y, 32, 64, n, True, phase_train, scope='group_3')
    # y = conv2d(y, 64, n_classes, 1, 1, 'SAME', True, scope='conv_last')
    y = tf.nn.avg_pool(y, [1, 8, 8, 1], [1, 1, 1, 1], 'VALID', name='avg_pool')
    y = tf.reshape(y,[-1,64])
    w = tf.get_variable(name='weight_fc', shape=[64,10],initializer=tf.contrib.layers.xavier_initializer_conv2d())
    b = tf.get_variable(name='weight_biase', shape=[10],initializer=tf.constant_initializer(0)  )
    y = tf.matmul(y,w)+b
  return y

def _loss(logits, labels, scope='loss'):
  with tf.variable_scope(scope):
    # entropy loss
    targets = one_hot_embedding(labels, 10)
    entropy_loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=targets),
      name='entropy_loss')
    tf.add_to_collection('losses', entropy_loss)
    # weight l2 decay loss
    weight_l2_losses = [tf.nn.l2_loss(o) for o in tf.get_collection('weights')]
    weight_decay_loss = FLAGS.weight_decay*tf.add_n(weight_l2_losses)
    tf.add_to_collection('losses', weight_decay_loss)
  # for var in tf.get_collection('losses'):
    # tf.scalar_summary('losses/' + var.op.name, var)
  # total loss
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _accuracy(logits, gt_label, scope='accuracy'):
  with tf.variable_scope(scope):
    pred_label = tf.argmax(logits, 1)
    acc = 1.0 - tf.nn.zero_fraction(
      tf.cast(tf.equal(pred_label, gt_label), tf.int32))
  return acc

def _train_op(loss, global_step, learning_rate):
  params = tf.trainable_variables()
  gradients = tf.gradients(loss, params, name='gradients')
  optim = tf.train.MomentumOptimizer(learning_rate, 0.9)
  update = optim.apply_gradients(zip(gradients, params))
  with tf.control_dependencies([update]):
    train_op = tf.no_op(name='train_op')
  return train_op

def cifar10_input_stream(records_path):
  reader = tf.TFRecordReader()
  filename_queue = tf.train.string_input_producer([records_path], None)
  _, record_value = reader.read(filename_queue)
  features = tf.parse_single_example(record_value,
    {
      'image_raw': tf.FixedLenFeature([], tf.string),
      'label': tf.FixedLenFeature([], tf.int64),
    })
  image = tf.decode_raw(features['image_raw'], tf.uint8)
  image = tf.reshape(image, [32,32,3])
  image = tf.cast(image, tf.float32)
  label = tf.cast(features['label'], tf.int64)
  return image, label

def normalize_image(image):
  mean = [ 125.30690002,122.95014954,113.86599731]
  std = [ 62.9932518,62.08860397,66.70500946]
  normed_image = (image - mean) / std
  return normed_image

def random_distort_image(image):
  distorted_image = image
  distorted_image = tf.image.pad_to_bounding_box(
    image, 4, 4, 40, 40)  # pad 4 pixels to each side
  distorted_image = tf.random_crop(distorted_image, [32, 32, 3])
  distorted_image = tf.image.random_flip_left_right(distorted_image)
  return distorted_image

def make_train_batch(train_records_path, batch_size):
  with tf.variable_scope('train_batch'):
    with tf.device('/cpu:0'):
      train_image, train_label = cifar10_input_stream(train_records_path)
      train_image = normalize_image(train_image)
      train_image = random_distort_image(train_image)
      train_image_batch, train_label_batch = tf.train.shuffle_batch(
        [train_image, train_label], batch_size=batch_size, num_threads=4,
        capacity=50000,
        min_after_dequeue=1000)
  return train_image_batch, train_label_batch

def make_validation_batch(test_records_path, batch_size):
  with tf.variable_scope('evaluate_batch'):
    with tf.device('/cpu:0'):
      test_image, test_label = cifar10_input_stream(test_records_path)
      test_image = normalize_image(test_image)
      test_image_batch, test_label_batch = tf.train.batch(
        [test_image, test_label], batch_size=batch_size, num_threads=1,
        capacity=10000)
  return test_image_batch, test_label_batch

phase_train = tf.placeholder(tf.bool, name='phase_train')
learning_rate = tf.placeholder(tf.float32, name='learning_rate')
global_step = tf.Variable(0, trainable=False, name='global_step')


train_image_batch, train_label_batch = make_train_batch(FLAGS.train_tf_path, FLAGS.train_batch_size)
val_image_batch, val_label_batch = make_validation_batch(FLAGS.val_tf_path, FLAGS.val_batch_size)

image_batch, label_batch = control_flow_ops.cond(phase_train,lambda: (train_image_batch, train_label_batch),lambda: (val_image_batch, val_label_batch))


logits = residual_net(image_batch, FLAGS.residual_net_n, 10, phase_train)


loss = _loss(logits, label_batch)
accuracy = _accuracy(logits, label_batch)

# train one step
train_op = _train_op(loss, global_step, learning_rate)


sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

init_op = tf.global_variables_initializer()
print('Initializing...')
sess.run(init_op, {phase_train.name: True})

print('Start training...')

tf.train.start_queue_runners(sess=sess)
curr_lr = 0.0
for step in range(FLAGS.max_steps):
  if step <= 48000:
    _lr = 1e-2
  elif step <=65000:
    _lr = 1e-3
  else:
    _lr = 1e-4
  if curr_lr != _lr:
    curr_lr = _lr
    print('Learning rate set to %f' % curr_lr)

  # train
  fetches = [train_op, loss]
  if step > 0 and step % FLAGS.summary_interval == 0:
    fetches += [accuracy]
  sess_outputs = sess.run(
    fetches, {phase_train.name: True, learning_rate.name: curr_lr})


  if step > 0 and step % FLAGS.summary_interval == 0:
    train_loss_value, train_acc_value= sess_outputs[1:]
    print('[%s] Iteration %d, train loss = %f, train accuracy = %f' %
        (datetime.now(), step, train_loss_value, train_acc_value))

  # validation
  if step > 0 and step % FLAGS.val_interval == 0:
    print('Evaluating...')
    n_val_samples = 10000
    val_batch_size = FLAGS.val_batch_size
    n_val_batch = int(n_val_samples / val_batch_size)
    val_logits = np.zeros((n_val_samples, 10), dtype=np.float32)
    val_labels = np.zeros((n_val_samples), dtype=np.int64)
    val_losses = []
    for i in range(n_val_batch):
      fetches = [logits, label_batch, loss]
      session_outputs = sess.run(
        fetches, {phase_train.name: False})
      val_logits[i*val_batch_size:(i+1)*val_batch_size, :] = session_outputs[0]
      val_labels[i*val_batch_size:(i+1)*val_batch_size] = session_outputs[1]
      val_losses.append(session_outputs[2])
    pred_labels = np.argmax(val_logits, axis=1)
    val_accuracy = np.count_nonzero(
      pred_labels == val_labels) / n_val_samples
    val_loss = float(np.mean(np.asarray(val_losses)))
    print('Test accuracy = %f' % val_accuracy)

saver = tf.train.Saver()
saver.save(sess,'F:/dynamic quantization/code/model/res.ckpt')