import tensorflow as tf
import tensorflow.contrib.layers as layers
import config
# 文件路径查找模块
import glob
import cv2
import numpy as np
import re


def batch_norm(x, is_training, decay=0.9, eps=1e-4):
    shape = x.get_shape().as_list()
    assert len(shape) in [2, 4]

    n_out = shape[-1]

    shift = tf.Variable(tf.zeros([n_out]))
    scale = tf.Variable(tf.ones([n_out]))

    if len(shape) == 2:
        batch_mean, batch_var = tf.nn.moments(x, [0])
    else:
        batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])

    ema = tf.train.ExponentialMovingAverage(decay=decay)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(is_training, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))

    return tf.nn.batch_normalization(x, mean, var, shift, scale, eps)


def code_to_vec(code):
    def char_to_vec(c):
        y = np.zeros((len((config.CHARS)),))
        y[(config.CHARS).index(c)] = 1.0
        return y

    # 生成N-dimension的one-hot矩阵
    c = np.vstack([char_to_vec(c) for c in code])

    # 不足20位的用全0的矩阵
    if len(code) < config.LABEL_SIZE:
        a = np.zeros((config.LABEL_SIZE - len(code), 10))
        # 增加行
        c = np.row_stack((c, a))

    return c.flatten()


def read_test_img(img_glob, max):
    num = 0
    for fname in sorted(glob.glob(img_glob)):
        if num >= max:
            break
        im = cv2.imread(fname)[:, :, 0].astype(np.float32) / 255.

        code = re.split('_|\.', fname)[2]

        yield im, code_to_vec(code)

        # break

        num += 1


def count_params():
    "print number of trainable variables"
    n = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()])

    print("Model size: %dM" % (n / 1000000,))

def count_params_detail():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        print(shape,'total:',variable_parametes)
        total_parameters += variable_parametes
    print("Model size: %dM" % (total_parameters / 1000000,))
