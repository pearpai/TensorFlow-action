# coding=utf-8
import tensorflow as tf
import numpy as np

# M = np.array([
#     [[1], [-1], [0]],
#     [[-1], [2], [1]],
#     [[0], [2], [-2]]
# ])
#
# print "Matrix shape is ", M.shape
#
# # 定义卷积过滤深度为1
# filter_weight = tf.get_variable('weights', [2, 2, 1, 1],
#                                 initializer=tf.constant_initializer([
#                                     [1, -1],
#                                     [0, 2]
#                                 ]))

# 通过tf.get_variable的方式创建过滤器的权重和偏置项变量。上面介绍的卷积层的参数个数之和
# 过滤的尺寸、深度以及当前层节点矩阵的深度有关，所以这里什么的参数变量时一个四维矩阵
# 签名两个维度代表了过滤尺寸的维度，第三个维度表示当前的深度，第四个维度表示过滤的深度
filter_weight = tf.get_variable('weights', [5, 5, 3, 16],
                                initializer=tf.truncated_normal_initializer(stddev=0.1))

# 和卷积层的权重类似，当前层矩阵上不同位置的偏置项也是共享的，所有总归有下一层深度个不同
# 的偏置项。本样例代码中16位过滤器的深度，也是神经网络中下一层节点矩阵的深度
biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0.1))

conv = tf.nn.conv2d(input, filter_weight, strides=[1, 1, 1, 1], padding='SAME')
bias = tf.nn.bias_add(conv, biases)

actived_conv = tf.nn.relu(bias)
