# coding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import LeNet5_infernece
import os
import numpy as np

# 配置神经网络参数
INPUT_NODE = 784
OUTPUT_NODE = 10

IMAGE_SIZE = 28
# 步长
NUM_CHANNELS = 1
NUM_LABELS = 10

# 第一层卷积的此处和深度
CONV1_DEEP = 32
CONV1_SIZE = 5

# 第二层卷积的尺寸和深度
CONV2_DEEP = 64
CONV2_SIZE = 5

# 全连接层的节点个数
FC_SIZE = 512

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 6000
MOVING_AVERAGE_DECAY = 0.99


# 定义卷积神经网络的前向传播过程。这里添加了一个新的参数train，用于区分训练过程和测试过程。
# 这个程序中将用到dropout方法，dropout可以进一步提升模型的可靠性并防止过拟合，
# dropout过程只在训练时使用
def inference(input_tensor, train, regularizer):
    # 声明第一次卷积层的变量并实现向前传播过程。这个过程和6.3.1小节中介绍的一致
    # 通过使用不同的命名空间来隔离不同层的变量。这样可以让每一层中的变量命名只需要考虑当前层的作用
    # 而不需要担心重命名的问题。和标准的LeNet-5模型不大一样，这里
    # 定义的卷积输入为28*28*1的元素MNIST图片像素。因为卷积中使用了全0填充
    # 所以输出为28*28*32
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable(
            "weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        # 使用边长为5，深度为32的过滤器 过滤器移动的步长为1，且使用全0填充
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 第二层 池化层的前向传播过程。这里选用最大池化层，池化层过滤器的边长为2
    # 使用全0填充移动步长为2，这一层的输入时上一层的输出也就是[28 * 28 * 32] 的矩阵
    # 输出为14 * 14 * 32 的矩阵
    with tf.name_scope("layer2-pool1"):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 声明第三层卷积层的变量 并实现向前传播过程。这一层输入为14*14*32的矩阵
    # 输出为14*14*64
    with tf.variable_scope("layer3-conv2"):
        conv2_weights = tf.get_variable(
            "weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        # 使用边长为5 深度为64的过滤器，过来移动步长为1，且使用全0填充
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 实现第四层 池化层的向前传播过程。这一层和第二层的结构是一样的。这一层的输入为
    # 输入 14*14*64
    # 输出 7 * 7 * 64
    with tf.name_scope("layer4-pool2"):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 将第四层池化层的输出转化为第五层全连接处的输入格式。第四层的输出为7*7*64
        # 然而第五次全连接层需要的输入格式为向量，所有这里需要将这个7*7*64的矩阵拉直成一个向量。pool2.get_shape()
        # 函数可以得到第四层输出矩阵的维度而不需要手工计算。注意因为每一层神经网络的输入都为一个batch的矩阵，所以这里得到的维度也包含了
        # 一个batch钟的数据个数
        pool_shape = pool2.get_shape().as_list()
        # 计算将矩阵拉直成向量之后的长度，这个长度就是矩阵长款及深度的乘积。注意这里
        # pool_shape[0]为一个batch中的个数
        # 精妙的地方
        nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
        # 通过tf.reshape函数将第四层的输出变成一个batch的向量
        reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    # 声明第五层全连接层的变量 并实现向量传播过程。这一层的输入时拉直之后的一个数组向量
    # 向量的长度为3136，输出的是一组长度为512的向量。这一层和之前的第五章中介绍的基本一组，唯一的却别就是引入了dropout的概率
    # dropout在训练时会随机将部分节点的输出改为0.dropout可以避免过度拟合问题，从而使得模型在测试数据上的效果更好
    # dropout一般只在全连接层而不是卷积层或者池化层使用
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        # 只有全连接层的权重需要加入正则化
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 声明第六层全连接层的变量并实现前向传播过程，这一层的输入为一组长度为512的向量，
    # 输出为一组长度为10的向量。这一层的输出通过softmax之后就等到最后的分类结果
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit


def train(mnist):
    # 定义输出为4维矩阵的placeholder
    x = tf.placeholder(tf.float32, [
        BATCH_SIZE,
        IMAGE_SIZE,
        IMAGE_SIZE,
        NUM_CHANNELS],
                       name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = inference(x, False, regularizer)
    global_step = tf.Variable(0, trainable=False)

    # 定义损失函数、学习率、滑动平均操作以及训练过程。
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类。
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            reshaped_xs = np.reshape(xs, (
                BATCH_SIZE,
                IMAGE_SIZE,
                IMAGE_SIZE,
                NUM_CHANNELS))
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: reshaped_xs, y_: ys})

            if i % 1000 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))


def main(argv=None):
    mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
