import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST 数据集的相关常数
# 输入层的节点数。对于MNIST数据集，这个就等于图片的像素
INPUT_NODE = 784
# 输出层的节点数。这个等于类别的数目。因为在MNIST数据集中
# 需要区分的是0~9这10个数字，所以这里输出层的节点数为10
OUTPUT_NODE = 10

# 配置神经网络参数
# 隐藏层节点数。这里使用只有一个隐藏层的网络结构作为样例
# 这个隐藏层有500个节点
LAYER1_NODE = 500

# 一个训练batch中的训练数据个数。数字越小时，训练过程约接近随机梯度下降；
# 数字越大时，训练越接近梯度下降
BATCH_SIZE = 100

# 基础学习率
LEARNING_RATE_BASE = 0.8
# 学习率的衰减率
LEARNING_RATE_DECAY = 0.99
# 描述模型复杂度的正则化项在损失函数中的系数
REGULARAZTION_RATE = 0.0001
# 训练轮数
TRAINING_STEPS = 30000
# 滑动平均衰减率
MOVING_AVERAGE_DECAY = 0.99


# 一个辅助函数，给定神经网络的输入和所有参数，计算神经网络的前向传播结果。在这里
# 定义了一个使用ReLU激活函数的三层全连接神经网络，通过加入隐藏层时间了多层神经网络结构
# 通过ReLU激活函数实现了去线性化。在这个函数中也支持传入用于计算参数平均值的类
# 这样方便在测试时使用滑动平均模型
def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
    # 当没有提供欢动平均类时，直接使用参数当前的取值
    if avg_class is None:
        # 计算隐藏层的前向传播结果，这里使用了ReLU激活函数
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
        # 计算输出层的前向传播结果。因为在计算损失函数时会一并计算 sofmax函数，
        # 所以这里不需要加入激活函数。而且不加入softmax不会影响预测结果。因为预测时
        # 使用的是不同类别的节点输出值的相对大小，有没有softmax层对最后的分类结果的
        # 计算没有影响。预算在计算整个神经网络的前向传播时可以不加入最后的softmax层。
        return tf.matmul(layer1, weights2) + biases2
    else:
        # 首先使用avg_class.average函数来计算得出变量的滑动平均值
        # 然后再计算相应的神经网络前向传播结果
        layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
        return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)


def train(mnist):
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
    # 生成隐藏层参数
    weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
    biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

    # 生成输出层参数
    weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
    biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

    # 计算不含滑动平均类的前向传播结果。这里给出的用于计算滑动平均的类为None，
    # 所以函数不会使用参数的滑动平均值
    y = inference(x, None, weights1, biases1, weights2, biases2)

    # 定义训练轮数及相关滑动平均类
    # 定义存储训练轮数的变量。这个变量不需要计算滑动平均值，所以这里制定这个变量为
    # 不可训练的变量（trainable=False）。在使用TensorFlow训练神经网络时，
    # 一般会将代表训练轮数的变量制定为不可训练的参数
    global_step = tf.Variable(0, trainable=False)
    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类。在第4章中介绍过给
    # 定训练轮数的变量可以加快训练早起变量的更新速度
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均。其他辅助变量（比如global_step）就
    # 不需要了。tf.trainable_variables返回的就是图上的集合
    # GraphKeys.TRAINABLE_VARIABLES中的元素。这个集合的元素就是所有没有指定
    # trainable=False 的参数
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的向前传播结果。第4章中介绍过滑动平均不会改变本事的取值
    # 而是维护了一个影子变量来记录滑动平均值。所以当需要使用这个滑动平均值时，需要明确调用average函数
    average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

    # 计算交叉熵机器平均值
    # 计算交叉熵作为刻画预测值和真实值之间差距的损失函数。这里使用了TensorFlow中提供的
    # sparse_softmax_cross_entropy_with_logits函数来计算交叉熵。当分类
    # 问题只有一个正确答案时，可以使用这个函数来加速交叉熵的计算。MNIST问题的图片中
    # 参数是神经网络不包括softmax层的前向传播结果，第二个是训练数据的正确答案。因为标准大难是长度为10的一维数组
    # 而该函数需要提供的是正确答案🔢，所以需要使用tf.arg_max来得到正确答案对于的类别编号
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.arg_max(y_, 1))
    # 计算在当前batch中所有样例的交叉熵平均值
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 计算L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
    # 计算模型的正则化损失。一般子计算神经网络边上权重的正则化损失，而不使用偏置项。
    regularaztion = regularizer(weights1) + regularizer(weights2)
    # 总损失等于交叉熵损失和正抓损失的和
    loss = cross_entropy_mean + regularaztion

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,  # 基础的学习率，随着迭代的进行，更新变量时使用学习率在这个基础上递减
        global_step,  # 当前迭代的轮数
        mnist.train.num_examples / BATCH_SIZE,  # 过完所有的训练数据需要的迭代次数
        LEARNING_RATE_DECAY,  # 学习率衰减速度
        staircase=True
    )

    # 优化损失函数
    # 使用 tf.train.GradientDescentOptimizer 优化算法来优化损失函数。注意这里损失函数
    # 包含了交叉熵损失和L2正则化损失
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    #
    # 反向传播更新参数和更新每一个参数的滑动平均值
    with tf.control_dependencies([train_step, variable_averages_op]):
        train_op = tf.no_op(name='train')

    # 计算正确率
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 初始化会话并开始训练过程
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        test_feed = {x: mnist.test.images, y_: mnist.test.labels}
        for i in range(TRAINING_STEPS):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                test_acc = sess.run(accuracy, feed_dict=test_feed)
                print("After %d training step(s), validation accuracy using average model is %g,"
                      "test accuracy using average model is %g " % (i, validate_acc, test_acc))

            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 计算滑动平均模型咋测试数据和验证数据上的正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))


def main(argv=None):
    mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    main()
