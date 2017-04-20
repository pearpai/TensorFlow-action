import tensorflow as tf
# numpy 是个科学计算的工具包，这里通过Numpy生成模拟数据
from numpy.random import RandomState

# 训练数据batch的大小
batch_size = 8

# 定义神经网络的参数，这里还是沿用3.4.2 小结中给出的神经网络结构
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 在shape的维度上使用None可以方便使用不打的batch大小，在训练时需要把数据
# 分成比较小的batch，但是在测试时，可以一次性使用全部数据，当数据集比较小时这样比较
# 方便测试，但是数据集比较大时放入一个batch会导致内存溢出
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# 定义神经网络向前传播的过程 x  w1  w2 两层神经
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数和反向传播的算法
# tf.clip_by_value 因为 log 会产生 none (如 log-3 ), 用它来限定不出现none
# 替代方法 cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv + 1e-10))
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
X = rdm.rand(128, 2)

# Y 为对数据集数据 进行 结果收集分类 和大于1 为1 小于 1为0
# 定义规则来给样本的标签。在这里所有x1 + x2 < 1 的样本都被认为是正样本（比如零件合格）
# 而其他为负样本（比如样本不合格）。和TensorFlow 游乐场中的表示法不大一样的地方是，
# 这里的0表示负样本，1 表示正样本。大部分解决分类问题的神经网络都采用
# 0 和 1 的表示方法
Y = [[int(x1 + x2) < 1] for (x1, x2) in X]

# 创建一个会话运行TensorFlow程序
with tf.Session() as sess:
    # 初始化变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 在训练之前神经网络参数
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))
    print("\n")
    '''
    训练之前神经网络参数的值
    w1: [[-0.81131822  1.48459876  0.06532937]
     [-2.44270396  0.0992484   0.59122431]]
    w2: [[-0.81131822]
     [ 1.48459876]
     [ 0.06532937]]
    '''
    # 设定训练的轮数
    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % 128
        end = (i * batch_size) % 128 + batch_size
        # 通过选取样本训练神经网络并更新参数
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training steps(s), cross entropy on all data is %g" % (i, total_cross_entropy))
            '''
            输出结果
            After 0 training steps(s), cross entropy on all data is 0.0674925
            After 1000 training steps(s), cross entropy on all data is 0.0163385
            After 2000 training steps(s), cross entropy on all data is 0.00907547
            After 3000 training steps(s), cross entropy on all data is 0.00714436
            After 4000 training steps(s), cross entropy on all data is 0.00578471

            通过这个结果可以发现随着训练的进行，交叉熵是逐渐减小的。交叉熵越小说明预测的结果和真实的结果差距越小
            '''

    print("\n")
    print("w1:", sess.run(w1))
    print("w2:", sess.run(w2))
    '''
    w1: [[-1.9618274   2.58235407  1.68203783]
     [-3.4681716   1.06982327  2.11788988]]
    w2: [[-1.8247149 ]
     [ 2.68546653]
     [ 1.41819501]]
     可以发现这两个参数的取值已经发生了编发，这个变化是训练的结果
     它使得这个神经网络能根号的拟合提供的训练数据
    '''

'''
1、定义神经网络的结构和前向传播的输出结果
2、定义损失函数以及选择反向传播的优化算法
3、生成会话（tf.Session）并且在训练数据上反复运行反向传播优化算法
'''
