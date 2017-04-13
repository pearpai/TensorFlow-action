import tensorflow as tf

# 定义一个变量用于计算滑动平均，这个变量的初始值为0.注意这里手动指定变量的类型
# 为float32，因为所有需要计算滑动平均的变量必须是实数型
v1 = tf.Variable(0, dtype=tf.float32)
# 这里的step变量模拟神经网络中迭代的轮数，可以用于动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均类（class）。初始化时给定了衰减率（0.99）和控制衰减的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作。这里需要给定一个列表，每次执行这个操作时
# 这个列表中的变量都会被更新
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run([v1, ema.average(v1)]))

    # 更新变量v1的取值
    sess.run(tf.assign(v1, 5))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 更新step和v1的取值
    sess.run(tf.assign(step, 10000))
    sess.run(tf.assign(v1, 10))
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 更新一次v1的滑动平均值
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
