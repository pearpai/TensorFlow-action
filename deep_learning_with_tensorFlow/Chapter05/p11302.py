# coding=utf-8
import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name='v3')

# 在没有声明滑动平均模型时只有一个变量v，所以下面的语句只会输出v:0
for variables in tf.global_variables():
    print(variables.name)

ema = tf.train.ExponentialMovingAverage(0.99)
# 加入命名空间中
maintain_averages_op = ema.apply(tf.global_variables())
# 在申明滑动平均模型之后，TensorFlow会自动生成一个影子变量
# v/ExponentialMovingAverage。于是下面的语句输出
# v:0 和 v/ExponentialMovingAverage:0
for variables in tf.global_variables():
    print(variables.name)

saver = tf.train.Saver()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(v, 10))
    sess.run(maintain_averages_op)
    # 保存时候会将v0， v/ExponentialMovingAverage:0 这两个变量保存下来
    saver.save(sess, "Saved_model/model2.ckpt")
    print(sess.run([v, ema.average(v)]))
