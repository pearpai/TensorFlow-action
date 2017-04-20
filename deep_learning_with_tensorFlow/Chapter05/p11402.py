import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name='v3')
ema = tf.train.ExponentialMovingAverage(0.99)
print(ema.variables_to_restore())

# saver = tf.train.Saver()
# saver = tf.train.Saver({"v/ExponentialMovingAverage": v})
# v3/ExponentialMovingAverage
saver = tf.train.Saver(ema.variables_to_restore())

with tf.Session() as sess:
    saver.restore(sess, "Saved_model/model2.ckpt")
    for variables in tf.global_variables():
        print(variables.name)
    print(sess.run(v))
