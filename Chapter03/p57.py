import tensorflow as tf

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1))

tf.assign(w1, w2, validate_shape=False)
# w1.assign(w2)
