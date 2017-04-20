import tensorflow as tf

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='b')
result = a + b

w = tf.random_normal([2, 3], stddev=2)

print(result)
with tf.Session() as sess:
    print(sess.run(result))

sess = tf.Session()
with sess.as_default():
    print(result.eval())
    print(w.eval())
