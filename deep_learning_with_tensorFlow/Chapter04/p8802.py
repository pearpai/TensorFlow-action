import tensorflow as tf

weight = tf.constant([[1.0, -2.0], [-3.0, 4.0]])
with tf.Session() as sess:
    # 输出为(|1| + |-2| + |-3| + |-4|) * 0.5 = 5 其中0.5 为正则化权重
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weight)))
    # 输出为(|1|^2 + |-2|^2 + |-3|^2 + |-4|^2)/2 * 0.5 = 7.5 其中0.5 为正则化权重
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weight)))
