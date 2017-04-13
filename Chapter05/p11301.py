import tensorflow as tf

# 这里声明的变量名称和已经保存的模型中名称不同
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='other-v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='other-v2')
result = v1 + v2
'''
如果使用tf.train.Saver()来加载模型会报变量找不到的错误。
tensorflow.python.framework.errors_impl.NotFoundError: Key other-v1 not found in checkpoint
'''
# saver = tf.train.Saver()

# 声明tf.train.Saver类用于保存模型
'''
使用一个字典(dictionary)来重命名变量就可以加载原来的模型了。这个字典制定了原来名称为
v1的变量现在加载到变量v1中（名称为other-v1）v2 同理
'''
saver = tf.train.Saver({"v1": v1, "v2": v2})

'''
参数 初始化
'''
with tf.Session() as sess:
    # 加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法
    saver.restore(sess, "Saved_model/model.ckpt")
    print(sess.run(result))
