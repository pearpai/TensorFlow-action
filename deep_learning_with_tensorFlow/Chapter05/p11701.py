import tensorflow as tf

# 声明两个变量并计算他们的和
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name='v1')
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name='v2')
result = v1 + v2

# 声明tf.train.Saver类用于保存模型
saver = tf.train.Saver()

saver.export_meta_graph("Saved_model/model.ckpt.meta.json", as_text=True)
