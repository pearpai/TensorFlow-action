import tensorflow as tf

# 直接加载持久化图
saver = tf.train.import_meta_graph('Saved_model/model.ckpt.meta')

'''
参数 初始化
'''
with tf.Session() as sess:
    # 加载已经保存的模型，并通过已经保存的模型中变量的值来计算加法
    saver.restore(sess, "Saved_model/model.ckpt")
    # 通过张量的名称来获取张量
    print(sess.run(tf.get_default_graph().get_tensor_by_name("add:0")))
