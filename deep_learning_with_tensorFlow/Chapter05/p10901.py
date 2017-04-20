import tensorflow as tf

with tf.variable_scope("root"):
    # 可以通过tf.get_variable_scope().reuse函数来获取当前上下文管理器中reuse参数的取值

    # 输出False，即最外层reuse是False
    print(tf.get_variable_scope().reuse)
    # 新建一个嵌套的上下文管理，并指定reuse为True
    with tf.variable_scope("foo", reuse=True):
        # 输出为True
        print(tf.get_variable_scope().reuse)
        # 新建一个嵌套的上下文管理器不指定reuse的取值会和外面一层的保持一致 输出True
        with tf.variable_scope("bar"):
            print(tf.get_variable_scope().reuse)
    # 外层的还是不变
    print(tf.get_variable_scope().reuse)
