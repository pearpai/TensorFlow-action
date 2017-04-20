import tensorflow as tf

with tf.variable_scope("foo"):
    v = tf.get_variable('v', [1], initializer=tf.constant_initializer(1.0))

'''
因为在命名空间foo中已经存在名字为v的变量，所以下面的代码将会报错
ValueError: Variable foo/v already exists, disallowed. Did you mean to set reuse=True in VarScope?
'''
# with tf.variable_scope("foo"):
#     v = tf.get_variable("v", [1])


'''
在生成上下文管理器时，将参赛reuse设置为True。这样tf.get_variable函数将直接获取
已经声明的变量
'''
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])

print(v)
print(v == v1)

'''
将参数reuse设置为TRUE时，tf.variable_scope将只能获取已经创建过的变量。因为在命名空间bar中没有创建变量v
所以下面的相会报错
ValueError: Variable bar/v does not exist, or was not created with tf.get_variable().
Did you mean to set reuse=None in VarScope?
'''
# with tf.variable_scope("bar", reuse=True):
#     v = tf.get_variable('v', [1])
