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

v1 = tf.get_variable("v", [1])

print(v1.name)  # v:0 v 变量的名称 0 表示这个变量是生成变量这个运算的第一个结果

with tf.variable_scope("foo", reuse=True):
    v2 = tf.get_variable("v", [1])
    print(v2.name)
    # 输出foo/v:0。 在tf.variable_scope中创建的变量，名称签名会加入命名空间的名称
    # 并通过/来分割命名空间的名称和变量的名称

with tf.variable_scope('foo'):
    with tf.variable_scope('bar'):
        v3 = tf.get_variable('v', [1])
        # 输出foo/bar/v:0
        # 命名空间可以嵌套，同事变量的名称也会加入所有命名空间的名称作为前缀
        print(v3.name)
    v4 = tf.get_variable('v1', [1])
    # 输出foo/v1:0
    # 当命名空间退出后，变量名称也就不会再被加入其前缀了
    print(v4.name)

# 创建一个名称为空的空间，并设置reuse=True
with tf.variable_scope("", reuse=True):
    # 可以直接通过命名空间的名称来获取其他命名空间下的变量。比如这里通过制定名称foo/bar/v
    # 来获取在命名空间foo/bar/中创建的变量
    v5 = tf.get_variable("foo/bar/v", [1])
    # 输出 True
    print(v5 == v3)
    v6 = tf.get_variable("foo/v1", [1])
    # 输出 True
    print(v6 == v4)
