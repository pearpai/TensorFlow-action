import tensorflow as tf

# 可以读取checkpoint文件中保存的所有变量
reader = tf.train.NewCheckpointReader("Saved_model/model.ckpt")

# 获取所有变量列表。 这个是一个冲变量名到变量维度的字典
all_variables = reader.get_variable_to_shape_map()

for variable_name in all_variables:
    # variable_name为变量名称，all_variables[variable_name]为变量维度
    print(variable_name, all_variables[variable_name])

print("Value for variable v1 is ", reader.get_tensor('v1'))
print("Value for variable v2 is ", reader.get_tensor('v2'))
'''
v1 [1]
v2 [1]
Value for variable v1 is  [ 1.]
Value for variable v2 is  [ 2.]
'''
