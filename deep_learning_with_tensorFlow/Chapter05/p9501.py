from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../datasets/MNIST_data/", one_hot=True)

# 训练数据分为两部分 Training Validating
print("Training data size: ", mnist.train.num_examples)
print("Validating data size: ", mnist.validation.num_examples)
# 测试数据
print("Testing data size: ", mnist.test.num_examples)

print("Example training data: ", mnist.train.images[0])
print("Example training data label: ", mnist.train.labels[0])

# 获取image 单条数据的长度
print(len(mnist.train.images[0]))

batch_size = 100
xs, ys = mnist.train.next_batch(batch_size)
# 从train的集合中选取batch_size个数据
# 训练数据集 X shape: (100, 784)
print("X shape:", xs.shape)
# 训练数据结果集 Y shape: (100, 10)
print("Y shape:", ys.shape)
