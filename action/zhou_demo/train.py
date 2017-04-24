import tensorflow as tf
import time
import model
import config
import numpy
import gen
import os
import itertools

batch_size = 100


def unzip(b):
    xs, ys = zip(*b)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys


def read_batches(batch_size):
    def gen_vecs():
        for img, code in itertools.islice(gen.generate_ims(), batch_size):
            # print(code,model.code_to_vec(code))
            yield img, model.code_to_vec(code)

    while True:
        yield unzip(gen_vecs())


with tf.name_scope("inputs"):
    # 图像输入
    images = tf.placeholder(tf.float32, [None, None, None], name="images")
    # 标签输入
    labels = tf.placeholder(tf.float32, [None, config.LABEL_SIZE * 10], name="labels")

    # drop out rate
    # keep_prob = tf.placeholder(tf.float32, name="keep_prob")

    is_training = tf.placeholder(tf.bool, name="is_training")

conv1_channels = 32
conv2_channels = 64
conv3_channels = 256

full_conn_neurals = 1024

with tf.name_scope("layer1_cnn"):
    input_expanded = tf.expand_dims(images, 3)
    # input_expanded = tf.reshape(input_expanded, [-1,512,320,1])

    # 第一层 weight
    W_layer1 = tf.Variable(tf.truncated_normal([5, 5, 1, conv1_channels], stddev=0.1), dtype=tf.float32,
                           name="layer1_weight")

    # 记录直方图
    tf.summary.histogram('layer1_weight', W_layer1)

    # 第一层 bias
    B_layer1 = tf.Variable(tf.constant(0.1, tf.float32, [conv1_channels]), dtype=tf.float32,
                           name="layer1_bias")

    # 记录直方图
    tf.summary.histogram('layer1_bias', B_layer1)

    layer1_conv = tf.nn.conv2d(input_expanded, W_layer1, strides=[1, 1, 1, 1],
                               padding="SAME", name="layer1_conv")

    layer1_act = tf.nn.relu(layer1_conv + B_layer1, name="layer1_relu")

    layer1_normal = model.batch_norm(layer1_act, is_training)

    layer1_pooling = tf.nn.max_pool(layer1_normal, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                    name="layer1_pooling")
    # layer1_dropout = tf.nn.dropout(layer1_pooling, keep_prob=keep_prob, name="layer1_dropout")
    # 记录直方图
    # tf.summary.histogram('layer1_output', layer1_dropout)

with tf.name_scope("layer2_cnn"):
    # 第二层 weight
    W_layer2 = tf.Variable(tf.truncated_normal([5, 5, conv1_channels, conv2_channels], stddev=0.1),
                           dtype=tf.float32, name="layer2_weight")
    # 记录直方图
    tf.summary.histogram('layer2_weight', W_layer2)

    # 第二层 bias
    B_layer2 = tf.Variable(tf.constant(0.1, tf.float32, [conv2_channels]), dtype=tf.float32,
                           name="layer2_bias")
    # 记录直方图
    tf.summary.histogram('layer2_bias', B_layer2)

    layer2_conv = tf.nn.conv2d(layer1_pooling, W_layer2, strides=[1, 1, 1, 1],
                               padding="SAME", name="layer2_conv")

    layer2_act = tf.nn.relu(layer2_conv + B_layer2, name="layer2_act")

    layer2_normal = model.batch_norm(layer2_act, is_training)

    layer2_pooling = tf.nn.max_pool(layer2_normal, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                    name="layer2_pooling")

    # layer2_dropout = tf.nn.dropout(layer2_pooling, keep_prob=keep_prob, name="layer2_dropout")

    # tf.summary.histogram('layer2_output', layer2_dropout)

with tf.name_scope("layer3_cnn"):
    # 第二层 weight
    W_layer3 = tf.Variable(tf.truncated_normal([5, 5, conv2_channels, conv3_channels], stddev=0.1),
                           dtype=tf.float32, name="layer3_weight")
    # 记录直方图
    tf.summary.histogram('layer3_weight', W_layer3)

    # 第二层 bias
    B_layer3 = tf.Variable(tf.constant(0.1, tf.float32, [conv3_channels]), dtype=tf.float32,
                           name="layer3_bias")
    # 记录直方图
    tf.summary.histogram('layer3_bias', B_layer3)

    layer3_conv = tf.nn.conv2d(layer2_pooling, W_layer3, strides=[1, 1, 1, 1],
                               padding="SAME", name="layer3_conv")

    layer3_act = tf.nn.relu(layer3_conv + B_layer3, name="layer3_act")

    layer3_normal = model.batch_norm(layer3_act, is_training)

    layer3_pooling = tf.nn.max_pool(layer3_normal, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                    name="layer3_pooling")

with tf.name_scope("layer_reshape"):
    # 经过conv后的输出神经元数
    conv_n_output = config.IMG_SHAPE[0] * config.IMG_SHAPE[1] * conv3_channels // 64
    layer_reshape = tf.reshape(layer3_pooling, [-1, conv_n_output], name="layer1_reshape")

layer3_neurals = 1024

# 全连接层
with tf.name_scope("layer_fullyConnect"):
    # 第三层 weight
    W_full = tf.Variable(tf.truncated_normal([conv_n_output, layer3_neurals], stddev=0.1), dtype=tf.float32,
                         name="weight")
    # 记录直方图
    tf.summary.histogram('weights', W_full)

    # 第三层 bias
    B_full = tf.Variable(tf.constant(0.1, tf.float32, [layer3_neurals]), dtype=tf.float32, name="bias")
    # 记录直方图
    tf.summary.histogram('bias', B_full)

    layer_full = tf.nn.relu(tf.matmul(layer_reshape, W_full) + B_full, name="layer3")

    layer_full_normal = model.batch_norm(layer_full, is_training)

    # layer3_dropout = tf.nn.dropout(layer3, keep_prob=keep_prob, name="layer3_dropout")

with tf.name_scope("layer_output"):
    # 输出层 weight
    W_output = tf.Variable(tf.truncated_normal([conv_n_output, config.LABEL_SIZE * 10], stddev=0.1),
                           dtype=tf.float32, name="output_weight")
    # 记录直方图
    tf.summary.histogram('output_weight', W_output)

    # 输出层 bias
    B_output = tf.Variable(tf.constant(0.1, tf.float32, [config.LABEL_SIZE * 10]), dtype=tf.float32,
                           name="output_bias")
    # 记录直方图
    tf.summary.histogram('output_bias', B_output)

    outputs = tf.matmul(layer_full_normal, W_output) + B_output

    # 记录直方图
    tf.summary.histogram('outputs', outputs)

with tf.name_scope("cost"):
    cost = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=tf.reshape(outputs, [-1, config.LABEL_SIZE]),
                                                                 labels=tf.reshape(labels, [-1, config.LABEL_SIZE])))
    tf.summary.scalar('cross_entropy', cost)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.reshape(outputs, [-1, config.LABEL_SIZE]), 1),
                                               tf.arg_max(tf.reshape(labels, [-1, config.LABEL_SIZE]), 1)), tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    optimizer = tf.train.AdamOptimizer(learning_rate=config.LEARNING_RATE).minimize(cost, global_step=global_step)

test_image, test_label = unzip(list(model.read_test_img("./test/*.png", 100)))

# 合并 summary操作，提供给session
merged = tf.summary.merge_all(key="summaries")

start = time.time()
with  tf.Session() as sess:
    saver = tf.train.Saver()
    train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
    sess.run(tf.global_variables_initializer())

    model.count_params()

    model.count_params_detail()

    model_file = "./models/card_recog.ckpt"

    if os.path.isfile(model_file + ".index"):
        start_time = time.time()
        saver.restore(sess, model_file)
        print("restore model cost:", time.time() - start_time)

    batch_iter = enumerate(read_batches(batch_size))

    num = 0

    for batch_idx, (batch_xs, batch_ys) in batch_iter:
        start_time = time.time()
        #
        _, c, a, summary, step = sess.run([optimizer, cost, accuracy, merged, global_step],
                                          feed_dict={images: batch_xs, labels: batch_ys, is_training: True})

        num += 1
        train_writer.add_summary(summary, step)
        print(num, c, a, time.time() - start_time)

        if num % 10 == 0:
            start_time = time.time()
            saver.save(sess, model_file)
            train_accuracy = accuracy.eval(feed_dict={images: test_image, labels: test_label, is_training: False})
            print("save model cost:", time.time() - start_time, "accuracy", train_accuracy)
