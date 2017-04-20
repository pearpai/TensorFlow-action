# coding=utf-8
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

image_raw_data = tf.gfile.FastGFile("../datasets/demo_picture/cat.jpg", 'r').read()
img_data = tf.image.decode_jpeg(image_raw_data)

with tf.Session() as sess:
    resized = tf.image.resize_images(img_data, [300, 300], method=0)

    print resized.get_shape()
    print img_data.get_shape()
    img_data.set_shape([300, 300, 3])
    print img_data.get_shape()
    # TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片。
    print "Digital type: ", resized.dtype
    cat = np.asarray(resized.eval(), dtype='uint8')
    # tf.image.convert_image_dtype(rgb_image, tf.float32)
    plt.imshow(cat)
    plt.show()
