# coding=utf-8
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image

path = "../datasets/demo_picture/cat.jpg"
img = Image.open(path)
# 获取长宽
print img.size

image_raw_data = tf.gfile.FastGFile("../datasets/demo_picture/cat.jpg", 'r').read()

img_data = tf.image.decode_jpeg(image_raw_data)

with tf.Session() as sess:
    croped = tf.image.resize_image_with_crop_or_pad(img_data, 1000, 1000)
    padded = tf.image.resize_image_with_crop_or_pad(img_data, 3000, 3000)
    print 'croped ', croped.get_shape()
    print 'padded ', croped.get_shape()

    # TensorFlow的函数处理图片后存储的数据是float32格式的，需要转换成uint8才能正确打印图片。

    # plt.imshow(croped.eval())
    # plt.show()
    # plt.imshow(padded.eval())
    # plt.show()
    # central_cropped = tf.image.central_crop(img_data, 0.5)
    # plt.imshow(central_cropped.eval())
    # plt.show()
    jt = tf.image.crop_to_bounding_box(img_data, 1500, 0, 200, 2673)
    print 'jt ', jt.get_shape()
    plt.imshow(jt.eval())
    plt.show()
