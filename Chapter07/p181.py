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
    img_data = tf.image.resize_images(img_data, [180, 267], method=0)
    # img_data = tf.image.resize_images(img_data, [300, 300], method=0)
    img_data = np.asarray(img_data.eval(), dtype='uint8')
    plt.imshow(img_data)
    plt.show()

    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
    boxes = tf.constant([[[0.05, 0.05, 0.0, 0.7], [0.35, 0.47, 0.5, 0.56]]])
    result = tf.image.draw_bounding_boxes(batched, boxes)
    img_data = result[0]
    begin, size, bbox_for_draw = tf.image.sample_distorted_bounding_box(
        tf.shape(img_data), bounding_boxes=boxes)
    batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
    image_with_box = tf.image.draw_bounding_boxes(batched, bbox_for_draw)

    distorted_image = tf.slice(img_data, begin, size)

    plt.imshow(distorted_image.eval())
    plt.show()
