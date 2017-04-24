# coding=utf-8
from captcha_image import gen_captcha_text_and_image
from train import convert2gray, crack_captcha


text, image = gen_captcha_text_and_image()
image = convert2gray(image)
image = image.flatten() / 255
predict_text = crack_captcha(image)
print("正确: {}  预测: {}".format(text, predict_text))