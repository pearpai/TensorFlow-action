#!/usr/bin/python
# coding=utf-8
import random

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']


def random_captcha_text(char_set=number, captcha_size=19):
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captcha_text.append(c)
    return captcha_text


# im_text 图片中数据
# im_text_v 图片验证数据
def random_text():
    card_text = random_captcha_text(captcha_size=16)
    num = 0
    im_text = " "
    for char in card_text:
        if num % 4 == 0 and num < 16:
            im_text = im_text + " " + char
        else:
            im_text += char
        num += 1
    im_text += "  "
    im_text_v = "".join(card_text)
    return im_text, im_text_v


if __name__ == '__main__':
    print random_text()
