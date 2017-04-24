# -*- coding: utf-8 -*-

import os
import StringIO
from PIL import Image, ImageFont, ImageDraw
import pygame
import random


def demo1():
    pygame.init()

    text = '  6231 6260 3100 3992  '

    bgcolor = (int(random.uniform(0, 255)), int(random.uniform(0, 255)), int(random.uniform(0, 255)))

    card_no_color = (int(random.uniform(0, 255)), int(random.uniform(0, 255)), int(random.uniform(0, 255)))

    im = Image.new("RGB", (400, 50), bgcolor)
    # dr = ImageDraw.Draw(im)
    # font = ImageFont.truetype(os.path.join("fonts", "simsun.ttc"), 18)
    font = pygame.font.SysFont('Microsoft YaHei', 50)
    # font = pygame.font.SysFont('Farrington-7B-Qiqi', 50)

    # font = ImageFont.truetype("font/Farrington-7B-Qiqi.ttf", 50)

    # dr.text((10, 5), text, font=font, fill="#000000")
    rtext = font.render(text, True, card_no_color, bgcolor)

    # pygame.image.save(rtext, "t.gif")
    sio = StringIO.StringIO()
    pygame.image.save(rtext, sio)
    sio.seek(0)

    line = Image.open(sio)
    im.paste(line, (10, 10))

    img_d = ImageDraw.Draw(im)
    x_len, y_len = im.size
    print im.size
    for _ in range(15):
        noise_color = (int(random.uniform(0, 255)), int(random.uniform(0, 255)), int(random.uniform(0, 255)))
        img_d.line(((random.uniform(1, x_len), random.uniform(1, y_len)),
                    (random.uniform(1, x_len), random.uniform(1, y_len))), noise_color)

    # im.show()
    im.save("t.jpg")


def demo2():
    # 打开图像
    img = Image.open('t.jpg')
    img_d = ImageDraw.Draw(img)
    # 获取 图片的 x轴，y轴 像素
    x_len, y_len = img.size
    for _ in range(15):
        noise_color = (int(random.uniform(0, 255)), int(random.uniform(0, 255)), int(random.uniform(0, 255)))
        img_d.line(((random.uniform(1, x_len), random.uniform(1, y_len)),
                    (random.uniform(1, x_len), random.uniform(1, y_len))), noise_color)
    # 保存图片
    img.save('ii.jpg')


if __name__ == '__main__':
    demo1()
    # demo2()
