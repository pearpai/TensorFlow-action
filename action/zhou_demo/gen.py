import os
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import numpy
import itertools
import random
import cv2
import math
import config
import tensorflow as tf

version = "ABCD"

CHARS_W_BLANK = config.CHARS + " "

BG_PATH = "/python/chinese_rec/bgs"

FONT_HEIGHT = 32


# 创建字体文件的图像集，高度32pix
def make_char_ims(font_path, output_height):
    font_size = output_height * 4

    # 加载字体文件，创建字体对象
    font = ImageFont.truetype(font_path, font_size)

    # 所有字符的最大高度
    height = max(font.getsize(c)[1] for c in CHARS_W_BLANK)

    for c in CHARS_W_BLANK:
        # 字符宽度
        width = font.getsize(c)[0]

        # 黑色底图
        im = Image.new("RGBA", (width, height), (255, 255, 255, 0))

        draw = ImageDraw.Draw(im)
        # 白色文字
        draw.text((0, 0), c, (255, 255, 255), font=font)

        # 缩放比例
        scale = float(output_height) / height
        # 比例缩放，输出高度为32pix
        im = im.resize((int(width * scale), output_height), Image.ANTIALIAS)
        # numpy.array(im)返回一个 width * height * 4 的3D数组
        # [:,:,0]返回一个width * height的2D数组,值为 RGBA 层的 第1列
        yield c, numpy.array(im)[:, :, 0].astype(numpy.float32) / 255.


# 加载字体文件
def load_fonts(folder_path):
    font_char_ims = {}
    fonts = [f for f in os.listdir(folder_path) if f.endswith('.ttf')]
    for font in fonts:
        font_char_ims[font] = dict(make_char_ims(os.path.join(folder_path,
                                                              font),
                                                 FONT_HEIGHT))
    return fonts, font_char_ims


# 生成测试图片背景
def generate_bg(num_bg_images):
    fname = BG_PATH + "/{:08d}.jpg".format(random.randint(0, num_bg_images - 1))
    bg = cv2.imread(fname, cv2.IMREAD_GRAYSCALE) / 255.
    bg = cv2.resize(bg, config.IMG_SHAPE)
    return bg


def rounded_rect(shape, radius):
    out = numpy.ones(shape)
    out[:radius, :radius] = 0.0
    out[-radius:, :radius] = 0.0
    out[:radius, -radius:] = 0.0
    out[-radius:, -radius:] = 0.0

    cv2.circle(out, (radius, radius), radius, 1.0, -1)
    cv2.circle(out, (radius, shape[0] - radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, radius), radius, 1.0, -1)
    cv2.circle(out, (shape[1] - radius, shape[0] - radius), radius, 1.0, -1)

    return out


def generate_code():
    v = random.choice(version)

    # 16位，四位分隔
    if v == "A":
        return "{}{}{}{} {}{}{}{} {}{}{}{} {}{}{}{}".format(
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS))

    # 19位连续
    if v == "B":
        return "{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}{}".format(
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS))

    # 19位，前6，后13
    if v == "C":
        return "{}{}{}{}{}{} {}{}{}{}{}{}{}{}{}{}{}{}{}".format(
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS),
            random.choice(config.CHARS))

    # 默认，19位，4位分隔，最后三位
    return "{}{}{}{} {}{}{}{} {}{}{}{} {}{}{}{} {}{}{}".format(
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS),
        random.choice(config.CHARS))


def pick_colors():
    first = True
    while first or plate_color - text_color < 0.3:
        text_color = random.random()
        plate_color = random.random()
        if text_color > plate_color:
            text_color, plate_color = plate_color, text_color
        first = False
    return text_color, plate_color


def euler_to_mat(yaw, pitch, roll):
    # Rotate clockwise about the Y-axis
    c, s = math.cos(yaw), math.sin(yaw)
    M = numpy.matrix([[c, 0., s],
                      [0., 1., 0.],
                      [-s, 0., c]])

    # Rotate clockwise about the X-axis
    c, s = math.cos(pitch), math.sin(pitch)
    M = numpy.matrix([[1., 0., 0.],
                      [0., c, -s],
                      [0., s, c]]) * M

    # Rotate clockwise about the Z-axis
    c, s = math.cos(roll), math.sin(roll)
    M = numpy.matrix([[c, -s, 0.],
                      [s, c, 0.],
                      [0., 0., 1.]]) * M

    return M


# 生成车牌
def generate_plate(font_height, char_ims):
    # 水平偏移
    h_padding = random.uniform(0.2, 0.4) * font_height
    # 垂直偏移
    v_padding = random.uniform(0.1, 0.3) * font_height
    spacing = font_height * random.uniform(0.05, 0.1)
    radius = 1 + int(font_height * 0.1 * random.random())

    code = generate_code()[0:5]
    text_width = sum(char_ims[c].shape[1] for c in code)
    text_width += (len(code) - 1) * spacing

    out_shape = (int(font_height + v_padding * 2),
                 int(text_width + h_padding * 2))

    text_color, plate_color = pick_colors()

    text_mask = numpy.zeros(out_shape)

    x = h_padding
    y = v_padding
    for c in code:
        char_im = char_ims[c]
        ix, iy = int(x), int(y)
        text_mask[iy:iy + char_im.shape[0], ix:ix + char_im.shape[1]] = char_im
        x += char_im.shape[1] + spacing

    plate = (numpy.ones(out_shape) * plate_color * (1. - text_mask) +
             numpy.ones(out_shape) * text_color * text_mask)

    return plate, rounded_rect(out_shape, radius), code.replace(" ", "")


# 生成测试图片
def generate_im(char_ims, num_bg_images):
    # 生成背景图
    bg = generate_bg(num_bg_images)

    plate, plate_mask, code = generate_plate(FONT_HEIGHT, char_ims)

    x = numpy.random.randint(0, bg.shape[0] - plate.shape[0])
    y = numpy.random.randint(0, bg.shape[1] - plate.shape[1])

    bg[x:plate.shape[0] + x, y:plate.shape[1] + y] = plate

    return bg, code


# 创建测试图片（生成器）
def generate_ims():
    """
    Generate number plate images.
    :return:
        Iterable of number plate images.
    """
    variation = 1.0
    fonts, font_char_ims = load_fonts("./fonts")

    num_bg_images = len(os.listdir(BG_PATH))
    # 当islice到达stop位时，生成器不再收到next()调用，将会一直等待直到被垃圾回收
    while True:
        # print(font_char_ims[num])
        # 根据随机字体文件（现在只有一个）的图片集，生成测试图片
        yield generate_im(font_char_ims[random.choice(fonts)], num_bg_images)


if __name__ == "__main__":
    if not os.path.isdir("./test"):
        os.mkdir("test")

    im_gen = itertools.islice(generate_ims(), 100)

    for img_idx, (im, c) in enumerate(im_gen):
        fname = "test/{:08d}_{}.png".format(img_idx, c)
        print(fname)
        cv2.imwrite(fname, im * 255.)
