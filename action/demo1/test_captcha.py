import glob

import do_captcha

PATH = "/Users/wuyunfeng/Documents/machine_learn/TensorFlow-action/data/captcha/ZZIY-T201704209584200219.jpeg"

path = "/Users/wuyunfeng/Documents/machine_learn/TensorFlow-action/data/captcha/*.jpeg"

# do_captcha.predict_single(PATH)


# print glob.glob(path)
do_captcha.predict(glob.glob(path))
