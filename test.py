import tensorflow as tf
import numpy as np
import os
from os import listdir
from os.path import isfile, join
import cv2


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(14, 10)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

# 恢复权重
model.load_weights('saved_checkpoint')


def cut_image(image):
    h = image.shape[0]
    w = image.shape[1]
    target_w = [3, 13, 21, 31, 40]
    target_h = [5, 19]
    res = []
    for wi in range(len(target_w) - 1):
        target_w_left = target_w[wi]
        target_w_right = target_w[wi + 1]
        target_h_lower = target_h[0]
        target_h_upper = target_h[1]
        # padding or reshape
        if len(image.shape) > 2:
            res.append(image[target_h_lower:target_h_upper, target_w_left:target_w_right, :])
        else:
            res.append(image[target_h_lower:target_h_upper, target_w_left:target_w_right])
    return res

code_path = "codes/"

# whole test
# true_count = 0.0
# false_count = 0.0
# for f in listdir(code_path):
#     if isfile(join(code_path, f)):
#         code_file_path = join(code_path, f)

#         nums = []
#         nums.append(int(f[0]))
#         nums.append(int(f[1]))
#         nums.append(int(f[2]))
#         nums.append(int(f[3]))

#         img = cv2.imread(code_file_path)
#         gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#         ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

#         sub_imgs = cut_image(~thresh)
#         max_d0 = 14
#         max_d1 = 10
#         true_i = 0
#         for i in range(4):
#             pad_len = max_d1 - sub_imgs[i].shape[1]
#             if pad_len > 0:
#                 result = np.zeros([max_d0, max_d1])
#                 result[:sub_imgs[i].shape[0], :sub_imgs[i].shape[1]] = sub_imgs[i]
#                 sub_imgs[i] = result

#             predict = model.predict(np.array([sub_imgs[i]]))
#             # print(predict)
#             if predict[0][nums[i]] == 1:
#                 true_i += 1

#         if true_i >= 4:
#             true_count += 1

# print("accuarcy: %.0f%%" % (100 * true_count / (true_count + false_count)))



def recognize(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    max_d0 = 14
    max_d1 = 10

    sub_imgs = cut_image(~thresh)
    for i in range(4):
        pad_len = max_d1 - sub_imgs[i].shape[1]
        if pad_len > 0:
            result = np.zeros([max_d0, max_d1])
            result[:sub_imgs[i].shape[0], :sub_imgs[i].shape[1]] = sub_imgs[i]
            sub_imgs[i] = result

    predict = model.predict(np.array(sub_imgs))
    
    res = ''
    for i in range(4):
        v = np.where(predict[i] == 1)
        res += str(v[0][0])
    
    return res

# img = cv2.imread(code_path + '0106.jpg')
# print(recognize(img))




import random
import urllib
import numpy as np
REQUEST_PREFIX = "https://agrygl.gdcic.net/v1/api/open/getvalidcode?key=examsigning&v="
random_v = str(random.random())
url = REQUEST_PREFIX + random_v



req = urllib.request.urlopen(url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
img = cv2.imdecode(arr, -1) # 'Load it as it is'

cv2.imshow('image', img)
print(recognize(img))
cv2.waitKey(0)
