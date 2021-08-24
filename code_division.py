
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
        if len(image.shape) > 2:
            res.append(image[target_h_lower:target_h_upper, target_w_left:target_w_right, :])
        else:
            res.append(image[target_h_lower:target_h_upper, target_w_left:target_w_right])
    return res

from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import cv2

code_path = "codes/"
save_path = "codes/processed_codes/"

processed_count = 0
labels = {}
for f in listdir(code_path):
    if isfile(join(code_path, f)):
        code_file_path = join(code_path, f)

        nums = []
        nums.append(f[0])
        nums.append(f[1])
        nums.append(f[2])
        nums.append(f[3])

        img = cv2.imread(code_file_path)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        sub_imgs = cut_image(~thresh)
        max_d0 = 14
        max_d1 = 10
        for i in range(4):
            pad_len = max_d1 - sub_imgs[i].shape[1]
            if pad_len > 0:
                result = np.zeros([max_d0, max_d1])
                result[:sub_imgs[i].shape[0], :sub_imgs[i].shape[1]] = sub_imgs[i]
                sub_imgs[i] = result
            # print(sub_imgs[i].shape[0])
            # print(sub_imgs[i].shape[1])
            cv2.imwrite(save_path + str(processed_count) + '.jpg', sub_imgs[i])
            labels[str(processed_count)] = nums[i]
            processed_count += 1
    

data = {
    'index': list(labels.keys()),
    'number': list(labels.values())
}
df = pd.DataFrame(data)
df.to_csv(save_path + 'code_labels.csv', index=False)


            




