import tensorflow as tf
import numpy as np
import cv2

checkpoint_path = "saved_model/saved_checkpoint"

class ID_MODEL(object):
    def __init__(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(14, 10)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        # 恢复权重
        self.model.load_weights(checkpoint_path)

    def cut_image(self, image):
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

    # @img: opencv.Image
    def recognize(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        max_d0 = 14
        max_d1 = 10

        sub_imgs = self.cut_image(~thresh)
        for i in range(4):
            pad_len = max_d1 - sub_imgs[i].shape[1]
            if pad_len > 0:
                result = np.zeros([max_d0, max_d1])
                result[:sub_imgs[i].shape[0], :sub_imgs[i].shape[1]] = sub_imgs[i]
                sub_imgs[i] = result

        predict = self.model.predict(np.array(sub_imgs))
        
        res = ''
        for i in range(4):
            v = np.where(predict[i] == 1)
            res += str(v[0][0])
        return res