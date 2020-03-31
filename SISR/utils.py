import cv2
import numpy as np

import tensorflow

from keras.layers import Input,Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model, load_model

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras import backend as K

def subsample(IMG_NAME):
    img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) ## BGR to YcrCb
    shape = img.shape
    Y_img = cv2.resize(img[:, :, 0], (shape[1] // 2, shape[0] // 2), cv2.INTER_CUBIC)
    Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
    img[:, :, 0] = Y_img
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
    return img,Y_img

def predict(img,Y_img,srcnn_model):
    Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
    Y[0, :, :, 0] = Y_img.astype(float) / 255.
    pre = srcnn_model.predict(Y, batch_size=1) * 255.
    pre[pre[:] > 255] = 255
    pre[pre[:] < 0] = 0
    pre = pre.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
    img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)

    return img