from keras.layers import Input,Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model, load_model

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam

import prepare_data as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2


class NN():
    def __init__(self,epochs=200):
        self.epochs = epochs
        
        self.nn_train = self.build_nn()
        self.nn_test = self.build_nn(None,None)
        
    def build_nn(self,img_rows=32,img_cols=32):
        input_img = Input(shape=(img_rows,img_cols,1))
        x = Conv2D(filters=128,kernel_size=(9,9),kernel_initializer="glorot_uniform",
                   activation="relu",padding="valid",use_bias=True)(input_img)
        x = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='glorot_uniform',
                   activation='relu', padding='same', use_bias=True)(x)
        output_img = Conv2D(filters=1, kernel_size=(5,5), kernel_initializer='glorot_uniform',
                     activation='linear',padding='valid', use_bias=True)(x)
        
        model_training = Model(input_img, output_img)
        adam = Adam(lr=0.0003)
        model_training.compile(optimizer=adam,loss="mean_squared_error",metrics=['mean_squared_error'])

        return model_training

    def train(self,batch_size=128):
        data, label = pd.read_training_data("./train.h5")
        val_data, val_label = pd.read_training_data("./test.h5")
        
        checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='min')
        callbacks_list = [checkpoint]

        self.nn_train.fit(data, label, batch_size=batch_size, validation_data=(val_data, val_label),
                        callbacks=callbacks_list, shuffle=True, epochs=self.epochs) #, verbose=0)

        self.nn_train.save_weights("srcnn.h5")
        
    def test_img(self,img_name="./Test/Set14/flowers.bmp",load_weights=False):
        srcnn_model = self.nn_test
        
        if load_weights:
            srcnn_model.load_weights("srcnn_200.h5")
            
        IMG_NAME = img_name
        INPUT_NAME = "input.jpg"
        OUTPUT_NAME = "output.jpg"

        img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) ## BGR to YcrCb
        shape = img.shape
        Y_img = cv2.resize(img[:, :, 0], (shape[1] // 2, shape[0] // 2), cv2.INTER_CUBIC)
        Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
        img[:, :, 0] = Y_img
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        cv2.imwrite(INPUT_NAME, img)
        
        fig,ax = plt.subplots(1,2,figsize=(20,20))

        ax[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
        Y[0, :, :, 0] = Y_img.astype(float) / 255.
        pre = srcnn_model.predict(Y, batch_size=1) * 255.
        pre[pre[:] > 255] = 255
        pre[pre[:] < 0] = 0
        pre = pre.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)

        ax[1].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.show()

        cv2.imwrite(OUTPUT_NAME, img)

#         # psnr calculation:
#         im1 = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
#         im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
#         im2 = cv2.imread(INPUT_NAME, cv2.IMREAD_COLOR)
#         im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]
#         im3 = cv2.imread(OUTPUT_NAME, cv2.IMREAD_COLOR)
#         im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2YCrCb)[6: -6, 6: -6, 0]


#         print("bicubic:")
#         print(cv2.PSNR(im1, im2))
#         print("SRCNN:")
#         print(cv2.PSNR(im1, im3))






if __name__ == "__main__":
    
    model = NN(1)
    model.train()
    model.test_img()
    model.test_img(load_weights=True)
