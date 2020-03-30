import tensorflow

from keras.layers import Input,Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model, load_model

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam
from keras import backend as K

import prepare_data as pd
import numpy as np
# import math
import matplotlib.pyplot as plt

import cv2
import sys

class NN():
    def __init__(self,epochs=200):
        self.epochs = epochs
        
        self.nn_train = self.build_nn()
        self.nn_test = self.build_nn(None,None)
        
    def build_nn(self,img_rows=32,img_cols=32):
        input_img = Input(shape=(img_rows,img_cols,1))
        x = Conv2D(filters=128,kernel_size=(9,9),kernel_initializer="glorot_uniform",
                   activation="relu",padding="valid",use_bias=True#, kernel_regularizer=regularizers.l2(0.001)
                   )(input_img)
        x = Dropout(0.2)(x,training=True)
        noise = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='glorot_uniform',
                   activation='relu', padding='same',use_bias=True#,kernel_regularizer=regularizers.l2(0.001)
                   )(x)
        x = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='glorot_uniform',
                   activation='relu', padding='same',use_bias=True#,kernel_regularizer=regularizers.l2(0.001)
                   )(x)
        x = Dropout(0.2)(x,training=True)
        output_img = Conv2D(filters=1, kernel_size=(5,5), kernel_initializer='glorot_uniform',
                     activation='linear',padding='valid',use_bias=True#,kernel_regularizer=regularizers.l2(0.001)
                     )(x)
        log_noise = Conv2D(filters=1, kernel_size=(5,5), kernel_initializer='glorot_uniform',
             activation='linear',padding='valid',use_bias=True#,kernel_regularizer=regularizers.l2(0.001)
             ,name="noise")(noise)
        
        model_training = Model(input_img, output_img)
        adam = Adam(lr=0.0003)
        model_training.compile(optimizer=adam,loss=self.custom_loss(log_noise)) #,metrics=['mean_squared_error']) #self.custom_loss(log_noise))

        self.model_out = Model(input_img,[output_img,log_noise])

        return model_training

    def custom_loss(self,log_noise):
        def loss(y_true,y_pred):
            out = K.sum(0.5*K.square(y_true-y_pred)*K.exp(-log_noise)+0.5*log_noise)/1024
            return out
        return loss

    def train(self,batch_size=128):
        data, label = pd.read_training_data("./train.h5")
        val_data, val_label = pd.read_training_data("./test.h5")
        
        checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                     save_weights_only=False, mode='min')
        callbacks_list = [checkpoint]

        self.nn_train.fit(data, label, batch_size=batch_size, validation_data=(val_data, val_label),
                        callbacks=callbacks_list, shuffle=True, epochs=self.epochs) #, verbose=0)

        self.nn_train.save_weights("srcnn_dropout.h5")
        
    def test_img(self,img_name="./Test/Set14/flowers.bmp",load_weights=None):
        srcnn_model = self.nn_test
        
        if load_weights:
            srcnn_model.load_weights(load_weights)
        else:
            srcnn_model.set_weights(self.nn_train.get_weights())
            
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

    def test_combined(self,img_name="./Test/Set14/flowers.bmp",load_weights=None):
                
        self.noiseModel = Model(inputs=self.nn_test.input,outputs=self.model_out.get_layer("noise").output)
        
        srcnn_model = self.nn_test
        
        if load_weights:
            srcnn_model.load_weights(load_weights)
        else:
            srcnn_model.set_weights(self.nn_train.get_weights())
            
        IMG_NAME = img_name
        INPUT_NAME = "input.jpg"
        OUTPUT_NAME = "output.jpg"

        img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) ## BGR to YcrCb
        shape = img.shape
        Y_img = cv2.resize(img[:, :, 0], (shape[1] // 2, shape[0] // 2), cv2.INTER_CUBIC)
        Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
        img[:, :, 0] = Y_img
        img_original = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        img = img_original.copy()
        cv2.imwrite(INPUT_NAME, img)
        
        fig,ax = plt.subplots(1,3,figsize=(20,20))
        ax[0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        ax[0].set_title("original")

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
        ax[1].set_title("x2")

        
        T = 30
        var = np.zeros(img.shape)
        Ey = np.zeros(img.shape)
        
        for k in range(T):       
            Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
            Y[0, :, :, 0] = Y_img.astype(float) / 255.
            pre = srcnn_model.predict(Y, batch_size=1) * 255.
            pre[pre[:] > 255] = 255
            pre[pre[:] < 0] = 0
            pre = pre.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
            img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
            
            var += (img/255)**2/T
            Ey += img/(T*255)
            
            Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
            Y[0, :, :, 0] = Y_img.astype(float) / 255.
            noise = srcnn_model.predict(Y, batch_size=1) * 255.
            noise[noise[:] > 255] = 255
            noise[noise[:] < 0] = 0
            noise = pre.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
            img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)
            
            var += np.exp(img/255)/T
            
        var -= Ey**2
        
        cb = ax[2].imshow(var[:,:,0]+var[:,:,1]+var[:,:,2],"jet")
        ax[2].set_title("combined uncertainty")
        fig.colorbar(cb,ax=ax,shrink=0.2,location="right")
        fig.savefig("./dropout")
        plt.show()
                
        fig,ax = plt.subplots(1,3,figsize=(20,20))
        ax[0].imshow(var[:,:,0],"jet")
        ax[0].set_title("r")
        ax[1].imshow(var[:,:,1],"jet")
        ax[1].set_title("g")
        cb = ax[2].imshow(var[:,:,2],"jet")
        ax[2].set_title("b")
        
        fig.colorbar(cb,ax=ax,shrink=0.2,location="right")
        
        fig.savefig("./var-rgb")
        plt.show()
        
#         cv2.imwrite(OUTPUT_NAME, img)

    def test_aleatoric(self,img_name="./Test/Set14/flowers.bmp",load_weights=None):
        srcnn_model = self.nn_test
        
        if load_weights:
            srcnn_model.load_weights(load_weights)
        else:
            srcnn_model.set_weights(self.nn_train.get_weights())
            
        IMG_NAME = img_name
        INPUT_NAME = "input.jpg"
        OUTPUT_NAME = "output.jpg"

        img = cv2.imread(IMG_NAME, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb) ## BGR to YcrCb
        shape = img.shape
        Y_img = cv2.resize(img[:, :, 0], (shape[1] // 2, shape[0] // 2), cv2.INTER_CUBIC)
        Y_img = cv2.resize(Y_img, (shape[1], shape[0]), cv2.INTER_CUBIC)
        img[:, :, 0] = Y_img
        img_original = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
        img = img_original.copy()
        cv2.imwrite(INPUT_NAME, img)
        
        fig,ax = plt.subplots(1,3,figsize=(20,20))
        ax[0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
        ax[0].set_title("original")
        
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
        ax[1].set_title("x2")
        
        self.noiseModel = Model(inputs=self.nn_test.input,outputs=self.model_out.get_layer("noise").output)
        
        Y = np.zeros((1, img.shape[0], img.shape[1], 1), dtype=float)
        Y[0, :, :, 0] = Y_img.astype(float) / 255.
        pre = self.noiseModel.predict(Y, batch_size=1) * 255.
        pre[pre[:] > 255] = 255
        pre[pre[:] < 0] = 0
        pre = pre.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        img[6: -6, 6: -6, 0] = pre[0, :, :, 0]
        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2RGB)/255.
        
        cb = ax[2].imshow(img[:,:,0]+img[:,:,1]+img[:,:,2],"jet")
        ax[2].set_title("aleatoric uncertainty")
        fig.colorbar(cb,ax=ax,shrink=0.2,location="right")
        fig.savefig("./aleatoric")
        plt.show()
        
        fig,ax = plt.subplots(1,3,figsize=(20,20))
        ax[0].imshow(img[:,:,0],"jet")
        ax[0].set_title("r")
        ax[1].imshow(img[:,:,1],"jet")
        ax[1].set_title("g")
        cb = ax[2].imshow(img[:,:,2],"jet")
        ax[2].set_title("b")
        
        fig.colorbar(cb,ax=ax,shrink=0.2,location="right")
        
        fig.savefig("./aleatoric-rgb")
        plt.show()        


if __name__ == "__main__":
    
    model = NN()

    if(len(sys.argv)==1):
        model.train()
        model.test_img()
        model.test()
    else:
        model.test_img(load_weights=True)
        model.test(load_weights=True)

