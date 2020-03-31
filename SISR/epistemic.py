import tensorflow as tf

from keras.layers import Input,Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model, load_model

from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam

import prepare_data as pd
import numpy as np
import matplotlib.pyplot as plt

import cv2
import sys

from utils import *

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
        x = Conv2D(filters=64, kernel_size=(3,3), kernel_initializer='glorot_uniform',
                   activation='relu', padding='same',use_bias=True#,kernel_regularizer=regularizers.l2(0.001)
                   )(x)
        x = Dropout(0.2)(x,training=True)
        output_img = Conv2D(filters=1, kernel_size=(5,5), kernel_initializer='glorot_uniform',
                     activation='linear',padding='valid',use_bias=True#,kernel_regularizer=regularizers.l2(0.001)
                     )(x)
        
        model_training = Model(input_img, output_img)
        adam = Adam(lr=0.0003)
        model_training.compile(optimizer=adam,loss="mean_squared_error",metrics=['mean_squared_error']) #self.custom_loss(log_noise))

        return model_training

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

        img,Y_img = subsample(img_name)

        fig,ax = plt.subplots(1,2,figsize=(20,20))

        ax[0].imshow(img)
        ax[0].set_title("original")

        img_x2 = predict(img,Y_img,srcnn_model)

        ax[1].imshow(img_x2)
        ax[1].set_title("x2")
        plt.show()

    def test_epistemic(self,img_name="./Test/Set14/flowers.bmp",load_weights=None):
        srcnn_model = self.nn_test
        
        if load_weights:
            srcnn_model.load_weights(load_weights)
        else:
            srcnn_model.set_weights(self.nn_train.get_weights())
            
        img,Y_img = subsample(img_name)

        fig,ax = plt.subplots(1,3,figsize=(20,20))

        ax[0].imshow(img)
        ax[0].set_title("original")

        img_x2 = predict(img,Y_img,srcnn_model)

        ax[1].imshow(img_x2)
        ax[1].set_title("x2")
        
        T = 30
        var = np.zeros(img.shape)
        Ey = np.zeros(img.shape)
        
        for k in range(T):            
            img_pred = predict(img,Y_img,srcnn_model)
            var += (img_pred/255)**2/T
            Ey += img_pred/(T*255)
            
        var -= Ey**2
        
        cb = ax[2].imshow(var[:,:,0]+var[:,:,1]+var[:,:,2],"jet")
        ax[2].set_title("epistemic uncertainty")
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



if __name__ == "__main__":
    
    model = NN(400)

    if(len(sys.argv)==1):
        model.train()
        model.test_img()
        model.test_epistemic()
    else:
        model.test_img(load_weights=True)
        model.test_epistemic(load_weights=True)

