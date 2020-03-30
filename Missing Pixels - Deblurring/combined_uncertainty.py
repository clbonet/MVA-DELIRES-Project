import pdb
import matplotlib.pyplot as plt
import numpy as np
import os

from keras.datasets import mnist,cifar10
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, BatchNormalization, Activation, ZeroPadding2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
from keras.regularizers import l2
from keras import backend as K

import sys

class autoencoder():
    def __init__(self,dataset_name='mnist',architecture='mlp'):
        
        X_train,_ = self.load_data(dataset_name)

        # image parameters
        self.epochs = 100001
        self.error_list = np.zeros((self.epochs,1))
        self.img_rows = X_train.shape[1]
        self.img_cols = X_train.shape[2]
        self.img_channels = X_train.shape[3]
        self.img_size = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
        self.img_shape = (self.img_rows, self.img_cols, self.img_channels)
        self.z_dim = 12
        self.architecture = architecture
        self.dataset_name = dataset_name

        # Build and compile the autoencoder
        self.ae = self.build_ae()
        self.ae.summary()

    def build_ae(self):

        n_pixels = self.img_rows*self.img_cols*self.img_channels

        if (self.architecture == 'mlp'):
            # FULLY CONNECTED (MLP)

            #BEGIN INSERT CODE
            #encoder
            input_img = Input(shape=(self.img_rows,self.img_cols,self.img_channels))
            x_flatten = Flatten()(input_img)
            x_flatten = Dense(512,activation=LeakyReLU(0.2))(x_flatten)
            x_flatten = Dropout(0.2)(x_flatten,training=True)
            x_flatten = Dense(256,activation=LeakyReLU(0.2))(x_flatten)
            x_flatten = Dropout(0.2)(x_flatten,training=True)
            z = Dense(self.z_dim,activation=LeakyReLU(0.2))(x_flatten)
            #decoder
            z = Dense(256,activation=LeakyReLU(0.2))(z)
            z = Dropout(0.2)(z,training=True)
            noise = Dense(512,activation=LeakyReLU(0.2))(z)
            z = Dense(512,activation=LeakyReLU(0.2))(z)
            z = Dropout(0.2)(z,training=True)
            log_noise = Reshape((self.img_rows,self.img_cols,1),name="noise")(Dense(784,activation="relu")(noise))
            output_img = Reshape((self.img_rows,self.img_cols,1))(Dense(784,activation='sigmoid')(z))

        #output the model
        model_training = Model(input_img, output_img)
        model_training.compile(optimizer="adadelta",loss=self.custom_loss(log_noise))

        self.model_out = Model(input_img,[output_img,log_noise])
        
        return model_training

    def custom_loss(self,log_noise):
        def loss(y_true,y_pred):
            out = K.sum(0.5*K.square(y_true-y_pred)*K.exp(-log_noise)+0.5*log_noise)/self.img_size
            return out
        return loss


    def load_data(self,dataset_name):
        # Load the dataset
        if(dataset_name == 'mnist'):
            (X_train, _), (X_test, _) = mnist.load_data()
        elif(dataset_name == 'cifar'):
            (X_train,_),(_,_) = cifar10.load_data()
        else:
            print('Error, unknown database')

        # normalise images between 0 and 1
        X_train = X_train/255.0
        X_test = X_test/255.0
        #add a channel dimension, if need be (for mnist data)
        if(X_train.ndim ==3):
            X_train = np.expand_dims(X_train, axis=3)
        if(X_test.ndim==3):
            X_test = np.expand_dims(X_test,axis=3)
        return X_train,X_test

    def train(self, epochs, batch_size=128, sample_interval=50):
        
        #load dataset
        X_train,X_test = self.load_data(self.dataset_name)

        sigma = 20.0/255.0 # 40.0/255.0

        for i in range(0,epochs):

            # ---------------------
            #  Autoencoder
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            curr_batch = X_train[idx,:,:,:]
            # Autoencoder training
            
            ## Denoising Autoencoder (add noise on the input)
            noise = np.random.randn(curr_batch.shape[0],curr_batch.shape[1],curr_batch.shape[2],curr_batch.shape[3])*sigma
            
            # curr_batch_masked = curr_batch*np.random.randint(2,size=curr_batch.shape)
            curr_batch_masked = curr_batch*np.random.binomial(1,0.75,size=curr_batch.shape)
    
            loss = self.ae.train_on_batch(curr_batch_masked,curr_batch)

            # print the losses
            print("%d [Loss: %f]" % (i, loss))
            self.error_list[i] = loss

            # Save some random generated images and the models at every sample_interval iterations
            if (i % sample_interval == 0):
                n_images = 5
                idx = np.random.randint(0, X_train.shape[0], n_images)
                test_imgs = X_train[idx,:,:,:]
                curr_batch = test_imgs
                self.test_images(curr_batch,'images/'+self.dataset_name+'_reconstruction_%06d.png' % i)

    def test_images(self, test_imgs, image_filename):
        n_images = test_imgs.shape[0]
        #get output imagesq
        masked_imgs = test_imgs * np.random.binomial(1,0.75,size=test_imgs.shape)
        output_imgs = self.ae.predict(masked_imgs)
        
        r = 3
        c = n_images
        fig, axs = plt.subplots(r, c)
        for j in range(c):
            #black and white images
            axs[0,j].imshow(masked_imgs[j, :,:,0], cmap='gray')
            axs[0,j].axis('off')
            axs[1,j].imshow(output_imgs[j, :,:,0], cmap='gray')
            axs[1,j].axis('off')
            axs[2,j].imshow(test_imgs[j,:,:,0],'gray')
            axs[2,j].axis('off')
        fig.savefig(image_filename)
        plt.close()
        
    def test(self,n_img=5):
        self.noiseModel = Model(inputs=self.ae.input,outputs=self.model_out.get_layer("noise").output)

        imgs = self.load_data(self.dataset_name)[1][:n_img]
        masked_imgs = imgs*np.random.binomial(1,0.75,size=imgs.shape) # np.random.randint(2,size=img.shape)
        output_imgs = self.ae.predict(masked_imgs.reshape(n_img,self.img_rows,self.img_cols,1))

        fig, axes = plt.subplots(nrows=n_img,ncols=4,figsize=(15,15))

        T = 30
        var = np.zeros((n_img,self.img_rows,self.img_cols))
        Ey = np.zeros((n_img,self.img_rows,self.img_cols))

        for t in range(T):
            output_imgs = self.ae.predict(masked_imgs.reshape(n_img,self.img_rows,self.img_cols,1)).reshape(n_img,self.img_rows,self.img_cols)
            output_noise = self.noiseModel.predict(masked_imgs.reshape(n_img,self.img_rows,self.img_cols,1)).reshape(n_img,self.img_rows,self.img_cols)
            var += output_imgs**2 /T
            var += K.exp(output_noise)/T
            Ey += output_imgs /T

        var -= Ey**2

        if n_img>=2:
            for i in range(n_img):
                # print(var[i].mean(),var[i].mean()**(1/2))
                axes[i,0].imshow(imgs[i].reshape(self.img_rows,self.img_cols),'gray',vmin=0,vmax=1)
                if i==0:
                    axes[i,0].set_title("image")
                axes[i,1].imshow(masked_imgs[i].reshape(self.img_rows,self.img_cols),'gray',vmin=0,vmax=1)
                if i==0:
                    axes[i,1].set_title("masked")
                axes[i,2].imshow(output_imgs[i].reshape(self.img_rows,self.img_cols),'gray',vmin=0,vmax=1)
                if i==0:
                    axes[i,2].set_title("reconstructed")
                cb = axes[i,3].imshow(var[i],'jet')
                if i==0:
                    axes[i,3].set_title("combined")
            fig.colorbar(cb,ax=axes[:,3],location="right")
                
        else:
            axes[0].imshow(imgs[0].reshape(self.img_rows,self.img_cols),'gray',vmin=0,vmax=1)
            axes[0].set_title("image")
            axes[1].imshow(masked_imgs[0].reshape(self.img_rows,self.img_cols),'gray',vmin=0,vmax=1)
            axes[1].set_title("masked")
            axes[2].imshow(output_imgs[0].reshape(self.img_rows,self.img_cols),'gray',vmin=0,vmax=1)
            axes[2].set_title("reconstructed")
            cb = axes[3].imshow(var[0],'jet')#,vmin=0,vmax=1)
            axes[3].set_title("combined")
            fig.colorbar(cb, ax=axes,shrink=0.2,location="right")

        fig.savefig("./images/results")
        plt.show()
        
        
        

if __name__ == '__main__':

    #create the output image directory
    if (os.path.isdir('images')==0):
        os.mkdir('images')

    #choose dataset
    dataset_name = 'mnist'
    # dataset_name = 'cifar'

    #create AE model
    architecture = 'mlp' #'convolutional'

    ae = autoencoder(dataset_name,architecture)

    if(len(sys.argv)==1):
        ae.train(epochs=ae.epochs, batch_size=64, sample_interval=100)
        plt.plot(ae.error_list[30:])
        plt.show()
        
        ae.test()

        ae.ae.save_weights('dense_combined.h5')
    else:
        if sys.argv[1] == "-l":
            ae.ae.load_weights(sys.argv[2])
            
            ae.test()