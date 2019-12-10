import glob
import shutil

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras import layers
from keras.layers import Input, Convolution2D, MaxPooling2D, ZeroPadding2D,Convolution2D,Activation,BatchNormalization
from keras.models import load_model,Model
from keras.layers import Add, AveragePooling2D, Dense, Flatten
from keras.initializers import glorot_uniform

''' Here we're trying to build the model for Image Classification with ResNet 50 Architecture  '''

''' First we've write Identity block, to skip the connections, and then we build convolution block '''

class layer_identification():

    def identity_block(self,X, f, filters, stage, block):
        #Defining name basis

        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        #Retrieve Filters . The no of filters that to be used in the conv mention them here.
        F1,F2,F3 = filters

        #Save the input value
        X_shortcut = X

        #First component of main path
        X = Convolution2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base+'2a')(X)
        X = Activation('relu')(X)

        #Second component of main path
        X = Convolution2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding='same',name= conv_name_base + '2b', kernel_initializer=glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis=3, name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        #Third component of main path

        X = Convolution2D(filters=F3, kernel_size=(1,1),strides=(1,1),padding='valid',name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name= bn_name_base + '2c')(X)
        
        X = Add()([X,X_shortcut])
        X = Activation('relu')(X)

        return X

    def convolutional_block(self,X,f,filters,stage,block,s=2):
        #Defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        #Retrieve Filters
        F1,F2,F3 = filters

        #Save the input value
        X_shortcut = X

        ######### MAIN PATH #####
        
        #First component of main path
        X = Convolution2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding='valid', 
        name=conv_name_base + '2a',kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3,name=bn_name_base+'2a')(X)
        X = Activation('relu')(X)

        #Second component of main path
        X = Convolution2D(filters=F2,kernel_size=(f,f),strides=(1,1),padding='same',name= conv_name_base + '2b', kernel_initializer=glorot_uniform(seed = 0))(X)
        X = BatchNormalization(axis=3,name = bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        #Third component of main path

        X = Convolution2D(filters=F3, kernel_size=(1,1),strides=(1,1),padding='valid',name=conv_name_base + '2c',kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3,name= bn_name_base + '2c')(X)
        
        ###### Shortcut Path #######
        X_shortcut = Convolution2D(filters= F3, kernel_size= (1,1), strides= (s,s), padding= 'valid',
        name = conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(axis=3, name = bn_name_base + '1')(X_shortcut)

        #Final step : Add shortcut value to main path, and pass it through a RELU Activation
        X = Add()([X,X_shortcut])
        X = Activation('relu')(X)

        return X

    def ResNet50(self,input_shape = (64, 64, 3), classes = 6):
    
        # Define the input as a tensor with shape input_shape
        X_input = Input(input_shape)

        
        # Zero-Padding
        X = ZeroPadding2D((3, 3))(X_input)
        
        # Stage 1
        X = Convolution2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
        X = Activation('relu')(X)
        X = MaxPooling2D((3, 3), strides=(2, 2))(X)

        # Stage 2
        X = self.convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='b')
        X = self.identity_block(X, 3, [64, 64, 256], stage=2, block='c')

        # Stage 3
        X = self.convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block='a', s=2)
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='b')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='c')
        X = self.identity_block(X, 3, [128, 128, 512], stage=3, block='d')

        # Stage 4
        X = self.convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block='a', s=2)
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='b')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='c')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='d')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='e')
        X = self.identity_block(X, 3, [256, 256, 1024], stage=4, block='f')

        # Stage 5
        X = self.convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
        X = self.identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

        # AVGPOOL
        X = AveragePooling2D(pool_size=(2,2), padding='same')(X)

        # Output layer
        X = Flatten()(X)
        X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
        
        
        # Create model
        model = Model(inputs = X_input, outputs = X, name='ResNet50')

        return model


model = layer_identification().ResNet50(input_shape = (64, 64, 3), classes = 6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

                                                                                                                                                                                                                                                                                                                                                                                                                                                        
