import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dropout
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import keras.models
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import wandb
from wandb.keras import WandbCallback


#Class to build the Model
class Model_builder:
    def __init__(self, input_shape, num_classes, encoder_filters=32, encoder_kernel_size=3, encoder_l1 = 0.00001,
        units=32, dropout=0.3, weight_initialization='he_normal', batch_size=32, learning_rate= 0.00001, num_maxpool=3,
        before_encoder_loop=1, num_loops=1, neurons_after_flatten=1024, activation= 'sigmoid'):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.encoder_filters = encoder_filters
        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_l1 = encoder_l1
        self.units = units
        self.dropout = dropout
        self.weight_initialization = weight_initialization
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_maxpool = num_maxpool
        self.before_encoder_loop = before_encoder_loop
        self.num_loops = num_loops
        self.neurons_after_flatten = neurons_after_flatten
        self.activation = activation
        self.model = None
        
    def build(self):
        input_img = Input(shape=self.input_shape)
        x= input_img
        # AUTOENCODER 
        for _ in range(self.num_maxpool): #3 or 4 or 5 or 6
            x = Conv2D(
                filters=self.encoder_filters,
                kernel_size=self.encoder_kernel_size,
                activation='relu',
                padding='same',
                activity_regularizer=regularizers.l1(self.encoder_l1),
                kernel_initializer=self.weight_initialization
            )(x)
            x = MaxPooling2D((2, 2), padding='same')(x) #max pooling layer
        for _ in range(self.before_encoder_loop): #0 or 1 or 2
            x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        #max pooling layer to give us the result of the encoding process: latent space
        encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        print(encoded.shape)

        #CLASSIFIER
        x = tf.keras.layers.Flatten()(encoded)
        n = self.neurons_after_flatten #128, 256, 512, 1024
        for _ in range(self.num_loops): #1 or 2 or 3  
            x = tf.keras.layers.Dense(n, activation='relu')(x)
            x = Dense(32,kernel_initializer=self.weight_initialization)(x)
            x = tf.keras.layers.Dropout(self.dropout)(x)
            n = n/2 
        output = tf.keras.layers.Dense(1, activation=self.activation)(x) 
        self.model = tf.keras.models.Model(inputs=input_img, outputs=output)  
        
    #compile the model
    def compile(self):
        self.model.compile(optmizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), 
            loss='binary_crossentropy', 
            metrics=['accuracy']
        )
        return self.model
    
