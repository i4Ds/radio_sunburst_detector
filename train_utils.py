import model_builder_class

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
import yaml


#Load YAML config file
def create_config(config_path):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config


#Initialize wandb using the config
def init_wandb_from_config(config):
        
    run = wandb.init(
        entity=config['entity'],
        project=config['project'],
        name=config['name'],
        config=config['parameters']
    )
    return run


def build_and_train(run, train_ds, validation_ds):
    
    #Create a model_builder object, build and compile the model inside it and retrieve it as "model"
    model_builder_object = model_builder_class.Model_builder(input_shape=(256, 256, 1) ,num_classes=2, encoder_filters=run.config.encoder_filters, encoder_kernel_size=run.config.encoder_kernel_size, encoder_l1 =run.config.encoder_l1,
        units=run.config.units, dropout=run.config.dropout, weight_initialization=run.config.weight_initialization, batch_size=run.config.batch_size, learning_rate= run.config.learning_rate, num_maxpool=run.config.num_maxpool,
        before_encoder_loop=run.config.before_encoder_loop, num_loops=run.config.num_loops, neurons_after_flatten=run.config.neurons_after_flatten, activation= run.config.activation
        )
    model_builder_object.build() 
    model = model_builder_object.compile()
    
    history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=run.config.epochs,
    batch_size=run.config.batch_size,
    callbacks=[WandbCallback()]  # WandbCallback logs metrics and final model
    )
    
    return model, history
    
if __name__ == "__main__":
    print("hello")