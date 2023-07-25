from model_builder_class import Model_builder
from data_preparation import get_imgs_directory, resize_and_save_img_in_directory, get_datasets, get_img_num
from train_utils import create_config, init_wandb_from_config, build_and_train

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

def main():
    #Fix the random generator seeds for better reproducibility
    tf.random.set_seed(67)
    np.random.seed(67)
    
    #Obtain the path of the YAML file and get a config file
    c_dir = os.getcwd
    config_path = os.path.join(c_dir, 'config.yaml')
    config = create_config(config_path)
    
    #Obtain the datasets, using get_datasets function from data_preparation
    train_ds, validation_ds = get_datasets()

    #Use config file to obtain sweep_id
    sweep_id = wandb.sweep(config, entity=config['entity'], project=config['project'])

    #Define a training function that uses the config file to initialize wandb and build/train the model using the datasets
    def train_fn():
        run = init_wandb_from_config(config)
        build_and_train(run, train_ds, validation_ds)

    #Calling the defined function for wandb to do hyperparameter tuning
    wandb.agent(sweep_id, train_fn)

if __name__ == "__main__":
    main()