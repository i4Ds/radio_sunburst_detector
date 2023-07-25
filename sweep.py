import os

import keras.models
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
import yaml
from PIL import Image
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    UpSampling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from wandb.keras import WandbCallback

from data_preparation_utils import get_datasets
from train_utils import build_and_train, init_wandb_from_config, load_config


def main():
    # Fix the random generator seeds for better reproducibility
    tf.random.set_seed(67)
    np.random.seed(67)
    print("checkpoint 1")
    # Obtain the path of the YAML file and get a config file
    c_dir = os.getcwd()
    config_path = os.path.join(c_dir, "config.yaml")
    config = load_config(config_path)
    print("checkpoint 2")
    # Obtain the datasets, using get_datasets function from data_preparation
    train_ds, validation_ds = get_datasets()
    print("checkpoint 3")
    # Use config file to obtain sweep_id
    sweep_id = wandb.sweep(config, entity=config["entity"], project=config["project"])
    print("checkpoint 4")

    # Define a training function that uses the config file to initialize wandb and build/train the model using the datasets
    def train_fn():
        run = init_wandb_from_config(config)
        build_and_train(run, train_ds, validation_ds)

    # Calling the defined function for wandb to do hyperparameter tuning
    wandb.agent(sweep_id, train_fn)
    print("checkpoint 5")


if __name__ == "__main__":
    main()
