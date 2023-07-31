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

from modelbuilder import ModelBuilder


# Load YAML config file
def load_config(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    return config


# Initialize wandb using the config
def init_wandb_from_config(config):
    run = wandb.init(
        entity=config["entity"],
        project=config["project"],
        name=config["name"],
        config=config["parameters"],
    )
    return run


if __name__ == "__main__":
    print("hello")
