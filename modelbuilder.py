import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import filters
from tensorflow.keras import regularizers
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Flatten,
    Input,
    MaxPooling2D,
)
from tensorflow.keras.models import Model


def elimwrongchannels(
    df,
    channel_std_mult=5,
    jump_std_mult=2,
    nan_interpolation_method="pchip",
    interpolate_created_nans=True,
    verbose=False,
):
    """
    Remove Radio Frequency Interference (RFI) from a spectrogram represented by a pandas DataFrame.
    This function works even when there is missing data thanks to interpolation of missing values.
    However, it could lead to some false or different results.
    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame where the index represents time and the columns represent frequency channels.
    channel_std_mult : float, optional
        Multiplicative factor for the standard deviation threshold used in the first RFI elimination step.
        Channels with standard deviation less than this threshold times the mean standard deviation across all channels are retained.
        Default is 5.
    jump_std_mult : float, optional
        Multiplicative factor for the standard deviation threshold used in the second RFI elimination step which deals with sharp jumps between channels.
        Channels with the absolute difference from the mean value less than this threshold times the standard deviation of differences are retained.
        Default is 2.
    nan_interpolation_method : str, optional
        Interpolation method to use for missing values. See pandas.DataFrame.interpolate for more details.
        Default is "pchip".
    interpolate_created_nans : bool, optional
        Whether to interpolate NaNs created by the first RFI elimination step.
        Default is True.
    verbose : bool, optional
        Whether to print out the number of eliminated channels.

    Returns
    -------
    pandas.DataFrame
        DataFrame with RFI removed. The DataFrame is oriented in the same way as the input DataFrame (time on index and frequency on columns).

    """
    df = df.copy()

    # Store original NaN positions
    nan_positions = df.isna()

    # Fill missing data with interpolation
    df.interpolate(method=nan_interpolation_method, inplace=True)
    df.fillna(
        method="bfill", inplace=True
    )  # for cases where NaNs are at the start of a series

    # Transpose df so that rows represent channels and columns represent time
    df = df.T

    # Calculate standard deviation for each channel and scale it to 0-255
    std = df.std(axis=1).fillna(0)
    std = ((std - std.min()) * 255) / (std.max() - std.min())
    std = std.clip(upper=255).astype(int)

    mean_sigma = std.mean()
    positions = std < channel_std_mult * mean_sigma
    eliminated_channels = (~positions).sum()
    if verbose:
        print(f"{eliminated_channels} channels eliminated")

    if np.sum(positions) > 0:
        # Replace the line with nans
        df.iloc[~positions, :] = np.nan

    if interpolate_created_nans:
        # Interpolate the nans
        df = df.interpolate(axis=0, limit_direction="both")

    if verbose:
        print("Eliminating sharp jumps between channels ...")
    y_profile = np.average(filters.roberts(df.values.astype(float)), axis=1)
    y_profile = pd.Series(y_profile - y_profile.mean(), index=df.index)
    mean_sigma = y_profile.std()

    positions = np.abs(y_profile) < jump_std_mult * mean_sigma
    eliminated_channels = (~positions).sum()
    if verbose:
        print(f"{eliminated_channels} channels eliminated")

    if np.sum(positions) > 0:
        # Replace the line with nans
        df.iloc[~positions, :] = np.nan
    else:
        if verbose:
            print("Sorry, all channels are bad ...")
        df = pd.DataFrame()
    if interpolate_created_nans:
        # Interpolate the nans
        df = df.interpolate(axis=0, limit_direction="both")
    # Transpose df back to original orientation
    df = df.T

    # Drop nans
    df.dropna(inplace=True)

    # Bring back original NaN values
    df[nan_positions] = np.nan

    return df

# Class to build the Model
class ModelBuilder:
    def __init__(
        self,
        model_params,
    ):
        self.model_params = model_params
        self.model = None

    def build(self):
        input_img = Input(shape=self.model_params["input_shape"])
        x = input_img

        # AUTOENCODER
        k = self.model_params["encoder_kernel_size"]
        for _ in range(self.model_params["num_hidden_layers"]):  # 3 or 4 or 5 or 6
            x = Conv2D(
                filters=self.model_params["encoder_filters"],
                kernel_size=(k, k),
                activation="relu",
                padding="same",
                activity_regularizer=regularizers.l1(self.model_params["encoder_l1"]),
                kernel_initializer=self.model_params["weight_initialization"],
            )(x)
            x = BatchNormalization()(x)
            x = tf.keras.activations.relu(x)
            x = MaxPooling2D((2, 2), padding="same")(x)  # max pooling layer

        # max pooling layer to give us the result of the encoding process: latent space
        encoded = Conv2D(32, (3, 3), padding="same")(x)
        encoded = BatchNormalization()(encoded)
        encoded = tf.keras.activations.relu(encoded)

        # CLASSIFIER
        x = tf.keras.layers.Flatten()(encoded)
        for _ in range(self.model_params["num_dense_layers"]):  # 1 or 2 or 3
            x = tf.keras.layers.Dense(
                int(self.model_params["neurons_dense_layer"]),
                kernel_initializer=self.model_params["weight_initialization"],
            )(x)
            x = BatchNormalization()(x)
            x = tf.keras.activations.relu(x)
            x = tf.keras.layers.Dropout(self.model_params["dropout"])(x)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(x)
        self.model = tf.keras.models.Model(inputs=input_img, outputs=output)

    # compile the model
    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.model_params["learning_rate"]
            ),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.metrics.F1Score(threshold=0.5),
            ],
        )
        return self.model


class TransferLearningModelBuilder:
    def __init__(self, model_params):
        self.input_shape = model_params["input_shape"]
        self.learning_rate = model_params.get("learning_rate", 0.001)
        self.l1 = model_params.get("l1", 0.0)
        self.base_model_name = model_params.get("base_model_name", "EfficientNetV2B3")
        self.weight_initialization = model_params.get("weight_initialization", "he_normal")
        self.model = None
        self.base_model = None
    
    def build_base_model(self):
        if self.base_model_name == "EfficientNetV2B3":
            self.base_model = EfficientNetV2B3(
                weights="imagenet", include_top=False, input_shape=self.input_shape
            )
        else:
            raise ValueError("Invalid base model name")
        # Freeze all layers of base model for transfer learning
        for layer in self.base_model.layers:
            layer.trainable = False
        x = self.base_model.output
        return x

    def build(self):
        x = self.build_base_model()
        x = Flatten()(x)  # Flatten the output to connect with Dense layer

        # Classifier with L1 regularization
        output = Dense(
            1, activation="sigmoid", kernel_regularizer=regularizers.l1(self.l1), kernel_initializer=self.weight_initialization
        )(x)

        self.model = Model(inputs=self.base_model.input, outputs=output)

    def compile(self):
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=[
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.metrics.F1Score(threshold=0.5),
            ],
        )
        return self.model

    @staticmethod
    def preprocess_input(x, ewc=False):
        if ewc:
            x = np.squeeze(x[:,:, :, 0])
            x = pd.DataFrame(x).T
            x = elimwrongchannels(x, verbose=False).T.values
            x = np.expand_dims(x, axis=-1)
            x = np.repeat(x, 3, axis=-1)
            x = np.expand_dims(x, axis=0)
        return preprocess_input(x)
