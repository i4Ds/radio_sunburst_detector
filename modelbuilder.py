import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D


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
            x = MaxPooling2D((2, 2), padding="same")(x)  # max pooling layer

        # max pooling layer to give us the result of the encoding process: latent space
        encoded = Conv2D(32, (3, 3), activation="relu", padding="same")(x)
        print(encoded.shape)

        # CLASSIFIER
        x = tf.keras.layers.Flatten()(encoded)
        n = int(self.model_params["neurons_dense_layer"])  # 128, 256, 512, 1024
        for _ in range(self.model_params["num_dense_layers"]):  # 1 or 2 or 3
            x = tf.keras.layers.Dense(
                n,
                activation="relu",
                kernel_initializer=self.model_params["weight_initialization"],
            )(x)
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
            metrics=["accuracy"],
        )
        return self.model
