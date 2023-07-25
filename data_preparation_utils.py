import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Obtain training and validation datasets from image directories
def get_datasets():
    directory = os.path.join(os.getcwd(), "data")
    # Create training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=["no_burst", "burst"],
        color_mode="grayscale",
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,
        seed=42,  # can change
        validation_split=0.3,  # can change
        subset="training",
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
    )

    # Create validation dataset
    validation_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=["no_burst", "burst"],
        color_mode="grayscale",
        batch_size=32,
        image_size=(256, 256),
        shuffle=True,  # Because of evaluation, each time you access it, the dataset is shuffeled -> impossible to extract labels
        seed=42,  # can change
        validation_split=0.3,  # can change
        subset="validation",
        interpolation="bilinear",
        follow_links=False,
        crop_to_aspect_ratio=False,
    )

    # Create test dataset
    validation_ds, test_ds = tf.keras.utils.split_dataset(
        validation_ds,
        left_size=0.5,
        shuffle=False,
        seed=42,
    )
    return train_ds, validation_ds, test_ds


if __name__ == "__main__":
    # Test the function
    train_ds, validation_ds, test_ds = get_datasets()

    # Create class names
    class_names = train_ds.class_names

    # Print out class balance in training and validation datasets
    for ds in [train_ds, validation_ds, test_ds]:
        # Plot some images
        plt.figure(figsize=(10, 10))
        for images, labels in ds.take(1):
            for i in range(9):
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(images[i].numpy().astype("uint8"), cmap="gray")
                plt.title(f"{class_names[int(labels[i])]} ({labels[i][0]}).")
                plt.axis("off")
        plt.show()

        # Calculate class balance
        y_true = np.concatenate([y for x, y in ds], axis=0)
        print("Class balance in dataset:")
        print(f"{np.unique(y_true, return_counts=True)}")
