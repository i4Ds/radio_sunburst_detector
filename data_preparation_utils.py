import os

import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image


# Obtain training and validation datasets from image directories
def get_datasets():
    directory = os.path.join(os.getcwd(), "data")
    # Create training dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=["burst", "no_burst"],
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
        class_names=["burst", "no_burst"],
        color_mode="grayscale",
        batch_size=32,
        image_size=(256, 256),
        shuffle=False, # Because of evaluation, each time you access it, the dataset is shuffeled -> impossible to extract labels
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
    print("get datasets works")
    return train_ds, validation_ds, test_ds


if __name__ == "__main__":
    # Get the directories of burst and non-burst images
    def get_imgs_directory():
        c_dir = os.getcwd()
        relative_burst_dir = os.path.join("data", "burst")
        relative_nburst_dir = os.path.join("data", "no_burst")

        burst_image_dir = os.path.join(c_dir, relative_burst_dir)
        nburst_image_dir = os.path.join(c_dir, relative_nburst_dir)

        print(burst_image_dir)
        print(nburst_image_dir)
        print("get imgs direcotry works")

        return burst_image_dir, nburst_image_dir

    # resize and fill the images to be the target size: likely 256x256 for possible use in CNN's
    def resize_and_fill(image_path, target_size):
        img = Image.open(image_path)
        img.thumbnail(target_size)

        new_img = Image.new("L", target_size)
        new_img.paste(
            img,
            ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2),
        )
        print("resize and fill works")

        return new_img

    # Use the resize and fill function for each image in the image directory and save it on top of these images: Note that the old versions of images are deleted
    def resize_and_save_img_in_directory(img_directory, target_size):
        for file_name in os.listdir(img_directory):
            # print(file_name)
            if file_name.endswith(".png"):
                image_path = os.path.join(img_directory, file_name)
                new_img = resize_and_fill(image_path, target_size)
                new_img.save(os.path.join(img_directory, file_name))  # save new image

        plt.imshow(new_img)
        plt.axis("off")
        plt.show()
        print("image showed, resize and save img in directory works")

    # Returns the number of batches
    def get_img_num(ds):
        # Determine the number of images in train and validation datasets
        ds_total_size = tf.data.experimental.cardinality(ds).numpy()

        print("get img num works")
        return ds_total_size
