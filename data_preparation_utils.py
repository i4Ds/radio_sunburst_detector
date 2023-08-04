import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from configure_dataframes import directory_to_dataframe


# Obtain training, and test datasets from dataframes
def get_datasets(
    data_df,
    train_size=0.9,
    test_size=0.1,
    burst_frac=0.5,
    sort_by_time=True,
    only_unique_time_periods=False,
    return_dfs=False,
):
    data_df = data_df.copy()
    # Shuffle the data, to make sure that all instruments appear in all datasets
    data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)
    if only_unique_time_periods:
        # Take only the first image from each time period
        data_df = data_df.drop_duplicates(subset=["start_time"])
    if sort_by_time:
        # Sort by time, so that the first image is the first image in time
        data_df = data_df.sort_values("start_time")

    # Calculate the lengths of the datasets
    train_len = int(train_size * len(data_df))
    test_len = int(test_size * len(data_df))

    # Create the dataframes
    test_df = data_df.iloc[-test_len:]
    train_df = data_df.iloc[:train_len]

    # Assert that the dataframes are correct
    assert np.intersect1d(train_df["file_path"], test_df["file_path"]).size == 0
    
     # Print out class balance
    print("Class balance in train dataset:")
    print(train_df["label"].value_counts())
    print("Class balance in test dataset:")
    print(test_df["label"].value_counts())

    # Create class balance in the dataframes
    if burst_frac:
        train_df = update_class_balance(train_df, burst_frac)
        test_df = update_class_balance(test_df, burst_frac)

    if sort_by_time:
        train_df = train_df.sort_values("start_time")
        test_df = test_df.sort_values("start_time")

    # Print out class balance
    print("Class balance in train dataset:")
    print(train_df["label"].value_counts())
    print("Class balance in test dataset:")
    print(test_df["label"].value_counts())

    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    directory = os.getcwd()

    train_ds = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=directory,
        classes=["no_burst", "burst"],
        x_col="file_path",
        y_col="label",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="binary",
        target_size=(256, 256),
        color_mode="grayscale",
    )

    test_ds = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=directory,
        classes=["no_burst", "burst"],
        x_col="file_path",
        y_col="label",
        batch_size=32,
        seed=42,
        shuffle=False,
        class_mode="binary",
        target_size=(256, 256),
        color_mode="grayscale",
    )
    if return_dfs:
        return train_ds, test_ds, train_df, test_df
    else:
        return train_ds, test_ds


def update_class_balance(df, burst_frac):
    df_bursts = df[df["label"] == "burst"]
    df_nobursts = df[df["label"] == "no_burst"]
    curr_ratio = len(df_bursts) / (len(df_bursts) + len(df_nobursts))
    while burst_frac < curr_ratio:
        # Drop a non burst randomly
        df_to_drop = df_bursts.sample(n=1)
        df_bursts = df_bursts.drop(df_to_drop.index)
        curr_ratio = len(df_bursts) / (len(df_bursts) + len(df_nobursts))
    while burst_frac > curr_ratio:
        # Drop a burst randomly
        df_to_drop = df_nobursts.sample(n=1)
        df_nobursts = df_nobursts.drop(df_to_drop.index)
        curr_ratio = len(df_bursts) / (len(df_bursts) + len(df_nobursts))

    return pd.concat([df_bursts, df_nobursts])


if __name__ == "__main__":
    data_df = directory_to_dataframe()
    train_ds, test_ds, train_df, test_df = get_datasets(data_df, return_dfs=True)

    train_df.to_excel("train.xlsx")
    test_df.to_excel("test.xlsx")
    data_df.to_excel("data.xlsx")

    class_names = list(train_ds.class_indices.keys())
    for ds, ds_name, df in zip(
        [train_ds, test_ds],
        ["train", "test"],
        [train_df, test_df],
    ):
        # Plot some images
        images, labels = next(
            ds
        )  # get the next batch of images and labels from the generator
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i], cmap="gray")
            plt.title(f"{class_names[int(labels[i])]} ({labels[i]}).")
            plt.axis("off")
        plt.show()

        # Calculate class balance
        print(f"Class balance in {ds_name} dataset:")
        print(df["label"].value_counts())
