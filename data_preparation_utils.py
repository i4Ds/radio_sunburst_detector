import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from configure_dataframes import directory_to_dataframe


# Obtain training, and test datasets from dataframes
def get_datasets(
    data_df,
    instruments=[
        "australia_assa_02",
        "australia_assa_62",
        "india_ooty_01",
        "glasgow_59",
        "swiss_landschlacht_01",
        "alaska_haarp_62",
        "humain_59",
    ],
    train_size=0.9,
    test_size=0.1,
    burst_frac=0.5,
    sort_by_time=True,
    only_unique_time_periods=False,
):
    data_df = data_df.copy()
    # Select only the instruments we want
    data_df = data_df[data_df["instrument"].isin(instruments)]
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

    # Create class balance in the dataframes
    if burst_frac:
        train_df = update_class_balance_per_instrument(train_df, burst_frac)
        test_df = update_class_balance_per_instrument(test_df, burst_frac)

    # Sample them again to make sure that the files are still shuffled
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    if sort_by_time:
        train_df = train_df.sort_values("start_time")
        test_df = test_df.sort_values("start_time")

    # Print out class balance
    print_class_balance(train_df, "train")
    print_class_balance(test_df, "test")

    return train_df, test_df

def update_class_balance_per_instrument(df, burst_frac):
    # List to store processed dataframes per instrument
    dfs = []

    # Loop through each instrument
    for instrument, group in df.groupby("instrument"):
        df_bursts = group[group["label"] == "burst"]
        df_nobursts = group[group["label"] == "no_burst"]

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

        # Append the processed dataframe for this instrument
        dfs.append(pd.concat([df_bursts, df_nobursts]))

    # Concatenate all processed dataframes
    return pd.concat(dfs)


def print_class_balance(df, dataset_name):
    print(f"Class balance in {dataset_name} dataset:")

    # Group by instrument and then get value counts for each label
    balance = (
        df.groupby("instrument")["label"].value_counts().unstack().fillna(0).astype(int)
    )

    print(balance)
    print("-" * 50)  # Print a separator for clarity


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
