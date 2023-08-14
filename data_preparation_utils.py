import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from configure_dataframes import directory_to_dataframe
from modelbuilder import TransferLearningModelBuilder


# Obtain training, and test datasets from dataframes
def get_datasets(
    data_df,
    instruments=[
        "australia_assa_02",
        "swiss_landschlacht_01",
        "alaska_haarp_62",
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

    # Sort so that no burst is 0 and burst is 1
    # This is because keras creates labeled after alphanumerics.
    data_df["label_keras"] = np.where(
        data_df["label"] == "no_burst", "_no_burst", "burst"
    )

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

    # Final asserts
    assert train_df.start_time.max() < test_df.start_time.min()
    # Assert that the dataframes are correct
    assert np.intersect1d(train_df["file_path"], test_df["file_path"]).size == 0
    
    # Final asserts
    assert train_df.start_time.max() < test_df.start_time.min()
    # Assert that the dataframes are correct
    assert np.intersect1d(train_df["file_path"], test_df["file_path"]).size == 0

    return train_df, test_df
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
    # Create datasets
    train_df, test_df = get_datasets(
        data_df,
        train_size=0.7,
        test_size=0.3,
        burst_frac=0.5,
        sort_by_time=True,
        only_unique_time_periods=True,
    )

    # Update datasets
    val_df, test_df = (
        test_df.iloc[: len(test_df) // 2],
        test_df.iloc[len(test_df) // 2 :],
    )

    # Get sample
    train_df = train_df.sample(n=100, random_state=42)
    val_df = val_df.sample(n=100, random_state=42)
    test_df = test_df.sample(n=100, random_state=42)

    train_df.to_excel("train.xlsx")
    test_df.to_excel("test.xlsx")
    data_df.to_excel("data.xlsx")

    # Data gen
    ppf = lambda x: TransferLearningModelBuilder.preprocess_input(x, ewc=False)
    datagen = ImageDataGenerator(preprocessing_function=ppf)

    # Create datasets
    train_ds = datagen.flow_from_dataframe(
        train_df,
        x_col="file_path",
        y_col="label_keras",
        batch_size=64,
        seed=42,
        shuffle=True,
        class_mode="binary",
        target_size=(256, 256),
        color_mode="grayscale",
    )
    val_ds = datagen.flow_from_dataframe(
        val_df,
        x_col="file_path",
        y_col="label_keras",
        batch_size=64,
        seed=42,
        shuffle=False,
        class_mode="binary",
        target_size=(256, 256),
        color_mode="grayscale",
    )

    test_ds = datagen.flow_from_dataframe(
        test_df,
        x_col="file_path",
        y_col="label_keras",
        batch_size=64,
        seed=42,
        shuffle=False,
        class_mode="binary",
        target_size=(256, 256),
        color_mode="grayscale",
    )

    class_names = list(train_ds.class_indices.keys())
    for ds, ds_name, df in zip(
        [train_ds, val_ds, test_ds],
        ["train", "val", "test"],
        [train_df, val_df, test_df],
        [train_ds, val_ds, test_ds],
        ["train", "val", "test"],
        [train_df, val_df, test_df],
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
            print(f"Image Min: {images[i].min()}, Image Max: {images[i].max()}")
            plt.axis("off")
        plt.show()

        # Calculate class balance
        print(f"Class balance in {ds_name} dataset:")
        print(df["label"].value_counts())
