import os
import numpy as np
import tensorflow as tf
import pandas as pd
import wandb
from data_preparation_utils import directory_to_dataframe
from datetime import datetime
from datetime import timedelta


def directory_to_dataframe(label, dataframe=None):
    # Initialize the lists to store the data
    start_times = []
    file_paths = []
    labels = []

    directory = os.getcwd()
    directory_path = os.path.join(directory, "data", label)

    # Iterate over all files in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith(".png"):
            start_time_str = filename.split("_")[0]
            start_time_str = start_time_str.replace(" ", "_")
            start_time_str = start_time_str.replace("-", ":")
            start_time = datetime.strptime(start_time_str, "%Y:%m:%d_%H:%M:%S")

            # Append data to the lists
            start_times.append(start_time)
            file_paths.append(os.path.join("data", label, filename))
            labels.append(label)

    if dataframe == None:
        df = pd.DataFrame(
            {"start_time": start_times, "file_path": file_paths, "label": labels}
        )
    else:
        new_df = pd.DataFrame(
            {"time_period": start_times, "file_path": file_paths, "label": labels}
        )
        df = pd.concat([dataframe, new_df], ignore_index=True)

    return df


def configure_data_frame(df, max_image_num, sorted=True):
    c_df = df.drop_duplicates(subset=["start_time"])

    if sorted:
        c_df = c_df.sort_values("start_time")

    c_df = c_df.reset_index(drop=True)
    c_df = c_df.iloc[:max_image_num]

    return c_df


if __name__ == "__main__":
    burst_df = directory_to_dataframe("burst")
    noburst_df = directory_to_dataframe("no_burst")

    IMAGE_NUM_PER_LABEL = 1000

    configured_burst_df = configure_data_frame(
        burst_df, IMAGE_NUM_PER_LABEL
    )  # sorted=False if you do not want sorting
    configured_noburst_df = configure_data_frame(
        noburst_df, IMAGE_NUM_PER_LABEL
    )  # sorted=False if you do not want sorting

    configured_burst_df.to_excel("configured_burst.xlsx", index=False)
    configured_noburst_df.to_excel("configured_noburst.xlsx", index=False)

    print(configured_burst_df.dtypes)
    print(configured_burst_df)
