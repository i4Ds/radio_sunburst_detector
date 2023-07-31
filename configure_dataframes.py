import os
from glob import glob

import pandas as pd


def directory_to_dataframe(directory="data"):
    files = glob(
        directory + "/**/*.png"
    )  # get all files in data with any subdirectory and return the path, starting with data
    return pd.DataFrame([extract_information_from_path(file) for file in files])


def extract_information_from_path(path):
    label = "no_burst" if "no_burst" in path else "burst"
    start_time_str = path.split(os.sep)[-1].split("_")[0]
    start_time = pd.to_datetime(start_time_str, format="%Y-%m-%d %H-%M-%S")
    return {"label": label, "start_time": start_time, "file_path": path}


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
