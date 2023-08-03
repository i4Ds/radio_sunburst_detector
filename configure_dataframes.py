import os
from glob import glob

import pandas as pd


def directory_to_dataframe(directory="data"):
    files = glob(
        directory + "/**/*.png"
    )  # get all files in data with any subdirectory and return the path, starting with data
    return pd.DataFrame([extract_information_from_path(file) for file in files])


def extract_information_from_path(path, time_bucket_agg='_None_'):
    label = "no_burst" if "no_burst" in path else "burst"
    file_name = path.split(os.sep)[-1]
    start_time_str = file_name.split("_")[0]

    # Create instrument
    tmp_ = file_name.split("_")[2:]
    instrument = "_".join(tmp_).split(time_bucket_agg)[0]

    # Create start time
    start_time = pd.to_datetime(start_time_str, format="%Y-%m-%d %H-%M-%S")

    # Extract burst type
    burst_type = file_name.split(time_bucket_agg)[-1].replace(".png", "")

    return {"label": label, "start_time": start_time, "file_path": path, "instrument": instrument, "burst_type": burst_type}


def configure_data_frame(df, max_image_num, sorted=True):
    c_df = df.drop_duplicates(subset=["start_time"])

    if sorted:
        c_df = c_df.sort_values("start_time")

    c_df = c_df.reset_index(drop=True)
    c_df = c_df.iloc[:max_image_num]

    return c_df


if __name__ == "__main__":
    print(directory_to_dataframe('data').sample(5))
