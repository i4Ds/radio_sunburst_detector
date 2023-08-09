import os
import random
import sys
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ecallisto_ng.data_fetching.get_data import extract_instrument_name, get_data
from ecallisto_ng.data_fetching.get_information import (
    check_table_data_availability,
    get_tables,
)
from tqdm import tqdm

np.random.seed(52)


def remove_id_from_instrument_name(instrument_name):
    return "_".join(instrument_name.split("_")[:-1])


# Functionality to hide print statements


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_data_save_as_img(
    instrument,
    start_datetime,
    end_datetime,
    time_bucket,
    agg_function="MAX",
    burst_type="no_burst",
    min_shape=(200, 200),
    data_folder="data",
):
    """
    Retrieves data for a specific instrument within a given time range, aggregates it using the specified function,
    normalizes the data, and saves it as an image file.

    Args:
        instrument (str): Name of the instrument for which data is to be retrieved.
        start_datetime (datetime.datetime): Start date and time of the data range.
        end_datetime (datetime.datetime): End date and time of the data range.
        time_bucket (str): Time granularity for data aggregation (e.g., '1H' for hourly, '30T' for every 30 minutes).
        agg_function (str, optional): Aggregation function to apply to the data. Defaults to 'MAX'.
        burst_type (str, optional): Label to be included in the file name. Defaults to 'no_burst'.
        data_folder (str, optional): Folder path where the data will be saved. Defaults to 'data'.
        min_shape (tuple, optional): Minimum shape of the image. Defaults to (200, 200).

    Returns:
        None

    Raises:
        None

    Examples:
        # Retrieve data for instrument 'instrument_name' from 'start_datetime' to 'end_datetime' and save it as an image
        get_data_save_as_img('instrument_name', start_datetime, end_datetime, '1H', 'MAX', 'no_burst', 'data')

    """
    sd_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
    ed_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

    # Generate path
    dir = os.path.join(data_folder, burst_type)
    if not os.path.exists(dir):
        os.makedirs(dir)

    # Create file path
    file_path = os.path.join(
        dir,
        sd_str.replace(":", "-")
        + "_"
        + ed_str.replace(":", "-")
        + "_"
        + instrument
        + "_"
        + str(time_bucket)
        + "_"
        + str(burst_type)
        + ".png",
    )
    if os.path.exists(file_path):
        print("File already exists.")
        return True

    df = get_data(
        instrument_name=instrument,
        start_datetime=sd_str,
        end_datetime=ed_str,
        timebucket=time_bucket,
        agg_function=agg_function,
    )

    img_data = df.to_numpy().astype(np.int16)
    if not img_data.shape[0] >= min_shape[0] and img_data.shape[1] >= min_shape[1]:
        print("Image shape is too small.")
        return False

    plt.imsave(file_path, img_data.T, cmap="gray")
    return True


if __name__ == "__main__":
    # Get burst list
    burst_list = pd.read_excel("burst_list.xlsx").dropna(subset=["instruments"])

    # Extract instrument name
    burst_list.loc[:, "instruments"] = burst_list.instruments.apply(
        extract_instrument_name
    )

    ### PARAMETERS ###
    IMAGE_NUM_BURST = 5000
    IMAGE_LENGTH = timedelta(minutes=1)
    PIXEL_PER_IMAGE_OVER_TIME = 200
    PIXEL_PER_IMAGE_OVER_FREQUENCY = 200
    INSTRUMENTS_TO_INCLUDE = [
        "australia_assa_02",
    ]
    ###

    # Because burst list only contains the antenna, not the unique instruments, translate
    instruments_to_include_sql_table_compatible = [
        remove_id_from_instrument_name(instrument)
        for instrument in INSTRUMENTS_TO_INCLUDE
    ]

    # Drop duplicate list
    instruments_to_include_sql_table_compatible = list(
        set(instruments_to_include_sql_table_compatible)
    )

    # Now keep only relevant bursts
    burst_list_filtered = burst_list[
        burst_list.instruments.isin(instruments_to_include_sql_table_compatible)
    ]

    # Create TQDM progress bar
    image_num = 0

    # Create a tqdm progress bar.
    progress_bar = None
    progress_bar = tqdm(
        total=IMAGE_NUM_BURST, desc="Processing Images", dynamic_ncols=True
    )
    # Sort the burst list by instrument, type, and datetime_start to allow that the highest type is at the end and kept, when we drop duplicates (not sure if this is necessary)
    burst_list_filtered = burst_list_filtered.sort_values(
        by=["instruments", "type", "datetime_start"], ascending=False
    )
    # Drop duplicates, keep the last (highest type)
    burst_list_filtered = burst_list_filtered.drop_duplicates(
        subset=["instruments", "type", "datetime_start"], keep="last"
    )
    # Iterate through each row in the filtered burst list
    burst_list_filtered = burst_list_filtered.sample(
        frac=1, random_state=42
    ).reset_index(drop=True)

    # Iterate through each row in the filtered burst list
    for index, burst_row in burst_list_filtered.iterrows():
        # Get the start and end datetime of the burst
        burst_start = burst_row.datetime_start
        burst_end = burst_row.datetime_end

        # Create a date range from the start to end datetime with a frequency of IMAGE_LENGTH, including the left endpoint
        burst_date_range = pd.date_range(
            burst_start, burst_end, freq=IMAGE_LENGTH, inclusive="left"
        )
        for date in burst_date_range:
            end_date = date + timedelta(minutes=1)
            # Iterate through each instrument table to include
            for instrument_table in INSTRUMENTS_TO_INCLUDE:
                if check_table_data_availability(
                    instrument_table, str(date), str(end_date)
                ):
                    # Attempt to retrieve the data and save it as an image
                    # Parameters: instrument_table, start date, end date, x-limits, y-limits, burst category, data type
                    try:
                        with HiddenPrints():
                            result = get_data_save_as_img(
                                instrument=instrument_table,
                                start_datetime=date,
                                end_datetime=end_date,
                                time_bucket=None,
                                agg_function=None,
                                burst_type=str(burst_row.type),
                                min_shape=(200, 200),
                                data_folder="data",
                            )
                        if result:
                            image_num += 1
                            progress_bar.update(1)
                            progress_bar.set_postfix(
                                {"Image": f"{instrument_table}_{date}_{end_date}"},
                                refresh=True,
                            )
                    except Exception as e:
                        print(e)
                else:
                    pass

            if image_num >= IMAGE_NUM_BURST:
                break
    progress_bar.close()
