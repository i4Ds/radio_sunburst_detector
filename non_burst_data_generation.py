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


def random_date(start, end):
    """Generate a random datetime between `start` and `end`"""
    return start + timedelta(
        # Get a random amount of seconds between `start` and `end`
        minutes=random.randint(0, int((end - start).total_seconds() // 60)),
    )


if __name__ == "__main__":
    # Get burst list
    burst_list = pd.read_excel("burst_list.xlsx").dropna(subset=["instruments"])

    # Extract instrument name
    burst_list.loc[:, "instruments"] = burst_list.instruments.apply(
        extract_instrument_name
    )

    ### PARAMETERS ###
    IMAGE_NUM_NON_BURST = 25000
    IMAGE_LENGTH = timedelta(minutes=1)
    PIXEL_PER_IMAGE_OVER_TIME = 200
    PIXEL_PER_IMAGE_OVER_FREQUENCY = 200
    INSTRUMENTS_TO_INCLUDE = [
        "australia_assa_02",
        "swiss_landschlacht_01",
        "alaska_haarp_62",
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

    # Drop duplicates, keep the last (highest type)
    burst_list_filtered = burst_list_filtered.drop_duplicates(
        subset=["instruments", "type", "datetime_start"], keep="last"
    )

    ###
    MIN_START_TIME = (
        burst_list_filtered.datetime_start.min()
    )  # burst_list_filtered.datetime_start.apply(lambda dt: dt.replace(hour=8, minute=0, second=0)).min()
    MAX_START_TIME = (
        burst_list_filtered.datetime_start.max() - IMAGE_LENGTH
    )  # MIN_START_TIME + timedelta(hours=12) - IMAGE_LENGTH

    print(MIN_START_TIME)
    print("----")
    print(MAX_START_TIME)

    # Tqdm progress bar
    non_burst_img_processed = 0
    # Create a tqdm progress bar.
    progress_bar = None
    progress_bar = tqdm(
        total=IMAGE_NUM_NON_BURST, desc="Processing Images", dynamic_ncols=True
    )

    # Continue processing until the required number of images is reached
    while non_burst_img_processed < IMAGE_NUM_NON_BURST:
        # Generate a random start date between a minimum and maximum start time
        random_start_date = random_date(MIN_START_TIME, MAX_START_TIME)
        end_date = random_start_date + IMAGE_LENGTH
        # Iterate through each instrument table to include
        for instrument_table in INSTRUMENTS_TO_INCLUDE:
            if check_table_data_availability(
                instrument_table, str(random_start_date), str(end_date)
            ):
                # Remove ID from the instrument name to get the base name
                base_instrument_name = remove_id_from_instrument_name(instrument_table)

                # Filter the burst list for entries that match the current instrument's base name
                burst_list_for_instrument = burst_list_filtered[
                    burst_list_filtered.instruments == base_instrument_name
                ]

                # Check that the random_start_date is not within any burst period for the current instrument
                non_burst_in_burst_df = burst_list_for_instrument[
                    (burst_list_for_instrument.datetime_start <= random_start_date)
                    & (random_start_date <= burst_list_for_instrument.datetime_end)
                ]

                # If the random start date falls within any burst period for the current instrument, continue to next iteration
                if len(non_burst_in_burst_df) > 0:
                    continue
                else:
                    try:
                        # Attempt to retrieve the data and save it as an image
                        # Parameters: instrument_table, start date, end date, x-limits, y-limits, burst category, data type
                        with HiddenPrints():
                            get_data_save_as_img(
                                instrument_table,
                                random_start_date,
                                random_start_date + timedelta(minutes=1),
                                time_bucket=None,
                                agg_function=None,
                                burst_type="no_burst",
                                min_shape=(200, 200),
                                data_folder="data",
                            )
                        # Increment the count of images processed
                        non_burst_img_processed += 1
                        progress_bar.update(1)
                        progress_bar.set_postfix(
                            {"Image": f"{instrument_table}_{random_start_date}"},
                            refresh=True,
                        )
                    except ValueError as e:
                        # Handle exception: print the error and skip to next date
                        print(e)
                        pass
