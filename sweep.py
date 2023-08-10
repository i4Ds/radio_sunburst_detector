"""
This script runs a sweep on parameters defined in `sweep_model_label.yaml`.
Add your parameters to the `sweep.yaml` file. Don't forget to put the base model
config in the `sweep.yaml` file as well.

The sweep has to be initated as follows:

    `wandb sweep sweep.yaml`

Then check the output of the Terminal and paste the comand to run the sweep.

The ID can be found in the command line output.
"""

import argparse
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold, TimeSeriesSplit
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm.keras import TqdmCallback

import wandb
from configure_dataframes import directory_to_dataframe
from data_preparation_utils import get_datasets
from modelbuilder import ModelBuilder, TransferLearningModelBuilder
from train_utils import load_config


def main(config_name: str, batch_size: int) -> None:
    # Wandb login
    wandb.init()

    # Fix the random generator seeds for better reproducibility
    tf.random.set_seed(67)
    np.random.seed(67)

    if not os.path.exists("models"):
        os.makedirs("models")

    # Send config to wandb
    config = load_config(os.path.join("model_base_configs", config_name + ".yaml"))
    wandb.config.update(config)
    del config

    # Load dataframes
    data_df = directory_to_dataframe()

    # Filter DF, if you want
    if 'instrument_to_use' in wandb.config:
        data_df = data_df[data_df.instrument.isin(wandb.config['instrument_to_use'])]

    # Create datasets
    train_ds, test_ds, train_df, test_df = get_datasets(
        data_df, 
        train_size=0.7,
        sort_by_time=True, return_dfs=True, only_unique_time_periods=True, burst_frac=wandb.config['burst_frac']
    )

    # Build and train the model
    mb = ModelBuilder(model_params=wandb.config["model_params"])
    #mb = TransferLearningModelBuilder(model_params=wandb.config)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_f1_score",
        patience=10,
        verbose=1,
        mode="max",
        restore_best_weights=True,
    )  # or val_recall, experiment
    # wandbcallback saves epoch by epoch every metric we gave on modelbuilder to wandb
    # checkpoints to save the best model in all epochs for every sweep, may be based on recall or accuracy
    # can also load the best parameters of the model before doing an evaluation, this enables us to give the best parameters as single instance to wandb as well

    n_splits = 4
    _s = TimeSeriesSplit(n_splits=n_splits)
    X = train_df["file_path"].values
    y = train_df["label"].values
    #pp_f = lambda x: TransferLearningModelBuilder.preprocess_input(x, ewc=wandb.config['elim_wrong_channels'])
    #datagen = ImageDataGenerator(preprocessing_function=pp_f)
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    evals = []

    for fold, (train_index, val_index) in enumerate(_s.split(X, y)):
        print("----------------------------------------")
        print(f"Training on Fold: {fold + 1}/{n_splits}")
        print("----------------------------------------")

        train_data = train_df.iloc[train_index]
        val_data = train_df.iloc[val_index]
        val_data, test_data = val_data.iloc[: len(val_data) // 2], val_data.iloc[
            len(val_data) // 2 :
        ]

        # Print out class balance
        print("Class balance in training set:")
        print(train_data.label.value_counts())
        print("Class balance in validation set:")
        print(val_data.label.value_counts())
        print("Class balance in test set:")
        print(test_data.label.value_counts())


        val_start = val_data.start_time.min()
        val_end = val_data.start_time.max()

        # Some asserts to make sure that the data is correct
        train_in_val = train_data[
            (train_data.start_time > val_start) & (train_data.start_time < val_end)
        ]
        assert (
            train_in_val.empty
        ), "Training data is in training data"  # gives error on second fold
        assert len(np.unique(train_data.start_time)) == len(train_data.start_time)
        assert len(np.unique(val_data.start_time)) == len(val_data.start_time)

        new_train_ds = datagen.flow_from_dataframe(
            train_data,
            x_col="file_path",
            y_col="label",
            batch_size=batch_size,
            seed=42,
            shuffle=True,
            class_mode="binary",
            target_size=(256, 256),
            color_mode="rgb",
        )
        val_ds = datagen.flow_from_dataframe(
            val_data,
            x_col="file_path",
            y_col="label",
            batch_size=batch_size,
            seed=42,
            shuffle=False,
            class_mode="binary",
            target_size=(256, 256),
            color_mode="rgb",
        )

        mb.build()
        model = mb.compile()

        _ = model.fit(
            new_train_ds,
            validation_data=val_ds,
            epochs=wandb.config["epochs"],
            verbose=0,
            callbacks=[early_stopping_callback, TqdmCallback(verbose=0)],
        )

        # Eval
        val_loss, val_acc, val_precision, val_recall, val_f1_score = model.evaluate(
            test_data, verbose=0
        )

        # Log metrics
        evals.append(
            {
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1_score": val_f1_score[0],
            }
        )

        # Print out all results to one line for easier comparison
        for metric, value in evals[-1].items():
            print(f"{metric}: {value:.4f}", end=" | ")

    # Calculate averages and standard deviations
    metrics_avg = {
        metric: np.mean([eval[metric] for eval in evals]) for metric in evals[0].keys()
    }
    metrics_std = {
        metric: np.std([eval[metric] for eval in evals]) for metric in evals[0].keys()
    }

    # Log averages
    for metric, value in metrics_avg.items():
        wandb.log({f"{metric}_avg": value})

    # Log standard deviations
    for metric, value in metrics_std.items():
        wandb.log({f"{metric}_std": value})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep for a specific model.")
    parser.add_argument(
        "--config_name", metavar="config_name", required=True, help="Name of model."
    )
    parser.add_argument(
        "--batch_size",
        metavar="batch_size",
        required=True,
        help="Batch size.",
    )
    # Cast to int
    batch_size = int(parser.parse_args().batch_size)
    args = parser.parse_args()
    main(config_name=args.config_name, batch_size=batch_size)
