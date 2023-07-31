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
from itertools import islice

import numpy as np
import tensorflow as tf
from wandb.keras import WandbCallback, WandbModelCheckpoint

import wandb
from configure_dataframes import directory_to_dataframe
from data_preparation_utils import get_datasets
from modelbuilder import ModelBuilder
from train_utils import load_config


def main(config_name: str) -> None:
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

    # Create datasets
    train_ds, _, _ = get_datasets(
        data_df, val_size=0.0, test_size=0.15, sort_by_time=True
    )

    # Build and train the model
    mb = ModelBuilder(model_params=wandb.config["model_params"])
    mb.build()
    model = mb.compile()
    # TODO: Early stopping
    # https://keras.io/api/callbacks/early_stopping/ I guess you need to add it to the model compile? Also, we should do early stopping on val_recall or accuracy, not val_loss
    # TODO: KFold cross validation
    # I would probably work with the updated get_dataset_function, such as below:
    # It works because it's sorted by time
    # Create datasets
    train_ds, val_ds, _ = get_datasets(
        data_df, train_size=0.5, val_size=0.1, test_size=0.15, sort_by_time=True
    )
    train_ds, val_ds, _ = get_datasets(
        data_df, train_size=0.6, val_size=0.1, test_size=0.15, sort_by_time=True
    )
    train_ds, val_ds, _ = get_datasets(
        data_df, train_size=0.7, val_size=0.1, test_size=0.15, sort_by_time=True
    )
    # Next one will fail because of assert statement because 0.8 + 0.1 + 0.15 > 1.0
    train_ds, val_ds, _ = get_datasets(
        data_df, train_size=0.8, val_size=0.1, test_size=0.15, sort_by_time=True
    )
    # Create a checkpoint for max val_accuracy
    checkpoint_acc = WandbModelCheckpoint(
        "models/best_model_acc.keras",
        monitor="val_accuracy",
        mode="max",
        verbose=1,
        save_best_only=True,
        save_format="tf",
    )

    # Create a checkpoint for max val_recall
    checkpoint_recall = WandbModelCheckpoint(
        "models/best_model_recall.keras",
        monitor="val_recall",
        mode="max",
        verbose=1,
        save_best_only=True,
        save_format="tf",
    )

    # wandbcallback saves epoch by epoch every metric we gave on modelbuilder to wandb
    # checkpoints to save the best model in all epochs for every sweep, may be based on recall or accuracy
    # can also load the best parameters of the model before doing an evaluation, this enables us to give the best parameters as single instance to wandb as well

    history = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=wandb.config["training_params"]["epochs"],
        callbacks=[WandbCallback()],
    )

    # Load one of the saved model
    # model.load_weights('models/best_model_acc.keras')
    # model.load_weights('models/best_model_recall.keras')

    evaluation = model.evaluate(validation_ds, verbose=2)
    val_loss, val_acc, val_precision, val_recall, val_f1_score = evaluation

    print("-----------------------------------")
    print("evaluation:")
    print(
        f"val_acc: {val_acc} val_loss: {val_loss} val_precision: {val_precision} val_recall: {val_recall} val_f1_score: {val_f1_score}"
    )
    print("----")
    print("history:")
    print(
        f"val_acc: {history.history['val_accuracy']} val_loss: {history.history['val_loss']} val_precision: {history.history['val_precision']} val_recall: {history.history['val_recall']} val_f1_score: {history.history['val_f1_score']}"
    )
    # Calculate metrics, Log to wandb
    wandb.log(
        {
            "training_acc": history.history["accuracy"][-1],
            "val_acc": float(val_acc),
            "loss": float(val_loss),
            "precision": history.history["val_precision"][-1],
            "recall": history.history["val_recall"][-1],
            "f1": history.history["val_f1_score"][-1],
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep for a specific model.")
    parser.add_argument(
        "--config_name", metavar="config_name", required=True, help="Name of model."
    )

    args = parser.parse_args()
    main(config_name=args.config_name)
