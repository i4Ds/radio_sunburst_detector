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
from sklearn.metrics import f1_score, precision_score, recall_score

import wandb
from data_preparation_utils import get_datasets
from modelbuilder import ModelBuilder
from train_utils import load_config


def main(config_name: str) -> None:
    # Wandb login
    wandb.init()

    # Fix the random generator seeds for better reproducibility
    tf.random.set_seed(67)
    np.random.seed(67)

    # Send config to wandb
    config = load_config(os.path.join("model_base_configs", config_name + ".yaml"))
    wandb.config.update(config)
    del config

    train_ds, validation_ds, _ = get_datasets()

    # Build and train the model
    mb = ModelBuilder(model_params=wandb.config["model_params"])
    mb.build()
    model = mb.compile()
    model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=wandb.config["training_params"]["epochs"],
    )
    # Predict
    y_pred_proba = model.predict(validation_ds).flatten()
    y_pred = np.where(y_pred_proba > 0.5, 1, 0)
    y_true = np.concatenate([y for x, y in validation_ds], axis=0).flatten()

    # Calculate metrics, Log to wandb
    wandb.log(
        {
            "precision": precision_score(y_true, y_pred, average=None),
            "recall": recall_score(y_true, y_pred, average=None),
            "f1": f1_score(y_true, y_pred, average=None),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep for a specific model.")
    parser.add_argument(
        "--config_name", metavar="config_name", required=True, help="Name of model."
    )

    args = parser.parse_args()
    main(config_name=args.config_name)
