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
from sklearn.model_selection import KFold, StratifiedKFold
import tensorflow as tf
from wandb.keras import WandbCallback, WandbModelCheckpoint

import wandb
from configure_dataframes import directory_to_dataframe
from data_preparation_utils import get_datasets
from modelbuilder import ModelBuilder
from train_utils import load_config

from tensorflow.keras.preprocessing.image import ImageDataGenerator


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
    train_ds, test_ds, train_df, test_df = get_datasets(
        data_df, sort_by_time=True, return_dfs=True, only_unique_time_periods=True
    )

    # Build and train the model
    mb = ModelBuilder(model_params=wandb.config["model_params"])
    mb.build()
    model = mb.compile()

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 3, verbose=1) #or val_recall, experiment 
    # wandbcallback saves epoch by epoch every metric we gave on modelbuilder to wandb
    # checkpoints to save the best model in all epochs for every sweep, may be based on recall or accuracy
    # can also load the best parameters of the model before doing an evaluation, this enables us to give the best parameters as single instance to wandb as well
    
    n_splits=5
    skf = KFold(n_splits = n_splits)
    X = train_df['file_path'].values
    y = train_df['label'].values
    datagen = ImageDataGenerator(rescale=1./255)
    
    recalls = []

    for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
        
        print("----------------------------------------")
        print(f"Training on Fold: {fold + 1}/{n_splits}")
        print("----------------------------------------")
        
        train_data = train_df.iloc[train_index]
        val_data = train_df.iloc[val_index]
        
        val_start = val_data.start_time.min()
        print("val_start:", val_start)
        val_end = val_data.start_time.max()
        train_in_val = train_data[(train_data.start_time > val_start) & (train_data.start_time < val_end)]
        assert train_in_val.empty, "Training data is in training data"  #gives error on second fold
        assert len(np.unique(train_data.start_time)) == len(train_data.start_time)
        assert len(np.unique(val_data.start_time)) == len(val_data.start_time)
                       
        train_data.to_excel("train_data.xlsx")
        val_data.to_excel("val_data.xlsx")
        
        new_train_ds = datagen.flow_from_dataframe(
            train_data,
            x_col='file_path',
            y_col='label',
            batch_size=32,
            seed=42,
            shuffle=True,
            class_mode="binary",
            target_size=(256, 256),
            color_mode="grayscale",
        )
        val_ds = datagen.flow_from_dataframe(
            val_data,
            x_col='file_path',
            y_col='label',
            batch_size=32,
            seed=42,
            shuffle=False,
            class_mode="binary",
            target_size=(256, 256),
            color_mode="grayscale",
        )
        
        history = model.fit(
            new_train_ds,
            validation_data=val_ds,
            epochs=wandb.config["training_params"]["epochs"],
            callbacks=[WandbCallback(),early_stopping_callback],
        )

        # Load one of the saved model
        # model.load_weights('models/best_model_acc.keras')
        # model.load_weights('models/best_model_recall.keras')

        evaluation = model.evaluate(val_ds, verbose=2)
        
        val_loss, val_acc, val_precision, val_recall, val_f1_score = evaluation

        print("-----------------------------------")
        print("evaluation:")
        print(
            f"val_acc: {val_acc} val_loss: {val_loss} val_precision: {val_precision} val_recall: {val_recall} val_f1_score: {val_f1_score}"
        )
  
        recalls.append(evaluation)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sweep for a specific model.")
    parser.add_argument(
        "--config_name", metavar="config_name", required=True, help="Name of model."
    )

    args = parser.parse_args()
    main(config_name=args.config_name)
