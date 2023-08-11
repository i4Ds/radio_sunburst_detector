import argparse
import os
from itertools import islice

import numpy as np
import tensorflow as tf
from wandb.keras import WandbMetricsLogger

import wandb
from configure_dataframes import directory_to_dataframe
from data_preparation_utils import get_datasets
from metric_utils import log_wandb_print_class_report, plot_roc_curve
from modelbuilder import ModelBuilder, TransferLearningModelBuilder
from train_utils import load_config
<<<<<<< HEAD
=======
from keras.utils import split_dataset
>>>>>>> e964e87 (Fixed main script.)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm.keras import TqdmCallback


def main(config_name):
    # Fix the random generator seeds for better reproducibility
    tf.random.set_seed(67)
    np.random.seed(67)

    # Send config to wandb
    config = load_config(os.path.join("model_base_configs", config_name + ".yaml"))
    wandb.init(
        project="radio_sunburst_detection_main",
        config=config,
        entity="i4ds_radio_sunburst_detection",
    )
    del config

    # Load dataframes
    data_df = directory_to_dataframe()

=======
>>>>>>> e964e87 (Fixed main script.)
    # Filter if you want
    if "instrument_to_use" in wandb.config:
        data_df = data_df[data_df.instrument.isin(wandb.config["instrument_to_use"])]

    # Create datasets
<<<<<<< HEAD
    train_df, test_df  = get_datasets(data_df, train_size=0.7, test_size=0.3, burst_frac=wandb.config["burst_frac"], sort_by_time=True, only_unique_time_periods=True)
                                            
    # Update datasets
    val_df, test_df = test_df.iloc[:len(test_df)//2], test_df.iloc[len(test_df)//2:]

    # Create label encoder for the labels


    # To excel for manual inspection
    train_df.to_excel("train_df.xlsx")
    val_df.to_excel("val_df.xlsx")
    test_df.to_excel("test_df.xlsx")
=======
    _, _, train_df, test_df  = get_datasets(data_df, train_size=0.7, test_size=0.3, return_dfs=True)
                                            
    # Update datasets
    val_df, test_df = train_df.iloc[:len(train_df)//2], train_df.iloc[len(train_df)//2:]
>>>>>>> e964e87 (Fixed main script.)

    # Get model
    if wandb.config["model"] == "transfer":
        mb = TransferLearningModelBuilder(model_params=wandb.config)
<<<<<<< HEAD
    else:
        raise ValueError("Model not implemented.")
=======
>>>>>>> e964e87 (Fixed main script.)
    
    # Create image generator
    ppf = lambda x: mb.preprocess_input(x, ewc=wandb.config['elim_wrong_channels'])
    datagen = ImageDataGenerator(preprocessing_function=ppf)
    
    # Create datasets
    train_ds = datagen.flow_from_dataframe(
        train_df,
        x_col="file_path",
        y_col="label_keras",
        batch_size=wandb.config["batch_size"],
        seed=42,
        shuffle=True,
        class_mode="binary",
        target_size=(256, 256),
        color_mode="rgb",
    )
    val_ds = datagen.flow_from_dataframe(
        val_df,
        x_col="file_path",
        y_col="label_keras",
        batch_size=wandb.config["batch_size"],
        seed=42,
        shuffle=False,
        class_mode="binary",
        target_size=(256, 256),
        color_mode="rgb",
    )

    test_ds = datagen.flow_from_dataframe(
        test_df,
        x_col="file_path",
        y_col="label_keras",
        batch_size=wandb.config["batch_size"],
        seed=42,
        shuffle=False,
        class_mode="binary",
        target_size=(256, 256),
        color_mode="rgb",
    )
<<<<<<< HEAD

    # Print out labels and their indices

=======
>>>>>>> e964e87 (Fixed main script.)

    # Log number of images in training and validation datasets
    # TODO: Log number of images in test dataset

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_f1_score", patience=10, verbose=1
    )  # or val_loss, experiment

    # Build and train the model
    mb.build()
    model = mb.compile()
    _ = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=wandb.config["epochs"],
        verbose=0,
        callbacks=[WandbMetricsLogger(), early_stopping_callback, TqdmCallback(verbose=1)],
    )

<<<<<<< HEAD
=======
    # Save the model
    # model.save(os.path.join(wandb.run.dir, "model.keras"))

    # Upload the model to wandb
    artifact = wandb.Artifact(
        config_name,
        type="model",
        description="trained model",
        metadata=dict(config_name=config_name),
    )
    artifact.add_file(os.path.join(wandb.run.dir, "model.keras"))

>>>>>>> e964e87 (Fixed main script.)
    # Evaluate model
    eval = model.evaluate(test_ds)
    # Create nice metrics names
    test_metric_names = ["test_" + metric for metric in model.metrics_names]
    # Create a dictionary of metrics
    eval_metrics = dict(zip(test_metric_names, eval))  # Python magic
    wandb.log(eval_metrics)

    # Do more things if it's not a sweep.
    if not wandb.run.sweep_id:
        # Save the model
        model.save(os.path.join(wandb.run.dir, "model.keras"))
        artifact = wandb.Artifact(
            config_name,
            type="model",
            description="trained model",
            metadata=dict(config_name=config_name),
        )
        artifact.add_file(os.path.join(wandb.run.dir, "model.keras"))

        # Calculate other things
        y_pred_proba = model.predict(test_ds).flatten()
        y_pred = np.where(y_pred_proba > 0.5, 1, 0)
        steps = len(test_ds)  # This will give the number of batches in the test_ds
        y_true = np.concatenate([y for x, y in islice(test_ds, steps)], axis=0).flatten()

        # Plot ROC curve
        fig = plot_roc_curve(y_true, y_pred_proba)
        wandb.log({"ROC Curve": [wandb.Image(fig)]})

        # Plot confusion matrix
        wandb.log(
            {
                "Confusion Matrix": wandb.plot.confusion_matrix(
                    y_true=y_true,
                    preds=y_pred,
                    class_names=["no_burst", "burst"],
                )
            }
        )

        # Upload classification report to wandb
        log_wandb_print_class_report(
            y_true, y_pred, target_names=list(train_ds.class_indices.keys())
        )

if __name__ == "__main__":
    """
    Runs the main script. Use it as follows:
    python main.py --config_name <config_name>
    """
    parser = argparse.ArgumentParser(description="Run a specific model.")

    parser.add_argument(
        "--config_name",
        metavar="config_name",
        help="Name of config-file.",
    )

    args = parser.parse_args()

    # Run the main function
    main(args.config_name)
