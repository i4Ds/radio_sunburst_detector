import argparse
from itertools import islice
import os

import numpy as np
import tensorflow as tf
from wandb.keras import WandbMetricsLogger

import wandb
from data_preparation_utils import get_datasets
from metric_utils import log_wandb_print_class_report, plot_roc_curve
from modelbuilder import ModelBuilder
from train_utils import load_config
import pandas as pd

def main(config_name):
    # Fix the random generator seeds for better reproducibility
    tf.random.set_seed(67)
    np.random.seed(67)

    # Send config to wandb
    config = load_config(os.path.join("model_base_configs", config_name + ".yaml"))
    wandb.init(
        project="radio_sunburst_detection",
        config=config,
        entity="i4ds_radio_sunburst_detection",
    )
    del config
    
    configured_burst_df = pd.read_excel('configured_burst.xlsx')
    configured_noburst_df = pd.read_excel('configured_noburst.xlsx')
        
    train_ds, validation_ds, test_ds, train_df, val_df, test_df = get_datasets(configured_burst_df, configured_noburst_df)
    # Log number of images in training and validation datasets
    # TODO: Log number of images in test dataset

    # Build and train the model
    mb = ModelBuilder(model_params=wandb.config["model_params"])
    mb.build()
    model = mb.compile()
    model.summary()
    _ = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=wandb.config["training_params"]["epochs"],
        callbacks=[WandbMetricsLogger()],
    )

    # Save the model
    model.save(os.path.join(wandb.run.dir, "model.keras"))

    # Upload the model to wandb
    artifact = wandb.Artifact(
        "model",
        type="model",
        description="trained model",
        metadata=dict(config_name=config_name),
    )
    artifact.add_file(os.path.join(wandb.run.dir, "model.keras"))

    print("befor evaluate")
    # Evaluate model
    eval = model.evaluate(test_ds)
    eval_metrics = dict(zip(model.metrics_names, eval))  # Python magic
    wandb.log(eval_metrics)
    
    print("before predict")
    # Calculate other things
    y_pred_proba = model.predict(test_ds).flatten()
    print("predict-1")
    y_pred = np.where(y_pred_proba > 0.5, 1, 0)
    print("predict-2")
    steps = len(test_ds)  # This will give the number of batches in the test_ds
    y_true = np.concatenate([y for x, y in islice(test_ds, steps)], axis=0).flatten()   
    print("after predict")

    # Plot ROC curve
    fig = plot_roc_curve(y_true, y_pred_proba)
    wandb.log({"ROC Curve": [wandb.Image(fig)]})

    # Plot confusion matrix
    wandb.log(
        {
            "Confusion Matrix": wandb.plot.confusion_matrix(
                y_true=y_true, preds=y_pred, class_names=list(train_ds.class_indices.keys())
            )
        }
    )

    # Upload classification report to wandb
    log_wandb_print_class_report(y_true, y_pred, target_names=list(train_ds.class_indices.keys()))


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
