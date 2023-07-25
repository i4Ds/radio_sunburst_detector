import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn import metrics
from wandb.keras import WandbMetricsLogger

import wandb
from data_preparation_utils import get_datasets
from modelbuilder import ModelBuilder
from train_utils import load_config


def main(config_name):
    # Fix the random generator seeds for better reproducibility
    tf.random.set_seed(67)
    np.random.seed(67)

    # Send config to wandb
    config = load_config(os.path.join("model_base_configs", config_name + ".yaml"))
    wandb.init(project="radio_sunburst_detection", config=config)
    del config

    train_ds, validation_ds, test_ds = get_datasets()
    # Log number of images in training and validation datasets
    wandb.log({"N Train Images (rounded up by batch size)": len(train_ds) * wandb.config["training_params"]["batch_size"]})
    wandb.log({"N Val Images (rounded up by batch size)": len(validation_ds) * wandb.config["training_params"]["batch_size"]})

    # Build and train the model
    mb = ModelBuilder(
        model_params=wandb.config['model_params']
    )
    mb.build()
    model = mb.compile()
    model.summary()
    _ = model.fit(
        train_ds,
        validation_data=validation_ds,
        epochs=wandb.config['training_params']['epochs'],
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

    # Evaluate model
    eval = model.evaluate(test_ds)
    eval_metrics = dict(zip(model.metrics_names, eval)) # Python magic
    wandb.log(eval_metrics)

    # Calculate other things
    pred = model.predict(test_ds)
    print(pred)
    y_pred = np.where(pred > 0.5, 1, 0)
    y_true = np.concatenate([y for x, y in test_ds], axis=0)


    # Classification report
    classification_report = metrics.classification_report(y_true, y_pred, output_dict=True)
    wandb.log({"classification_report": classification_report})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a specific model.")

    parser.add_argument(
        "--config_name",
        metavar="config_name",
        help="Name of config-file.",
    )

    args = parser.parse_args()

    # Run the main function
    main(args.config_name)