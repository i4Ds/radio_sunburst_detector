import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve

import wandb


def plot_roc_curve(y_true, y_pred_proba):
    """
    Function to plot ROC curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver operating characteristic")
    ax.legend(loc="lower right")

    return fig


def log_wandb_print_class_report(y_true, y_pred, target_names, verbose=True):
    label_bot_class_report = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    # Make Dataframe pretty:
    label_bot_class_report = pd.DataFrame(label_bot_class_report).T
    label_bot_class_report.loc[
        "accuracy",
        [
            "precision",
            "recall",
        ],
    ] = None
    label_bot_class_report.reset_index(inplace=True)

    # Upload to Wandb
    label_bot_class_report = wandb.Table(dataframe=label_bot_class_report)
    wandb.log({f"Classification Report": label_bot_class_report})

    if verbose:
        print(classification_report(y_true, y_pred, target_names=target_names))
