# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np

def extra_plot_3_confusion_matrices(results_df, task_name, class_names, save_path=None):
    """
    Plots confusion matrices for a given task:
    acoustic vs spectral vs combined
    """

    feature_order = ["acoustic", "spectral", "all"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, feat in zip(axes, feature_order):

        row = results_df[
            (results_df["task"] == task_name) &
            (results_df["features"] == feat)
        ].iloc[0]

        cm = row["confusion_matrix"]

        cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-12)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

        ax.set_title(f"{feat.capitalize()}")
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        # annotate cells
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i, j]:.2f}",
                        ha="center", va="center",
                        color=color)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(task_name)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()