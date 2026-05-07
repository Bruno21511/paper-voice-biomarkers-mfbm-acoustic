# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_acoustic_features(
    df,
    features,
    class_col="group",
    db_transform=("localJitter", "localShimmer"),
    palette="Set2",
    figsize=(12, 6),
    save_path=None,
    dpi=300
):
    """
    Plots boxplots of acoustic features per class.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing features and class labels.

    features : list of tuples
        List of (column_name, display_name), e.g.
        [("localJitter", "Jitter (dB)")]

    class_col : str
        Column with class labels (default: "group")

    db_transform : tuple
        Features to convert to dB scale for visualization.

    palette : str or list
        Seaborn color palette.

    figsize : tuple
        Figure size.

    save_path : str or None
        If provided, saves figure to file.

    dpi : int
        Resolution for saving figure.
    """

    sns.set(style="whitegrid")

    fig, axes = plt.subplots(1, len(features), figsize=figsize)

    if len(features) == 1:
        axes = [axes]

    for ax, (feat, title) in zip(axes, features):

        df_plot = df.copy()

        # Convert to dB if needed
        if feat in db_transform:
            df_plot[feat] = 20 * np.log10(df_plot[feat] + 1e-10)

        sns.boxplot(
            data=df_plot,
            x=class_col,
            y=feat,
            hue=class_col,
            palette=palette,
            dodge=False,
            legend=False,
            ax=ax
        )

        ax.set_title(title)
        ax.set_xlabel("Class")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    plt.show()