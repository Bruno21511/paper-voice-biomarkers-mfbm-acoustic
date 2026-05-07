# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


def extra_plot_results_bar(results_df, save_path=None, dpi=300):
    """
    Plot a grouped bar chart of classification accuracy across tasks and feature sets.

    For each classification task, three bars are drawn side by side — one per feature
    set (acoustic, spectral, combined). Error bars show the standard deviation of
    accuracy across cross-validation iterations.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results table as returned by the experiment loop. Expected columns:
            - "task"          : classification task identifier (e.g. "Control_vs_PhLP")
            - "features"      : feature set identifier ("acoustic", "spectral", or "all")
            - "accuracy_mean" : mean accuracy across iterations (in [0, 1])
            - "accuracy_std"  : std of accuracy across iterations (in [0, 1])
    save_path : str or None, optional
        If provided, the figure is saved to this path before display.
        The format is inferred from the file extension (e.g. ".png", ".pdf").
        If None, the figure is only displayed (default).
    dpi : int, optional
        Resolution in dots per inch for the saved figure (default: 300).
        Only used when save_path is provided.

    Returns
    -------
    None
    """

    # ------------------------------------------------------------------
    # Label maps
    # ------------------------------------------------------------------
    feature_map = {
        "acoustic": "Acoustic",
        "spectral": "Spectral",
        "all":      "Combined"
    }
    task_map = {
        "Control_vs_PhLP": "HEAL vs. PhLP",
        "Control_vs_UVFP": "HEAL vs. UVFP",
        "PhLP_vs_UVFP":    "PhLP vs. UVFP",
        "All_classes":     "All classes"
    }

    # ------------------------------------------------------------------
    # Task and feature ordering
    # ------------------------------------------------------------------
    tasks    = list(results_df["task"].unique())
    features = ["acoustic", "spectral", "all"]

    groups = [task_map.get(t, t) for t in tasks]   # x-axis group labels
    labels = [feature_map[f] for f in features]     # legend labels

    # ------------------------------------------------------------------
    # Build accuracy and std matrices — shape (n_features, n_tasks)
    # ------------------------------------------------------------------
    acc = []
    std = []

    for feat in features:
        acc_row = []
        std_row = []
        for task in tasks:
            row = results_df[
                (results_df["features"] == feat) &
                (results_df["task"]     == task)
            ]
            acc_row.append(row["accuracy_mean"].values[0] * 100)  # convert to %
            std_row.append(row["accuracy_std"].values[0]  * 100)
        acc.append(acc_row)
        std.append(std_row)

    acc = np.array(acc)
    std = np.array(std)

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    positions = np.arange(len(groups))
    width   = 0.2
    colors  = ["darkorange", "lightcyan", "darkgreen"]
    hatches = ["//", "\\\\", "oo"]

    fig, ax = plt.subplots(figsize=(8, 4))

    for i in range(len(features)):
        pos = positions + (i - 1) * width  # centre the three bars around each group
        ax.bar(
            pos,
            acc[i],
            width,
            yerr=std[i],
            capsize=5,
            color=colors[i],
            edgecolor="black",
            hatch=hatches[i],
            label=labels[i]
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(groups)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy [%]")
    ax.set_title("Classification Performance")

    # major and minor horizontal gridlines for readability
    ax.grid(which="major", linewidth=0.6, axis="y", color="black")
    ax.grid(which="minor", linewidth=0.3, axis="y", color="black")
    ax.minorticks_on()
    ax.set_axisbelow(True)

    ax.legend()
    plt.tight_layout()

    # ------------------------------------------------------------------
    # Save and/or display
    # ------------------------------------------------------------------
    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight", dpi=300)

    plt.show()