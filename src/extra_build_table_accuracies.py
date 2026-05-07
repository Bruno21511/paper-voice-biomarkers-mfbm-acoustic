# -*- coding: utf-8 -*-
import pandas as pd

def extra_build_table_accuracies(results_df):

    feature_order = ["acoustic", "spectral", "all"]

    task_order = [
        "Control_vs_PhLP",
        "Control_vs_UVFP",
        "PhLP_vs_UVFP",
        "All_classes"
    ]

    feature_map = {
        "acoustic": "Acoustic",
        "spectral": "Spectral",
        "all": "Combined"
    }

    task_map = {
        "Control_vs_PhLP": "HE vs. PhLP",
        "Control_vs_UVFP": "HE vs. UVFP",
        "PhLP_vs_UVFP": "PhLP vs. UVFP",
        "All_classes": "3-Class"
    }

    rows = []

    # -------------------------
    # MAIN TABLE
    # -------------------------
    for feat in feature_order:

        row = {"Classification": feature_map[feat]}

        for task in task_order:

            df_sel = results_df[
                (results_df["features"] == feat) &
                (results_df["task"] == task)
            ]

            if len(df_sel) == 0:
                row[task_map[task]] = "n/a"
            else:
                acc = df_sel["accuracy_mean"].values[0] * 100
                std = df_sel["accuracy_std"].values[0] * 100

                row[task_map[task]] = f"{acc:.2f} ± {std:.2f}"

        rows.append(row)

    table = pd.DataFrame(rows)

    # -------------------------
    # OVA (correct per feature set)
    # -------------------------
    for feat in feature_order:
        df_sel = results_df[
            (results_df["features"] == feat) &
            (results_df["task"] == "All_classes")
        ].iloc[0]

        for i, col_name in enumerate(["HE vs. All (*)", "PhLP vs. All (*)", "UVFP vs. All (*)"]):
            mean = df_sel[f"ova_class_{i}_mean"] * 100
            std  = df_sel[f"ova_class_{i}_std"]  * 100
            table.loc[
                table["Classification"] == feature_map[feat],
                col_name
            ] = f"{mean:.2f} ± {std:.2f}"

    return table