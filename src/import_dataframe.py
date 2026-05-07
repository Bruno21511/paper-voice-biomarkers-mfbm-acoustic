# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from pathlib import Path


def import_dataframe(dataset_name,
                     input_root="../data/processed",
                     rebuild_mfbm=True):
    """
    Load parquet dataset and optionally rebuild MFBM matrices.

    Parameters
    ----------
    dataset_name : str
        Name of dataset file (without extension)
    input_root : str
        Base directory
    rebuild_mfbm : bool
        If True, reconstruct MFBM (20 x N)

    Returns
    -------
    df : pandas.DataFrame
    """

    # --- 1. Load parquet
    input_path = Path(input_root) / f"{dataset_name}.parquet"
    df = pd.read_parquet(input_path)

    # --- 2. Rebuild MFBM
    if rebuild_mfbm:

        # find all MFBM columns
        mfbm_cols = [col for col in df.columns if col.startswith("MFBM_")]

        # sort to guarantee order
        mfbm_cols = sorted(mfbm_cols, key=lambda x: int(x.split("_")[1]))

        def rebuild(row):
            bands = [np.array(row[col]) for col in mfbm_cols]
            return np.vstack(bands)

        df['MFBM'] = df.apply(rebuild, axis=1)
        
        # remove duplicate columns
        df = df.drop(columns=mfbm_cols)

    return df