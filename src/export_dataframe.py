# -*- coding: utf-8 -*-
from pathlib import Path

def export_dataframe(df,
                     dataset_name,
                     output_root="../data/processed",
                     expand_mfbm=True,
                     drop_columns=None):
    """
    Export processed DataFrame to parquet.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe
    dataset_name : str
        Name used for output file
    output_root : str
        Base directory to save file
    expand_mfbm : bool
        Whether to split MFBM into per-band columns
    drop_columns : list or None
        Columns to remove before saving
    """

    df_out = df.copy()

    # --- 1. Expand MFBM if requested
    if expand_mfbm and 'MFBM' in df_out.columns:
        n_bands = df_out['MFBM'].iloc[0].shape[0]

        for i in range(n_bands):
            df_out[f'MFBM_{i}'] = df_out['MFBM'].apply(
                lambda x: x[i, :].tolist()
            )

    # --- 2. Drop columns
    if drop_columns is not None:
        df_out = df_out.drop(columns=drop_columns, errors='ignore')

    # --- 3. Reset index
    df_out = df_out.reset_index(drop=True)

    # --- 4. Save
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{dataset_name}.parquet"
    df_out.to_parquet(output_path, index=False)

    print(f"Saved to: {output_path}")

    return df_out