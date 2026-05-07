# -*- coding: utf-8 -*-
import numpy as np
from sklearn.decomposition import PCA

def apply_block_pca(X, train_idx, test_idx):
    """
    Applies PCA per feature block, replicating original paper logic
    using explicit structure.
    """

    # --------------------------
    # define blocks explicitly
    # --------------------------
    blocks = {
        "mfbm_mean": (0, 12, 2),
        "mfbm_std":  (12, 24, 1),
    }

    Xtr_out = []
    Xte_out = []

    # --------------------------
    # PCA blocks
    # --------------------------
    for start, end, n_comp in blocks.values():

        pca = PCA(n_components=n_comp)

        Xtr_block = pca.fit_transform(X[train_idx, start:end])
        Xte_block = pca.transform(X[test_idx, start:end])

        Xtr_out.append(Xtr_block)
        Xte_out.append(Xte_block)

    # --------------------------
    # append non-PCA features
    # --------------------------
    Xtr_out.append(X[train_idx, 24:])  # jitter, shimmer, HNR
    Xte_out.append(X[test_idx, 24:])

    # --------------------------
    # final concat
    # --------------------------
    Xtr_final = np.hstack(Xtr_out)
    Xte_final = np.hstack(Xte_out)

    return Xtr_final, Xte_final