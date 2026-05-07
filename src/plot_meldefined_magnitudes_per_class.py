# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def plot_meldefined_magnitudes_per_class(df, save_path=None, classes=None):
    """
    Plot mean and std of MFBM per class.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns:
        - 'group'
        - 'MFBM' (array shape: 20 x N per sample)

    classes : list or None
        List of classes to plot. If None, uses unique values in df['group']
    """

    # --- Classes
    if classes is None:
        classes = sorted(df['group'].unique())

    # --- Aggregate per class
    mean_dict = {}
    std_dict = {}

    for c in classes:
        subset = df[df['group'] == c]

        mean_stack = np.vstack(subset['mean_MFBM'].values)
        std_stack  = np.vstack(subset['std_MFBM'].values)

        mean_dict[c] = np.mean(mean_stack, axis=0)
        std_dict[c]  = np.mean(std_stack, axis=0)

    # --- Plot styles
    linestyles = {
        'control': '-',
        'physio': '-.',
        'neuro': '--'
    }

    banda_x = np.arange(len(mean_dict[classes[0]])) +1

    plt.figure(figsize=(12,6))

    # --- Mean plot
    plt.subplot(121)
    plt.title('Mean MFBM average per Class')

    for c in classes:
        plt.plot(banda_x, mean_dict[c],
                 linestyle=linestyles.get(c, '-'),
                 linewidth=2,
                 label=c)

    plt.xlabel('Band')
    plt.ylabel('Mean magnitude')
    plt.grid(True)
    plt.xticks(range(len(banda_x)+1))
    plt.legend()

    # --- Std plot
    plt.subplot(122)
    plt.title('Std MFBM average per Class')

    for c in classes:
        plt.plot(banda_x, std_dict[c],
                 linestyle=linestyles.get(c, '-'),
                 linewidth=2,
                 label=c)

    plt.xlabel('Band')
    plt.ylabel('Std magnitude')
    plt.grid(True)
    plt.xticks(range(len(banda_x)+1))
    plt.legend()

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

    # return mean_dict, std_dict