# -*- coding: utf-8 -*-
import numpy as np
import parselmouth
from parselmouth.praat import call


def extract_voice_features(df, audio_col="signal", fs_col="fs"):
    """
    Extracts voice features and appends them as new DataFrame columns.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing audio signals

    Returns
    -------
    pandas.DataFrame
        DataFrame with added acoustic feature columns
    """

    meanF0_list = []
    stdF0_list = []
    jitter_list = []
    shimmer_list = []
    hnr_list = []

    f0min = 75
    f0max = 400
    unidade = "Hertz"
    inicio = 0.0

    janela = 0.03

    for _, row in df.iterrows():

        signal = row[audio_col]
        fs = row[fs_col]

        sound = parselmouth.Sound(signal, sampling_frequency=fs)

        # -------------------------
        # PITCH
        # -------------------------
        pitch = call(sound, "To Pitch", inicio, f0min, f0max)
        meanF0 = call(pitch, "Get mean", 0, 0, unidade)
        stdF0 = call(pitch, "Get standard deviation", 0, 0, unidade)

        # -------------------------
        # POINT PROCESS
        # -------------------------
        pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

        # -------------------------
        # JITTER (only local kept)
        # -------------------------
        localJitter = call(
            pointProcess,
            "Get jitter (local)",
            0, 0, 0.0001, 0.02, 1.3
        )

        # -------------------------
        # SHIMMER (only local kept)
        # -------------------------
        localShimmer = call(
            [sound, pointProcess],
            "Get shimmer (local)",
            0, 0, 0.0001, janela, 1.3, 1.6
        )

        # -------------------------
        # HNR
        # -------------------------
        harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0, 0)

        # -------------------------
        # STORE RESULTS
        # -------------------------
        meanF0_list.append(meanF0)
        stdF0_list.append(stdF0)
        jitter_list.append(localJitter)
        shimmer_list.append(localShimmer)
        hnr_list.append(hnr)

    # Add to dataframe
    df["meanF0"] = meanF0_list
    df["stddevF0"] = stdF0_list
    df["localJitter"] = jitter_list
    df["localShimmer"] = shimmer_list
    df["HNR"] = hnr_list

    return df