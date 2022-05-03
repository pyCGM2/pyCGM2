# -*- coding: utf-8 -*-
#APIDOC: /Low level/Signal


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyCGM2
LOGGER = pyCGM2.LOGGER


figsize = (7, 2.75)
kw = dict(marker='o', linestyle='none', color='r', alpha=0.3)


def anomaly_rolling(values, aprioriError=0, window=10, threshold=3, method="median", plot=False, label="Unknow", referenceValues=None):
    """anomaly detection from rolling windows

    Args:
        values (np.array): values
        aprioriError (int,Optional[0]): a priori error.
        window (int,Optional[10]): size of the window.
        threshold (int,Optional[3]):  standard deviation factor
        method (str,Optional[median]): descriptive statistic method
        plot (bool,Optional[False]): enable plot
        label (str,Optional[Unknown]): Description of parameter `label`
        referenceValues (np.array,Optional[None]): values used as reference instead of the values computing from the rolling windows

    """

    df = pd.DataFrame({'Values': values})

    df["Values0"] = df["Values"]
    indices0 = list()

    outlier0_idx = pd.Series([False] * df.Values.shape[0])
    if len(df[df.Values == 0]) != 0:
        LOGGER.logger.warning('[pyCGM2]  zeros found for label [%s]' % (label))
        outlier0_idx = df.Values == 0
        indices0 = df['Values'][df["Values"] == 0].index.to_list()
        df["Values"] = df.Values.replace(0, np.nan)  # .fillna(method='ffill')

    df["Values_upper"] = df["Values"] + aprioriError  # np.random.random()*3
    df["Values_lower"] = df["Values"] - aprioriError  # np.random.random()*3

    if method == "median":
        df['rolling'] = df['Values'].rolling(
            window=window, center=True).median()  # .fillna(df["Values"])

    elif method == "mean":
        df['rolling'] = df['Values'].rolling(
            window=window, center=True).mean()  # .fillna(df["Values"])

    if referenceValues is not None:
        df['rolling'] = referenceValues

    difference = np.abs(df['Values'] - df['rolling'])
    outlier_idx = difference > threshold

    difference_p3 = np.abs(df['Values_upper'] - df['rolling'])
    difference_m3 = np.abs(df['Values_lower'] - df['rolling'])
    outlier_idx_p3 = difference_p3 > threshold
    outlier_idx_m3 = difference_m3 > threshold
    outlier_idx = outlier_idx_m3.replace(True, np.nan).fillna(
        outlier_idx_p3.replace(True, np.nan)).replace(np.nan, True).replace(0, False)

    indexes = []
    if np.any(outlier_idx.values) != False:
        indexes = df['Values'][outlier_idx].index.to_list()

    for i in range(0, outlier_idx.shape[0]):
        if outlier0_idx.values[i]:
            outlier_idx.values[i] = True

    allIndexes0 = indices0 + indexes
    allIndexes0.sort()

    allIndexes = []
    [allIndexes.append(x) for x in allIndexes0 if x not in allIndexes]

    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        df['Values0'].plot()
        df['rolling'].plot(style="-r")
        df['Values0'][outlier_idx].plot(**kw)
        plt.title(label)
        plt.show()

    return allIndexes
