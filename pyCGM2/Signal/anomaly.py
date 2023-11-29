import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyCGM2
LOGGER = pyCGM2.LOGGER

from typing import Optional

figsize = (7, 2.75)
kw = dict(marker='o', linestyle='none', color='r', alpha=0.3)


def anomaly_rolling(values:np.ndarray, aprioriError:int=0, window:int=10, threshold:int=3, method:str="median", plot:bool=False, label:str="Unknow", referenceValues:Optional[float]=None):
    
    """
    Detects anomalies in a sequence of values using a rolling window approach.

    This function identifies points in the data that are statistically different from the values in a moving window. Anomalies are determined based on a threshold applied to a specified descriptive statistic (median or mean) over a rolling window.

    Args:
        values (np.ndarray): The array of values for anomaly detection.
        aprioriError (int, optional): An a priori error margin added and subtracted from each value. Defaults to 0.
        window (int, optional): The size of the rolling window to compute the descriptive statistic. Defaults to 10.
        threshold (int, optional): The threshold for identifying anomalies, based on the standard deviation factor. Defaults to 3.
        method (str, optional): The method of computing the descriptive statistic ('median' or 'mean'). Defaults to "median".
        plot (bool, optional): If True, a plot of the values, rolling statistic, and anomalies is displayed. Defaults to False.
        label (str, optional): A label for the data, used in plotting. Defaults to "Unknown".
        referenceValues (Optional[float], optional): An array of reference values to be used instead of the values computed from the rolling window. If None, the rolling window values are used. Defaults to None.

    Returns:
        list: A list of indices where anomalies are detected in the input values.

    Note:
        - Zero values in the input are considered anomalies and are warned about.
        - The function plots the original values, the rolling statistic, and the detected anomalies if `plot` is True.
    """

    df = pd.DataFrame({'Values': values})

    df["Values0"] = df["Values"]
    indices0 = []

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
