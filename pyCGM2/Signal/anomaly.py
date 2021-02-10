# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

figsize = (7, 2.75)
kw = dict(marker='o', linestyle='none', color='r', alpha=0.3)



def anomaly_rolling(values,window=10 , threshold = 3, method = "median", plot=False,label="Unknow"):

    df = pd.DataFrame({'Values': values})
    df["Values0"] = df["Values"]
    indices0 = list()

    outlier0_idx = pd.Series( [False] * df.Values.shape[0])
    if len(df[df.Values == 0]) != 0:
        logging.warning('[pyCGM2]  zeros found for label [%s]'%(label))
        outlier0_idx =  df.Values == 0
        indices0 = df['Values'][df["Values"] == 0].index.to_list()
        df["Values"] = df.Values.replace(0, np.nan).fillna(method='ffill')

    if method == "median":
        df['rolling'] = df['Values'].rolling(window=window, center=True).median().fillna(method='bfill').fillna(method='ffill')
    elif method == "mean":
        df['rolling'] = df['Values'].rolling(window=window, center=True).mean().fillna(method='bfill').fillna(method='ffill')

    difference = np.abs(df['Values'] - df['rolling'])
    outlier_idx = difference > threshold

    indexes = []
    if np.any(outlier_idx.values) != False:
        indexes = df['Values'][outlier_idx].index.to_list()

    for i in range(0,outlier_idx.shape[0]):
        if outlier0_idx.values[i]:
            outlier_idx.values[i] = True

    allIndexes0 = indices0 + indexes
    allIndexes0.sort()

    allIndexes = []
    [allIndexes.append(x) for x in allIndexes0 if x not in allIndexes]

    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        df['Values0'].plot()
        df['rolling'].plot(style = "-r")
        df['Values0'][outlier_idx].plot(**kw)
        plt.title(label)
        plt.show()


    return allIndexes



def anomaly_ewm(values,window_size = 3,threshold=3,plot=True):
    df = pd.DataFrame({'Values': values})

    mean = df['Values'].ewm(window_size).mean()
    std = df['Values'].ewm(window_size).std()
    std[0] = 0 #the first value turns into NaN because of no data

    df['rolling'] = mean + std

    difference = np.abs(df['Values'] - df['rolling'])
    outlier_idx = difference > threshold

    success =  False
    indexes = None
    if np.any(outlier_idx.values) == False:
        logging.warning('[pyCGM2]  No outlier found')
        success =  True
    else:
        logging.warning('[pyCGM2] outliers found !!!')
        indexes = df['Values'][outlier_idx].index.to_list()


    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        df['Values'].plot()
        df['Values'][outlier_idx].plot(**kw)

        plt.show()

    return indexes


def anomaly_Isolation(values,plot=True):

    df = pd.DataFrame({'Values': values})

    from sklearn.ensemble import IsolationForest
    clf = IsolationForest( max_samples="auto", random_state = 1, contamination= 0.1)
    preds = clf.fit_predict(values.reshape((values.shape[0],1)))


    outlier_idx = pd.Series(preds).replace({ -1 : True, 1 : False })


    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        df['Values'].plot()
        df['Values'][outlier_idx].plot(**kw)
        plt.show()

    plt.show()
