# coding: utf-8
# pytest -s --disable-pytest-warnings --log-cli-level=INFO  test_forcePlateReconstruction.py

import pytest

import pandas as pd
import btk
import numpy as np
from streamlit import dataframe
from pyCGM2.Tools import btkTools
import pyCGM2

import matplotlib.pyplot as plt

from pyCGM2.Model.Models import singleBody


class Test_bertec:
    def test_basket_csv(self):
        
        dataPath = DATA_PATH =  pyCGM2.TEST_DATA_PATH + "LowLevel\\ForcePlate\\ForcePlateReader\\"
        filename = "bertec-yannisDataSample.csv"
        dataframe =   pd.read_csv(dataPath+filename,sep=",",skiprows=4)

        dataframe["Time (s)"] = dataframe["Time (s)"] - dataframe["Time (s)"].iloc[0]
        dataframe_100Hz = dataframe.iloc[::10]



class Test_bioware:
    def test_basket_txt(self):

        dataPath = pyCGM2.TEST_DATA_PATH + "LowLevel\\ForcePlate\\ForcePlateReader\\"
        filename = "BASKET_001_01.txt"
        dataframe = pd.read_csv(dataPath+filename, sep="\t", skiprows=17)
        dataframe = dataframe.drop(dataframe.index[0]).reset_index(drop=True)




