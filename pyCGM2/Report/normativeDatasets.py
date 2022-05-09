# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Report
#APIDOC["Draft"]=False
#--end--

"""
Module deals with normative datasets.

Normative datasets are embedded in pyCGM2. they are placed in the folder *pyCGM2/Data* as a json file.

The construction of a `NormativeData` instance automatically populate its attribute
`data`, an intuitive dictionary with model ouput as key, reporting mean and
standard deviation as sub-key.

The `NormativeSTP` class replicates the process for the spatio-temporal excel file (normal_stp.xlsx) placed in
the folder *pyCGM2/Data/stp*
"""

import numpy as np
import pandas as pd
import pyCGM2

from pyCGM2.Utils import files


class NormativeData(object):
    """Normative dataset.

    The instance contruction populates the attribute `data` which reports normative data as a dictionary

    Args:
        filenameNoExt (str): filename of the targeted json file.
        modality (str): modality.

    """

    def __init__(self,filenameNoExt,modality):


        fullJsonDict = files.openFile(pyCGM2.NORMATIVE_DATABASE_PATH,filenameNoExt+".json")
        keys = list(fullJsonDict.keys())
        self._jsonDict = fullJsonDict[keys[0]][modality]

        self.data = dict()
        self._construct()

    def _construct(self):

        for key in self._jsonDict:

            self.data[key] = dict()
            self.data[key].update({"mean":None, "sd":None})

        for key in self._jsonDict:
            valueX_m2sd = np.array(self._jsonDict[key]["X"])[:,1]
            valueX_p2sd = np.array(self._jsonDict[key]["X"])[:,2]
            valueX_mean = 0.5 * (valueX_p2sd+valueX_m2sd)
            valueX_sd = valueX_p2sd-valueX_mean


            valueY_m2sd = np.array(self._jsonDict[key]["Y"])[:,1]
            valueY_p2sd = np.array(self._jsonDict[key]["Y"])[:,2]
            valueY_mean = 0.5 * (valueY_p2sd+valueY_m2sd)
            valueY_sd = valueY_p2sd-valueY_mean

            valueZ_m2sd = np.array(self._jsonDict[key]["Z"])[:,1]
            valueZ_p2sd = np.array(self._jsonDict[key]["Z"])[:,2]
            valueZ_mean = 0.5 * (valueZ_p2sd+valueZ_m2sd)
            valueZ_sd = valueZ_p2sd-valueZ_mean

            self.data[key]["mean"] = np.array([valueX_mean,valueY_mean,valueZ_mean]).T
            self.data[key]["sd"] = np.array([valueX_sd,valueY_sd,valueZ_sd]).T


class NormalSTP(object):
    """Normative spatio-temporal dataset.

    The instance contruction populates the attribute `data` which reports normative data as a dictionary
    """

    def __init__(self):

        self.m_filename = pyCGM2.NORMATIVE_DATABASE_PATH+"stp\\normal_stp.xlsx"
        self.data = dict()
        self._construct()

    def _construct(self):



        values =pd.read_excel(self.m_filename,sheet_name = "Nantes")

        for index, row in values.iterrows():
            self.data[row["Label"]]={}
            self.data[row["Label"]]["Mean"] = row["Mean"]
            self.data[row["Label"]]["Std"] = row["Std"]
