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
    """
    Represents a normative dataset loaded from a JSON file.

    The dataset is automatically populated into the `data` attribute, which is a dictionary with model output as keys.
    Each key reports mean and standard deviation values.

    Attributes:
        data (dict): Dictionary with model outputs as keys and mean, standard deviation as sub-keys.

    Args:
        filenameNoExt (str): Filename of the targeted JSON file, without extension.
        modality (str): Specific modality of the normative data to load.
    """

    def __init__(self,filenameNoExt:str,modality:str):
        """
        Initializes the NormativeData instance and populates the `data` attribute.
        """

        fullJsonDict = files.openFile(pyCGM2.NORMATIVE_DATABASE_PATH,filenameNoExt+".json")
        keys = list(fullJsonDict.keys())
        self._jsonDict = fullJsonDict[keys[0]][modality]

        self.data = {}
        self._construct()

    def _construct(self):
        """
        Constructs the normative data dictionary from the loaded JSON data.
        """
        for key in self._jsonDict:

            self.data[key] = {}
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
    """
    Represents normative spatio-temporal data loaded from an Excel file.

    The instance construction populates the attribute `data`, which reports the normative spatio-temporal data
    as a dictionary with labels as keys, and mean and standard deviation as sub-values.

    Attributes:
        data (dict): Dictionary with spatio-temporal labels as keys and mean, standard deviation as sub-values.
    """

    def __init__(self):
        """
        Initializes the NormalSTP instance and populates the `data` attribute from an Excel file.
        """

        self.m_filename = pyCGM2.NORMATIVE_DATABASE_PATH+"stp\\normal_stp.xlsx"
        self.data = {}
        self._construct()

    def _construct(self):
        """
        Constructs the normative spatio-temporal data dictionary from the loaded Excel data.
        """
        values =pd.read_excel(self.m_filename,sheet_name = "Nantes")

        for index, row in values.iterrows():
            self.data[row["Label"]]={}
            self.data[row["Label"]]["Mean"] = row["Mean"]
            self.data[row["Label"]]["Std"] = row["Std"]
