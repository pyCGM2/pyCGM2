# -*- coding: utf-8 -*-
"""Processing of several Analysis instances (e.g. normative data generation)."""
import numpy as np

from pyCGM2.Utils import files



class NormativeDataGeneratorFilter(object):
    """Filter generating normative data from several Analysis instances.

    Args:
        name (str): Name of the normative dataset.
        modality (str): Modality (e.g. walking condition).
        analyses (list): Analysis instances.
        kinematicLabelsDict (dict, optional): Kinematic labels by context (``Left``, ``Right``).
        kineticLabelsDict (dict, optional): Kinetic labels by context.
        pointlabelSuffix (str, optional): Suffix of the point labels.
        metadata (dict, optional): Metadata of the dataset.
    """
    def __init__(self,
                name,modality,
                analyses,
                kinematicLabelsDict=None ,
                kineticLabelsDict =None,
                pointlabelSuffix = None,
                metadata=None):

        self.m_name = name
        self.m_modality = modality
        self.m_analyses = analyses
        self.m_kinematicLabelsDict = kinematicLabelsDict
        self.m_kineticLabelsDict = kineticLabelsDict
        self.m_pointlabelSuffix = pointlabelSuffix
        self.m_metadata =metadata

        self.__normativeData = None
        self.__normalCorridor =  {}
        self.__allCycles = None

    def getNormativeData(self):
        """Return the normative data.

        Returns:
            dict: Normative data (``{name: {modality: data}, "metadata": metadata}``).
        """
        return self.__normativeData

    def getNormativeCorridor(self):
        """Return the normative corridor (mean - sd, mean + sd) of each label and axis.

        Returns:
            dict: Corridors by label and axis.
        """
        return self.__normalCorridor

    def getAllTraces(self):
        """Return all cycles gathered by label.

        Returns:
            dict: Cycles by label.
        """
        return self.__allCycles


    def generate(self):
        """Gather the cycles of all analyses and compute the normative data and corridors."""

        cyclesBySide={}
        # cycles : ncycle of array(101,3)
        for label in self.m_kinematicLabelsDict["Left"]:
            cyclesBySide[label]=[]
            for analysisIt in self.m_analyses:
                cyclesBySide[label] = cyclesBySide[label]+ analysisIt.kinematicStats.data[label,"Left"]["values"]

        for label in self.m_kinematicLabelsDict["Right"]:
            cyclesBySide[label]=[]
            for analysisIt in self.m_analyses:
                cyclesBySide[label] = cyclesBySide[label]+ analysisIt.kinematicStats.data[label,"Right"]["values"]

        # cycles : ncycle of array(101,3)
        for label in self.m_kineticLabelsDict["Left"]:
            cyclesBySide[label]=[]
            for analysisIt in self.m_analyses:
                cyclesBySide[label] = cyclesBySide[label]+ analysisIt.kineticStats.data[label,"Left"]["values"]

        for label in self.m_kineticLabelsDict["Right"]:
            cyclesBySide[label]=[]
            for analysisIt in self.m_analyses:
                cyclesBySide[label] = cyclesBySide[label]+ analysisIt.kineticStats.data[label,"Right"]["values"]

        # gather both sides
        cycles = {}
        for key in cyclesBySide.keys():
            new_key = key[1:]
            if new_key in cycles.keys():
                cycles[new_key] = cycles[new_key] + cyclesBySide[key]
            else:
                cycles[new_key] = cyclesBySide[key]

        self.__allCycles = cycles

        out={}

        for key in cycles.keys():
            out[key] = {}
            out[key]["X"] = []
            out[key]["Y"] = []
            out[key]["Z"] = []

            self.__normalCorridor[key] = {}
            self.__normalCorridor[key]["X"] = []
            self.__normalCorridor[key]["Y"] = []
            self.__normalCorridor[key]["Z"] = []

            X =np.zeros((101,len(cycles[key])))
            Y =np.zeros((101,len(cycles[key])))
            Z =np.zeros((101,len(cycles[key])))

            i=0
            for cycleIt in cycles[key]:
                X[:,i] = cycleIt[:,0]
                Y[:,i] = cycleIt[:,1]
                Z[:,i] = cycleIt[:,2]
                i+=1

            meanX_minusSd = np.mean(X,axis=1) - np.std(X,axis=1)
            meanX_plusSd = np.mean(X,axis=1) + np.std(X,axis=1)

            meanY_minusSd = np.mean(Y,axis=1) - np.std(Y,axis=1)
            meanY_plusSd = np.mean(Y,axis=1) + np.std(Y,axis=1)

            meanZ_minusSd = np.mean(Z,axis=1) - np.std(Z,axis=1)
            meanZ_plusSd = np.mean(Z,axis=1) + np.std(Z,axis=1)

            self.__normalCorridor[key]["X"] = [meanX_minusSd, meanX_plusSd]
            self.__normalCorridor[key]["Y"] = [meanY_minusSd, meanY_plusSd]
            self.__normalCorridor[key]["Z"] = [meanZ_minusSd, meanZ_plusSd]

            for j in range(0,101):
                out[key]["X"].append([j,meanX_minusSd[j], meanX_plusSd[j]])
                out[key]["Y"].append([j,meanY_minusSd[j], meanY_plusSd[j]])
                out[key]["Z"].append([j,meanZ_minusSd[j], meanZ_plusSd[j]])

        self.__normativeData =  {self.m_name :
                                    {self.m_modality :  out},
                                    "metadata": self.m_metadata }

    def save(self,DATA_PATH,filename):
        """Save the normative data as a json file.

        Args:
            DATA_PATH (str): Folder path.
            filename (str): Filename.

        Raises:
            Exception: If the normative data have not been generated.
        """
        if self.__normativeData is not None:
            files.saveJson(DATA_PATH, filename, self.__normativeData)
        else:
            raise Exception("[pyCGM2] Normative data is empty")
