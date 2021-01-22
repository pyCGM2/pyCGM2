# -*- coding: utf-8 -*-
import numpy as np

from pyCGM2.Utils import files



class NormativeDataGeneratorFilter(object):
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
        self.__normalCorridor =  dict()
        self.__allCycles = None

    def getNormativeData(self):
        return self.__normativeData

    def getNormativeCorridor(self):
        return self.__normalCorridor

    def getAllTraces(self):
        return self.__allCycles


    def generate(self):

        cyclesBySide=dict()
        # cycles : ncycle of array(101,3)
        for label in self.m_kinematicLabelsDict["Left"]:
            cyclesBySide[label]=list()
            for analysisIt in self.m_analyses:
                cyclesBySide[label] = cyclesBySide[label]+ analysisIt.kinematicStats.data[label,"Left"]["values"]

        for label in self.m_kinematicLabelsDict["Right"]:
            cyclesBySide[label]=list()
            for analysisIt in self.m_analyses:
                cyclesBySide[label] = cyclesBySide[label]+ analysisIt.kinematicStats.data[label,"Right"]["values"]

        # cycles : ncycle of array(101,3)
        for label in self.m_kineticLabelsDict["Left"]:
            cyclesBySide[label]=list()
            for analysisIt in self.m_analyses:
                cyclesBySide[label] = cyclesBySide[label]+ analysisIt.kineticStats.data[label,"Left"]["values"]

        for label in self.m_kineticLabelsDict["Right"]:
            cyclesBySide[label]=list()
            for analysisIt in self.m_analyses:
                cyclesBySide[label] = cyclesBySide[label]+ analysisIt.kineticStats.data[label,"Right"]["values"]

        # gather both sides
        cycles = dict()
        for key in cyclesBySide.keys():
            new_key = key[1:]
            if new_key in cycles.keys():
                cycles[new_key] = cycles[new_key] + cyclesBySide[key]
            else:
                cycles[new_key] = cyclesBySide[key]

        self.__allCycles = cycles

        out=dict()

        for key in cycles.keys():
            out[key] = dict()
            out[key]["X"] = list()
            out[key]["Y"] = list()
            out[key]["Z"] = list()

            self.__normalCorridor[key] = dict()
            self.__normalCorridor[key]["X"] = list()
            self.__normalCorridor[key]["Y"] = list()
            self.__normalCorridor[key]["Z"] = list()

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
        if self.__normativeData is not None:
            files.saveJson(DATA_PATH, filename, self.__normativeData)
        else:
            raise Exception("[pyCGM2] Normative data is empty")
