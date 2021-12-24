# -*- coding: utf-8 -*-
from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.Utils import prettyfier
from pyCGM2.Processing import progressionFrame
from pyCGM2.Model.Opensim import osimProcessing
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files
from bs4 import BeautifulSoup
import os
import numpy as np
import pyCGM2
LOGGER = pyCGM2.LOGGER

# pyCGM2
try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk
try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    import opensim


class opensimXmlInterface(object):
    def __init__(self, templateFullFilename, outFullFilename=None):

        if outFullFilename is None:
            outFullFilename = files.getFilename(templateFullFilename)

        self.m_out = outFullFilename
        self.m_soup = BeautifulSoup(open(templateFullFilename), "xml")

    def getSoup(self):
        return self.m_soup

    def set_one(self, labels, text):

        if isinstance(labels, str):
            labels = [labels]

        nitems = len(labels)
        if nitems == 1:
            self.m_soup.find(labels[0]).string = text

        else:
            count = 0
            for label in labels:
                if count == 0:
                    current = self.m_soup.find(label)
                else:
                    current = current.find(label)
                count += 1
            current.string = text

    # def set_one(self, label, text):
    #
    #     self.m_soup.find(label).string = text

    def set_many(self, label, text):
        items = self.m_soup.find_all(label)
        for item in items:
            item.string = text

    def set_many_inList(self, listName, label, text):
        items = self.m_soup.find_all(listName)
        for item in items:
            item.find(label).string = text

    def set_inList_fromAttr(self, listName, label, attrKey, attrValue, text):
        items = self.m_soup.find_all(listName)
        for item in items:
            if item.attrs[attrKey] == attrValue:
                item.find(label).string = text

    def update(self):
        ugly = self.m_soup.prettify()
        pretty = prettyfier.prettify_xml(ugly)
        with open(self.m_out, "w") as f:
            f.write(pretty)
            pass


class opensimInterfaceScalingFilter(object):
    def __init__(self, procedure):
        self.m_procedure = procedure

    def run(self):
        self.m_procedure.run()

    def getOsimName(self):
        return self.m_procedure.m_osimModel_name

    def getOsim(self):
        return self.m_procedure.m_osimModel


class opensimInterfaceInverseKinematicsFilter(object):
    def __init__(self, procedure):
        self.m_procedure = procedure

    def run(self):
        self.m_procedure.run()

    def getAcq(self):
        return self.m_procedure.m_acqMotionFinal

    def motToC3d(self, osimConverter):

        storageDataframe = opensimIO.OpensimDataFrame(
            self.m_procedure.m_DATA_PATH, self.m_procedure.m_dynamicFile+".mot")

        for jointIt in osimConverter["Angles"]:

            values = np.zeros(
                (self.m_procedure.m_acqMotionFinal.GetPointFrameNumber(), 3))

            osimlabel_X = osimConverter["Angles"][jointIt]["X"]
            osimlabel_Y = osimConverter["Angles"][jointIt]["Y"]
            osimlabel_Z = osimConverter["Angles"][jointIt]["Z"]

            serie_X = storageDataframe.getDataFrame()[osimlabel_X]
            serie_Y = storageDataframe.getDataFrame()[osimlabel_Y]
            serie_Z = storageDataframe.getDataFrame()[osimlabel_Z]

            values[:, 0] = [+1*x for x in serie_X.to_list()]
            values[:, 1] = [+1*x for x in serie_Y.to_list()]
            values[:, 2] = [+1*x for x in serie_Z.to_list()]

            btkTools.smartAppendPoint(self.m_procedure.m_acqMotionFinal, jointIt
                                      + "_osim", values, PointType=btk.btkPoint.Angle, desc="opensim angle")


class opensimInterfaceInverseDynamicsFilter(object):
    def __init__(self, procedure):
        self.m_procedure = procedure

    def run(self):
        self.m_procedure.run()

    def getAcq(self):
        return self.m_procedure.m_acq

    def stoToC3d(self, bodymass, osimConverter):

        storageDataframe = opensimIO.OpensimDataFrame(
            self.m_procedure.m_DATA_PATH+self.m_procedure._resultsDir+"\\",
            self.m_procedure.m_dynamicFile+"-"+self.m_procedure.m_modelVersion+"-inverse_dynamics.sto")

        for jointIt in osimConverter["Moments"]:

            values = np.zeros(
                (self.m_procedure.m_acq.GetPointFrameNumber(), 3))

            osimlabel_X = osimConverter["Moments"][jointIt]["X"]
            osimlabel_Y = osimConverter["Moments"][jointIt]["Y"]
            osimlabel_Z = osimConverter["Moments"][jointIt]["Z"]

            serie_X = storageDataframe.getDataFrame()[osimlabel_X]
            serie_Y = storageDataframe.getDataFrame()[osimlabel_Y]
            serie_Z = storageDataframe.getDataFrame()[osimlabel_Z]

            values[:, 0] = [+1*x*1000/bodymass for x in serie_X.to_list()]
            values[:, 1] = [+1*x*1000/bodymass for x in serie_Y.to_list()]
            values[:, 2] = [+1*x*1000/bodymass for x in serie_Z.to_list()]

            btkTools.smartAppendPoint(self.m_procedure.m_acq, jointIt+"_osim",
                                      values, PointType=btk.btkPoint.Moment, desc="opensim moment")


class opensimInterfaceStaticOptimizationFilter(object):
    def __init__(self, procedure):
        self.m_procedure = procedure

    def run(self):
        self.m_procedure.run()

    def getAcq(self):
        return self.m_procedure.m_acq

    def stoToC3d(self):
        storageDataframe = opensimIO.OpensimDataFrame(
            self.m_procedure.m_DATA_PATH+self.m_procedure._resultsDir+"\\",
            self.m_procedure.m_dynamicFile+"-"+self.m_procedure.m_modelVersion + "-analyses_StaticOptimization_force.sto")

        values = np.zeros(
            (self.m_procedure.m_acq.GetPointFrameNumber(), 3))

        for muscle in storageDataframe.m_dataframe.columns[1:]:
            serie = storageDataframe.getDataFrame()[muscle]
            values[:, 0] = serie.to_list()

            btkTools.smartAppendPoint(self.m_procedure.m_acq, muscle
                                      + "[StaticOptForce]", values, PointType=btk.btkPoint.Scalar, desc="StaticOptForce")

        storageDataframe = opensimIO.OpensimDataFrame(
            self.m_procedure.m_DATA_PATH+self.m_procedure._resultsDir+"\\",
            self.m_procedure.m_dynamicFile+"-"+self.m_procedure.m_modelVersion + "-analyses_StaticOptimization_activation.sto")

        values = np.zeros(
            (self.m_procedure.m_acq.GetPointFrameNumber(), 3))

        for muscle in storageDataframe.m_dataframe.columns[1:]:
            serie = storageDataframe.getDataFrame()[muscle]
            values[:, 0] = serie.to_list()

            btkTools.smartAppendPoint(self.m_procedure.m_acq, muscle
                                      + "[StaticOptActivation]", values, PointType=btk.btkPoint.Scalar, desc="StaticOptActivation")


class opensimInterfaceAnalysesFilter(object):
    def __init__(self, procedure):
        self.m_procedure = procedure

    def run(self):
        self.m_procedure.run()

    def getAcq(self):
        return self.m_procedure.m_acq

    def stoToC3d(self):

        # "-analyses_MuscleAnalysis_MuscleActuatorPower.sto"
        # "-analyses_MuscleAnalysis_NormalizedFiberLength.sto"
        # "-analyses_MuscleAnalysis_NormFiberVelocity.sto"
        # "-analyses_MuscleAnalysis_PassiveFiberForce.sto"
        # "-analyses_MuscleAnalysis_PassiveFiberForceAlongTendon.sto"
        # "-analyses_MuscleAnalysis_PennationAngle.sto"
        # "-analyses_MuscleAnalysis_PennationAngularVelocity.sto"
        # "-analyses_MuscleAnalysis_TendonForce.sto"
        # "-analyses_MuscleAnalysis_TendonLength.sto"
        # "-analyses_MuscleAnalysis_TendonPower.sto"
        # "-analyses_MuscleAnalysis_ActiveFiberForce.sto"
        # "-analyses_MuscleAnalysis_ActiveFiberForceAlongTendon.sto"
        # "-analyses_MuscleAnalysis_FiberActivePower.sto"
        # "-analyses_MuscleAnalysis_FiberForce.sto"
        # "-analyses_MuscleAnalysis_FiberLength.sto"
        # "-analyses_MuscleAnalysis_FiberPassivePower.sto"
        # "-analyses_MuscleAnalysis_FiberVelocity.sto"
        # "-analyses_MuscleAnalysis_Length.sto"

        storageDataframe = opensimIO.OpensimDataFrame(
            self.m_procedure.m_DATA_PATH+self.m_procedure._resultsDir+"\\",
            self.m_procedure.m_dynamicFile+"-"+self.m_procedure.m_modelVersion + "-analyses_MuscleAnalysis_Length.sto")

        values = np.zeros(
            (self.m_procedure.m_acq.GetPointFrameNumber(), 3))

        for muscle in storageDataframe.m_dataframe.columns[1:]:
            serie = storageDataframe.getDataFrame()[muscle]
            values[:, 0] = serie.to_list()

            btkTools.smartAppendPoint(self.m_procedure.m_acq, muscle
                                      + "[MuscleLength]", values, PointType=btk.btkPoint.Scalar, desc="MuscleLength")
