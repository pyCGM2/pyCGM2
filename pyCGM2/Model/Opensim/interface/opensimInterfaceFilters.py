# -*- coding: utf-8 -*-
from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.Utils import prettyfier
from pyCGM2.Tools import  btkTools,opensimTools
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
        self.m_acq = None

    def run(self):
        self.m_procedure.run()
        self.m_acq = self.m_procedure.m_acq0

    def getAcq(self):
        return self.m_acq

    def pushFittedMarkersIntoAcquisition(self):
        marker_location_filename = self.m_procedure.m_DATA_PATH + self.m_procedure.m_resultsDir+"\\"+ self.m_procedure.m_dynamicFile+"_ik_model_marker_locations.sto"
        if os.path.isfile(marker_location_filename):
            os.remove(marker_location_filename)
        os.rename(self.m_procedure.m_DATA_PATH + self.m_procedure.m_resultsDir+ "\\_ik_model_marker_locations.sto",
                    marker_location_filename)

        marker_errors_filename = self.m_procedure.m_DATA_PATH + self.m_procedure.m_resultsDir+"\\"+ self.m_procedure.m_dynamicFile+"_ik_marker_errors.sto"
        if os.path.isfile(marker_errors_filename):
            os.remove(marker_errors_filename)
        os.rename(self.m_procedure.m_DATA_PATH + self.m_procedure.m_resultsDir+"\\_ik_marker_errors.sto",
                 marker_errors_filename)

        acqMotionFinal = btk.btkAcquisition.Clone(self.m_procedure.m_acq0)

        # TODO : worl with storage datframe instead of the opensim sorage instance

        storageDataframe = opensimIO.OpensimDataFrame(
            self.m_procedure.m_DATA_PATH+self.m_procedure.m_resultsDir+"\\", self.m_procedure.m_dynamicFile+"_ik_model_marker_locations.sto")

        storageObject = opensim.Storage(self.m_procedure.m_DATA_PATH + self.m_procedure.m_resultsDir+"\\"+self.m_procedure.m_dynamicFile +"_ik_model_marker_locations.sto")
        for marker in self.m_procedure.m_weights.keys():
            if self.m_procedure.m_weights[marker] != 0:
                values =opensimTools.sto2pointValues(storageObject,marker,self.m_procedure.m_R_LAB_OSIM)
                btkTools.smartAppendPoint(acqMotionFinal,marker+"_m", acqMotionFinal.GetPoint(marker).GetValues(), desc= "measured" )
                modelled = acqMotionFinal.GetPoint(marker).GetValues()
                ff = acqMotionFinal.GetFirstFrame()

                modelled[self.m_procedure.m_frameRange[0]-ff:self.m_procedure.m_frameRange[1]-ff+1,:] = values
                btkTools.smartAppendPoint(acqMotionFinal,marker, modelled, desc= "kinematic fitting" ) # new acq with marker overwrited


        self.m_acq = acqMotionFinal


    def pushMotToAcq(self, osimConverter):

        storageDataframe = opensimIO.OpensimDataFrame(
            self.m_procedure.m_DATA_PATH+self.m_procedure.m_resultsDir+"\\", self.m_procedure.m_dynamicFile+".mot")

        for jointIt in osimConverter["Angles"]:

            values = np.zeros(
                (self.m_acq.GetPointFrameNumber(), 3))

            osimlabel_X = osimConverter["Angles"][jointIt]["X"]
            osimlabel_Y = osimConverter["Angles"][jointIt]["Y"]
            osimlabel_Z = osimConverter["Angles"][jointIt]["Z"]

            serie_X = storageDataframe.getDataFrame()[osimlabel_X]
            serie_Y = storageDataframe.getDataFrame()[osimlabel_Y]
            serie_Z = storageDataframe.getDataFrame()[osimlabel_Z]

            values[:, 0] = [+1*x for x in serie_X.to_list()]
            values[:, 1] = [+1*x for x in serie_Y.to_list()]
            values[:, 2] = [+1*x for x in serie_Z.to_list()]

            btkTools.smartAppendPoint(self.m_acq, jointIt
                                      + "_osim", values, PointType="Angle", desc="opensim angle")


class opensimInterfaceInverseDynamicsFilter(object):
    def __init__(self, procedure):
        self.m_procedure = procedure

    def run(self):
        self.m_procedure.run()

    def getAcq(self):
        return self.m_procedure.m_acq

    def pushStoToAcq(self, bodymass, osimConverter):

        if self.m_procedure.m_resultsDir == "":
            path = self.m_procedure.m_DATA_PATH
        else:
            path = self.m_procedure.m_DATA_PATH+self.m_procedure.m_resultsDir+"\\"


        storageDataframe = opensimIO.OpensimDataFrame(path,
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
                                      values, PointType="Moment", desc="opensim moment")


class opensimInterfaceStaticOptimizationFilter(object):
    def __init__(self, procedure):
        self.m_procedure = procedure

    def run(self):
        self.m_procedure.run()

    def getAcq(self):
        return self.m_procedure.m_acq

    def pushStoToAcq(self):
        storageDataframe = opensimIO.OpensimDataFrame(
            self.m_procedure.m_DATA_PATH+self.m_procedure.m_resultsDir+"\\",
            self.m_procedure.m_dynamicFile+"-"+self.m_procedure.m_modelVersion + "-analyses_StaticOptimization_force.sto")

        values = np.zeros(
            (self.m_procedure.m_acq.GetPointFrameNumber(), 3))

        for muscle in storageDataframe.m_dataframe.columns[1:]:
            serie = storageDataframe.getDataFrame()[muscle]
            values[:, 0] = serie.to_list()

            btkTools.smartAppendPoint(self.m_procedure.m_acq, muscle
                                      + "[StaticOptForce]", values, PointType="Scalar", desc="StaticOptForce")

        storageDataframe = opensimIO.OpensimDataFrame(
            self.m_procedure.m_DATA_PATH+self.m_procedure.m_resultsDir+"\\",
            self.m_procedure.m_dynamicFile+"-"+self.m_procedure.m_modelVersion + "-analyses_StaticOptimization_activation.sto")

        values = np.zeros(
            (self.m_procedure.m_acq.GetPointFrameNumber(), 3))

        for muscle in storageDataframe.m_dataframe.columns[1:]:
            serie = storageDataframe.getDataFrame()[muscle]
            values[:, 0] = serie.to_list()

            btkTools.smartAppendPoint(self.m_procedure.m_acq, muscle
                                      + "[StaticOptActivation]", values, PointType="Scalar", desc="StaticOptActivation")


class opensimInterfaceAnalysesFilter(object):
    def __init__(self, procedure):
        self.m_procedure = procedure

    def run(self):
        self.m_procedure.run()

    def getAcq(self):
        return self.m_procedure.m_acq

    def pushStoToAcq(self):

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
            self.m_procedure.m_DATA_PATH+self.m_procedure.m_resultsDir+"\\",
            self.m_procedure.m_dynamicFile+"-"+self.m_procedure.m_modelVersion + "-analyses_MuscleAnalysis_Length.sto")

        values = np.zeros(
            (self.m_procedure.m_acq.GetPointFrameNumber(), 3))

        for muscle in storageDataframe.m_dataframe.columns[1:]:
            serie = storageDataframe.getDataFrame()[muscle]
            values[:, 0] = serie.to_list()

            btkTools.smartAppendPoint(self.m_procedure.m_acq, muscle
                                      + "[MuscleLength]", values*1000, PointType="Scalar", desc="MuscleLength")
