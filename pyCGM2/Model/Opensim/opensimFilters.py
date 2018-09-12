# -*- coding: utf-8 -*-
import os
import numpy as np
import logging

from pyCGM2 import btk

# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.Opensim import osimProcessing



# ---- PROCEDURES -----

class GeneralOpensimCalibrationProcedure(object):
    """
        General procedure for calibrating an opensim model
    """
    def __init__(self):

        self.markers=dict()
        self.geometry=dict()

    def setMarkers(self, segmentLabel, trackingMarkers):
        """
            Add tracking markers to a pycgm2-model segment

            :Parameters:
                - `segmentLabel` (str) - segment label of a pyCGM2 model
                - `trackingMarkers` (list) - list of traking markers

        """
        self.markers[segmentLabel] = trackingMarkers

    def setGeometry(self, opensimJointLabel, jointLabel, proximalSegmentLabel, distalSegmentLabel ):
        """
            Design the opensim model geometry

            :Parameters:
                - `opensimJointLabel` (str) - joint label of the opensim model (see inside osim file)
                - `jointLabel` (str) - joint label of a pyCGM2 model
                - `proximalSegmentLabel` (str) - proximal segment label of a pyCGM2 model
                - `distalSegmentLabel` (str) - distal segment label of a pyCGM2 model

        """

        self.geometry[opensimJointLabel] = {"joint label":jointLabel, "proximal segment label":proximalSegmentLabel, "distal segment label":distalSegmentLabel}


class GeneralOpensimFittingProcedure(object):
    """
        General procedure for Fitting a motion on the osim file
    """
    def __init__(self):

        self.ikTags=dict()

    def setIkTags(self,markerLabel,weight):
        # TODO : remove this method ( check if used before)
        self.ikTags[markerLabel]=weight


    def setMarkerWeight(self,markerLabel,weight):
        """
            Set weigth of a tracking marker

            :Parameters:
                - `markerLabel` (str) - marker label
                - `weight` (double) - joint label of a pyCGM2 model

        """
        self.ikTags[markerLabel] = weight
        logging.warning( "marker (%s) weight altered. New weight = %d" %(markerLabel,weight))



class CgmOpensimCalibrationProcedures(object):
    """ Model-embedded Procedure for calibrating an opensim model """
    def __init__(self,model):
        """
            :Parameters:
                - `model` (Model) - instance of a pyCGM2 Model

        """
        self.model=model
        self.markers=dict()
        self.geometry=dict()

        self.__setMarkers()
        self.__setGeometry()

    def __setMarkers(self):
        self.markers=self.model.opensimTrackingMarkers()

    def __setGeometry(self):
        self.geometry=self.model.opensimGeometry()



class CgmOpensimFittingProcedure(object):
    """ Model-embedded Procedure for fitting an opensim model """

    def __init__(self,model,expertMode = False):
        """
            :Parameters:
           - `model` (Model) - instance of Model

        """
        self.model=model
        self.ikTags=dict()#
        self.m_expertMode = expertMode

        self.__setIkTags()


    def __setIkTags(self):
        self.ikTags=self.model.opensimIkTask(expert=self.m_expertMode)

    def updateMarkerWeight(self,markerLabel,weight):
        """
            Update weigth of a tracking marker

            :Parameters:
                - `markerLabel` (str) - marker label
                - `weight` (double) - joint label of a pyCGM2 model

        """

        if self.ikTags[markerLabel] != weight:
            self.ikTags[markerLabel] = weight
            logging.warning( "marker (%s) weight altered. New weight = %d" %(markerLabel,weight))


# ---- FILTERS -----

class opensimCalibrationFilter(object):

    def __init__(self,osimFile,model,procedure,dataDir):
        """
            :Parameters:
                - `osimFile` (str) - full filename of the osim file
                - `model` (pyCGM2.Model) - joint label of a pyCGM2 model
                - `procedure` (pyCGM2.opensim.procedure) - calibration procedure

        """
        self.m_osimFile = osimFile
        self.m_model = model
        self.m_procedure = procedure
        self.m_toMeter = 1000.0

        self._osimModel = osimProcessing.opensimModel(osimFile,model)
        self.opensimOutputDir = dataDir if dataDir[-1:] =="\\" else str(dataDir+"\\")

    def addMarkerSet(self,markerSetFile):
        """
            Add a marker set file
            :Parameters:
                - `markerSetFile` (str) - full filename of the opensim marker set file
        """
        self._osimModel.addMarkerSet(markerSetFile)


    def build(self,exportOsim=True):
        """
            Build the calibrated opensim model
        """


        # IMPORTANT update marker from procedure ( need markers in osim)
        for segName in self.m_procedure.markers.keys():
            for marker in self.m_procedure.markers[segName]:
                self._osimModel.updateMarkerInMarkerSet( marker, modelSegmentLabel = segName, rotation_osim_model=osimProcessing.R_OSIM_CGM[segName], toMeter=self.m_toMeter)

        # set joint centre geometry
        for openSimJointLabel in self.m_procedure.geometry.keys():
            self._osimModel.setOsimJoinCentres(osimProcessing.R_OSIM_CGM,
                                                openSimJointLabel,
                                                self.m_procedure.geometry[openSimJointLabel]["proximal segment label"],
                                                self.m_procedure.geometry[openSimJointLabel]["distal segment label"],
                                                self.m_procedure.geometry[openSimJointLabel]["joint label"],
                                                toMeter = self.m_toMeter)

        self._osimModel.m_model.setName("pyCGM2 calibrated model")

        if exportOsim:
            if os.path.isfile(self.opensimOutputDir +"scaledModel.osim"):
                os.remove(self.opensimOutputDir +"scaledModel.osim")
            self.exportXml("scaledModel.osim", path = self.opensimOutputDir)

        return self._osimModel



    def exportXml(self,filename, path=None):
        """
            Export the calibrated model as an osim file
            :Parameters:
                - `filename` (str) - filename of the calibrated osim file


        """
        filename = filename if path is None else str(path+filename)
        self._osimModel.m_model.printToXML(filename)


class opensimFittingFilter(object):
    def __init__(self,ikToolFile,calibratedOsim, ikTagProcedure,dataDir,accuracy = 1e-8 ):
        """
            :Parameters:
                - `ikToolFile` (str) - full filename of the opensim inverse kinematic tool file
                - `calibratedOsim` (osim file) - calibrated opensim file
                - `ikTagProcedure` (pyCGM2.opensim.procedure) - fitting procedure
                - `dataDir` (str) - path to opensim result directory
                - `accuracy` (double) - accuracy of the kinematic fitter

        """
        self.m_calibratedOsim = calibratedOsim
        self.m_ikToolFile = ikToolFile
        self.m_procedure = ikTagProcedure

        self.accuracy = accuracy


        self.opensimOutputDir = dataDir if dataDir[-1:] =="\\" else str(dataDir+"\\")

        self._osimIK = osimProcessing.opensimKinematicFitting(self.m_calibratedOsim.m_model,self.m_ikToolFile)
        self._osimIK.setAccuracy(self.accuracy)
        self._osimIK.setResultsDirectory(self.opensimOutputDir)

    def run(self,acqMotion, acqMotionFilename,exportSetUp=True):
        """
            Run kinematic fitting
            :Parameters:
                - `acqMotion` (btk.Acquisition) - acquisition of a motion trial
                - `acqMotionFilename` (filename) - filename of the motion trial

        """


        acqMotion_forIK = btk.btkAcquisition.Clone(acqMotion)
        progressionAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqMotion,["LASI","RASI","RPSI","LPSI"])

        # --- ikTasks
        #  UPDATE method - ik tags ( need task in the initial iktools)
        for markerIt in self.m_procedure.ikTags.keys():
            self._osimIK.updateIKTask(markerIt,self.m_procedure.ikTags[markerIt])


        # --- configuration and run IK
        if os.path.isfile(self.opensimOutputDir +"ik_model_marker_locations.sto"):
            os.remove(self.opensimOutputDir +"ik_model_marker_locations.sto")

        R_LAB_OSIM = osimProcessing.setGlobalTransormation_lab_osim(progressionAxis,forwardProgression)
        self._osimIK.config(R_LAB_OSIM, acqMotion_forIK, acqMotionFilename )

        if exportSetUp:
            if os.path.isfile(self.opensimOutputDir +"scaledModel-ikSetUp.xml"):
                os.remove(self.opensimOutputDir +"scaledModel-ikSetUp.xml")
            self.exportXml("scaledModel-ikSetUp.xml",path = self.opensimOutputDir)


        self._osimIK.run()

        # --- gernerate acq with rigid markers
        acqMotionFinal = btk.btkAcquisition.Clone(acqMotion)
        for marker in self.m_procedure.ikTags.keys():
            if self.m_procedure.ikTags[marker] != 0:
                values =osimProcessing.sto2pointValues(self.opensimOutputDir + "ik_model_marker_locations.sto",marker,R_LAB_OSIM)
                lenOsim  = len(values)

                lenc3d  = acqMotion.GetPoint(marker).GetFrameNumber()
                if lenOsim < lenc3d:
                    logging.warning(" size osim (%i) inferior to c3d (%i)" % (lenOsim,lenc3d))
                    values2 = np.zeros((lenc3d,3))
                    values2[0:lenOsim,:]=values
                    values2[lenOsim:lenc3d,:]=acqMotion.GetPoint(marker).GetValues()[lenOsim:lenc3d,:]

                    btkTools.smartAppendPoint(acqMotionFinal,marker+"_m", acqMotionFinal.GetPoint(marker).GetValues(), desc= "measured" ) # new acq with marker overwrited
                    btkTools.smartAppendPoint(acqMotionFinal,marker, values2, desc= "kinematic fitting" ) # new acq with marker overwrited
                else:
                    btkTools.smartAppendPoint(acqMotionFinal,marker+"_m", acqMotionFinal.GetPoint(marker).GetValues(), desc= "measured" ) # measured marker suffix with _m
                    btkTools.smartAppendPoint(acqMotionFinal,marker, values, desc= "kinematic fitting" ) # new acq with marker overwrited

        return acqMotionFinal


    def exportXml(self,filename, path=None):
        """
            Export the generated inverse kinematic Tool
            :Parameters:
                - `filename` (str) - filename of the generated inverse kinematic Tool


        """
        filename = filename if path is None else str(path+filename)
        self._osimIK.m_ikTool.printToXML(filename)
