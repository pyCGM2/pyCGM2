# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Model/Opensim
#APIDOC["Draft"]=False
#--end--

"""
This module contains  Filters and procedures used as interface for working with
the opensim API
"""

import os
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
from bs4 import BeautifulSoup

# pyCGM2
try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    try:
        import btk
    except:
        LOGGER.logger.error("[pyCGM2] btk not found on your system. install it for working with the API")

from pyCGM2.Tools import  btkTools
from pyCGM2.Model.Opensim import osimProcessing
from pyCGM2.Processing.ProgressionFrame import progressionFrameFilters
from pyCGM2.Processing.ProgressionFrame import progressionFrameProcedures

try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    try:
        import opensim
    except:
        LOGGER.logger.error("[pyCGM2] : opensim not find on your system. Install it for working with the API")

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

            Args:
                segmentLabel (str): segment label of a pyCGM2 model
                trackingMarkers (list):  traking markers

        """
        self.markers[segmentLabel] = trackingMarkers

    def setGeometry(self, opensimJointLabel, jointLabel, proximalSegmentLabel, distalSegmentLabel ):
        """
            Design the opensim model geometry

            Args:
                opensimJointLabel (str): joint label of the opensim model (see inside osim file)
                jointLabel (str): joint label of the pyCGM2 model
                proximalSegmentLabel (str): proximal segment label of LTHI_perturbTraj pyCGM2 model
                distalSegmentLabel (str): distal segment label of the pyCGM2 model

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

            Args:
                markerLabel (str): marker label
                weight (double): joint label of a pyCGM2 model

        """
        self.ikTags[markerLabel] = weight
        LOGGER.logger.info( "marker (%s) weight altered. New weight = %d" %(markerLabel,weight))



class CgmOpensimCalibrationProcedures(object):
    """ Model-embedded Procedure for calibrating an opensim model """
    def __init__(self,model):
        """
            Args:
                model (Model): instance of a pyCGM2 Model

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

    def __init__(self,model):
        """
            Args:
           model (Model): instance of Model

        """
        self.model=model
        self.ikTags=dict()#

        self.__setIkTags()


    def __setIkTags(self):
        self.ikTags=self.model.opensimIkTask()

    def updateMarkerWeight(self,markerLabel,weight):
        """
            Update weigth of a tracking marker

            Args:
                markerLabel (str): marker label
                weight (double): joint label of a pyCGM2 model

        """

        if self.ikTags[markerLabel] != weight:
            self.ikTags[markerLabel] = weight
            LOGGER.logger.info( "marker (%s) weight altered. New weight = %d" %(markerLabel,weight))


# ---- FILTERS -----

class opensimCalibrationFilter(object):

    def __init__(self,osimFile,model,procedure,dataDir):
        """
            Args:
                osimFile (str): full filename of the osim file
                model (pyCGM2.Model): joint label of a pyCGM2 model
                procedure (pyCGM2.opensim.procedure): calibration procedure

        """
        self.m_osimFile = osimFile
        self.m_model = model
        self.m_procedure = procedure
        self.m_toMeter = 1000.0

        self._osimModel = osimProcessing.opensimModel(osimFile,model)
        self.opensimOutputDir = dataDir if dataDir[-1:] =="\\" else dataDir+"\\"

    def addMarkerSet(self,markerSetFile):
        """
            Add a marker set file
            Args:
                markerSetFile (str): full filename of the opensim marker set file
        """
        self._osimModel.addMarkerSet(markerSetFile)


    def build(self,exportOsim=False):
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

        self._osimModel.m_model.setName("pyCGM2CalibratedModel")

        if exportOsim:
            if os.path.isfile(self.opensimOutputDir +"scaledModel.osim"):
                os.remove(self.opensimOutputDir +"scaledModel.osim")
            self.exportXml("scaledModel.osim", path = self.opensimOutputDir)

        return self._osimModel



    def exportXml(self,filename, path=None):
        """
            Export the calibrated model as an osim file
            Args:
                filename (str): filename of the calibrated osim file


        """
        filename = filename if path is None else path+filename
        self._osimModel.m_model.printToXML(filename)


class opensimFittingFilter(object):
    def __init__(self,ikToolFile,calibratedOsim, ikTagProcedure,dataDir,acqMotion, accuracy = 1e-8 ):
        """
            Args:
                ikToolFile (str): full filename of the opensim inverse kinematic tool file
                calibratedOsim (osim file): calibrated opensim file
                ikTagProcedure (pyCGM2.opensim.procedure): fitting procedure
                dataDir (str): path to opensim result directory
                accuracy (double): accuracy of the kinematic fitter

        """
        self.m_calibratedOsim = calibratedOsim
        self.m_ikToolFile = ikToolFile
        self.m_ikSoup = BeautifulSoup(open(self.m_ikToolFile), "xml")
        self.m_procedure = ikTagProcedure

        self.accuracy = accuracy
        self.m_acqMotion = acqMotion

        self.opensimOutputDir = dataDir if dataDir[-1:] =="\\" else dataDir+"\\"

        self.setAccuracy(self.accuracy)
        self.setResultsDirectory(self.opensimOutputDir)
        self.setTimeRange(acqMotion)

        self.updateConfig()

    def setAccuracy(self,value):
        self.accuracy = value
        self.m_ikSoup.accuracy.string = str(value)

        self.updateConfig()

    def setResultsDirectory(self,path):
        self.m_ikSoup.results_directory.string = path.replace("\\","/")

    def setTimeRange(self,acq,beginFrame=None,lastFrame=None):
        ff = acq.GetFirstFrame()
        freq = acq.GetPointFrequency()
        beginTime = 0.0 if beginFrame is None else (beginFrame-ff)/freq
        endTime = (acq.GetLastFrame() - ff)/freq  if lastFrame is  None else (lastFrame-ff)/freq
        self.m_ikSoup.time_range.string = str(beginTime) + " " + str(endTime)

        self.m_frameRange = [int((beginTime*freq)+ff),int((endTime*freq)+ff)]

        self.updateConfig()

    def updateConfig(self):
        newIkFile = self.opensimOutputDir + self.m_ikToolFile[self.m_ikToolFile.rfind("\\")+1:]
        with open(newIkFile, "w") as f:
            f.write(self.m_ikSoup.prettify())

        self._osimIK = osimProcessing.opensimKinematicFitting(self.m_calibratedOsim.m_model,newIkFile)



    def run(self, acqMotionFilename,exportSetUp=False,**kwargs):
        """
            Run kinematic fitting
            Args:
                acqMotion (btk.Acquisition): acquisition of a motion trial
                acqMotionFilename (filename): filename of the motion trial

        """


        acqMotion_forIK = btk.btkAcquisition.Clone(self.m_acqMotion)

        if "progressionAxis" in kwargs:
            progressionAxis = kwargs["progressionAxis"]
            if "forwardProgression" in kwargs:
                forwardProgression = kwargs["forwardProgression"]
            else:
                forwardProgression = True

        else:
            pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure()
            pff = progressionFrameFilters.ProgressionFrameFilter(self.m_acqMotion,pfp)
            pff.compute()
            progressionAxis = pff.outputs["progressionAxis"]
            forwardProgression = pff.outputs["forwardProgression"]

        # --- ikTasks
        #  UPDATE method - ik tags ( need task in the initial iktools)
        for markerIt in self.m_procedure.ikTags.keys():
            self._osimIK.updateIKTask(markerIt,self.m_procedure.ikTags[markerIt])


        # --- configuration and run IK
        if os.path.isfile(self.opensimOutputDir +"_ik_model_marker_locations.sto"):
            os.remove(self.opensimOutputDir +"_ik_model_marker_locations.sto")

        R_LAB_OSIM = osimProcessing.setGlobalTransormation_lab_osim(progressionAxis,forwardProgression)
        self._osimIK.config(R_LAB_OSIM, acqMotion_forIK, acqMotionFilename )

        if exportSetUp:
            if os.path.isfile(self.opensimOutputDir +"scaledModel-ikSetUp.xml"):
                os.remove(self.opensimOutputDir +"scaledModel-ikSetUp.xml")
            self.exportXml("scaledModel-ikSetUp.xml",path = self.opensimOutputDir)

        self._osimIK.run()

        # --- gernerate acq with rigid markers
        acqMotionFinal = btk.btkAcquisition.Clone(self.m_acqMotion)
        storageObject = opensim.Storage(self.opensimOutputDir + "_ik_model_marker_locations.sto")
        for marker in self.m_procedure.ikTags.keys():
            if self.m_procedure.ikTags[marker] != 0:
                values =osimProcessing.sto2pointValues(storageObject,marker,R_LAB_OSIM)
                btkTools.smartAppendPoint(acqMotionFinal,marker+"_m", acqMotionFinal.GetPoint(marker).GetValues(), desc= "measured" )

                modelled = acqMotionFinal.GetPoint(marker).GetValues()
                residuals = acqMotionFinal.GetPoint(marker).GetResiduals()

                ff = acqMotionFinal.GetFirstFrame()
                modelled[self.m_frameRange[0]-ff:self.m_frameRange[1]-ff+1,:] = values
                btkTools.smartAppendPoint(acqMotionFinal,marker, modelled, desc= "kinematic fitting",residuals=residuals ) # new acq with marker overwrited


                # btkTools.smartAppendPoint(acqMotionFinal,marker+"_m", measuredValues, desc= "measured" ) # measured marker suffix with _m
                # btkTools.smartAppendPoint(acqMotionFinal,marker, acqMotionFinal.GetPoint(marker).GetValues(), desc= "kinematic fitting" ) # new acq with marker overwrited
                #
                # lenOsim  = len(values)
                # lenc3d  = self.m_acqMotion.GetPoint(marker).GetFrameNumber()
                # if lenOsim < lenc3d:
                #     LOGGER.logger.warning(" size osim (%i) inferior to c3d (%i)" % (lenOsim,lenc3d))
                #     values2 = np.zeros((lenc3d,3))
                #     values2[0:lenOsim,:]=values
                #     values2[lenOsim:lenc3d,:]=self.m_acqMotion.GetPoint(marker).GetValues()[lenOsim:lenc3d,:]
                #
                #     btkTools.smartAppendPoint(acqMotionFinal,marker+"_m", acqMotionFinal.GetPoint(marker).GetValues(), desc= "measured" ) # new acq with marker overwrited
                #     btkTools.smartAppendPoint(acqMotionFinal,marker, values2, desc= "kinematic fitting" ) # new acq with marker overwrited
                # else:
                #     btkTools.smartAppendPoint(acqMotionFinal,marker+"_m", acqMotionFinal.GetPoint(marker).GetValues(), desc= "measured" ) # measured marker suffix with _m
                #     btkTools.smartAppendPoint(acqMotionFinal,marker, values, desc= "kinematic fitting" ) # new acq with marker overwrited

        return acqMotionFinal


    def exportXml(self,filename, path=None):
        """
            Export the generated inverse kinematic Tool
            Args:
                filename (str): filename of the generated inverse kinematic Tool


        """
        filename = filename if path is None else path+filename
        self._osimIK.m_ikTool.printToXML(filename)
