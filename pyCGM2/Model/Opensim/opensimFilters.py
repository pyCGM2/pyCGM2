# -*- coding: utf-8 -*-
import os
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
from bs4 import BeautifulSoup

# pyCGM2
try:
    from pyCGM2 import btk
except:
    LOGGER.logger.info("[pyCGM2] pyCGM2-embedded btk not imported")
    import btk
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.Opensim import osimProcessing
from pyCGM2.Processing import progressionFrame

try:
    from pyCGM2 import opensim4 as opensim
except:
    LOGGER.logger.info("[pyCGM2] : pyCGM2-embedded opensim4 not imported")
    import opensim

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
        LOGGER.logger.info( "marker (%s) weight altered. New weight = %d" %(markerLabel,weight))



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

    def __init__(self,model):
        """
            :Parameters:
           - `model` (Model) - instance of Model

        """
        self.model=model
        self.ikTags=dict()#

        self.__setIkTags()


    def __setIkTags(self):
        self.ikTags=self.model.opensimIkTask()

    def updateMarkerWeight(self,markerLabel,weight):
        """
            Update weigth of a tracking marker

            :Parameters:
                - `markerLabel` (str) - marker label
                - `weight` (double) - joint label of a pyCGM2 model

        """

        if self.ikTags[markerLabel] != weight:
            self.ikTags[markerLabel] = weight
            LOGGER.logger.info( "marker (%s) weight altered. New weight = %d" %(markerLabel,weight))


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
        self.opensimOutputDir = dataDir if dataDir[-1:] =="\\" else dataDir+"\\"

    def addMarkerSet(self,markerSetFile):
        """
            Add a marker set file
            :Parameters:
                - `markerSetFile` (str) - full filename of the opensim marker set file
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
            :Parameters:
                - `filename` (str) - filename of the calibrated osim file


        """
        filename = filename if path is None else path+filename
        self._osimModel.m_model.printToXML(filename)


class opensimFittingFilter(object):
    def __init__(self,ikToolFile,calibratedOsim, ikTagProcedure,dataDir,acqMotion, accuracy = 1e-8 ):
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
            :Parameters:
                - `acqMotion` (btk.Acquisition) - acquisition of a motion trial
                - `acqMotionFilename` (filename) - filename of the motion trial

        """


        acqMotion_forIK = btk.btkAcquisition.Clone(self.m_acqMotion)

        if "progressionAxis" in kwargs:
            progressionAxis = kwargs["progressionAxis"]
            if "forwardProgression" in kwargs:
                forwardProgression = kwargs["forwardProgression"]
            else:
                forwardProgression = True

        else:
            pfp = progressionFrame.PelvisProgressionFrameProcedure()
            pff = progressionFrame.ProgressionFrameFilter(self.m_acqMotion,pfp)
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
                ff = acqMotionFinal.GetFirstFrame()
                modelled[self.m_frameRange[0]-ff:self.m_frameRange[1]-ff+1,:] = values
                btkTools.smartAppendPoint(acqMotionFinal,marker, modelled, desc= "kinematic fitting" ) # new acq with marker overwrited


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
            :Parameters:
                - `filename` (str) - filename of the generated inverse kinematic Tool


        """
        filename = filename if path is None else path+filename
        self._osimIK.m_ikTool.printToXML(filename)


class opensimScalingFilter(object):
    def __init__(self,model_input,markerSetFile,scaleToolFile, static_path, xml_output,required_mp,model_output = "OutPutModel.osim"):


        self._osimModel = osimProcessing.opensimModel2(model_input)

        self._osimModel.m_model.setName("pyCGM2-CGM-scaled")

        markerSet= opensim.MarkerSet(markerSetFile)
        self._osimModel.m_model.updateMarkerSet(markerSet)

        self.model_output = model_output
        self.model_with_markers_output = model_output.replace(".osim", "_markers.osim")
        self.static_path = static_path
        self.xml_output = xml_output

        self.time_range = self.time_range_from_static()


        # initialize scale tool from setup file
        self.scale_tool = opensim.ScaleTool(scaleToolFile)
        self.set_anthropometry(required_mp["Bodymass"], required_mp["Height"])
        # Tell scale tool to use the loaded model
        self.scale_tool.getGenericModelMaker().setModelFileName(model_input)

        # self.scale_tool.getModelScaler().processModel(self.model, "", required_mp["Bodymass"])
        self.run_model_scaler(required_mp["Bodymass"])
        self.run_marker_placer()

    def time_range_from_static(self):
        static = opensim.MarkerData(self.static_path)
        initial_time = static.getStartFrameTime()
        final_time = static.getLastFrameTime()
        range_time = opensim.ArrayDouble()
        range_time.set(0, initial_time)
        range_time.set(1, final_time)
        return range_time

    def set_anthropometry(self, mass, height):#, age):
        """
        Set basic anthropometric parameters in scaling model
        Parameters
        ----------
        mass : Double
            mass (kg)
        height : Double
            height (mm)
        age : int
            age (year)
        """
        self.scale_tool.setSubjectMass(mass)
        self.scale_tool.setSubjectHeight(height)
        # self.scale_tool.setSubjectAge(age)

    def run_model_scaler(self, mass):
        model_scaler = self.scale_tool.getModelScaler()
        # Whether or not to use the model scaler during scale
        model_scaler.setApply(True)
        # Set the marker file to be used for scaling
        model_scaler.setMarkerFileName(self.static_path)

        # set time range
        model_scaler.setTimeRange(self.time_range)

        # Indicating whether or not to preserve relative mass between segments
        model_scaler.setPreserveMassDist(True)

        # Name of model file (.osim) to write when done scaling
        model_scaler.setOutputModelFileName(self.model_output)

        # Filename to write scale factors that were applied to the unscaled model (optional)
        model_scaler.setOutputScaleFileName(
            self.xml_output.replace(".xml", "_scaling_factor.xml")
        )

        model_scaler.processModel(self._osimModel.m_model, "", mass)

        # self.scale_tool.printToXML(self.xml_output)


    def run_marker_placer(self):
        # load a scaled model
        scaled_model = opensim.Model(self.model_output)

        self._osimScaledModel = osimProcessing.opensimModel2(self.model_output)

        marker_placer = self.scale_tool.getMarkerPlacer()
        # Whether or not to use the model scaler during scale`
        marker_placer.setApply(True)
        marker_placer.setTimeRange(self.time_range)

        marker_placer.setStaticPoseFileName(self.static_path)

        # Name of model file (.osim) to write when done scaling
        marker_placer.setOutputModelFileName(self.model_with_markers_output)

        # Maximum amount of movement allowed in marker data when averaging
        marker_placer.setMaxMarkerMovement(-1)

        marker_placer.processModel(self._osimScaledModel.m_model)

        # save processed model
        self._osimScaledModel.m_model.printToXML(self.model_output)

        # print scale config to xml
        self.scale_tool.printToXML(self.xml_output)

    def getScaledModel(self):
        return self._osimScaledModel
