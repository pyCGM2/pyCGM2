# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 14:52:00 2016

@author: Fabien Leboeuf ( Salford Univ, UK)
"""

import numpy as np
import logging
import btk


# pyCGM2
from pyCGM2.Tools import  btkTools
from pyCGM2.Model.Opensim import osimProcessing
import pyCGM2.enums as pyCGM2Enums

import pdb


# ---- PROCEDURES -----

class GeneralOpensimCalibrationProcedures(object):
    def __init__(self):
        """
        """
        self.markers=dict()
        self.geometry=dict()
               
    def setMarkers(self, segmentLabel, trackingMarkers):
        self.markers[segmentLabel] = trackingMarkers
    
    def setGeometry(self, opensimJointLabel, jointLabel, proximalSegmentLabel, distalSegmentLabel ):
         self.geometry[opensimJointLabel] = {"joint label":jointLabel, "proximal segment label":proximalSegmentLabel, "distal segment label":distalSegmentLabel}


class GeneralOpensimReconstructionProcedure(object):
    def __init__(self):
        """
        """
        self.ikTags=dict()
        
    def setIkTags(self,markerLabel,weight):
        self.ikTags[markerLabel]=weight
            

    def setMarkerWeight(self,markerLabel,weight):
        self.ikTags[markerLabel] = weight
        logging.warning( "marker (%s) weight altered. New weight = %d" %(markerLabel,weight))


        
class CgmOpensimCalibrationProcedures(object):
    """ model embedded Procedure """
    def __init__(self,model):
        """Constructor
        :Parameters:
           - `model` (Model) - instance of Model  
        
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



class CgmOpensimReconstructionProcedure(object):
    """ model embedded Procedure """
    def __init__(self,model):
        """Constructor
        :Parameters:
           - `model` (Model) - instance of Model  
        
        """
        self.model=model
        self.ikTags=dict()
        
        self.__setIkTags()
        
    def __setIkTags(self):
        self.ikTags=self.model.opensimIkTask()

    def setMarkerWeight(self,label,weight):
        self.ikTags[label] = weight
        logging.warning( "marker (%s) weight altered. New weight = %d" %(label,weight))


# ---- FILTERS -----

class opensimCalibrationFilter(object):
    """ act as an adaptator  """
    def __init__(self,osimFile,model,procedure):
        self.m_osimFile = osimFile
        self.m_model = model        
        self.m_procedure = procedure
        self.m_toMeter = 1000.0

        self._osimModel = osimProcessing.opensimModel(osimFile,model)    
        
    def addMarkerSet(self,markerSetFile):
         self._osimModel.addMarkerSet(markerSetFile)
        

    def build(self):

        # update marker from procedure ( need markers in osim)        
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

        self._osimModel.m_model.setName("fitted model")
        
        return self._osimModel
        
        
        
    def exportXml(self,filename, path=None):
         self._osimModel.m_model.printToXML(filename)


class opensimReconstructionFilter(object):
    def __init__(self,ikToolFiles,fittingOsim, ikTagProcedure,dataDir,accuracy = 1e-8 ):
        self.m_fittingOsim = fittingOsim
        self.m_ikToolFiles = ikToolFiles
        self.m_procedure = ikTagProcedure
        
        self.accuracy = accuracy
        self.opensimOutputDir=dataDir
        

        self._osimIK = osimProcessing.opensimKinematicFitting(self.m_fittingOsim.m_model,self.m_ikToolFiles)
        self._osimIK.setAccuracy(self.accuracy)
        self._osimIK.setResultsDirectory(self.opensimOutputDir)
    
    def reconstruct(self,acqMotion, acqMotionFilename):

                
        acqMotion_forIK = btk.btkAcquisition.Clone(acqMotion)
        progressionAxis,forwardProgression,globalFrame = btkTools.findProgressionFromPoints(acqMotion,"SACR", "midASIS","RPSI")
        
        
        # --- ikTasks 
        #  UPDATE method - ik tags ( need task in the initial iktools)
        for markerIt in self.m_procedure.ikTags.keys():
            self._osimIK.updateIKTask(markerIt,self.m_procedure.ikTags[markerIt])
        
        # --- configuration and run IK
        R_LAB_OSIM = osimProcessing.setGlobalTransormation_lab_osim(progressionAxis,forwardProgression) 
        self._osimIK.config(R_LAB_OSIM, acqMotion_forIK, acqMotionFilename )
        self._osimIK.run()
        
        # --- gernerate acq with rigid markers
        acqMotionFinal = btk.btkAcquisition.Clone(acqMotion)
        for marker in self.m_procedure.ikTags.keys():
            print marker
            values =osimProcessing.sto2pointValues(self.opensimOutputDir + "ik_model_marker_locations.sto",marker,R_LAB_OSIM)
            lenOsim  = len(values)
            lenc3d  = acqMotion.GetPoint(marker).GetFrameNumber()
            if lenOsim < lenc3d: 
                logging.warnings(" size osim (%i) inferior to c3d (%i)" % (lenOsim,lenc3d))
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
        self._osimIK.m_ikTool.printToXML(filename)
        