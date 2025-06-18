
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
from typing import List, Tuple, Dict, Optional,Union

# pyCGM2 libraries
from pyCGM2.Model import model
from pyCGM2.Model import modelFilters
from pyCGM2.Tools import btkTools

from pyCGM2 import enums



class SingleBody():
    def __init__(self,calibration_markers:List[str], tracking_markers:List[str],sequence="XYZ"):
        self.m_calibration_markers = calibration_markers
        self.m_tracking_markers = tracking_markers
        self.m_sequence = sequence

        self.m_model = None
        self.m_gcp=modelFilters.GeneralCalibrationProcedure()
        self.m_gcp.setDefinition('Body',
                          "TF",
                          sequence=self.m_sequence,
                          pointLabel1=self.m_calibration_markers[0],
                          pointLabel2=self.m_calibration_markers[1],
                          pointLabel3=self.m_calibration_markers[2],
                          pointLabelOrigin=self.m_calibration_markers[3])
         

    def calibrate(self,acq,calibFramesOfInterest:Optional[List[str]]=None):
         

        self.m_model=model.Model()
        self.m_model.addSegment("Body",0,enums.SegmentSide.Central,
                       calibration_markers=self.m_calibration_markers, 
                       tracking_markers=self.m_tracking_markers)

        modCal=modelFilters.ModelCalibrationFilter(self.m_gcp,acq,self.m_model)
        if calibFramesOfInterest is not None:
            modCal.setFrames(calibFramesOfInterest[0],calibFramesOfInterest[1])
    
        modCal.compute()

    def fit(self,acq):
        # # Motion FILTER
        modMotion=modelFilters.ModelMotionFilter(self.m_gcp,acq,self.m_model,enums.motionMethod.Sodervisk)
        modMotion.setNoAnatomicalMotion(True)
        modMotion.compute()

    def getModel(self):
        return self.m_model

    def getBody(self):
        return self.m_model.getSegment("Body")
 
    def addNode(self,label,values,positionType):
        self.getBody().getReferential("TF").static.addNode(label,values,positionType=positionType)

    def getTrajectory(self,label):
        traj = self.getBody().getReferential("TF").getNodeTrajectory(label)

        return traj
    
    def displayAxis(self,acq,label):
        X = np.array([100,0,0])
        Y = np.array([0,100,0])
        Z = np.array([0,0,100])


        valuesX = np.zeros((len(self.getBody().getReferential("TF").motion), 3))
        valuesY = np.zeros((len(self.getBody().getReferential("TF").motion), 3))
        valuesZ = np.zeros((len(self.getBody().getReferential("TF").motion), 3))

        for i in range(0, len(self.getBody().getReferential("TF").motion)):
            valuesX[i, :] = np.dot(self.getBody().getReferential("TF").motion[i].getRotation(),X) +  self.getBody().getReferential("TF").motion[i].getTranslation()
            valuesY[i, :] = np.dot(self.getBody().getReferential("TF").motion[i].getRotation(),Y) +  self.getBody().getReferential("TF").motion[i].getTranslation()
            valuesZ[i, :] = np.dot(self.getBody().getReferential("TF").motion[i].getRotation(),Z) +  self.getBody().getReferential("TF").motion[i].getTranslation()

        btkTools.smartAppendPoint(acq,label + "_X", valuesX)
        btkTools.smartAppendPoint(acq,label + "_Y", valuesY)
        btkTools.smartAppendPoint(acq,label + "_Z", valuesZ)

