"""
This module contains filters and associated procedures which can be applied on a model

"""


import pyCGM2
import numpy as np
import matplotlib.pyplot as plt
import copy
from typing import List, Tuple, Dict, Optional,Union

LOGGER = pyCGM2.LOGGER
import btk

from pyCGM2.Model import frame
from pyCGM2.Model import motion

from pyCGM2 import enums
from  pyCGM2.Math import euler
from  pyCGM2.Math import numeric
from pyCGM2.Tools import  btkTools
from pyCGM2.Utils import timer

from pyCGM2.Model.model import Model
from pyCGM2.Model.Procedures.forcePlateIntegrationProcedures import ForcePlateIntegrationProcedure
from pyCGM2.Model.Procedures.modelMotionCorrection import ModelCorrectionProcedure
from pyCGM2.Model.Procedures.modelQuality import QualityProcedure 

#-------- MODEL PROCEDURE  ----------

# --- calibration procedure
class GeneralCalibrationProcedure(object):
    """General Procedure to load from the Model Calibration Filter for custom models."""


    def __init__(self):
       self.definition={}
       self.anatomicalDefinition={}

    def setDefinition(self, segmentName: str, referentialLabel: str, sequence: str = "",
                      pointLabel1: str = "", pointLabel2: str = "", pointLabel3: str = "",
                      pointLabelOrigin: str = "") -> None:
        
        """Define rules for constructing a 'technical' coordinate system.

        Args:
            segmentName (str): Name of the segment.
            referentialLabel (str): Label of the referential.
            sequence (str, optional): Construction sequence (XYZ, XYiZ, ...). Defaults to "".
            pointLabel1 (str, optional): Marker label used for constructing axes v1 and v2. Defaults to "".
            pointLabel2 (str, optional): Marker label used for constructing axis v1. Defaults to "".
            pointLabel3 (str, optional): Marker label used for constructing axis v2. Defaults to "".
            pointLabelOrigin (str, optional): Marker label used as origin of the coordinate system. Defaults to "".
        """

        if segmentName in self.definition:
            self.definition[segmentName][referentialLabel]={'sequence':sequence, 'labels':[pointLabel1,pointLabel2,pointLabel3,pointLabelOrigin]}
        else:
            self.definition[segmentName]={}
            self.definition[segmentName][referentialLabel]={'sequence':sequence, 'labels':[pointLabel1,pointLabel2,pointLabel3,pointLabelOrigin]}

    def setAnatomicalDefinition(self, segmentName: str, sequence: str = "",
                                nodeLabel1: str = "", nodeLabel2: str = "", nodeLabel3: str = "",
                                nodeLabelOrigin: str = "") -> None:
        """Define rules for constructing the 'anatomical' coordinate system.

        Args:
            segmentName (str): Name of the segment.
            sequence (str, optional): Construction sequence (XYZ, XYiZ, ...). Defaults to "".
            nodeLabel1 (str, optional): Node label used for constructing axes v1 and v2. Defaults to "".
            nodeLabel2 (str, optional): Node label used for constructing axis v1. Defaults to "".
            nodeLabel3 (str, optional): Node label used for constructing axis v2. Defaults to "".
            nodeLabelOrigin (str, optional): Node label used as origin of the coordinate system. Defaults to "".
        """

        if segmentName in self.anatomicalDefinition:
            self.anatomicalDefinition[segmentName]={'sequence':sequence, 'labels':[nodeLabel1,nodeLabel2,nodeLabel3,nodeLabelOrigin]}
        else:
            self.anatomicalDefinition[segmentName]={}
            self.anatomicalDefinition[segmentName]={'sequence':sequence, 'labels':[nodeLabel1,nodeLabel2,nodeLabel3,nodeLabelOrigin]}


class StaticCalibrationProcedure(object):
    """Procedure for calibration using a pyCGM2-embedded model instance.
    Args:
            model (Model): A pyCGM2-embedded model instance.
    """
    def __init__(self,model:Model):

        self.model=model
        self.definition={}

        self.__setDefinition()

    def __setDefinition(self):
        """Internal method to set calibration definitions based on the model."""
        self.definition=self.model.calibrationProcedure()



# ---- inverse dynamic procedure
class InverseDynamicProcedure(object):
    def __init__(self):
        pass

class CGMLowerlimbInverseDynamicProcedure(InverseDynamicProcedure):
    """Procedure for calculating Inverse dynamics of the CGM lower limbs."""
    def __init__(self):
        super(CGMLowerlimbInverseDynamicProcedure,self).__init__()
        pass


    def _externalDeviceForceContribution(self, wrenchs:List[btk.btkWrench]):
        """Calculate external device force contribution.

        Args:
            wrenchs (List[btk.btkWrench]): List of wrenches.

        Returns:
            np.ndarray: Calculated force values.
        """

        nf = wrenchs[0].GetForce().GetValues().shape[0]
        forceValues = np.zeros((nf,3))
        for wrIt in wrenchs:
            forceValues = forceValues + wrIt.GetForce().GetValues()

        return forceValues

    def _externalDeviceMomentContribution(self, wrenchs: List[btk.btkWrench], Oi: np.ndarray, scaleToMeter: float) -> np.ndarray:
        
        """Calculate external device moment contribution.

        Args:
            wrenchs (btk.btkWrenchCollection): Collection of wrenches.
            Oi (np.ndarray): Origin positions.
            scaleToMeter (float): Scale factor to meter.

        Returns:
            np.ndarray: Calculated moment values.
        """

        nf = wrenchs[0].GetMoment().GetValues().shape[0]
        momentValues = np.zeros((nf,3))

        for wrIt in wrenchs:
             Fext = wrIt.GetForce().GetValues()
             Mext = wrIt.GetMoment().GetValues()
             posExt = wrIt.GetPosition().GetValues()

             for i in range(0,nf):
                 Fext_i = np.matrix(Fext[i,:])
                 Mext_i = np.matrix(Mext[i,:])
                 di = posExt[i,:] - Oi[i].getTranslation()
                 wrMomentValues = (Mext_i.T*scaleToMeter + numeric.skewMatrix(di*scaleToMeter)*Fext_i.T)
                 momentValues[i,:] = momentValues[i,:] + np.array(wrMomentValues.T)

        return momentValues


    def _distalMomentContribution(self, wrench: btk.btkWrench, Oi: np.ndarray, scaleToMeter: float, source: str = "Wrench") -> np.ndarray:
        """Calculate distal moment contribution.

        Args:
            wrench (btk.btkWrench): Wrench instance.
            Oi (np.ndarray): Origin positions.
            scaleToMeter (float): Scale factor to meter.
            source (str, optional): Source type. Defaults to "Wrench".

        Returns:
            np.ndarray: Calculated moment values.
        """

        nf = wrench.GetMoment().GetValues().shape[0]
        momentValues = np.zeros((nf,3))

        Fext = wrench.GetForce().GetValues()
        Mext = wrench.GetMoment().GetValues()
        posExt = wrench.GetPosition().GetValues()

        for i in range(0,nf):
            Fext_i = np.matrix(Fext[i,:])
            Mext_i = np.matrix(Mext[i,:])
            di = posExt[i,:] - Oi[i].getTranslation()

            if source == "Wrench":
                wrMomentValues = (- 1.0*Mext_i.T*scaleToMeter + - 1.0*numeric.skewMatrix(di*scaleToMeter)*Fext_i.T)
            elif source == "Force":
                wrMomentValues = ( - 1.0*numeric.skewMatrix(di*scaleToMeter)*Fext_i.T)
            elif source == "Moment":
                wrMomentValues = (- 1.0*Mext_i.T*scaleToMeter)

            momentValues[i,:] =  np.array(wrMomentValues.T)

        return momentValues

    def _forceAccelerationContribution(self, mi: float, ai: np.ndarray, g: np.ndarray, scaleToMeter: float) -> np.ndarray:
        """Calculate force acceleration contribution.

        Args:
            mi (float): Mass.
            ai (np.ndarray): Acceleration.
            g (np.ndarray): Gravity vector.
            scaleToMeter (float): Scale factor to meter.

        Returns:
            np.ndarray: Calculated force values.
        """

        nf = ai.shape[0]
        g= np.matrix(g)
        accelerationContribution = np.zeros((nf,3))

        for i in range(0,nf):
            ai_i = np.matrix(ai[i,:])
            val = mi * ai_i.T*scaleToMeter - mi*g.T
            accelerationContribution[i,:] = np.array(val.T)

        return  accelerationContribution


    def _inertialMomentContribution(self, Ii: np.ndarray, alphai: np.ndarray, omegai: np.ndarray, Ti: np.ndarray, scaleToMeter: float) -> np.ndarray:
        """Calculate inertial moment contribution.

        Args:
            Ii (np.ndarray): Inertia.
            alphai (np.ndarray): Angular acceleration.
            omegai (np.ndarray): Angular velocity.
            Ti (np.ndarray): Transformation matrices.
            scaleToMeter (float): Scale factor to meter.

        Returns:
            np.ndarray: Calculated moment values.
        """

        nf = alphai.shape[0]

        accelerationContribution = np.zeros((nf,3))
        coriolisContribution = np.zeros((nf,3))

        Ii = np.matrix(Ii)


        for i in range(0,nf):
            alphai_i = np.matrix(alphai[i,:])
            omegai_i = np.matrix(omegai[i,:])
            Ri_i = np.matrix(Ti[i].getRotation())

            accContr_i = Ri_i*(Ii*np.power(scaleToMeter,2))*Ri_i.T * alphai_i.T
            accelerationContribution[i,:]=np.array(accContr_i.T)

            corCont_i = numeric.skewMatrix(omegai_i) * Ri_i*(Ii*np.power(scaleToMeter,2))*Ri_i.T * omegai_i.T
            coriolisContribution[i,:] = np.array(corCont_i.T)


        return   accelerationContribution + coriolisContribution

    def _accelerationMomentContribution(self, mi: float, ci: np.ndarray, ai: np.ndarray, Ti: np.ndarray, scaleToMeter: float) -> np.ndarray:
        """Calculate acceleration moment contribution.

        Args:
            mi (float): Mass.
            ci (np.ndarray): Center of mass.
            ai (np.ndarray): Acceleration.
            Ti (np.ndarray): Transformation matrices.
            scaleToMeter (float): Scale factor to meter.

        Returns:
            np.ndarray: Calculated moment values.
        """

        nf = ai.shape[0]

        accelerationContribution = np.zeros((nf,3))

        for i in range(0,nf):
            ai_i = np.matrix(ai[i,:])
            Ri_i = np.matrix(Ti[i].getRotation())
            ci = np.matrix(ci)

            val = -1.0*mi*numeric.skewMatrix(ai_i*scaleToMeter) * Ri_i*(ci.T*scaleToMeter)

            accelerationContribution[i,:] = np.array(val.T)

        return accelerationContribution


    def _gravityMomentContribution(self, mi: float, ci: np.ndarray, g: np.ndarray, Ti: np.ndarray, scaleToMeter: float) -> np.ndarray:
        """Calculate gravity moment contribution.

        Args:
            mi (float): Mass.
            ci (np.ndarray): Center of mass.
            g (np.ndarray): Gravity vector.
            Ti (np.ndarray): Transformation matrices.
            scaleToMeter (float): Scale factor to meter.

        Returns:
            np.ndarray: Calculated moment values.
        """

        nf = len(Ti)

        gravityContribution = np.zeros((nf,3))


        g= np.matrix(g)
        ci = np.matrix(ci)
        for i in range(0,nf):
            Ri_i = np.matrix(Ti[i].getRotation())
            val = - 1.0 *mi*numeric.skewMatrix(g) * Ri_i*(ci.T*scaleToMeter)

            gravityContribution[i,:] = np.array(val.T)
        return  gravityContribution


    def computeSegmental(self, model: Model, segmentLabel: str, btkAcq: btk.btkAcquisition, gravity: np.ndarray, scaleToMeter: float, distalSegmentLabel: Optional[str] = None) -> np.ndarray:
        """Compute segmental dynamics.

        Args:
            model (Model): Model instance.
            segmentLabel (str): Label of the segment.
            btkAcq (btk.btkAcquisition): Acquisition instance.
            gravity (np.ndarray): Gravity vector.
            scaleToMeter (float): Scale factor to meter.
            distalSegmentLabel (Optional[str], optional): Label of the distal segment. Defaults to None.

        Returns:
            np.ndarray: Calculated segmental dynamics.
        """
        N = btkAcq.GetPointFrameNumber()

        # initialisation
        model.getSegment(segmentLabel).zeroingProximalWrench()

        forceValues = np.zeros((N,3))
        momentValues = np.zeros((N,3))
        positionValues = np.zeros((N,3))

        wrench = btk.btkWrench()
        ForceBtkPoint = btk.btkPoint(N)
        MomentBtkPoint = btk.btkPoint(N)
        PositionBtkPoint = btk.btkPoint(N)

        Ti = model.getSegment(segmentLabel).anatomicalFrame.motion
        mi = model.getSegment(segmentLabel).m_bsp["mass"]
        ci = model.getSegment(segmentLabel).m_bsp["com"]
        Ii = model.getSegment(segmentLabel).m_bsp["inertia"]


        # external devices
        extForces = np.zeros((N,3))
        extMoment = np.zeros((N,3))
        if model.getSegment(segmentLabel).isExternalDeviceWrenchsConnected():
            extForces = self._externalDeviceForceContribution(model.getSegment(segmentLabel).m_externalDeviceWrenchs)
            extMoment = self._externalDeviceMomentContribution(model.getSegment(segmentLabel).m_externalDeviceWrenchs, Ti, scaleToMeter)

        # distal
        distSegMoment = np.zeros((N,3))
        distSegForce = np.zeros((N,3))
        distSegMoment_forceDistalContribution = np.zeros((N,3))
        distSegMoment_momentDistalContribution = np.zeros((N,3))


        if distalSegmentLabel != None:
            distalWrench = model.getSegment(distalSegmentLabel).m_proximalWrench

            distSegForce = distalWrench.GetForce().GetValues()
            distSegMoment = self._distalMomentContribution(distalWrench, Ti, scaleToMeter)

            distSegMoment_forceDistalContribution = self._distalMomentContribution(distalWrench, Ti, scaleToMeter, source ="Force")
            distSegMoment_momentDistalContribution = self._distalMomentContribution(distalWrench, Ti, scaleToMeter, source ="Moment")

        # Force
        ai = model.getSegment(segmentLabel).getComAcceleration(btkAcq.GetPointFrequency(), order=4, fc=6 )
        force_accContr = self._forceAccelerationContribution(mi,ai,gravity,scaleToMeter)
        forceValues  = force_accContr - ( extForces) - ( - distSegForce)

        # moment
        alphai = model.getSegment(segmentLabel).getAngularAcceleration(btkAcq.GetPointFrequency())
        omegai = model.getSegment(segmentLabel).getAngularVelocity(btkAcq.GetPointFrequency())

        inertieCont = self._inertialMomentContribution(Ii, alphai,omegai, Ti ,scaleToMeter)
        accCont = self._accelerationMomentContribution(mi,ci, ai, Ti, scaleToMeter)
        grCont = self._gravityMomentContribution(mi,ci, gravity, Ti, scaleToMeter)

        momentValues = inertieCont + accCont -  grCont - extMoment - distSegMoment

        for i in range(0,N):
            positionValues[i,:] = Ti[i].getTranslation()

        ForceBtkPoint.SetValues(forceValues)
        MomentBtkPoint.SetValues(momentValues/scaleToMeter)
        PositionBtkPoint.SetValues(positionValues)

        wrench.SetForce(ForceBtkPoint)
        wrench.SetMoment(MomentBtkPoint)
        wrench.SetPosition(PositionBtkPoint)

        model.getSegment(segmentLabel).m_proximalWrench = wrench
        model.getSegment(segmentLabel).m_proximalMomentContribution["internal"] = (inertieCont+accCont)/scaleToMeter
        model.getSegment(segmentLabel).m_proximalMomentContribution["external"] = (-  grCont - extMoment - distSegMoment)/scaleToMeter
        model.getSegment(segmentLabel).m_proximalMomentContribution["inertia"] = inertieCont/scaleToMeter
        model.getSegment(segmentLabel).m_proximalMomentContribution["linearAcceleration"] = accCont/scaleToMeter
        model.getSegment(segmentLabel).m_proximalMomentContribution["gravity"] = - grCont/scaleToMeter
        model.getSegment(segmentLabel).m_proximalMomentContribution["externalDevices"] = - extMoment/scaleToMeter
        model.getSegment(segmentLabel).m_proximalMomentContribution["distalSegments"] = - distSegMoment/scaleToMeter
        model.getSegment(segmentLabel).m_proximalMomentContribution["distalSegmentForces"] = - distSegMoment_forceDistalContribution/scaleToMeter
        model.getSegment(segmentLabel).m_proximalMomentContribution["distalSegmentMoments"] = - distSegMoment_momentDistalContribution/scaleToMeter

        return momentValues

    def compute(self, model: Model, btkAcq: btk.btkAcquisition, gravity: np.ndarray, scaleToMeter: float):
        """Run the computation of segmental dynamics for the lower limbs.

        Args:
            model (Model): Model instance.
            btkAcq (btk.btkAcquisition): Acquisition instance.
            gravity (np.ndarray): Gravity vector.
            scaleToMeter (float): Scale factor to meter.
        """

        self.computeSegmental(model,"Left Foot",btkAcq, gravity, scaleToMeter)
        self.computeSegmental(model,"Right Foot",btkAcq, gravity, scaleToMeter)

        self.computeSegmental(model,"Left Shank",btkAcq, gravity, scaleToMeter,distalSegmentLabel = "Left Foot")
        self.computeSegmental(model,"Right Shank",btkAcq, gravity, scaleToMeter,distalSegmentLabel = "Right Foot")

        model.getSegment("Left Shank Proximal").m_proximalWrench = model.getSegment("Left Shank").m_proximalWrench
        model.getSegment("Left Shank Proximal").m_proximalMomentContribution = model.getSegment("Left Shank").m_proximalMomentContribution
        self.computeSegmental(model,"Left Thigh",btkAcq, gravity, scaleToMeter,distalSegmentLabel = "Left Shank")


        model.getSegment("Right Shank Proximal").m_proximalWrench = model.getSegment("Right Shank").m_proximalWrench
        model.getSegment("Right Shank Proximal").m_proximalMomentContribution = model.getSegment("Right Shank").m_proximalMomentContribution
        self.computeSegmental(model,"Right Thigh",btkAcq, gravity, scaleToMeter,distalSegmentLabel = "Right Shank")

#-------- FILTERS ----------


#-------- MODEL CALIBRATION FILTER ----------

class ModelCalibrationFilter(object):
    """Calibrate a model from a static acquisition.

    The calibration consists of constructing both technical and anatomical coordinate systems for each segment constituting the model.

    Args:
        procedure (Union[GeneralCalibrationProcedure,StaticCalibrationProcedure]): Calibration procedure to be used.
        acq (btk.btkAcquisition): Acquisition instance containing static trial data.
        iMod (Model): Model instance to be calibrated.

    Kwargs:
        markerDiameter (float): Diameter of the markers used.
        basePlate (float): Thickness of the base plate.
        viconCGM1compatible (bool): If True, replicate the Vicon Plugin-gait error related to the proximal and distal tibia.
        leftFlatFoot (bool): If True, set the longitudinal axis of the left foot parallel to the ground.
        rightFlatFoot (bool): If True, set the longitudinal axis of the right foot parallel to the ground.
        headFlat (bool): If True, set the longitudinal axis of the head parallel to the ground.
    """

    def __init__(self,
                 procedure:Union[GeneralCalibrationProcedure,StaticCalibrationProcedure], 
                 acq:btk.btkAcquisition, iMod:Model,**options):
        self.m_aqui=acq
        self.m_procedure=procedure
        self.m_model=iMod
        self.m_options=options
        self.m_noAnatomicalCalibration = False

    def setOption(self, label: str, value):
        """Set or update an option for the calibration.

        Args:
            label (str): Option label.
            value: Value of the option.
        """
        self.m_options[label] = value

    def setBoolOption(self, label: str):
        """Enable a boolean option.

        Args:
            label (str): Option label.
        """
        self.m_options[label] = True

    def setNoAnatomicalCalibration(self, boolFlag: bool):
        """Set whether to perform anatomical calibration or not.

        Args:
            boolFlag (bool): Flag to enable or disable anatomical calibration.
        """
        self.m_noAnatomicalCalibration = boolFlag


    def compute(self, firstFrameOnly: bool = True):
        """Run the calibration filter.

        Args:
            firstFrameOnly (bool, optional): Use only the first frame for calibration. Defaults to True.
        """

        ff=self.m_aqui.GetFirstFrame()

        if firstFrameOnly :
            frameInit=0
            frameEnd=1
        else :
            frameInit=frameInit-ff
            frameEnd=frameEnd-ff+1

        if str(self.m_model) != "Basis Model":
            for segName in self.m_procedure.definition[0]:
                segPicked=self.m_model.getSegment(segName)
                for tfName in self.m_procedure.definition[0][segName]:
                    segPicked.addTechnicalReferential(tfName)

            self.m_model.calibrate(self.m_aqui, self.m_procedure.definition[0], self.m_procedure.definition[1], options=self.m_options)

        else :
            for segName in self.m_procedure.definition:
                segPicked=self.m_model.getSegment(segName)
                for tfName in self.m_procedure.definition[segName]:

                    segPicked.addTechnicalReferential(tfName)

                    pt1=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName][tfName]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
                    pt2=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName][tfName]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
                    pt3=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName][tfName]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

                    ptOrigin=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName][tfName]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)


                    a1=(pt2-pt1)
                    a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

                    v=(pt3-pt1)
                    v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

                    a2=np.cross(a1,v)
                    a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

                    x,y,z,R=frame.setFrameData(a1,a2,self.m_procedure.definition[segName][tfName]['sequence'])

                    segPicked.referentials[-1].static.m_axisX=x
                    segPicked.referentials[-1].static.m_axisY=y
                    segPicked.referentials[-1].static.m_axisZ=z

                    segPicked.referentials[-1].static.setRotation(R)
                    segPicked.referentials[-1].static.setTranslation(ptOrigin)

                    #  - add Nodes in segmental static(technical)Frame -
                    for label in segPicked.m_markerLabels:
                        globalPosition=self.m_aqui.GetPoint(str(label)).GetValues()[frameInit:frameEnd,:].mean(axis=0)
                        segPicked.referentials[-1].static.addNode(label,globalPosition,positionType="Global")

            # calibration of anatomicalFrame
            if not self.m_noAnatomicalCalibration:

                for segName in self.m_procedure.anatomicalDefinition:

                    segPicked=self.m_model.getSegment(segName)
                    tf=segPicked.getReferential("TF")


                    nd1 = str(self.m_procedure.anatomicalDefinition[segName]['labels'][0])
                    pt1 = tf.static.getNode_byLabel(nd1).m_global

                    nd2 = str(self.m_procedure.anatomicalDefinition[segName]['labels'][1])
                    pt2 = tf.static.getNode_byLabel(nd2).m_global

                    nd3 = str(self.m_procedure.anatomicalDefinition[segName]['labels'][2])
                    pt3 = tf.static.getNode_byLabel(nd3).m_global

                    ndO = str(self.m_procedure.anatomicalDefinition[segName]['labels'][3])
                    ptO = tf.static.getNode_byLabel(ndO).m_global

                    a1=(pt2-pt1)
                    a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

                    v=(pt3-pt1)
                    v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

                    a2=np.cross(a1,v)
                    a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

                    x,y,z,R=frame.setFrameData(a1,a2,self.m_procedure.anatomicalDefinition[segName]['sequence'])

                    segPicked.anatomicalFrame.static.m_axisX=x
                    segPicked.anatomicalFrame.static.m_axisY=y
                    segPicked.anatomicalFrame.static.m_axisZ=z

                    segPicked.anatomicalFrame.static.setRotation(R)
                    segPicked.anatomicalFrame.static.setTranslation(ptO)


                    # --- relative rotation Technical Anatomical
                    tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,segPicked.anatomicalFrame.static.getRotation()))


#-------- MOTION FILTER  ----------



class ModelMotionFilter(object):
    """Updates the model with the pose of both technical and anatomical coordinate systems at each frame.

    Args:
        procedure (Union[GeneralCalibrationProcedure,StaticCalibrationProcedure]): The motion procedure to be used.
        acq (btk.btkAcquisition): Acquisition instance containing motion data.
        iMod (Model): The model instance to be updated.
        method (enums.motionMethod): Pose method to be used.

    Kwargs:
        markerDiameter (float): Diameter of the markers used. It helps in determining the exact location of the markers.
        basePlate (float): Thickness of the base plate. Used in calculations where ground interaction is considered.
        viconCGM1compatible (bool): If true, replicates the Vicon Plugin-gait error related to proximal and distal tibia.
        useLeftKJCmarker (str): Label of the left knee joint center, present in the c3d as a virtual marker.
        useLeftAJCmarker (str): Label of the left ankle joint center, present in the c3d as a virtual marker.
        useLeftSJCmarker (str): Label of the left shoulder joint center, present in the c3d as a virtual marker.
        useLeftEJCmarker (str): Label of the left elbow joint center, present in the c3d as a virtual marker.
        useLeftWJCmarker (str): Label of the left wrist joint center, present in the c3d as a virtual marker.
        useRightKJCmarker (str): Label of the right knee joint center, present in the c3d as a virtual marker.
        useRightAJCmarker (str): Label of the right ankle joint center, present in the c3d as a virtual marker.
        useRightSJCmarker (str): Label of the right shoulder joint center, present in the c3d as a virtual marker.
        useRightEJCmarker (str): Label of the right elbow joint center, present in the c3d as a virtual marker.
        useRightWJCmarker (str): Label of the right wrist joint center, present in the c3d as a virtual marker.
    """


    def __init__(self,procedure:Union[GeneralCalibrationProcedure,StaticCalibrationProcedure],acq:btk.btkAcquisition, iMod:Model,method:enums.motionMethod, **options ):


        self.m_aqui = acq
        self.m_procedure = procedure
        self.m_model = iMod
        self.m_method = method
        self.m_options = options
        self.m_noAnatomicalMotion = False


    def setOption(self, label: str, value):
        """Set or update an option for the motion filter.

        Args:
            label (str): The option label.
            value: The value of the option.
        """
        self.m_options[label] = value

    def setBoolOption(self, label: str):
        """Activate a boolean option.

        Args:
            label (str): The option label.
        """
        self.m_options[label] = True

    def setNoAnatomicalMotion(self, boolFlag: bool):
        """Determines whether or not anatomical motion should be computed.

        Args:
            boolFlag (bool): Flag to activate or deactivate anatomical motion.
        """
        self.m_noAnatomicalMotion = boolFlag

    def segmentalCompute(self, segments: List[str]):
        """Computes motion for the given segments.

        Args:
            segments (List[str]): Labels of the segments to process.
        """

        if str(self.m_model) != "Basis Model":
            self.m_model.computeOptimizedSegmentMotion(self.m_aqui,
                                             segments,
                                             self.m_procedure.definition[0],
                                             self.m_procedure.definition[1],
                                             self.m_method,
                                             self.m_options)
        else:

            for segName in segments:
                segPicked=self.m_model.getSegment(segName)
                segPicked.getReferential("TF").motion =[]

                if self.m_method == enums.motionMethod.Determinist :
                    for i in range(0,self.m_aqui.GetPointFrameNumber()):

                        pt1=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName]["TF"]['labels'][0])).GetValues()[i,:]
                        pt2=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName]["TF"]['labels'][1])).GetValues()[i,:]
                        pt3=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName]["TF"]['labels'][2])).GetValues()[i,:]
                        ptOrigin=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName]["TF"]['labels'][3])).GetValues()[i,:]

                        a1=(pt2-pt1)
                        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

                        v=(pt3-pt1)
                        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

                        a2=np.cross(a1,v)
                        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

                        x,y,z,R=frame.setFrameData(a1,a2,self.m_procedure.definition[segName]["TF"]['sequence'])

                        cframe=frame.Frame()
                        cframe.m_axisX=x
                        cframe.m_axisY=y
                        cframe.m_axisZ=z
                        cframe.setRotation(R)
                        cframe.setTranslation(ptOrigin)

                        segPicked.getReferential("TF").addMotionFrame(copy.deepcopy(cframe) )


                if self.m_method == enums.motionMethod.Sodervisk :
                    tms= segPicked.m_tracking_markers
                    for i in range(0,self.m_aqui.GetPointFrameNumber()):
                        visibleMarkers = btkTools.getVisibleMarkersAtFrame(self.m_aqui,tms,i)

                        arrayStatic = np.zeros((len(visibleMarkers),3))
                        arrayDynamic = np.zeros((len(visibleMarkers),3))

                        j=0
                        for vm in visibleMarkers:
                            arrayStatic[j,:] = segPicked.getReferential("TF").static.getNode_byLabel(vm).m_global
                            arrayDynamic[j,:] = self.m_aqui.GetPoint(vm).GetValues()[i,:]
                            j+=1

                        Ropt, Lopt, RMSE, Am, Bm=motion.segmentalLeastSquare(arrayStatic,arrayDynamic)
                        R=np.dot(Ropt,segPicked.getReferential("TF").static.getRotation())
                        tOri=np.dot(Ropt,segPicked.getReferential("TF").static.getTranslation())+Lopt

                        cframe=frame.Frame()
                        cframe.setRotation(R)
                        cframe.setTranslation(tOri)
                        cframe.m_axisX=R[:,0]
                        cframe.m_axisY=R[:,1]
                        cframe.m_axisZ=R[:,2]
                        segPicked.getReferential("TF").addMotionFrame(copy.deepcopy(cframe) )

            if not self.m_noAnatomicalMotion:
                for segName in segments:
                    segPicked=self.m_model.getSegment(segName)

                    segPicked.anatomicalFrame.motion=[]

                    ndO = str(self.m_procedure.anatomicalDefinition[segName]['labels'][3])
                    ptO = segPicked.getReferential("TF").getNodeTrajectory(ndO)

                    csFrame=frame.Frame()
                    for i in range(0,self.m_aqui.GetPointFrameNumber()):
                        R = np.dot(segPicked.getReferential("TF").motion[i].getRotation(), segPicked.getReferential("TF").relativeMatrixAnatomic)
                        csFrame.update(R,ptO)
                        segPicked.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))
            else:
                for segName in self.m_procedure.definition:
                    segPicked=self.m_model.getSegment(segName)

                    segPicked.anatomicalFrame.motion=[]

                    ndO = str(self.m_procedure.definition[segName]["TF"]['labels'][3])
                    ptO = segPicked.getReferential("TF").getNodeTrajectory(ndO)

                    csFrame=frame.Frame()
                    for i in range(0,self.m_aqui.GetPointFrameNumber()):
                        R = np.dot(segPicked.getReferential("TF").motion[i].getRotation(), segPicked.getReferential("TF").relativeMatrixAnatomic)
                        csFrame.update(R,ptO)
                        segPicked.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))
    def compute(self):
        """Runs the motion filter."""

        if str(self.m_model) != "Basis Model":
           self.m_model.computeMotion(self.m_aqui,
                                      self.m_procedure.definition[0],
                                      self.m_procedure.definition[1],
                                      self.m_method,
                                      self.m_options)
        else :
            for segName in self.m_procedure.definition:

                segPicked=self.m_model.getSegment(segName)
                segPicked.getReferential("TF").motion =[]

                if self.m_method == enums.motionMethod.Determinist :
                    for i in range(0,self.m_aqui.GetPointFrameNumber()):

                        pt1=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName]["TF"]['labels'][0])).GetValues()[i,:]
                        pt2=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName]["TF"]['labels'][1])).GetValues()[i,:]
                        pt3=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName]["TF"]['labels'][2])).GetValues()[i,:]
                        ptOrigin=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName]["TF"]['labels'][3])).GetValues()[i,:]

                        a1=(pt2-pt1)
                        a1=np.nan_to_num(np.divide(a1,np.linalg.norm(a1)))

                        v=(pt3-pt1)
                        v=np.nan_to_num(np.divide(v,np.linalg.norm(v)))

                        a2=np.cross(a1,v)
                        a2=np.nan_to_num(np.divide(a2,np.linalg.norm(a2)))

                        x,y,z,R=frame.setFrameData(a1,a2,self.m_procedure.definition[segName]["TF"]['sequence'])

                        cframe=frame.Frame()
                        cframe.m_axisX=x
                        cframe.m_axisY=y
                        cframe.m_axisZ=z
                        cframe.setRotation(R)
                        cframe.setTranslation(ptOrigin)

                        segPicked.getReferential("TF").addMotionFrame(copy.deepcopy(cframe) )

                if self.m_method == enums.motionMethod.Sodervisk :

                    tms= segPicked.m_tracking_markers

                    for i in range(0,self.m_aqui.GetPointFrameNumber()):
                        visibleMarkers = btkTools.getVisibleMarkersAtFrame(self.m_aqui,tms,i)

                        # constructuion of the input of sodervisk
                        arrayStatic = np.zeros((len(visibleMarkers),3))
                        arrayDynamic = np.zeros((len(visibleMarkers),3))

                        j=0
                        for vm in visibleMarkers:
                            arrayStatic[j,:] = segPicked.getReferential("TF").static.getNode_byLabel(vm).m_global
                            arrayDynamic[j,:] = self.m_aqui.GetPoint(vm).GetValues()[i,:]
                            j+=1

                        Ropt, Lopt, RMSE, Am, Bm=motion.segmentalLeastSquare(arrayStatic,arrayDynamic)
                        R=np.dot(Ropt,segPicked.getReferential("TF").static.getRotation())
                        tOri=np.dot(Ropt,segPicked.getReferential("TF").static.getTranslation())+Lopt

                        cframe=frame.Frame()
                        cframe.setRotation(R)
                        cframe.setTranslation(tOri)
                        cframe.m_axisX=R[:,0]
                        cframe.m_axisY=R[:,1]
                        cframe.m_axisZ=R[:,2]

                        segPicked.getReferential("TF").addMotionFrame(copy.deepcopy(cframe) )



            if not self.m_noAnatomicalMotion:
                for segName in self.m_procedure.anatomicalDefinition:
                    segPicked=self.m_model.getSegment(segName)

                    segPicked.anatomicalFrame.motion=[]

                    ndO = str(self.m_procedure.anatomicalDefinition[segName]['labels'][3])
                    ptO = segPicked.getReferential("TF").getNodeTrajectory(ndO)

                    csFrame=frame.Frame()
                    for i in range(0,self.m_aqui.GetPointFrameNumber()):
                        R = np.dot(segPicked.getReferential("TF").motion[i].getRotation(), segPicked.getReferential("TF").relativeMatrixAnatomic)
                        csFrame.update(R,ptO[i])
                        segPicked.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))
            else:
                for segName in self.m_procedure.definition:
                    segPicked=self.m_model.getSegment(segName)

                    segPicked.anatomicalFrame.motion=[]
                    ndO = str(self.m_procedure.definition[segName]["TF"]['labels'][3])
                    ptO = segPicked.getReferential("TF").getNodeTrajectory(ndO)

                    csFrame=frame.Frame()
                    for i in range(0,self.m_aqui.GetPointFrameNumber()):
                        R = np.dot(segPicked.getReferential("TF").motion[i].getRotation(), segPicked.getReferential("TF").relativeMatrixAnatomic)
                        csFrame.update(R,ptO[i])
                        segPicked.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))



class TrackingMarkerDecompositionFilter(object):
    """Decompose tracking markers into their component directions.

    This filter separates tracking markers into three components (directions) and appends
    them to the acquisition as individual markers. The decomposition directions depend
    on the segment and marker names. For example, foot markers are decomposed into
    superior-inferior, medial-lateral, and proximal-distal directions.

    Args:
        iModel (Model): The model instance containing segment and marker information.
        iAcq (btk.btkAcquisition): The acquisition instance from which marker data will be extracted.

    Attributes:
        m_model (Model): The model instance.
        m_acq (btk.btkAcquisition): The acquisition instance.
    """

    def __init__(self, iModel: Model, iAcq: btk.btkAcquisition):
        self.m_model = iModel
        self.m_acq = iAcq

    def decompose(self):
        """Performs the decomposition of tracking markers.

        Decomposes each tracking marker in the model into three orthogonal components
        and appends these as new markers in the acquisition data.
        """
        for seg in self.m_model.m_segmentCollection:
            if  "Proximal" not in seg.name:
                if "Foot" in seg.name:
                    suffix = ["_supInf", "_medLat", "_proDis"]
                elif "Pelvis" in seg.name:
                    suffix = ["_posAnt", "_medLat", "_supInf"]
                else:
                    suffix = ["_posAnt", "_medLat", "_proDis"]

                copyTrackingMarkers = list(seg.m_tracking_markers) # copy of list

                # add direction point as tracking markers and copy node
                for marker in copyTrackingMarkers:
                    globalNodePos = seg.anatomicalFrame.static.getNode_byLabel(marker).m_global

                    seg.anatomicalFrame.static.addNode(marker+suffix[0],globalNodePos,positionType="Global")
                    seg.getReferential("TF").static.addNode(marker+suffix[0],globalNodePos,positionType="Global")

                    seg.anatomicalFrame.static.addNode(marker+suffix[1],globalNodePos,positionType="Global")
                    seg.getReferential("TF").static.addNode(marker+suffix[1],globalNodePos,positionType="Global")

                    seg.anatomicalFrame.static.addNode(marker+suffix[2],globalNodePos,positionType="Global")
                    seg.getReferential("TF").static.addNode(marker+suffix[2],globalNodePos,positionType="Global")

                    seg.addTrackingMarkerLabel(str(marker+suffix[0]))
                    seg.addTrackingMarkerLabel(str(marker+suffix[1]))
                    seg.addTrackingMarkerLabel(str(marker+suffix[2]))


                # decompose tracking marker in the acq
                for marker in copyTrackingMarkers:

                    nodeTraj= seg.anatomicalFrame.getNodeTrajectory(marker)
                    markersTraj =self.m_acq.GetPoint(marker).GetValues()

                    markerTrajectoryX=np.array( [ markersTraj[:,0], nodeTraj[:,1],    nodeTraj[:,2]]).T
                    markerTrajectoryY=np.array( [ nodeTraj[:,0],    markersTraj[:,1], nodeTraj[:,2]]).T
                    markerTrajectoryZ=np.array( [ nodeTraj[:,0],    nodeTraj[:,1],    markersTraj[:,2]]).T

                    btkTools.smartAppendPoint(self.m_acq,marker+suffix[0],markerTrajectoryX,PointType="Marker", desc="")
                    btkTools.smartAppendPoint(self.m_acq,marker+suffix[1],markerTrajectoryY,PointType="Marker", desc="")
                    btkTools.smartAppendPoint(self.m_acq,marker+suffix[2],markerTrajectoryZ,PointType="Marker", desc="")



# ----- Joint angles -----

class ModelJCSFilter(object):
    """Compute the relative joint angles using joint coordinate systems.

    This filter calculates the relative angles between segments at each joint using
    the anatomical and technical coordinate systems defined in the model. The angles
    are computed for each joint and stored as points in the acquisition instance.

    Args:
        iMod (Model): The model instance containing joint and segment information.
        acq (btk.btkAcquisition): The acquisition instance to which the computed angles will be added.

    Attributes:
        m_aqui (btk.btkAcquisition): The acquisition instance.
        m_model (Model): The model instance.
        m_fixEuler (bool): Flag to determine if Euler angles should be fixed.
    """

    def __init__(self, iMod: Model, acq: btk.btkAcquisition):
        self.m_aqui = acq
        self.m_model = iMod
        self.m_fixEuler = True

    def setFixEuler(self, fix: bool):
        """Set the flag to fix Euler angles.

        Args:
            fix (bool): If True, Euler angles will be fixed.
        """
        self.m_fixEuler =  bool

    def compute(self, description: str = "", pointLabelSuffix: str = None):
        """Run the joint coordinate system filter.

        Computes the relative joint angles for each joint defined in the model and
        stores them in the acquisition instance.

        Args:
            description (str, optional): Short description to be added to the point labels.
            pointLabelSuffix (str, optional): Suffix to be added to the point labels.
        """


        for it in  self.m_model.m_jointCollection:
            LOGGER.logger.debug("---Processing of %s---"  % it.m_label)
            LOGGER.logger.debug(" proximal : %s "% it.m_proximalLabel)
            LOGGER.logger.debug(" distal : %s "% it.m_distalLabel)


            jointLabel = it.m_label

            proxSeg = self.m_model.getSegment(it.m_proximalLabel)
            distSeg = self.m_model.getSegment(it.m_distalLabel)

            jointValues = np.zeros((self.m_aqui.GetPointFrameNumber(),3))
            for i in range (0, self.m_aqui.GetPointFrameNumber()):
                Rprox = proxSeg.anatomicalFrame.motion[i].getRotation()
                Rdist = distSeg.anatomicalFrame.motion[i].getRotation()
                Rrelative= np.dot(Rprox.T, Rdist)


                if it.m_sequence == "XYZ":
                    Euler1,Euler2,Euler3 = euler.euler_xyz(Rrelative)
                elif it.m_sequence == "XZY":
                    Euler1,Euler2,Euler3 = euler.euler_xzy(Rrelative)
                elif it.m_sequence == "YXZ":
                    Euler1,Euler2,Euler3 = euler.euler_yxz(Rrelative)
                elif it.m_sequence == "YZX":
                    Euler1,Euler2,Euler3 = euler.euler_yzx(Rrelative)
                elif it.m_sequence == "ZXY":
                    Euler1,Euler2,Euler3 = euler.euler_zxy(Rrelative)
                elif it.m_sequence == "ZYX":
                    Euler1,Euler2,Euler3 = euler.euler_zyx(Rrelative)
                else:
                    raise Exception("[pycga] joint sequence unknown ")

                jointValues[i,0] = Euler1
                jointValues[i,1] = Euler2
                jointValues[i,2] = Euler3



            descriptorInfos = self.m_model.getClinicalDescriptor(enums.DataType.Angle,jointLabel)
            if  descriptorInfos:
                LOGGER.logger.debug("joint label (%s) in clinical descriptors" %(jointLabel) )
                jointFinalValues = np.zeros((jointValues.shape))
                jointFinalValues[:,0] =  np.rad2deg(descriptorInfos["SaggitalCoeff"] * (jointValues[:,descriptorInfos["SaggitalIndex"]] + descriptorInfos["SaggitalOffset"]))
                jointFinalValues[:,1] =  np.rad2deg(descriptorInfos["CoronalCoeff"] * (jointValues[:,descriptorInfos["CoronalIndex"]] + descriptorInfos["CoronalOffset"]))
                jointFinalValues[:,2] =  np.rad2deg(descriptorInfos["TransversalCoeff"] * (jointValues[:,descriptorInfos["TransversalIndex"]] + descriptorInfos["TransversalOffset"]))
            else:
                LOGGER.logger.debug("no clinical descriptor for joint label (%s)" %(jointLabel) )
                jointFinalValues = np.rad2deg(jointValues)

            if self.m_fixEuler:
                dest = np.deg2rad(np.array([0,0,0]))
                for i in range (0, self.m_aqui.GetPointFrameNumber()):
                    jointFinalValues[i,:] = euler.wrapEulerTo(np.deg2rad(jointFinalValues[i,:]), dest)
                    dest = jointFinalValues[i,:]

                jointFinalValues = np.rad2deg(jointFinalValues)

            fulljointLabel  = jointLabel + "Angles_" + pointLabelSuffix if pointLabelSuffix is not None else jointLabel+"Angles"
            btkTools.smartAppendPoint(self.m_aqui,
                             fulljointLabel,
                             jointFinalValues,PointType="Angle", desc=description)


class ModelAbsoluteAnglesFilter(object):
    """Compute absolute joint angles.

    This filter calculates the absolute angles of specified segments in the model. The angles are 
    expressed relative to a global frame, making them 'absolute' in the context of the model's motion.
    The computed angles are stored as points in the acquisition instance.

    Args:
        iMod (Model): The model instance containing segment information.
        acq (btk.btkAcquisition): The acquisition instance where the angles will be stored.
        segmentLabels (List[str]): Labels of the segments for which the angles are computed.
        angleLabels (List[str]): Labels for the angles to be computed.
        eulerSequences (List[str]): Euler sequences for angle computations.
        globalFrameOrientation (str): Orientation of the global frame.
        forwardProgression (bool): Indicates the direction of subject's movement.
    """

    def __init__(self, iMod:Model, acq:btk.btkAcquisition, segmentLabels:List[str]=[],angleLabels:List[str]=[], eulerSequences:List[str]=[], 
                 globalFrameOrientation:str = "XYZ", forwardProgression:bool = True):

        self.m_aqui = acq
        self.m_model = iMod
        self.m_segmentLabels = segmentLabels
        self.m_angleLabels = angleLabels
        self.m_eulerSequences = eulerSequences
        self.m_globalFrameOrientation = globalFrameOrientation
        self.m_forwardProgression = forwardProgression


    def compute(self, 
                description: str = "absolute", 
                pointLabelSuffix: Optional[str] = None):
        """Run the absolute angles filter.

        Calculates and stores the absolute angles of specified segments in the model.

        Args:
            description (str, optional): Description added to the angle labels.
            pointLabelSuffix (Optional[str], optional): Suffix added to the angle labels.
        """

        for index in range (0, len(self.m_segmentLabels)):

            absoluteAngleValues = np.zeros((self.m_aqui.GetPointFrameNumber(),3))


            if self.m_globalFrameOrientation == "XYZ":
                if self.m_forwardProgression:
                    pt1=np.array([0,0,0])
                    pt2=np.array([1,0,0])
                    pt3=np.array([0,0,1])
                else:
                    pt1=np.array([0,0,0])
                    pt2=np.array([-1,0,0])
                    pt3=np.array([0,0,1])

                a1=(pt2-pt1)
                v=(pt3-pt1)
                a2=np.cross(a1,v)
                x,y,z,Rglobal=frame.setFrameData(a1,a2,"XYiZ")

            if self.m_globalFrameOrientation == "YXZ":
                if self.m_forwardProgression:

                    pt1=np.array([0,0,0])
                    pt2=np.array([0,1,0])
                    pt3=np.array([0,0,1])
                else:
                    pt1=np.array([0,0,0])
                    pt2=np.array([0,-1,0])
                    pt3=np.array([0,0,1])

                a1=(pt2-pt1)
                v=(pt3-pt1)
                a2=np.cross(a1,v)
                x,y,z,Rglobal=frame.setFrameData(a1,a2,"XYiZ")

            seg = self.m_model.getSegment(self.m_segmentLabels[index])
            side  = seg.side
            eulerSequence = self.m_eulerSequences[index]

            if eulerSequence == "TOR":
                LOGGER.logger.debug( "segment (%s) - sequence Tilt-Obliquity-Rotation used" %(seg.name) )
            elif eulerSequence == "TRO":
                LOGGER.logger.debug( "segment (%s) - sequence Tilt-Rotation-Obliquity used" %(seg.name) )
            elif eulerSequence == "ROT":
                LOGGER.logger.debug( "segment (%s) - sequence Rotation-Obliquity-Tilt used" %(seg.name) )
            elif eulerSequence == "RTO":
                LOGGER.logger.debug( "segment (%s) - sequence Rotation-Tilt-Obliquity used" %(seg.name) )
            elif eulerSequence == "OTR":
                LOGGER.logger.debug( "segment (%s) - sequence Obliquity-Tilt-Rotation used" %(seg.name) )
            elif eulerSequence == "ORT":
                LOGGER.logger.debug( "segment (%s) - sequence Obliquity-Rotation-Tilt used" %(seg.name) )
            else:
                pass



            for i in range (0, self.m_aqui.GetPointFrameNumber()):
                Rseg = seg.anatomicalFrame.motion[i].getRotation()
                Rrelative= np.dot(Rglobal.T,Rseg)

                if eulerSequence == "TOR":
                    tilt,obliquity,rotation = euler.euler_yxz(Rrelative)
                elif eulerSequence == "TRO":
                    tilt,rotation,obliquity = euler.euler_yzx(Rrelative)
                elif eulerSequence == "ROT":
                    rotation,obliquity,tilt = euler.euler_zxy(Rrelative)
                elif eulerSequence == "RTO":
                    rotation,tilt,obliquity = euler.euler_zyx(Rrelative)
                elif eulerSequence == "OTR":
                    obliquity,tilt,rotation = euler.euler_xyz(Rrelative)
                elif eulerSequence == "ORT":
                    obliquity,rotation,tilt = euler.euler_xzy(Rrelative)
                elif eulerSequence == "YXZ":
                    tilt,obliquity,rotation = euler.euler_yxz(Rrelative)#,similarOrder = False)
                elif eulerSequence == "YZX":
                    tilt,obliquity,rotation = euler.euler_yzx(Rrelative)#,similarOrder = False)
                elif eulerSequence == "ZXY":
                    tilt,obliquity,rotation = euler.euler_zxy(Rrelative)#,similarOrder = False)
                elif eulerSequence == "ZYX":
                    tilt,obliquity,rotation = euler.euler_zyx(Rrelative)#,similarOrder = False)
                elif eulerSequence == "XYZ":
                    tilt,obliquity,rotation = euler.euler_xyz(Rrelative)#,similarOrder = False)
                elif eulerSequence == "XZY":
                    tilt,obliquity,rotation = euler.euler_xzy(Rrelative)#,similarOrder = False)
                else:
                    LOGGER.logger.debug("no sequence defined for absolute angles. sequence YXZ selected by default" )
                    tilt,obliquity,rotation = euler.euler_yxz(Rrelative)

                absoluteAngleValues[i,0] = tilt
                absoluteAngleValues[i,1] = obliquity
                absoluteAngleValues[i,2] = rotation

            segName = self.m_segmentLabels[index]

            if side == enums.SegmentSide.Left or side == enums.SegmentSide.Right:

                if  self.m_model.getClinicalDescriptor(enums.DataType.Angle,segName):
                    descriptorInfos = self.m_model.getClinicalDescriptor(enums.DataType.Angle,segName)
                    absoluteAngleValuesFinal = np.zeros((absoluteAngleValues.shape))
                    absoluteAngleValuesFinal[:,0] =  np.rad2deg(descriptorInfos["SaggitalCoeff"] * (absoluteAngleValues[:,descriptorInfos["SaggitalIndex"]] + descriptorInfos["SaggitalOffset"]))
                    absoluteAngleValuesFinal[:,1] =  np.rad2deg(descriptorInfos["CoronalCoeff"] * (absoluteAngleValues[:,descriptorInfos["CoronalIndex"]] + descriptorInfos["CoronalOffset"]))
                    absoluteAngleValuesFinal[:,2] =  np.rad2deg(descriptorInfos["TransversalCoeff"] * (absoluteAngleValues[:,descriptorInfos["TransversalIndex"]] + descriptorInfos["TransversalOffset"]))

                    fullAngleLabel  = self.m_angleLabels[index] + "Angles_" + pointLabelSuffix if pointLabelSuffix is not None else self.m_angleLabels[index]+"Angles"

                    dest = np.deg2rad(np.array([0,0,0]))
                    for i in range (0, self.m_aqui.GetPointFrameNumber()):
                         absoluteAngleValuesFinal[i,:] = euler.wrapEulerTo(np.deg2rad(absoluteAngleValuesFinal[i,:]), dest)
                         dest = absoluteAngleValuesFinal[i,:]
                    absoluteAngleValuesFinal = np.rad2deg(absoluteAngleValuesFinal)

                    btkTools.smartAppendPoint(self.m_aqui, fullAngleLabel,
                                         absoluteAngleValuesFinal,PointType="Angle", desc=description)

                else:
                    LOGGER.logger.debug("no clinical descriptor for segment label (%s)" %(segName))
                    absoluteAngleValuesFinal = np.rad2deg(absoluteAngleValues)

                    fullAngleLabel  = self.m_angleLabels[index] + "Angles_" + pointLabelSuffix if pointLabelSuffix is not None else self.m_angleLabels[index]+"Angles"


                    btkTools.smartAppendPoint(self.m_aqui, fullAngleLabel,
                                         absoluteAngleValuesFinal,PointType="Angle", desc=description)

            if side == enums.SegmentSide.Central:
                descriptorInfos1 = self.m_model.getClinicalDescriptor(enums.DataType.Angle,segName)
                if  descriptorInfos1:
                    absoluteAngleValuesFinal = np.zeros((absoluteAngleValues.shape))
                    absoluteAngleValuesFinal[:,0] =  np.rad2deg(descriptorInfos1["SaggitalCoeff"] * (absoluteAngleValues[:,descriptorInfos1["SaggitalIndex"]] + descriptorInfos1["SaggitalOffset"]))
                    absoluteAngleValuesFinal[:,1] =  np.rad2deg(descriptorInfos1["CoronalCoeff"] * (absoluteAngleValues[:,descriptorInfos1["CoronalIndex"]] + descriptorInfos1["CoronalOffset"]))
                    absoluteAngleValuesFinal[:,2] =  np.rad2deg(descriptorInfos1["TransversalCoeff"] * (absoluteAngleValues[:,descriptorInfos1["TransversalIndex"]] + descriptorInfos1["TransversalOffset"]))

                    fullAngleLabel  = self.m_angleLabels[index] + "Angles_" + pointLabelSuffix if pointLabelSuffix is not None else self.m_angleLabels[index]+"Angles"

                    dest = np.deg2rad(np.array([0,0,0]))
                    for i in range (0, self.m_aqui.GetPointFrameNumber()):
                         absoluteAngleValuesFinal[i,:] = euler.wrapEulerTo(np.deg2rad(absoluteAngleValuesFinal[i,:]), dest)
                         dest = absoluteAngleValuesFinal[i,:]
                    absoluteAngleValuesFinal = np.rad2deg(absoluteAngleValuesFinal)


                    btkTools.smartAppendPoint(self.m_aqui, fullAngleLabel,
                                         absoluteAngleValuesFinal,PointType="Angle", desc=description)

                # case Left
                descriptorInfos2 = self.m_model.getClinicalDescriptor(enums.DataType.Angle,str("L"+segName))
                if  descriptorInfos2:
                    absoluteAngleValuesFinal = np.zeros((absoluteAngleValues.shape))
                    absoluteAngleValuesFinal[:,0] =  np.rad2deg(descriptorInfos2["SaggitalCoeff"] * (absoluteAngleValues[:,descriptorInfos2["SaggitalIndex"]] + descriptorInfos2["SaggitalOffset"]))
                    absoluteAngleValuesFinal[:,1] =  np.rad2deg(descriptorInfos2["CoronalCoeff"] * (absoluteAngleValues[:,descriptorInfos2["CoronalIndex"]] + descriptorInfos2["CoronalOffset"]))
                    absoluteAngleValuesFinal[:,2] =  np.rad2deg(descriptorInfos2["TransversalCoeff"] * (absoluteAngleValues[:,descriptorInfos2["TransversalIndex"]] + descriptorInfos2["TransversalOffset"]))

                    fullAngleLabel  = "L" + self.m_angleLabels[index] + "Angles_" + pointLabelSuffix if pointLabelSuffix is not None else "L" +self.m_angleLabels[index]+"Angles"


                    dest = np.deg2rad(np.array([0,0,0]))
                    for i in range (0, self.m_aqui.GetPointFrameNumber()):
                         absoluteAngleValuesFinal[i,:] = euler.wrapEulerTo(np.deg2rad(absoluteAngleValuesFinal[i,:]), dest)
                         dest = absoluteAngleValuesFinal[i,:]
                    absoluteAngleValuesFinal = np.rad2deg(absoluteAngleValuesFinal)



                    btkTools.smartAppendPoint(self.m_aqui, fullAngleLabel,
                                         absoluteAngleValuesFinal,PointType="Angle", desc=description)


                # case Right
                descriptorInfos3 = self.m_model.getClinicalDescriptor(enums.DataType.Angle,str("R"+segName))
                if descriptorInfos3 :
                    absoluteAngleValuesFinal = np.zeros((absoluteAngleValues.shape))
                    absoluteAngleValuesFinal[:,0] =  np.rad2deg(descriptorInfos3["SaggitalCoeff"] * (absoluteAngleValues[:,descriptorInfos3["SaggitalIndex"]] + descriptorInfos3["SaggitalOffset"]))
                    absoluteAngleValuesFinal[:,1] =  np.rad2deg(descriptorInfos3["CoronalCoeff"] * (absoluteAngleValues[:,descriptorInfos3["CoronalIndex"]] + descriptorInfos3["CoronalOffset"]))
                    absoluteAngleValuesFinal[:,2] =  np.rad2deg(descriptorInfos3["TransversalCoeff"] * (absoluteAngleValues[:,descriptorInfos3["TransversalIndex"]] + descriptorInfos3["TransversalOffset"]))

                    fullAngleLabel  = "R" + self.m_angleLabels[index] + "Angles_" + pointLabelSuffix if pointLabelSuffix is not None else "R" +self.m_angleLabels[index]+"Angles"

                    dest = np.deg2rad(np.array([0,0,0]))
                    for i in range (0, self.m_aqui.GetPointFrameNumber()):
                         absoluteAngleValuesFinal[i,:] = euler.wrapEulerTo(np.deg2rad(absoluteAngleValuesFinal[i,:]), dest)
                         dest = absoluteAngleValuesFinal[i,:]
                    absoluteAngleValuesFinal = np.rad2deg(absoluteAngleValuesFinal)

                    btkTools.smartAppendPoint(self.m_aqui, fullAngleLabel,
                                         absoluteAngleValuesFinal,PointType="Angle", desc=description)

                if not descriptorInfos1 and not descriptorInfos2 and not descriptorInfos3:
                    LOGGER.logger.debug("no clinical descriptor for segment label (%s)" %(segName))
                    absoluteAngleValuesFinal = np.rad2deg(absoluteAngleValues)

                    fullAngleLabel  = self.m_angleLabels[index] + "Angles_" + pointLabelSuffix if pointLabelSuffix is not None else self.m_angleLabels[index]+"Angles"
                    btkTools.smartAppendPoint(self.m_aqui, fullAngleLabel,
                                         absoluteAngleValuesFinal,PointType="Angle", desc=description)



# ----- Force plates -----

class ForcePlateAssemblyFilter(object):
    """
    Assemble force plates with the model for dynamic trials.

    This filter associates force plates with specified segments of the model during dynamic trials. 
    It calculates ground reaction forces and moments and appends them to the acquisition instance. 
    The association is based on mapped force plate letters indicating the body side in contact with the force plates.

    Args:
        iMod (Model): A model instance.
        btkAcq (btk.btkAcquisition): An acquisition instance of a dynamic trial.
        mappedForcePlateLetters (str): String indicating body side of the segment in contact with the force plate.
        leftSegmentLabel (str, optional): Left segment label to assemble with force plates. Defaults to "Left Foot".
        rightSegmentLabel (str, optional): Right segment label to assemble with force plates. Defaults to "Right Foot".

    """

    def __init__(self, 
                 iMod: Model, 
                 btkAcq: btk.btkAcquisition, 
                 mappedForcePlateLetters: str, 
                 leftSegmentLabel: str = "Left Foot", 
                 rightSegmentLabel: str = "Right Foot"):

        self.m_aqui = btkAcq
        self.m_model = iMod
        self.m_mappedForcePlate = mappedForcePlateLetters
        self.m_leftSeglabel = leftSegmentLabel
        self.m_rightSeglabel = rightSegmentLabel
        self.m_model = iMod

        # zeroing externalDevice
        self.m_model.getSegment(self.m_leftSeglabel).zeroingExternalDevice()
        self.m_model.getSegment(self.m_rightSeglabel).zeroingExternalDevice()


    def compute(self, pointLabelSuffix: Optional[str] = None):
        """
        Run the filter to associate force plates with model segments.

        Calculates ground reaction forces and moments and appends them to the acquisition instance.

        Args:
            pointLabelSuffix (Optional[str], optional): Suffix to append to the ground reaction force and moment labels.
        """

        pfe = btk.btkForcePlatformsExtractor()
        grwf = btk.btkGroundReactionWrenchFilter()
        pfe.SetInput(self.m_aqui)
        pfc = pfe.GetOutput()
        grwf.SetInput(pfc)
        grwc = grwf.GetOutput()
        grwc.Update()

        appf = self.m_aqui.GetNumberAnalogSamplePerFrame()
        pfn = self.m_aqui.GetPointFrameNumber()

        fpwf = btk.btkForcePlatformWrenchFilter() # the wrench of the center of the force platform data, expressed in the global frame
        fpwf.SetInput(pfe.GetOutput())
        fpwc = fpwf.GetOutput()
        fpwc.Update()


        left_forceValues =  np.zeros((pfn,3))
        left_momentValues =  np.zeros((pfn,3))

        right_forceValues =  np.zeros((pfn,3))
        right_momentValues =  np.zeros((pfn,3))

        i = 0
        for l in self.m_mappedForcePlate:

            if l == "L":
                self.m_model.getSegment(self.m_leftSeglabel).addExternalDeviceWrench(grwc.GetItem(i))
                left_forceValues = left_forceValues + fpwc.GetItem(i).GetForce().GetValues()[::appf]
                left_momentValues = left_momentValues + fpwc.GetItem(i).GetMoment().GetValues()[::appf]

            elif l == "R":
                self.m_model.getSegment(self.m_rightSeglabel).addExternalDeviceWrench(grwc.GetItem(i))
                right_forceValues = right_forceValues + fpwc.GetItem(i).GetForce().GetValues()[::appf]
                right_momentValues = right_momentValues + fpwc.GetItem(i).GetMoment().GetValues()[::appf]

            else:
                LOGGER.logger.debug("force plate %i with no data" %(i))
            i+=1

        if "L" in self.m_mappedForcePlate:
            self.m_model.getSegment(self.m_leftSeglabel).downSampleExternalDeviceWrenchs(appf)
        if "R" in self.m_mappedForcePlate:
            self.m_model.getSegment(self.m_rightSeglabel).downSampleExternalDeviceWrenchs(appf)

        # ground reaction force and moment
        if "Bodymass" in self.m_model.mp:
            bodymass = self.m_model.mp["Bodymass"]
        else:
            bodymass = 1.0
            LOGGER.logger.warning("[pyCGM2] - bodymass is not within your mp data. non-normalized GroundReaction Force and Moment ouput")

        forceLabel  = "LGroundReactionForce_"+pointLabelSuffix if pointLabelSuffix is not None else "LGroundReactionForce"
        momentLabel  = "LGroundReactionMoment_"+pointLabelSuffix if pointLabelSuffix is not None else "LGroundReactionMoment"

        btkTools.smartAppendPoint(self.m_aqui,forceLabel,
                         left_forceValues / bodymass,
                         PointType="Force", desc="")
        btkTools.smartAppendPoint(self.m_aqui,momentLabel,
                         left_momentValues / bodymass,
                         PointType="Moment", desc="")

        forceLabel  = "RGroundReactionForce_"+pointLabelSuffix if pointLabelSuffix is not None else "RGroundReactionForce"
        momentLabel  = "RGroundReactionMoment_"+pointLabelSuffix if pointLabelSuffix is not None else "RGroundReactionMoment"
        btkTools.smartAppendPoint(self.m_aqui,forceLabel,
                         right_forceValues / bodymass,
                         PointType="Force", desc="")
        btkTools.smartAppendPoint(self.m_aqui,momentLabel,
                         right_momentValues / bodymass,
                         PointType="Moment", desc="")
        
        
class GroundReactionIntegrationFilter(object):
    """
    Integrates ground reaction forces into the model during dynamic trials.

    This filter computes and integrates the ground reaction forces based on the mapped force plate letters 
    and the body mass of the subject. It's used to enhance the accuracy of force plate data integration 
    into the biomechanical model during dynamic trials.

    Args:
        procedure (ForcePlateIntegrationProcedure): An instance of a procedure defining specific computations.
        btkAcq (btk.btkAcquisition): An acquisition instance of a dynamic trial.
        mappedForcePlateLetters (str): String indicating body side of the segment in contact with the force plate.
        bodymass (float): The body mass of the subject.
        globalFrameOrientation (str, optional): Orientation of the global frame. Defaults to "XYZ".
        forwardProgression (bool, optional): Indicates if the subject moves in the same direction as the global longitudinal axis. Defaults to True.

    """

    def __init__(self, 
                 procedure: ForcePlateIntegrationProcedure, 
                 btkAcq: btk.btkAcquisition, 
                 mappedForcePlateLetters: str, 
                 bodymass: float, 
                 globalFrameOrientation: str = "XYZ", 
                 forwardProgression: bool = True):

    
        self.m_aqui = btkAcq
        self.m_mappedForcePlate = mappedForcePlateLetters
        self.m_bodymass = bodymass
        self.m_globalFrameOrientation = globalFrameOrientation
        self.m_forwardProgression = forwardProgression
        self.m_procedure = procedure

    def compute(self):
        """
        Execute the filter to integrate ground reaction forces into the model.

        This method runs the defined procedure to compute and integrate the ground reaction forces 
        into the biomechanical model based on the force plate data and subject's body mass.
        """

        self.m_procedure.compute(self.m_aqui,
                                 self.m_mappedForcePlate,
                                 self.m_bodymass,
                                 self.m_globalFrameOrientation,self.m_forwardProgression)



class GroundReactionForceAdapterFilter(object):
    """
    Filter to standardize ground reaction force data in a biomechanical model.

    This filter processes ground reaction force data to conform to a standardized nomenclature and coordinate system,
    making it consistent and easier to interpret across different analyses. The output force vectors are aligned with 
    the specified global frame orientation and take into account the direction of progression.

    Args:
        btkAcq (btk.btkAcquisition): Acquisition instance containing dynamic trial data.
        globalFrameOrientation (str, optional): Orientation of the global reference frame. Defaults to "XYZ".
        forwardProgression (bool, optional): Indicates if the subject moves in the same direction as the global longitudinal axis. Defaults to True.

    Example:

    ```python

        gaitFilename="gait1.c3d"
        acqGaitYf = btkTools.smartReader(data_path +  gaitFilename)
        mfpa = "RLX"
     
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGaitYf,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(acqGaitYf,mappedForcePlate)
        LOGGER.logger.warning("Manual Force plate assignment : %s" %mappedForcePlate)

        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,acqGaitYf,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix=None)
        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGaitYf)

        cgrff = modelFilters.GroundReactionForceAdapterFilter(acqGaitYf,globalFrameOrientation=globalFrame, forwardProgression=forwardProgression)
        cgrff.compute()
    ```
    
    """
    def __init__(self, 
                 btkAcq: btk.btkAcquisition, 
                 globalFrameOrientation: str = "XYZ", 
                 forwardProgression: bool = True):

        self.m_aqui = btkAcq
        self.m_globalFrameOrientation = globalFrameOrientation
        self.m_forwardProgression = forwardProgression

    def compute(self,pointLabelSuffix:Optional[str]=None):
        """
        Execute the filter to standardize ground reaction forces.

        The method adjusts ground reaction force vectors according to the global frame orientation and subject's 
        direction of progression. It appends standardized force vectors to the acquisition data.

        Args:
            pointLabelSuffix (str, optional): Suffix to be added to the label of the output ground reaction force data. Defaults to None.
        """       

        if self.m_globalFrameOrientation == "XYZ":
            if self.m_forwardProgression:
               Rglobal= np.array([[1, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]])
            else:
                Rglobal= np.array([[-1, 0, 0],
                                  [0, -1, 0],
                                  [0, 0, 1]])

        if self.m_globalFrameOrientation == "YXZ":
            if self.m_forwardProgression:
                Rglobal= np.array([[0, -1, 0],
                                  [1, 0, 0],
                                  [0, 0, 1]])
            else:
                Rglobal= np.array([[0, 1, 0],
                                  [-1, 0, 0],
                                  [0, 0, 1]])


        LGroundReactionForce_label  = "LGroundReactionForce"+pointLabelSuffix if pointLabelSuffix is not None else "LGroundReactionForce"
        RGroundReactionForce_label  = "RGroundReactionForce"+pointLabelSuffix if pointLabelSuffix is not None else "RGroundReactionForce"



        leftFlag= True
        try:
            LGroundReactionForceValues = self.m_aqui.GetPoint(LGroundReactionForce_label).GetValues()
        except:
            leftFlag= False
            LOGGER.logger.warning("No LGroundReactionForce") 
            
        if leftFlag:

            valuesL = np.zeros((self.m_aqui.GetPointFrameNumber(),3))
            for i in range (0, self.m_aqui.GetPointFrameNumber()):
                    valuesL[i,:] = np.dot(Rglobal.T,LGroundReactionForceValues[i,:])

            label =  LGroundReactionForce_label[0]+"Stan"+LGroundReactionForce_label[1:]
            btkTools.smartAppendPoint(self.m_aqui,label,
                         valuesL,
                         PointType="Force", desc="[0]forward(+)backward(-) [1]lateral(+)medial(-) [2] upward(+)downward(-)")


        rightFlag= True
        try:
            RGroundReactionForceValues = self.m_aqui.GetPoint(RGroundReactionForce_label).GetValues()
        except:
            rightFlag= False
            LOGGER.logger.warning("No RGroundReactionForce")

        if rightFlag:
            valuesR = np.zeros((self.m_aqui.GetPointFrameNumber(),3))
            for i in range (0, self.m_aqui.GetPointFrameNumber()):
                    valuesR[i,:] = np.dot(Rglobal.T,RGroundReactionForceValues[i,:])

            valuesR[:,1] = -1.0*valuesR[:,1] # +/- correct the lateral/medial for the right foot

            label =  RGroundReactionForce_label[0]+"Stan"+RGroundReactionForce_label[1:]
            btkTools.smartAppendPoint(self.m_aqui,label,
                            valuesR,
                            PointType="Force", desc="[0]forward(+)backward(-) [1]lateral(+)medial(-) [2] upward(+)downward(-)")
            

        
# ----- Inverse dynamics -----

class InverseDynamicFilter(object):
    """
    Compute joint forces and moments from inverse dynamics.

    This filter calculates joint forces and moments based on inverse dynamics analysis of a biomechanical model during a dynamic trial.

    Args:
        iMod (Model): A biomechanical model instance.
        btkAcq (btk.btkAcquisition): An acquisition instance of a dynamic trial.
        procedure (Optional[InverseDynamicProcedure]): An inverse dynamic procedure, if any. Defaults to None.
        gravityVector (np.array): The gravity vector, typically [0, 0, -1]. Defaults to np.array([0, 0, -1]).
        scaleToMeter (float): Scale factor to convert measurements to meters. Defaults to 0.001.
        projection (enums.MomentProjection): Coordinate system in which joint moments and forces are expressed. Defaults to enums.MomentProjection.Distal.
        globalFrameOrientation (str): Global frame orientation. Defaults to "XYZ".
        forwardProgression (bool): Flag indicating if subject moves in the same direction as the global longitudinal axis. Defaults to True.
        exportMomentContributions (bool): Flag to export moment contributions. Defaults to False.
        **options: Additional optional arguments.

    Kargs:
        viconCGM1compatible(bool): replicate the Vicon Plugin-gait error related to the proximal and distal tibia

    """

    def __init__(self, 
                 iMod: Model, 
                 btkAcq: btk.btkAcquisition, 
                 procedure: Optional[InverseDynamicProcedure] = None, 
                 gravityVector: np.array = np.array([0, 0, -1]), 
                 scaleToMeter: float = 0.001,
                 projection: enums.MomentProjection = enums.MomentProjection.Distal,
                 globalFrameOrientation: str = "XYZ",
                 forwardProgression: bool = True,
                 exportMomentContributions: bool = False,
                 **options):

        self.m_aqui = btkAcq
        self.m_model = iMod
        self.m_gravity = 9.81 * gravityVector
        self.m_scaleToMeter = scaleToMeter
        self.m_procedure = procedure
        self.m_projection = projection
        self.m_exportMomentContributions = exportMomentContributions
        self.m_globalFrameOrientation = globalFrameOrientation
        self.m_forwardProgression = forwardProgression

        self.m_options = options

    def compute(self, pointLabelSuffix: Optional[str] = None):
        """
        Execute the inverse dynamics analysis and store the results in the acquisition instance.

        Joint forces and moments are calculated and added to the acquisition data.

        Args:
            pointLabelSuffix (str, optional): Suffix to be added to the label of the output joint forces and moments data. Defaults to None.
        """


        self.m_procedure.compute(self.m_model,self.m_aqui,self.m_gravity,self.m_scaleToMeter)


        if self.m_globalFrameOrientation == "XYZ":
            if self.m_forwardProgression:
                pt1=np.array([0,0,0])
                pt2=np.array([1,0,0])
                pt3=np.array([0,0,1])
            else:
                pt1=np.array([0,0,0])
                pt2=np.array([-1,0,0])
                pt3=np.array([0,0,1])

            a1=(pt2-pt1)
            v=(pt3-pt1)
            a2=np.cross(a1,v)
            x,y,z,Rglobal=frame.setFrameData(a1,a2,"XYiZ")

        if self.m_globalFrameOrientation == "YXZ":
            if self.m_forwardProgression:

                pt1=np.array([0,0,0])
                pt2=np.array([0,1,0])
                pt3=np.array([0,0,1])
            else:
                pt1=np.array([0,0,0])
                pt2=np.array([0,-1,0])
                pt3=np.array([0,0,1])

            a1=(pt2-pt1)
            v=(pt3-pt1)
            a2=np.cross(a1,v)
            x,y,z,Rglobal=frame.setFrameData(a1,a2,"XYiZ")




        for it in  self.m_model.m_jointCollection:

            if it.m_label not in ["ForeFoot"]:  # TODO : clumpsy... :-(  Think about a new method
                LOGGER.logger.debug("kinetics of %s"  %(it.m_label))
                LOGGER.logger.debug("proximal label :%s" %(it.m_proximalLabel))
                LOGGER.logger.debug("distal label :%s" %(it.m_distalLabel))

                jointLabel = it.m_label
                nFrames = self.m_aqui.GetPointFrameNumber()

                if "viconCGM1compatible" in self.m_options.keys() and self.m_options["viconCGM1compatible"]:
                    if it.m_label == "LAnkle":
                        proximalSegLabel = "Left Shank Proximal"
                    elif it.m_label == "RAnkle":
                        proximalSegLabel = "Right Shank Proximal"
                    else:
                        proximalSegLabel = it.m_proximalLabel
                else:
                    proximalSegLabel = it.m_proximalLabel

                if self.m_model.getSegment(it.m_distalLabel).m_proximalWrench is not None:
                    if self.m_projection != enums.MomentProjection.JCS and  self.m_projection != enums.MomentProjection.JCS_Dual:
                        if self.m_projection == enums.MomentProjection.Distal:
                            mot = self.m_model.getSegment(it.m_distalLabel).anatomicalFrame.motion
                        elif self.m_projection == enums.MomentProjection.Proximal:
                            mot = self.m_model.getSegment(proximalSegLabel).anatomicalFrame.motion

                        forceValues = np.zeros((nFrames,3))
                        momentValues = np.zeros((nFrames,3))
                        for i in range(0,nFrames ):
                            if self.m_projection == enums.MomentProjection.Global:
                                forceValues[i,:] = (1.0 / self.m_model.mp["Bodymass"]) * np.dot(Rglobal.T,
                                                                                            self.m_model.getSegment(it.m_distalLabel).m_proximalWrench.GetForce().GetValues()[i,:].T)
                                momentValues[i,:] = (1.0 / self.m_model.mp["Bodymass"]) * np.dot(Rglobal.T,
                                                                                          self.m_model.getSegment(it.m_distalLabel).m_proximalWrench.GetMoment().GetValues()[i,:].T)
                            else:
                                forceValues[i,:] = (1.0 / self.m_model.mp["Bodymass"]) * np.dot(mot[i].getRotation().T,
                                                                                        self.m_model.getSegment(it.m_distalLabel).m_proximalWrench.GetForce().GetValues()[i,:].T)
                                momentValues[i,:] = (1.0 / self.m_model.mp["Bodymass"]) * np.dot(mot[i].getRotation().T,
                                                                                        self.m_model.getSegment(it.m_distalLabel).m_proximalWrench.GetMoment().GetValues()[i,:].T)


                    else:

                        F = (1.0 / self.m_model.mp["Bodymass"]) * self.m_model.getSegment(it.m_distalLabel).m_proximalWrench.GetForce().GetValues()
                        M = (1.0 / self.m_model.mp["Bodymass"]) * self.m_model.getSegment(it.m_distalLabel).m_proximalWrench.GetMoment().GetValues()

                        proxSeg = self.m_model.getSegment(proximalSegLabel)
                        distSeg = self.m_model.getSegment(it.m_distalLabel)

                        # WARNING : I keep X-Y-Z sequence in output
                        forceValues = np.zeros((nFrames,3))
                        momentValues = np.zeros((nFrames,3))

                        for i in range(0,nFrames ):

                            if it.m_sequence == "XYZ":
                                e1 = proxSeg.anatomicalFrame.motion[i].m_axisX
                                e3 = distSeg.anatomicalFrame.motion[i].m_axisZ
                                order=[0,1,2]

                            elif it.m_sequence == "XZY":
                                e1 = proxSeg.anatomicalFrame.motion[i].m_axisX
                                e3 = distSeg.anatomicalFrame.motion[i].m_axisY
                                order=[0,2,1]

                            elif it.m_sequence == "YXZ":
                                e1 = proxSeg.anatomicalFrame.motion[i].m_axisY
                                e3 = distSeg.anatomicalFrame.motion[i].m_axisZ
                                order=[1,0,2]

                            elif it.m_sequence == "YZX":
                                e1 = proxSeg.anatomicalFrame.motion[i].m_axisY
                                e3 = distSeg.anatomicalFrame.motion[i].m_axisX
                                order=[1,2,0]

                            elif it.m_sequence == "ZXY":
                                e1 = proxSeg.anatomicalFrame.motion[i].m_axisZ
                                e3 = distSeg.anatomicalFrame.motion[i].m_axisY
                                order=[2,0,1]

                            elif it.m_sequence == "ZYX":
                                e1 = proxSeg.anatomicalFrame.motion[i].m_axisZ
                                e3 = distSeg.anatomicalFrame.motion[i].m_axisX
                                order=[2,1,0]

                            e2= np.cross(e3,e1)
                            e2=np.nan_to_num(np.divide(e2,np.linalg.norm(e2)))

                            if self.m_projection == enums.MomentProjection.JCS_Dual:

                                forceValues[i,order[0]] = np.nan_to_num(np.divide(np.dot(np.cross(e2,e3),F[i]), np.dot(np.cross(e1,e2),e3)))
                                forceValues[i,order[1]] = np.nan_to_num(np.divide(np.dot(np.cross(e3,e1),F[i]), np.dot(np.cross(e1,e2),e3)))
                                forceValues[i,order[2]] = np.nan_to_num(np.divide(np.dot(np.cross(e1,e2),F[i]), np.dot(np.cross(e1,e2),e3)))

                                momentValues[i,order[0]] = np.nan_to_num(np.divide(np.dot(np.cross(e2,e3),M[i]), np.dot(np.cross(e1,e2),e3)))
                                momentValues[i,order[1]] = np.dot(M[i],e2) #np.nan_to_num(np.divide(np.dot(np.cross(e3,e1),M[i]), np.dot(np.cross(e1,e2),e3))
                                momentValues[i,order[2]] = np.nan_to_num(np.divide(np.dot(np.cross(e1,e2),M[i]), np.dot(np.cross(e1,e2),e3)))

                            if self.m_projection == enums.MomentProjection.JCS:

                                forceValues[i,order[0]] = np.dot(F[i],e1)
                                forceValues[i,order[1]] = np.dot(F[i],e2)
                                forceValues[i,order[2]] = np.dot(F[i],e3)

                                momentValues[i,order[0]] = np.dot(M[i],e1)
                                momentValues[i,order[1]] = np.dot(M[i],e2)
                                momentValues[i,order[2]] = np.dot(M[i],e3)


                    descriptorForceInfos = self.m_model.getClinicalDescriptor(enums.DataType.Force,jointLabel,projection = self.m_projection)
                    descriptorMomentInfos = self.m_model.getClinicalDescriptor(enums.DataType.Moment,jointLabel,projection = self.m_projection)
                    if descriptorForceInfos:
                        finalForceValues = np.zeros((forceValues.shape))
                        finalForceValues[:,0] =  descriptorForceInfos["SaggitalCoeff"] * (forceValues[:,descriptorForceInfos["SaggitalIndex"]] + descriptorForceInfos["SaggitalOffset"])
                        finalForceValues[:,1] =  descriptorForceInfos["CoronalCoeff"] * (forceValues[:,descriptorForceInfos["CoronalIndex"]] + descriptorForceInfos["CoronalOffset"])
                        finalForceValues[:,2] =  descriptorForceInfos["TransversalCoeff"] * (forceValues[:,descriptorForceInfos["TransversalIndex"]] + descriptorForceInfos["TransversalOffset"])
                    else:
                        finalForceValues = forceValues


                    if descriptorMomentInfos:
                        finalMomentValues = np.zeros((momentValues.shape))
                        finalMomentValues[:,0] =  descriptorMomentInfos["SaggitalCoeff"] * (momentValues[:,descriptorMomentInfos["SaggitalIndex"]] + descriptorMomentInfos["SaggitalOffset"])
                        finalMomentValues[:,1] =  descriptorMomentInfos["CoronalCoeff"] * (momentValues[:,descriptorMomentInfos["CoronalIndex"]] + descriptorMomentInfos["CoronalOffset"])
                        finalMomentValues[:,2] =  descriptorMomentInfos["TransversalCoeff"] * (momentValues[:,descriptorMomentInfos["TransversalIndex"]] + descriptorMomentInfos["TransversalOffset"])
                    else:
                        finalMomentValues = momentValues

                    fulljointLabel_force  = jointLabel + "Force_" + pointLabelSuffix if pointLabelSuffix is not None else jointLabel+"Force"

                    btkTools.smartAppendPoint(self.m_aqui,
                                     fulljointLabel_force,
                                     finalForceValues,PointType="Force", desc="")

                    fulljointLabel_moment  = jointLabel + "Moment_" + pointLabelSuffix if pointLabelSuffix is not None else jointLabel+"Moment"
                    btkTools.smartAppendPoint(self.m_aqui,
                                     fulljointLabel_moment,
                                     finalMomentValues,PointType="Moment", desc="")

                    # Todo - Validate
                    # if self.m_exportMomentContributions:
                    #     forceValues = np.zeros((nFrames,3)) # need only for finalizeKinetics
                    #
                    #     for contIt  in ["internal","external", "inertia", "linearAcceleration","gravity", "externalDevices", "distalSegments","distalSegmentForces","distalSegmentMoments"] :
                    #         momentValues = np.zeros((nFrames,3))
                    #         for i in range(0,nFrames ):
                    #             if self.m_projection == enums.MomentProjection.Global:
                    #                 momentValues[i,:] = (1.0 / self.m_model.mp["Bodymass"]) * self.m_model.getSegment(it.m_distalLabel).m_proximalMomentContribution[contIt][i,:]
                    #             else:
                    #                 momentValues[i,:] = (1.0 / self.m_model.mp["Bodymass"]) * np.dot(mot[i].getRotation().T,
                    #                                                                         self.m_model.getSegment(it.m_distalLabel).m_proximalMomentContribution[contIt][i,:].T)
                    #
                    #         finalForceValues,finalMomentValues = self.m_model.finalizeKinetics(jointLabel,forceValues,momentValues,self.m_projection)
                    #
                    #         fulljointLabel_moment  = jointLabel + "Moment_" + pointLabelSuffix + "_" + contIt if pointLabelSuffix!="" else jointLabel+"Moment" + "_" + contIt
                    #         btkTools.smartAppendPoint(self.m_aqui,
                    #                          fulljointLabel_moment,
                    #                          finalMomentValues,PointType="Moment", desc= contIt + " Moment contribution")



class JointPowerFilter(object):
    """Compute joint power

    Args:
        btkAcq (btk.btkAcquisition): an acquisition instance of a dynamic trial
        iMod (pyCGM2.Model.CGM2.model.Model): a model instance
        scaleToMeter (double,Optional[0.001]): scale to meter
    """

    def __init__(self, iMod, btkAcq, scaleToMeter =0.001):

        self.m_aqui = btkAcq
        self.m_model = iMod
        self.m_scale = scaleToMeter

    def compute(self, pointLabelSuffix=None):
        """Run the filter. Joint powers are stored in the acquisition instance

        Args:
           pointLabelSuffix (str,Optional[None]): suffix ending output labels
        """

        for it in  self.m_model.m_jointCollection:
            if "ForeFoot" not in it.m_label:
                LOGGER.logger.debug("power of %s"  %(it.m_label))
                LOGGER.logger.debug("proximal label :%s" %(it.m_proximalLabel))
                LOGGER.logger.debug("distal label :%s" %(it.m_distalLabel))

                if self.m_model.getSegment(it.m_distalLabel).m_proximalWrench is not None:
                    jointLabel = it.m_label

                    nFrames = self.m_aqui.GetPointFrameNumber()

                    prox_omegai = self.m_model.getSegment(it.m_proximalLabel).getAngularVelocity(self.m_aqui.GetPointFrequency())
                    dist_omegai = self.m_model.getSegment(it.m_distalLabel).getAngularVelocity(self.m_aqui.GetPointFrequency())

                    relativeOmega = prox_omegai - dist_omegai

                    power = np.zeros((nFrames,3))
                    for i in range(0, nFrames):
                        power[i,2] = -1.0*(1.0 / self.m_model.mp["Bodymass"]) * self.m_scale * np.dot(self.m_model.getSegment(it.m_distalLabel).m_proximalWrench.GetMoment().GetValues()[i,:] ,relativeOmega[i,:])#


                    fulljointLabel  = jointLabel + "Power_" + pointLabelSuffix if pointLabelSuffix is not None else jointLabel+"Power"
                    btkTools.smartAppendPoint(self.m_aqui,
                                     fulljointLabel,
                                     power,PointType="Power", desc="")


class GeneralCoordinateSystemProcedure(object):
    """
    Procedure for defining general coordinate systems for segments in a biomechanical model.

    This procedure allows setting up custom definitions for coordinate systems attached to various segments.

    Attributes:
        definitions (List[Dict]): A list of coordinate system definitions. Each definition is a dictionary containing segment label, coordinate system label, and referential type.
    """
    def __init__(self):
        self.definitions=[]

    def setDefinition(self, segmentLabel: str, coordinateSystemLabel: str, referentialType: str):
        """
        Set a definition for a coordinate system in a model segment.

        Args:
            segmentLabel (str): Label of the segment.
            coordinateSystemLabel (str): Label of the coordinate system.
            referentialType (str): Type of the referential.
        """

        dic = {"segmentLabel": segmentLabel,"coordinateSystemLabel": coordinateSystemLabel,"referentialType": referentialType}
        self.definitions.append(dic)

class ModelCoordinateSystemProcedure(object):
    """
    Procedure for handling model-specific coordinate systems.

    This procedure utilizes model's internal coordinate system definitions for further processing.

    Args:
        iMod (Model): A biomechanical model instance.

    Attributes:
        definitions (List[dict]): Coordinate system definitions from the model.
    """
    def __init__(self,iMod:Model):
        self.definitions = iMod.m_csDefinitions

class CoordinateSystemDisplayFilter(object):
    """
    Filter to display coordinate systems of a biomechanical model.

    This filter is used for visualizing the coordinate systems defined in a model, either statically or during motion.

    Args:
        iProc (GeneralCoordinateSystemProcedure or ModelCoordinateSystemProcedure): The procedure with coordinate system definitions.
        iMod (Model): The biomechanical model.
        btkAcq (btk.btkAcquisition): An acquisition instance.

    """

    def __init__(self, iProc, iMod: Model, btkAcq: btk.btkAcquisition):
        self.m_procedure = iProc
        self.model = iMod
        self.aqui = btkAcq
        self.static = False
    def __init__(self,iProc, iMod, btkAcq):

        self.m_procedure = iProc
        self.model = iMod
        self.aqui = btkAcq
        self.static = False

    def setStatic(self,boolean:bool):
        """
        Set the display mode to static or dynamic.

        Args:
            boolean (bool): True for static display, False for dynamic display.
        """
        self.static = boolean

    def display(self):
        """
        Execute the display of coordinate systems as per the procedure definitions.

        The coordinate systems are visualized either statically or dynamically based on the `static` attribute.
        """
        definitions = self.m_procedure.definitions

        if self.static:
            for definition in definitions:
                if definition["segmentLabel"] in self.model.getSegmentList():
                    self.model.displayStaticCoordinateSystem( self.aqui,
                                                        definition["segmentLabel"],
                                                        definition["coordinateSystemLabel"],
                                                        referential = definition["referentialType"] )
                else:
                    LOGGER.logger.info("[pyCGM2] - referential not display because the segment [%s] is not in the model segment list "%(definition["segmentLabel"]))

        else:
            for definition in definitions:
                if definition["segmentLabel"] in self.model.getSegmentList():
                    self.model.displayMotionCoordinateSystem( self.aqui,
                                                        definition["segmentLabel"],
                                                        definition["coordinateSystemLabel"],
                                                        referential = definition["referentialType"] )
                else:
                    LOGGER.logger.info("[pyCGM2] - referential not display because the segment [%s] is not in the model segment list "%(definition["segmentLabel"]))


class CentreOfMassFilter(object):
    """
    Filter for computing the center of mass (CoM) trajectory of a biomechanical model.

    This filter calculates the trajectory of the CoM for each segment and for the entire model.

    Args:
        iMod (Model): The biomechanical model instance.
        btkAcq (btk.btkAcquisition): The motion acquisition data.

    """

    def __init__(self, iMod:Model, btkAcq:btk.btkAcquisition):

        self.model = iMod
        self.aqui = btkAcq

    def compute(self, pointLabelSuffix: Optional[str] = None):
        """
        Compute and append the CoM trajectory to the acquisition data.

        Args:
            pointLabelSuffix (Optional[str]): An optional suffix to be added to the CoM point label.
        """

        count = 0
        bodymass= 0
        for itSegment in self.model.m_segmentCollection:
            if itSegment.m_bsp["mass"] != 0 and not itSegment.m_isCloneOf:
                count = count + itSegment.m_bsp["mass"]  *  self.model.getSegment(itSegment.name).getComTrajectory()
                bodymass = bodymass + itSegment.m_bsp["mass"]

        com = count / bodymass

        self.model.setCentreOfMass(com)

        for itSegment in self.model.m_segmentCollection:
            if itSegment.m_bsp["mass"] != 0 and not itSegment.m_isCloneOf:
                print(itSegment.name)
                comTraj = self.model.getSegment(itSegment.name).getComTrajectory()
                outLabel = "Com_"+itSegment.name
                btkTools.smartAppendPoint(self.aqui,outLabel,comTraj)


        outLabel  = "CentreOfMass_" + pointLabelSuffix if pointLabelSuffix is not None else "CentreOfMass"
        btkTools.smartAppendPoint(self.aqui,outLabel,self.model.getCentreOfMass())



class ModelMotionCorrectionFilter(object):
    """
    Filter for correcting the motion of a biomechanical model.

    This filter applies corrections to the motion attributes of the anatomical coordinate systems in the model.

    Args:
        procedure (ModelCorrectionProcedure): The model correction procedure to be applied.
    """
    
    def __init__(self,procedure:ModelCorrectionProcedure):

        self.m_procedure = procedure

    def correct(self):
        """
        Execute the correction procedure on the model.
        """
        self.m_procedure.correct()

class ModelQualityFilter(object):
    """
    Filter for assessing the quality of the biomechanical model's motion.

    This filter runs a specified procedure to evaluate the quality of the model's motion.

    Args:
        acq (btk.btkAcquisition): The motion acquisition data.
        procedure (QualityProcedure): The quality assessment procedure to be applied.
    """
    def __init__(self,acq:btk.btkAcquisition,procedure:QualityProcedure):
        self.m_procedure = procedure
        self.m_acq = acq

    def run(self):
        """
        Execute the quality assessment procedure.
        """
        self.m_procedure.run(self.m_acq)