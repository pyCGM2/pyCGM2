# -*- coding: utf-8 -*-
import logging
import numpy as np
import copy

from pyCGM2 import btk

import frame
import motion

from pyCGM2 import enums
from  pyCGM2.Math import euler,numeric
import pyCGM2.Signal.signal_processing as pyCGM2signal
from pyCGM2.Tools import  btkTools
from pyCGM2.Utils import timer
import matplotlib.pyplot as plt



#-------- MODEL PROCEDURE  ----------


# --- calibration procedure
class GeneralCalibrationProcedure(object):
    """
        General Procedure to call in the Model Calibration Filter if you work on a custom model
    """

    def __init__(self):
       self.definition=dict()
       self.anatomicalDefinition=dict()

    def setDefinition(self, segmentName,referentialLabel,
                      sequence=str(),
                      pointLabel1=0,pointLabel2=0, pointLabel3=0,
                      pointLabelOrigin=0):
        """
            method defining how to construct a referential for a given segment

            :Parameters:
                 - `segmentName` (str) - name of the segment
                 - `referentialLabel` (str) - label of the referential
                 - `sequence` (str) - construction sequence (XYZ,XYiZ)
                 - `pointLabel1` (str) - marker label
                 - `pointLabel2` (str) - marker label
                 - `pointLabel3` (str) - marker label
                 - `pointLabelOrigin` (str) - marker label

            .. todo::

               - add optional markers i.e marker only use for segmental least square computation
               - use an ENUM for sequence


        """
        if segmentName in self.definition:
            self.definition[segmentName][referentialLabel]={'sequence':sequence, 'labels':[pointLabel1,pointLabel2,pointLabel3,pointLabelOrigin]}
        else:
            self.definition[segmentName]=dict()
            self.definition[segmentName][referentialLabel]={'sequence':sequence, 'labels':[pointLabel1,pointLabel2,pointLabel3,pointLabelOrigin]}

    def setAnatomicalDefinition(self, segmentName,
                      sequence=str(),
                      nodeLabel1=0,nodeLabel2=0, nodeLabel3=0,
                      nodeLabelOrigin=0):
        """
            method defining how to construct a referential for a given segment

        """
        if segmentName in self.anatomicalDefinition:
            self.anatomicalDefinition[segmentName]={'sequence':sequence, 'labels':[nodeLabel1,nodeLabel2,nodeLabel3,nodeLabelOrigin]}
        else:
            self.anatomicalDefinition[segmentName]=dict()
            self.anatomicalDefinition[segmentName]={'sequence':sequence, 'labels':[nodeLabel1,nodeLabel2,nodeLabel3,nodeLabelOrigin]}


class StaticCalibrationProcedure(object):
    """
        Procedure to call if you work with pyCGM2 embedded-model instance ( like the cgm)

        This procedure internally calls the `calibrationProcedure` method of the model

    """
    def __init__(self,model):
        """
            :Parameters:
              - `iModel` (pyCGM2.Model.CGM2.cgm.CGM) - a CGM instance
        """
        self.model=model
        self.definition=dict()

        self.__setDefinition()

    def __setDefinition(self):
        self.definition=self.model.calibrationProcedure()



# ---- inverse dynamic procedure
class CGMLowerlimbInverseDynamicProcedure(object):
    def __init__(self):
        """
        # procedure = anticipation de nouveau Model
        """

    def _externalDeviceForceContribution(self, wrenchs):

        nf = wrenchs[0].GetForce().GetValues().shape[0]
        forceValues = np.zeros((nf,3))
        for wrIt in wrenchs:
            forceValues = forceValues + wrIt.GetForce().GetValues()

        return forceValues

    def _externalDeviceMomentContribution(self, wrenchs, Oi, scaleToMeter):

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


    def _distalMomentContribution(self, wrench, Oi, scaleToMeter, source = "Wrench"):

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

    def _forceAccelerationContribution(self,mi,ai,g,scaleToMeter):

        nf = ai.shape[0]
        g= np.matrix(g)
        accelerationContribution = np.zeros((nf,3))

        for i in range(0,nf):
            ai_i = np.matrix(ai[i,:])
            val = mi * ai_i.T*scaleToMeter - mi*g.T
            accelerationContribution[i,:] = np.array(val.T)

        return  accelerationContribution


    def _inertialMomentContribution(self,Ii, alphai,omegai, Ti ,scaleToMeter):
        """
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

    def _accelerationMomentContribution(self, mi,ci, ai, Ti, scaleToMeter):
        """
        SkewMatrix(ai_i*scaleToMeter) *mi * Ri_i*(ci*scaleToMeter
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


    def _gravityMomentContribution(self, mi,ci, g, Ti, scaleToMeter):

        nf = len(Ti)

        gravityContribution = np.zeros((nf,3))


        g= np.matrix(g)
        ci = np.matrix(ci)
        for i in range(0,nf):
            Ri_i = np.matrix(Ti[i].getRotation())
            val = - 1.0 *mi*numeric.skewMatrix(g) * Ri_i*(ci.T*scaleToMeter)

            gravityContribution[i,:] = np.array(val.T)
        return  gravityContribution



    def computeSegmental(self,model,segmentLabel,btkAcq, gravity, scaleToMeter,distalSegmentLabel=None):
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

    def compute(self,model, btkAcq, gravity, scaleToMeter):
        """
        calcul segment par segment, j automatiserai plus tard !

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
    """
        Perform model calibration from a static acquisition
    """
    def __init__(self,procedure, acq, iMod,**options):
        """
            :Parameters:
               - `procedure` (pyCGM2.Model.CGM2.ModelFilters.(_Procedure) ) - Model Procedure to use
               - `acq` (btkAcquisition) - a btk acquisition instance of the static c3d
               - `iMod` (pyCGM2.Model.CGM2.model.Model) - a model instance
               - `**options` (dict) - dictionnary passing calibration options

        """


        self.m_aqui=acq
        self.m_procedure=procedure
        self.m_model=iMod
        self.m_options=options
        self.m_noAnatomicalCalibration = False

    def setOption(self, label,value):
        self.m_options[label] = value

    def setBoolOption(self, label):
        self.m_options[label] = True

    def setNoAnatomicalCalibration(self,boolFlag):
        self.m_noAnatomicalCalibration = boolFlag


    def compute(self, firstFrameOnly = True):
        """
            Run Calibration filter

            :Parameters:
               - `firstFrameOnly` (bool) - flag indicating calibration on the first frame
        """

        ff=self.m_aqui.GetFirstFrame()

        if firstFrameOnly :
            frameInit=0 #frameInit-ff
            frameEnd=1 #frameEnd-ff+1
        else :
            frameInit=frameInit-ff
            frameEnd=frameEnd-ff+1



        if str(self.m_model) != "Basis Model":
            for segName in self.m_procedure.definition[0]:
                segPicked=self.m_model.getSegment(segName)
                for tfName in self.m_procedure.definition[0][segName]: # TF name
                    segPicked.addTechnicalReferential(tfName) # append Segment.technicalFrames ()

            self.m_model.calibrate(self.m_aqui, self.m_procedure.definition[0], self.m_procedure.definition[1], options=self.m_options)


        else :
            # calibration of technical Frames
            for segName in self.m_procedure.definition:
                print segName
                segPicked=self.m_model.getSegment(segName)
                for tfName in self.m_procedure.definition[segName]: # TF name

                    segPicked.addTechnicalReferential(tfName) # append Segment.technicalFrames ()

                    pt1=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName][tfName]['labels'][0])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
                    pt2=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName][tfName]['labels'][1])).GetValues()[frameInit:frameEnd,:].mean(axis=0)
                    pt3=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName][tfName]['labels'][2])).GetValues()[frameInit:frameEnd,:].mean(axis=0)

                    ptOrigin=self.m_aqui.GetPoint(str(self.m_procedure.definition[segName][tfName]['labels'][3])).GetValues()[frameInit:frameEnd,:].mean(axis=0)


                    a1=(pt2-pt1)
                    a1=np.divide(a1,np.linalg.norm(a1))

                    v=(pt3-pt1)
                    v=np.divide(v,np.linalg.norm(v))

                    a2=np.cross(a1,v)
                    a2=np.divide(a2,np.linalg.norm(a2))

                    x,y,z,R=frame.setFrameData(a1,a2,self.m_procedure.definition[segName][tfName]['sequence'])

                    segPicked.referentials[-1].static.m_axisX=x # work on the last TF in the list : thus index -1
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
                # calibration of technical Frames
                for segName in self.m_procedure.anatomicalDefinition:
                    print segName
                    segPicked=self.m_model.getSegment(segName)
                    tf=segPicked.getReferential("TF")
                    #if segName == "Right Shank":

                    nd1 = str(self.m_procedure.anatomicalDefinition[segName]['labels'][0])
                    pt1 = tf.static.getNode_byLabel(nd1).m_global

                    nd2 = str(self.m_procedure.anatomicalDefinition[segName]['labels'][1])
                    pt2 = tf.static.getNode_byLabel(nd2).m_global

                    nd3 = str(self.m_procedure.anatomicalDefinition[segName]['labels'][2])
                    pt3 = tf.static.getNode_byLabel(nd3).m_global

                    ndO = str(self.m_procedure.anatomicalDefinition[segName]['labels'][3])
                    ptO = tf.static.getNode_byLabel(ndO).m_global

                    a1=(pt2-pt1)
                    a1=np.divide(a1,np.linalg.norm(a1))

                    v=(pt3-pt1)
                    v=np.divide(v,np.linalg.norm(v))

                    a2=np.cross(a1,v)
                    a2=np.divide(a2,np.linalg.norm(a2))

                    x,y,z,R=frame.setFrameData(a1,a2,self.m_procedure.anatomicalDefinition[segName]['sequence'])

                    segPicked.anatomicalFrame.static.m_axisX=x # work on the last TF in the list : thus index -1
                    segPicked.anatomicalFrame.static.m_axisY=y
                    segPicked.anatomicalFrame.static.m_axisZ=z

                    segPicked.anatomicalFrame.static.setRotation(R)
                    segPicked.anatomicalFrame.static.setTranslation(ptO)


                    # --- relative rotation Technical Anatomical
                    tf.setRelativeMatrixAnatomic( np.dot(tf.static.getRotation().T,segPicked.anatomicalFrame.static.getRotation()))


#-------- MOTION FILTER  ----------



class ModelMotionFilter(object):
    """
        Compute Motion of each segment
    """
    def __init__(self,procedure,acq, iMod,method, **options ):
        """
            :Parameters:

               -- `procedure` (pyCGM2.Model.CGM2.ModelFilters.(_Procedure) ) - Model Procedure to use
               - `acq` (btkAcquisition) - btk acquisition instance of a dynamic trial
               - `iMod` (pyCGM2.Model.CGM2.model.Model) - a model instancel
               - `method` (pyCGM2.enums) -method uses for computing segment pose

        """

        self.m_aqui = acq
        self.m_procedure = procedure
        self.m_model = iMod
        self.m_method = method
        self.m_options = options
        self.m_noAnatomicalMotion = False


    def setOption(self, label,value):
        self.m_options[label] = value

    def setBoolOption(self, label):
        self.m_options[label] = True

    def setNoAnatomicalMotion(self,boolFlag):
        self.m_noAnatomicalMotion = boolFlag

    def segmentalCompute(self,segments):

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
                        a1=np.divide(a1,np.linalg.norm(a1))

                        v=(pt3-pt1)
                        v=np.divide(v,np.linalg.norm(v))

                        a2=np.cross(a1,v)
                        a2=np.divide(a2,np.linalg.norm(a2))

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
        """
            Run the motion filter
        """
        if str(self.m_model) != "Basis Model":
           self.m_model.computeMotion(self.m_aqui,
                                      self.m_procedure.definition[0],
                                      self.m_procedure.definition[1],
                                      self.m_method,
                                      self.m_options)
        else :
            with timer.Timer("test"):
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
                            a1=np.divide(a1,np.linalg.norm(a1))

                            v=(pt3-pt1)
                            v=np.divide(v,np.linalg.norm(v))

                            a2=np.cross(a1,v)
                            a2=np.divide(a2,np.linalg.norm(a2))

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



class TrackingMarkerDecompositionFilter(object):
    """
        decomposition of tracking markers
    """
    def __init__(self,iModel,iAcq):
        """
            :Parameters:

               -- `` ( ) -

        """
        self.m_model = iModel
        self.m_acq = iAcq



    def decompose(self):
        """
           Run decomposition.
           - add directionMarker as tracking markers
           - add nodes to bth Technical and Anatomical CS
           - decompose tracking marker in the motion trial

        TODO : revoir les suffix car depende de l orientation des referentiels

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

                    btkTools.smartAppendPoint(self.m_acq,marker+suffix[0],markerTrajectoryX,PointType=btk.btkPoint.Marker, desc="")
                    btkTools.smartAppendPoint(self.m_acq,marker+suffix[1],markerTrajectoryY,PointType=btk.btkPoint.Marker, desc="")
                    btkTools.smartAppendPoint(self.m_acq,marker+suffix[2],markerTrajectoryZ,PointType=btk.btkPoint.Marker, desc="")



# ----- Joint angles -----

class ModelJCSFilter(object):
    """
        Compute joint angles
    """

    def __init__(self, iMod, acq):
        """
            :Parameters:
               - `acq` (btkAcquisition) - btk acquisition instance of a dynamic trial
               - `iMod` (pyCGM2.Model.CGM2.model.Model) - a model instance
        """
        self.m_aqui = acq
        self.m_model = iMod
        self.m_fixEuler = True

    def setFixEuler(self,bool):
        self.m_fixEuler =  bool

    def compute(self,description="", pointLabelSuffix = None):
        """
            run `ModelJCSFilter`

            .. note:: All angles are stored as btk point

            :Parameters:
               - `description` (str) -short description added to the btkPoint storing joint angles
               - `pointLabelSuffix` (str) - suffix ending the angle label
        """


        for it in  self.m_model.m_jointCollection:
            logging.debug("---Processing of %s---"  % it.m_label)
            logging.debug(" proximal : %s "% it.m_proximalLabel)
            logging.debug(" distal : %s "% it.m_distalLabel)


            jointLabel = it.m_label

            proxSeg = self.m_model.getSegment(it.m_proximalLabel)
            distSeg = self.m_model.getSegment(it.m_distalLabel)

            jointValues = np.zeros((self.m_aqui.GetPointFrameNumber(),3))
            for i in range (0, self.m_aqui.GetPointFrameNumber()):
                Rprox = proxSeg.anatomicalFrame.motion[i].getRotation()
                Rdist = distSeg.anatomicalFrame.motion[i].getRotation()
#                Rprox = np.dot(proxSeg.getReferential("TF").motion[i].getRotation(), proxSeg.getReferential("TF").relativeMatrixAnatomic)
#                Rdist = np.dot(distSeg.getReferential("TF").motion[i].getRotation(), distSeg.getReferential("TF").relativeMatrixAnatomic)
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
                logging.debug("joint label (%s) in clinical descriptors" %(jointLabel) )
                jointFinalValues = np.zeros((jointValues.shape))
                jointFinalValues[:,0] =  np.rad2deg(descriptorInfos["SaggitalCoeff"] * (jointValues[:,descriptorInfos["SaggitalIndex"]] + descriptorInfos["SaggitalOffset"]))
                jointFinalValues[:,1] =  np.rad2deg(descriptorInfos["CoronalCoeff"] * (jointValues[:,descriptorInfos["CoronalIndex"]] + descriptorInfos["CoronalOffset"]))
                jointFinalValues[:,2] =  np.rad2deg(descriptorInfos["TransversalCoeff"] * (jointValues[:,descriptorInfos["TransversalIndex"]] + descriptorInfos["TransversalOffset"]))
            else:
                logging.debug("no clinical descriptor for joint label (%s)" %(jointLabel) )
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
                             jointFinalValues,PointType=btk.btkPoint.Angle, desc=description)


class ModelAbsoluteAnglesFilter(object):
    """
        Compute absolute angles

    """

    def __init__(self, iMod, acq, segmentLabels=list(),angleLabels=list(), eulerSequences=list(), globalFrameOrientation = "XYZ", forwardProgression = True  ):
        """
            :Parameters:
               - `acq` (btkAcquisition) - btk acquisition instance of a dynamic trial
               - `iMod` (pyCGM2.Model.CGM2.model.Model) - a model instance
               - `segmentLabels` (list of str) - segment labels
               - `angleLabels` (list of str) - absolute angles labels
               - `eulerSequences` (list of str) - euler sequence to use. ( nomenclature TOR (ie Tilt-Obliquity-Rotation) )
               - `globalFrameOrientation` (str) - global frame
               - `forwardProgression` (bool) - flag indicating subject moves in same direction than the global longitudinal axis

        """
        self.m_aqui = acq
        self.m_model = iMod
        self.m_segmentLabels = segmentLabels
        self.m_angleLabels = angleLabels
        self.m_eulerSequences = eulerSequences
        self.m_globalFrameOrientation = globalFrameOrientation
        self.m_forwardProgression = forwardProgression


    def compute(self,description="absolute", pointLabelSuffix = None):
        """
            run `ModelAbsoluteAnglesFilter`

            .. note:: All angles are stored as btk point

            :Parameters:
               - `description` (str) -short description added to the btkPoint storing joint angles
               - `pointLabelSuffix` (str) - suffix ending the angle label
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
                logging.debug( "segment (%s) - sequence Tilt-Obliquity-Rotation used" %(seg.name) )
            elif eulerSequence == "TRO":
                logging.debug( "segment (%s) - sequence Tilt-Rotation-Obliquity used" %(seg.name) )
            elif eulerSequence == "ROT":
                logging.debug( "segment (%s) - sequence Rotation-Obliquity-Tilt used" %(seg.name) )
            elif eulerSequence == "RTO":
                logging.debug( "segment (%s) - sequence Rotation-Tilt-Obliquity used" %(seg.name) )
            elif eulerSequence == "OTR":
                logging.debug( "segment (%s) - sequence Obliquity-Tilt-Rotation used" %(seg.name) )
            elif eulerSequence == "ORT":
                logging.debug( "segment (%s) - sequence Obliquity-Rotation-Tilt used" %(seg.name) )
            else:
                pass
                #logging.debug( "segment (%s) - sequence doest recognize - sequence Tilt-Obliquity-Rotation used by default" %(seg.name) )


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
                    logging.debug("no sequence defined for absolute angles. sequence YXZ selected by default" )
                    tilt,obliquity,rotation = euler.euler_yxz(Rrelative)

                absoluteAngleValues[i,0] = tilt
                absoluteAngleValues[i,1] = obliquity
                absoluteAngleValues[i,2] = rotation

            segName = self.m_segmentLabels[index]

            if side == enums.SegmentSide.Left or side == enums.SegmentSide.Right:
                # no alteration of the segment name
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
                                         absoluteAngleValuesFinal,PointType=btk.btkPoint.Angle, desc=description)

                else:
                    logging.debug("no clinical descriptor for segment label (%s)" %(segName))
                    absoluteAngleValuesFinal = np.rad2deg(absoluteAngleValues)

                    fullAngleLabel  = self.m_angleLabels[index] + "Angles_" + pointLabelSuffix if pointLabelSuffix is not None else self.m_angleLabels[index]+"Angles"


                    btkTools.smartAppendPoint(self.m_aqui, fullAngleLabel,
                                         absoluteAngleValuesFinal,PointType=btk.btkPoint.Angle, desc=description)

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
                                         absoluteAngleValuesFinal,PointType=btk.btkPoint.Angle, desc=description)

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
                                         absoluteAngleValuesFinal,PointType=btk.btkPoint.Angle, desc=description)


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
                                         absoluteAngleValuesFinal,PointType=btk.btkPoint.Angle, desc=description)

                if not descriptorInfos1 and not descriptorInfos2 and not descriptorInfos3:
                    logging.debug("no clinical descriptor for segment label (%s)" %(segName))
                    absoluteAngleValuesFinal = np.rad2deg(absoluteAngleValues)

                    fullAngleLabel  = self.m_angleLabels[index] + "Angles_" + pointLabelSuffix if pointLabelSuffix is not None else self.m_angleLabels[index]+"Angles"
                    btkTools.smartAppendPoint(self.m_aqui, fullAngleLabel,
                                         absoluteAngleValuesFinal,PointType=btk.btkPoint.Angle, desc=description)



# ----- Force plates -----

class ForcePlateAssemblyFilter(object):
    """
        Assemble Force plate with the model
    """

    def __init__(self, iMod, btkAcq, mappedForcePlateLetters, leftSegmentLabel="Left Foot", rightSegmentLabel="Right Foot" ):
        """
            :Parameters:
               - `btkAcq` (btkAcquisition) - btk acquisition instance of a dynamic trial
               - `iMod` (pyCGM2.Model.CGM2.model.Model) - a model instance
               - `mappedForcePlateLetters` (str) - string indicating body side of the segment in contact with the force plate
               - `leftSegmentLabel` (str) - left segment label to assemble with force plates
               - `rightSegmentLabel` (str) - right segment label to assemble with force plates

        """
        self.m_aqui = btkAcq
        self.m_model = iMod
        self.m_mappedForcePlate = mappedForcePlateLetters
        self.m_leftSeglabel = leftSegmentLabel
        self.m_rightSeglabel = rightSegmentLabel
        self.m_model = iMod

        # zeroing externalDevice
        self.m_model.getSegment(self.m_leftSeglabel).zeroingExternalDevice()
        self.m_model.getSegment(self.m_rightSeglabel).zeroingExternalDevice()


    def compute(self,pointLabelSuffix=None):
        """
            run `ForcePlateAssemblyFilter`
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
                logging.debug("force plate %i sans donnees" %(i))
            i+=1

        if "L" in self.m_mappedForcePlate:
            self.m_model.getSegment(self.m_leftSeglabel).downSampleExternalDeviceWrenchs(appf)
        if "R" in self.m_mappedForcePlate:
            self.m_model.getSegment(self.m_rightSeglabel).downSampleExternalDeviceWrenchs(appf)

        # ground reaction force and moment
        if self.m_model.mp.has_key("Bodymass"):
            bodymass = self.m_model.mp["Bodymass"]
        else:
            bodymass = 1.0
            logging.warning("[pyCGM2] - bodymass is not within your mp data. non-normalized GroundReaction Force and Moment ouput")

        forceLabel  = "LGroundReactionForce_"+pointLabelSuffix if pointLabelSuffix is not None else "LGroundReactionForce"
        momentLabel  = "LGroundReactionMoment_"+pointLabelSuffix if pointLabelSuffix is not None else "LGroundReactionMoment"

        btkTools.smartAppendPoint(self.m_aqui,forceLabel,
                         left_forceValues / bodymass,
                         PointType=btk.btkPoint.Force, desc="")
        btkTools.smartAppendPoint(self.m_aqui,momentLabel,
                         left_momentValues / bodymass,
                         PointType=btk.btkPoint.Moment, desc="")

        forceLabel  = "RGroundReactionForce_"+pointLabelSuffix if pointLabelSuffix is not None else "RGroundReactionForce"
        momentLabel  = "RGroundReactionMoment_"+pointLabelSuffix if pointLabelSuffix is not None else "RGroundReactionMoment"
        btkTools.smartAppendPoint(self.m_aqui,forceLabel,
                         right_forceValues / bodymass,
                         PointType=btk.btkPoint.Force, desc="")
        btkTools.smartAppendPoint(self.m_aqui,momentLabel,
                         right_momentValues / bodymass,
                         PointType=btk.btkPoint.Moment, desc="")



# ----- Inverse dynamics -----

class InverseDynamicFilter(object):
    """
        Compute joint Force and moment from Inverse dynamics
    """

    def __init__(self, iMod, btkAcq, procedure = None, gravityVector = np.array([0,0,-1]), scaleToMeter =0.001,
                 projection = enums.MomentProjection.Distal,
                 globalFrameOrientation="XYZ",forwardProgression=True,
                 exportMomentContributions = False, **options):
        """
           :Parameters:
               - `btkAcq` (btkAcquisition) - btk acquisition instance of a dynamic trial
               - `iMod` (pyCGM2.Model.CGM2.model.Model) - a model instance
               - `procedure` (pyCGM2.Model.CGM2.modelFilters.(_Procedure)) - an inverse dynamic procedure
               - `gravityVector` (numpy.array(3,)) - gravity vector
               - `scaleToMeter` (float) - values scaling to meter
               - `projection` (pyCGM2.enums) - method of moment projection
               - `exportMomentContributions` (bool) - enable export of moment contribution into the c32


        """
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

    def compute(self, pointLabelSuffix = None ):
        """
            Run`InverseDynamicFilter`

            .. note:: forces and Moments are stored as btk point

            :Parameters:
               - `pointLabelSuffix` (str) - suffix ending the force/moment label

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
                logging.debug("kinetics of %s"  %(it.m_label))
                logging.debug("proximal label :%s" %(it.m_proximalLabel))
                logging.debug("distal label :%s" %(it.m_distalLabel))

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
                            e2=np.divide(e2,np.linalg.norm(e2))

                            if self.m_projection == enums.MomentProjection.JCS_Dual:

                                forceValues[i,order[0]] = np.divide(np.dot(np.cross(e2,e3),F[i]), np.dot(np.cross(e1,e2),e3))
                                forceValues[i,order[1]] = np.divide(np.dot(np.cross(e3,e1),F[i]), np.dot(np.cross(e1,e2),e3))
                                forceValues[i,order[2]] = np.divide(np.dot(np.cross(e1,e2),F[i]), np.dot(np.cross(e1,e2),e3))

                                momentValues[i,order[0]] = np.divide(np.dot(np.cross(e2,e3),M[i]), np.dot(np.cross(e1,e2),e3))
                                momentValues[i,order[1]] = np.dot(M[i],e2) #np.divide(np.dot(np.cross(e3,e1),M[i]), np.dot(np.cross(e1,e2),e3))
                                momentValues[i,order[2]] = np.divide(np.dot(np.cross(e1,e2),M[i]), np.dot(np.cross(e1,e2),e3))

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
                                     finalForceValues,PointType=btk.btkPoint.Force, desc="")

                    fulljointLabel_moment  = jointLabel + "Moment_" + pointLabelSuffix if pointLabelSuffix is not None else jointLabel+"Moment"
                    btkTools.smartAppendPoint(self.m_aqui,
                                     fulljointLabel_moment,
                                     finalMomentValues,PointType=btk.btkPoint.Moment, desc="")

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
                    #                          finalMomentValues,PointType=btk.btkPoint.Moment, desc= contIt + " Moment contribution")



class JointPowerFilter(object):
    """
        Compute joint power
    """

    def __init__(self, iMod, btkAcq, scaleToMeter =0.001):
        """
            :Parameters:
               - `btkAcq` (btkAcquisition) - btk acquisition instance of a dynamic trial
               - `iMod` (pyCGM2.Model.CGM2.model.Model) - a model instance
               - `scaleToMeter` (float) - values scaling to meter

        """
        self.m_aqui = btkAcq
        self.m_model = iMod
        self.m_scale = scaleToMeter

    def compute(self, pointLabelSuffix=None):
        """
            Run `JointPowerFilter`

            .. note:: powers are stored as btk point

            :Parameters:
               - `pointLabelSuffix` (str) - suffix ending the power label
        """

        for it in  self.m_model.m_jointCollection:
            if "ForeFoot" not in it.m_label:
                logging.debug("power of %s"  %(it.m_label))
                logging.debug("proximal label :%s" %(it.m_proximalLabel))
                logging.debug("distal label :%s" %(it.m_distalLabel))

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
                                     power,PointType=btk.btkPoint.Power, desc="")


class GeneralCoordinateSystemProcedure(object):
    def __init__(self):
        self.definitions=list()

    def setDefinition(self, segmentLabel, coordinateSystemLabel, referentialType):
        dic = {"segmentLabel": segmentLabel,"coordinateSystemLabel": coordinateSystemLabel,"referentialType": referentialType}
        self.definitions.append(dic)

class ModelCoordinateSystemProcedure(object):
    def __init__(self,iMod):
        self.definitions = iMod.m_csDefinitions

class CoordinateSystemDisplayFilter(object):
    """
        display Coordinate system
    """

    def __init__(self,iProc, iMod, btkAcq):

        self.m_procedure = iProc
        self.model = iMod
        self.aqui = btkAcq
        self.static = False

    def setStatic(self,boolean):
        self.static = boolean

    def display(self):
        definitions = self.m_procedure.definitions

        if self.static:
            for definition in definitions:
                if definition["segmentLabel"] in self.model.getSegmentList():
                    self.model.displayStaticCoordinateSystem( self.aqui,
                                                        definition["segmentLabel"],
                                                        definition["coordinateSystemLabel"],
                                                        referential = definition["referentialType"] )
                else:
                    logging.info("[pyCGM2] - referential not display because the segment [%s] is not in the model segment list "%(definition["segmentLabel"]))

        else:
            for definition in definitions:
                if definition["segmentLabel"] in self.model.getSegmentList():
                    self.model.displayMotionCoordinateSystem( self.aqui,
                                                        definition["segmentLabel"],
                                                        definition["coordinateSystemLabel"],
                                                        referential = definition["referentialType"] )
                else:
                    logging.info("[pyCGM2] - referential not display because the segment [%s] is not in the model segment list "%(definition["segmentLabel"]))


class CentreOfMassFilter(object):
    """
        display Coordinate system
    """

    def __init__(self, iMod, btkAcq):

        self.model = iMod
        self.aqui = btkAcq

    def compute(self, pointLabelSuffix=None):

        count = 0
        bodymass= 0
        for itSegment in self.model.m_segmentCollection:
            if itSegment.m_bsp["mass"] != 0 and not itSegment.m_isCloneOf:
                count = count + itSegment.m_bsp["mass"]  *  self.model.getSegment(itSegment.name).getComTrajectory()
                bodymass = bodymass + itSegment.m_bsp["mass"]

        com = count / bodymass

        self.model.setCentreOfMass(com)

        outLabel  = "CentreOfMass_" + pointLabelSuffix if pointLabelSuffix is not None else "CentreOfMass"
        btkTools.smartAppendPoint(self.aqui,outLabel,self.model.getCentreOfMass())

class Naim2019ThighMisaligmentCorrectionProcedure(object):
    """
        This function corrects the thigh anatomical frame in respect with the method detailed in Naim et al, 2019.


        Naaim, A., Bonnefoy-Mazure, A., Armand, S., & Dumas, R. (2019).
        Correcting lower limb segment axis misalignment in gait analysis: A simple geometrical method.
        Gait and Posture, 72(May), 34–39
    """

    def __init__(self,model,side):
        """
        :param model [pyCGM2.Model.Model]: model
        :param side [str]: body side (Left, Right or Both as choice)
        """

        self.m_model =  model
        self.m_side = side


    def correct(self):
        """

        """
        side = self.m_side

        if self.m_side == "Both":
            sides = ["Left","Right"]
        else:
            sides = [self.m_side]

        for side in sides:
            if side== "Left":
                sequence = "ZXiY"
                letter = "L"
            elif side == "Right":
                sequence = "ZXY"
                letter = "R"

            else:
                raise Exception("[pyCGM2] : side not recognized ( Left or Right only)")

            seg = self.m_model.getSegment(side+" Thigh")

            hjc = self.m_model.getSegment(side+" Thigh").anatomicalFrame.getNodeTrajectory(letter+"HJC")
            kjc = self.m_model.getSegment(side+" Thigh").anatomicalFrame.getNodeTrajectory(letter+"KJC")
            ajc = self.m_model.getSegment(side+" Shank").anatomicalFrame.getNodeTrajectory(letter+"AJC")

            v = np.cross(hjc-kjc,ajc-kjc)
            v = np.divide(v.mean(axis =0),np.linalg.norm(v.mean(axis =0)))

            # virtual point along the mean axis
            virtual = np.zeros((len(ajc),3))
            for i in range(0,len(ajc)):
                virtual[i,:]= kjc[i,:] + -100 * v if side == "Left" else kjc[i,:] + 100 * v

            #alteration of the segmental motion
            seg.anatomicalFrame.motion=[] # erase all previous motion

            csFrame=frame.Frame()
            for i in range(0,len(ajc)):
                pt1=kjc[i,:]
                pt2=hjc[i,:]
                pt3=virtual[i,:]
                ptOrigin=hjc[i,:]

                a1=(pt2-pt1)
                a1=np.divide(a1,np.linalg.norm(a1))

                v=(pt3-pt1)
                v=np.divide(v,np.linalg.norm(v))

                a2=np.cross(a1,v)
                a2=np.divide(a2,np.linalg.norm(a2))

                x,y,z,R=frame.setFrameData(a1,a2,sequence)

                csFrame.m_axisX=x
                csFrame.m_axisY=y
                csFrame.m_axisZ=z
                csFrame.setRotation(R)
                csFrame.setTranslation(ptOrigin)

                seg.anatomicalFrame.addMotionFrame(copy.deepcopy(csFrame))

            logging.warning("[pyCGM2] : %s thigh anatomical frame motion corrected according Naim et al, 2019"%(side))


class ModelMotionCorrectionFilter(object):
    """
        Correction of the Motion of an anatomical frame
    """
    def __init__(self,procedure):
        self.m_procedure = procedure

    def correct(self):
        self.m_procedure.correct()
