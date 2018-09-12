# -*- coding: utf-8 -*-
import numpy as np
import logging
import copy
import ipdb


from pyCGM2 import btk

from pyCGM2 import enums
from pyCGM2.Model import model, modelDecorator,frame,motion
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Math import euler,geometry
from pyCGM2.Tools import  btkTools


class RCM(model.Model6Dof):
    """
    """

    def __init__(self):
        super(RCM, self).__init__()
        self.decoratedModel = False
        self.version = "RCM1.0"


    def __repr__(self):
        return "Running Clinics Model"

    def configure(self):
        self.addSegment("Pelvis",0,enums.SegmentSide.Central,calibration_markers=["SACR","midASIS"], tracking_markers = ["LASI","RASI","LPSI","RPSI"])
        self.addSegment("Left Thigh",1,enums.SegmentSide.Left,calibration_markers=["LKNE","LKNM"], tracking_markers = ["LTHI1","LTHI2","LTHI3","LTHI4"])
        self.addSegment("Right Thigh",4,enums.SegmentSide.Right,calibration_markers=["RKNE","RKNM"], tracking_markers = ["RTHI1","RTHI2","RTHI3","RTHI4"])
        self.addSegment("Left Shank",2,enums.SegmentSide.Left,calibration_markers=["LANK","LMED"], tracking_markers = ["LTIB1","LTIB2","LTIB3","LTIB4"])
        self.addSegment("Right Shank",5,enums.SegmentSide.Right,calibration_markers=["RANK","RMED"], tracking_markers = ["RTIB1","RTIB2","RTIB3","RTIB4"])
        self.addSegment("Left Foot",3,enums.SegmentSide.Left,calibration_markers=["LMET"], tracking_markers = ["LHEE","LFMH","LSMH","LVMH"] )
        self.addSegment("Right Foot",6,enums.SegmentSide.Right,calibration_markers=["RMET"], tracking_markers = ["RHEE","RFMH","RSMH","RVMH"])

        self.addJoint("LHip","Pelvis", "Left Thigh","YXZ")
        self.addJoint("LKnee","Left Thigh", "Left Shank","YXZ")
        self.addJoint("LAnkle","Left Shank", "Left Foot","YXZ")

        self.addJoint("RHip","Pelvis", "Right Thigh","YXZ")
        self.addJoint("RKnee","Right Thigh", "Right Shank","YXZ")
        self.addJoint("RAnkle","Right Shank", "Right Foot","YXZ")

        self.setClinicalDescriptor("LHip",enums.DataType.Angle, [0,1,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("LKnee",enums.DataType.Angle, [0,1,2],[+1.0,-1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("LAnkle",enums.DataType.Angle, [0,2,1],[-1.0,-1.0,-1.0], [ np.radians(90),0.0,0.0])
        self.setClinicalDescriptor("RHip", enums.DataType.Angle,[0,1,2],[-1.0,+1.0,+1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("RKnee", enums.DataType.Angle,[0,1,2],[+1.0,+1.0,+1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("RAnkle",enums.DataType.Angle, [0,2,1],[-1.0,+1.0,+1.0], [ np.radians(90),0.0,0.0])

        self.setClinicalDescriptor("Pelvis",enums.DataType.Angle,[0,1,2],[1.0,1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("Left Foot",enums.DataType.Angle,[0,2,1],[1.0,1.0,-1.0], [0.0,0.0,0.0])
        self.setClinicalDescriptor("Right Foot",enums.DataType.Angle,[0,2,1],[1.0,-1.0,1.0], [0.0,0.0,0.0])

    def calibrationProcedure(self):

        dictRef={}
        dictRef["Pelvis"]={"TF" : {'sequence':"YZX", 'labels':   ["RASI","LASI","SACR","midASIS"]} }
        dictRef["Left Thigh"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LTHI1","LTHI2","LTHI3","LTHI1"]} }
        dictRef["Right Thigh"]={"TF" : {'sequence':"ZXY", 'labels':   ["RTHI1","RTHI2","RTHI3","RTHI1"]} }
        dictRef["Left Shank"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LTIB1","LTIB2","LTIB3","LTIB1"]} }
        dictRef["Right Shank"]={"TF" : {'sequence':"ZXY", 'labels':   ["RTIB1","RTIB2","RTIB3","RTIB1"]} }
        dictRef["Left Foot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["LSMH","LHEE","LVMH","LHEE"]} }
        dictRef["Right Foot"]={"TF" : {'sequence':"ZXiY", 'labels':   ["RSMH","RHEE","RVMH","RHEE"]} }

        dictRefAnatomical={}
        dictRefAnatomical["Pelvis"]= {'sequence':"YZX", 'labels':  ["RASI","LASI","SACR","midASIS"]}
        dictRefAnatomical["Left Thigh"]= {'sequence':"ZXiY", 'labels':  ["LKJC","LHJC","LKNE","LHJC"]}
        dictRefAnatomical["Right Thigh"]= {'sequence':"ZXY", 'labels': ["RKJC","RHJC","RKNE","RHJC"]}
        dictRefAnatomical["Left Shank"]={'sequence':"ZXiY", 'labels':   ["LAJC","LKJC","LANK","LKJC"]}
        dictRefAnatomical["Right Shank"]={'sequence':"ZXY", 'labels':  ["RAJC","RKJC","RANK","RKJC"]}
        dictRefAnatomical["Left Foot"]={'sequence':"ZXiY", 'labels':  ["LMET","LAJC","LVMH","LAJC"]}
        dictRefAnatomical["Right Foot"]={'sequence':"ZXiY", 'labels':  ["RMET","RAJC","RVMH","RAJC"]}


        return dictRef,dictRefAnatomical


    def calibrate(self,aquiStatic, dictRef, dictAnatomic,  options=None):

        # add markers
        valSACR=(aquiStatic.GetPoint("LPSI").GetValues() + aquiStatic.GetPoint("RPSI").GetValues()) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"SACR",valSACR,desc="")
        valMidAsis=(aquiStatic.GetPoint("LASI").GetValues() + aquiStatic.GetPoint("RASI").GetValues()) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"midASIS",valMidAsis,desc="")
        valLMET=(aquiStatic.GetPoint("LFMH").GetValues() + aquiStatic.GetPoint("LVMH").GetValues()) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"LMET",valLMET,desc="")
        valRMET=(aquiStatic.GetPoint("RFMH").GetValues() + aquiStatic.GetPoint("RVMH").GetValues()) / 2.0
        btkTools.smartAppendPoint(aquiStatic,"RMET",valRMET,desc="")


        ff=aquiStatic.GetFirstFrame()
        lf=aquiStatic.GetLastFrame()
        frameInit=ff-ff
        frameEnd=lf-ff+1

        # calibrate technical
        self._calibrateTechnicalSegment(aquiStatic, "Pelvis", dictRef,frameInit,frameEnd, options=options)
        self._calibrateTechnicalSegment(aquiStatic, "Left Thigh", dictRef,frameInit,frameEnd, options=options)
        self._calibrateTechnicalSegment(aquiStatic, "Right Thigh", dictRef,frameInit,frameEnd, options=options)
        self._calibrateTechnicalSegment(aquiStatic, "Left Shank", dictRef,frameInit,frameEnd, options=options)
        self._calibrateTechnicalSegment(aquiStatic, "Right Shank", dictRef,frameInit,frameEnd, options=options)
        self._calibrateTechnicalSegment(aquiStatic, "Left Foot", dictRef,frameInit,frameEnd, options=options)
        self._calibrateTechnicalSegment(aquiStatic, "Right Foot", dictRef,frameInit,frameEnd, options=options)

        # # ---- decorator ----
        modelDecorator.HipJointCenterDecorator(self).greatTrochanterOffset(aquiStatic)
        modelDecorator.KneeCalibrationDecorator(self).midCondyles(aquiStatic)
        modelDecorator.AnkleCalibrationDecorator(self).midMaleolus(aquiStatic)



        # calibrate anatomic
        self._calibrateAnatomicalSegment(aquiStatic, "Pelvis", dictAnatomic,frameInit,frameEnd, options=options)
        self._calibrateAnatomicalSegment(aquiStatic, "Left Thigh", dictAnatomic,frameInit,frameEnd, options=options)
        self._calibrateAnatomicalSegment(aquiStatic, "Right Thigh", dictAnatomic,frameInit,frameEnd, options=options)
        self._calibrateAnatomicalSegment(aquiStatic, "Left Shank", dictAnatomic,frameInit,frameEnd, options=options)
        self._calibrateAnatomicalSegment(aquiStatic, "Right Shank", dictAnatomic,frameInit,frameEnd, options=options)
        self._calibrateAnatomicalSegment(aquiStatic, "Left Foot", dictAnatomic,frameInit,frameEnd, options=options)
        self._calibrateAnatomicalSegment(aquiStatic, "Right Foot", dictAnatomic,frameInit,frameEnd, options=options)



    def computeOptimizedSegmentMotion(self,aqui,segments, dictRef,dictAnat,motionMethod ):

        for seg in segments:
            self.computeMotionTechnicalFrame(aqui,seg,dictRef,motionMethod,options=options)
            self.computeMotionAnatomicalFrame(aqui,seg,dictAnat,options=options)


    def computeMotion(self,aqui, dictRef,dictAnat, motionMethod,options=None ):

        self.computeMotionTechnicalFrame(aqui,"Pelvis",dictRef,motionMethod,options=options)
        self.computeMotionTechnicalFrame(aqui,"Left Thigh",dictRef,motionMethod,options=options)
        self.computeMotionTechnicalFrame(aqui,"Right Thigh",dictRef,motionMethod,options=options)
        self.computeMotionTechnicalFrame(aqui,"Left Shank",dictRef,motionMethod,options=options)
        self.computeMotionTechnicalFrame(aqui,"Right Shank",dictRef,motionMethod,options=options)
        self.computeMotionTechnicalFrame(aqui,"Left Foot",dictRef,motionMethod,options=options)
        self.computeMotionTechnicalFrame(aqui,"Right Foot",dictRef,motionMethod,options=options)

        self.computeMotionAnatomicalFrame(aqui,"Pelvis",dictAnat,options=options)
        self.computeMotionAnatomicalFrame(aqui,"Left Thigh",dictAnat,options=options)
        self.computeMotionAnatomicalFrame(aqui,"Right Thigh",dictAnat,options=options)
        self.computeMotionAnatomicalFrame(aqui,"Left Shank",dictAnat,options=options)
        self.computeMotionAnatomicalFrame(aqui,"Right Shank",dictAnat,options=options)
        self.computeMotionAnatomicalFrame(aqui,"Left Foot",dictAnat,options=options)
        self.computeMotionAnatomicalFrame(aqui,"Right Foot",dictAnat,options=options)
