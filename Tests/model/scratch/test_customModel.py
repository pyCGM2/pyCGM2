# -*- coding: utf-8 -*-
import logging
import numpy as np

import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.DEBUG)

from pyCGM2.Model import modelFilters,model,modelDecorator,anthropometricMeasurement
from pyCGM2.Model.CGM2 import cgm

from pyCGM2 import enums
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files



if __name__ == "__main__":




    DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "MODEL\\Custom\\sample0\\"
    staticFilename = "static.c3d"
    dynamicFilename = "dynamic.c3d"

    acqStatic = btkTools.smartReader(str(DATA_PATH +  staticFilename))
    acqDynamic = btkTools.smartReader(str(DATA_PATH +  dynamicFilename))


    # new markers
    valSACR=(acqStatic.GetPoint("LPSI").GetValues() + acqStatic.GetPoint("RPSI").GetValues()) / 2.0
    btkTools.smartAppendPoint(acqStatic,"SACR",valSACR,desc="")
    valMidAsis=(acqStatic.GetPoint("LASI").GetValues() + acqStatic.GetPoint("RASI").GetValues()) / 2.0
    btkTools.smartAppendPoint(acqStatic,"midASIS",valMidAsis,desc="")
    valLMET=(acqStatic.GetPoint("LFMH").GetValues() + acqStatic.GetPoint("LVMH").GetValues()) / 2.0
    btkTools.smartAppendPoint(acqStatic,"LMET",valLMET,desc="")
    valRMET=(acqStatic.GetPoint("RFMH").GetValues() + acqStatic.GetPoint("RVMH").GetValues()) / 2.0
    btkTools.smartAppendPoint(acqStatic,"RMET",valRMET,desc="")

    # anthropometric Me


    # ---- Model configuration ----
    bioMechModel = model.Model()
    mp={
    'Bodymass'   : 71.0,
    'LeftKneeWidth' : anthropometricMeasurement.measureNorm(acqStatic,"LKNE","LKNM",markerDiameter =14),
    'RightKneeWidth' : anthropometricMeasurement.measureNorm(acqStatic,"RKNE","RKNM",markerDiameter =14),
    'LeftAnkleWidth' : anthropometricMeasurement.measureNorm(acqStatic,"LANK","LMED",markerDiameter =14),
    'RightAnkleWidth' : anthropometricMeasurement.measureNorm(acqStatic,"RANK","RMED",markerDiameter =14),
    }
    bioMechModel.addAnthropoInputParameters(mp)

    bioMechModel.addSegment("Pelvis",0,enums.SegmentSide.Central,calibration_markers=["SACR","midASIS"], tracking_markers = ["LASI","RASI","LPSI","RPSI"])
    bioMechModel.addSegment("Left Thigh",1,enums.SegmentSide.Left,calibration_markers=["LKNE","LKNM"], tracking_markers = ["LTHI1","LTHI2","LTHI3","LTHI4"])
    bioMechModel.addSegment("Right Thigh",4,enums.SegmentSide.Right,calibration_markers=["RKNE","RKNM"], tracking_markers = ["RTHI1","RTHI2","RTHI3","RTHI4"])
    bioMechModel.addSegment("Left Shank",2,enums.SegmentSide.Left,calibration_markers=["LANK","LMED"], tracking_markers = ["LTIB1","LTIB2","LTIB3","LTIB4"])
    bioMechModel.addSegment("Right Shank",5,enums.SegmentSide.Right,calibration_markers=["RANK","RMED"], tracking_markers = ["RTIB1","RTIB2","RTIB3","RTIB4"])
    bioMechModel.addSegment("Left Foot",3,enums.SegmentSide.Left,calibration_markers=["LMET"], tracking_markers = ["LHEE","LFMH","LSMH","LVMH"] )
    bioMechModel.addSegment("Right Foot",6,enums.SegmentSide.Right,calibration_markers=["RMET"], tracking_markers = ["RHEE","RFMH","RSMH","RVMH"])

    bioMechModel.addJoint("LHip","Pelvis", "Left Thigh","YXZ")
    bioMechModel.addJoint("LKnee","Left Thigh", "Left Shank","YXZ")
    bioMechModel.addJoint("LAnkle","Left Shank", "Left Foot","YXZ")

    bioMechModel.addJoint("RHip","Pelvis", "Right Thigh","YXZ")
    bioMechModel.addJoint("RKnee","Right Thigh", "Right Shank","YXZ")
    bioMechModel.addJoint("RAnkle","Right Shank", "Right Foot","YXZ")

    gcp=modelFilters.GeneralCalibrationProcedure()
    gcp.setDefinition("Pelvis","TF",sequence = "YZX" ,pointLabel1="RASI",pointLabel2="LASI", pointLabel3="SACR",pointLabelOrigin="midASIS")
    gcp.setDefinition("Left Thigh","TF",sequence = "ZXiY" ,pointLabel1="LTHI1",pointLabel2="LTHI2", pointLabel3="LTHI3",pointLabelOrigin="LTHI1")
    gcp.setDefinition("Right Thigh","TF",sequence = "ZXY" ,pointLabel1="RTHI1",pointLabel2="RTHI2", pointLabel3="RTHI3",pointLabelOrigin="RTHI1")
    gcp.setDefinition("Left Shank","TF",sequence = "ZXiY" ,pointLabel1="LTIB1",pointLabel2="LTIB2", pointLabel3="LTIB3",pointLabelOrigin="LTIB1")
    gcp.setDefinition("Right Shank","TF",sequence = "ZXY" ,pointLabel1="RTIB1",pointLabel2="RTIB2", pointLabel3="RTIB3",pointLabelOrigin="RTIB1")
    gcp.setDefinition("Left Foot","TF",sequence = "ZXiY" ,pointLabel1="LSMH",pointLabel2="LHEE", pointLabel3="LVMH",pointLabelOrigin="LHEE")
    gcp.setDefinition("Right Foot","TF",sequence = "ZXY" ,pointLabel1="RSMH",pointLabel2="RHEE", pointLabel3="RVMH",pointLabelOrigin="RHEE")

    #---Initial Calibration Filter---
    smf0 = modelFilters.ModelCalibrationFilter(gcp,acqStatic,bioMechModel)
    smf0.setBoolOption("technicalReferentialOnly")
    smf0.compute()

    # ---- decorator ----
    modelDecorator.HipJointCenterDecorator(bioMechModel).greatTrochanterOffset(acqStatic)
    modelDecorator.KneeCalibrationDecorator(bioMechModel).midCondyles(acqStatic)
    modelDecorator.AnkleCalibrationDecorator(bioMechModel).midMaleolus(acqStatic)


    gcp.setAnatomicalDefinition("Pelvis",sequence = "YZX" ,nodeLabel1="RASI",nodeLabel2="LASI", nodeLabel3="SACR",nodeLabelOrigin="midASIS")
    gcp.setAnatomicalDefinition("Left Thigh",sequence = "ZXiY" ,nodeLabel1="LKJC_mid",nodeLabel2="LHJC_gt", nodeLabel3="LKNE",nodeLabelOrigin="LHJC_gt")
    gcp.setAnatomicalDefinition("Right Thigh",sequence = "ZXY" ,nodeLabel1="RKJC_mid",nodeLabel2="RHJC_gt", nodeLabel3="RKNE",nodeLabelOrigin="RHJC_gt")
    gcp.setAnatomicalDefinition("Left Shank",sequence = "ZXiY" ,nodeLabel1="LAJC_mid",nodeLabel2="LKJC_mid", nodeLabel3="LANK",nodeLabelOrigin="LKJC_mid")
    gcp.setAnatomicalDefinition("Right Shank",sequence = "ZXY" ,nodeLabel1="RAJC_mid",nodeLabel2="RKJC_mid", nodeLabel3="RANK",nodeLabelOrigin="RKJC_mid")
    gcp.setAnatomicalDefinition("Left Foot",sequence = "ZXiY" ,nodeLabel1="LMET",nodeLabel2="LAJC_mid", nodeLabel3="LVMH",nodeLabelOrigin="LAJC_mid")
    gcp.setAnatomicalDefinition("Right Foot",sequence = "ZXiY" ,nodeLabel1="RMET",nodeLabel2="RAJC_mid", nodeLabel3="RVMH",nodeLabelOrigin="RAJC_mid")

    #---second Calibration Filter---
    smf1 = modelFilters.ModelCalibrationFilter(gcp,acqStatic,bioMechModel)
    smf1.compute()

    btkTools.smartWriter(acqStatic, "rcmStatic.c3d")


    #---Motion Filter---
    mmf=modelFilters.ModelMotionFilter(gcp,acqDynamic,bioMechModel,enums.motionMethod.Sodervisk)
    mmf.compute()


    #---JCS---
    bioMechModel.setClinicalDescriptor("LHip",enums.DataType.Angle, [0,1,2],[-1.0,-1.0,-1.0], [0.0,0.0,0.0])
    bioMechModel.setClinicalDescriptor("LKnee",enums.DataType.Angle, [0,1,2],[+1.0,-1.0,-1.0], [0.0,0.0,0.0])
    bioMechModel.setClinicalDescriptor("LAnkle",enums.DataType.Angle, [0,2,1],[-1.0,-1.0,-1.0], [ np.radians(90),0.0,0.0])
    bioMechModel.setClinicalDescriptor("RHip",enums.DataType.Angle, [0,1,2],[-1.0,+1.0,+1.0], [0.0,0.0,0.0])
    bioMechModel.setClinicalDescriptor("RKnee",enums.DataType.Angle, [0,1,2],[+1.0,+1.0,+1.0], [0.0,0.0,0.0])
    bioMechModel.setClinicalDescriptor("RAnkle",enums.DataType.Angle, [0,2,1],[-1.0,+1.0,+1.0], [ np.radians(90),0.0,0.0])

    jcsf = modelFilters.ModelJCSFilter(bioMechModel,acqDynamic)
    jcsf.compute(description="vectoriel", pointLabelSuffix="rcm")

    longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqDynamic,["LASI","LPSI","RASI","RPSI"])

    bioMechModel.setClinicalDescriptor("Pelvis",enums.DataType.Angle,[0,1,2],[1.0,1.0,-1.0], [0.0,0.0,0.0])
    bioMechModel.setClinicalDescriptor("Left Foot",enums.DataType.Angle,[0,2,1],[1.0,1.0,-1.0], [0.0,0.0,0.0])
    bioMechModel.setClinicalDescriptor("Right Foot",enums.DataType.Angle,[0,2,1],[1.0,-1.0,1.0], [0.0,0.0,0.0])

    aaf = modelFilters.ModelAbsoluteAnglesFilter(bioMechModel,acqDynamic,
                              segmentLabels=["Left Foot","Right Foot","Pelvis"],
                              angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                              eulerSequences=["TOR","TOR", "TOR"],
                              globalFrameOrientation = globalFrame,
                              forwardProgression = forwardProgression)


    aaf.compute(pointLabelSuffix="rcm")

    btkTools.smartWriter(acqDynamic, "rcmDynamic.c3d")
