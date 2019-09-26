# -*- coding: utf-8 -*-
import numpy as np
import logging
from pyCGM2 import enums
from pyCGM2.Model import modelFilters, modelDecorator



def applyBasicDecorators(dcm, model,acqStatic,optional_mp,markerDiameter,cgm1only=False):

    # native but thighRotation altered in mp
    if dcm["Left Knee"] == enums.JointCalibrationMethod.Basic and  dcm["Left Ankle"] == enums.JointCalibrationMethod.Basic and optional_mp["LeftThighRotation"] !=0:
        logging.warning("CASE FOUND ===> Left Side = NATIVE CGM1 + manual Thigh  ")
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",optional_mp["LeftThighRotation"],
            markerDiameter, optional_mp["LeftTibialTorsion"], optional_mp["LeftShankRotation"])

    if dcm["Right Knee"] == enums.JointCalibrationMethod.Basic and  dcm["Right Ankle"] == enums.JointCalibrationMethod.Basic and optional_mp["RightThighRotation"] !=0:
        logging.warning("CASE FOUND ===> Right Side = NATIVE CGM1 + manual Thigh  ")
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",optional_mp["RightThighRotation"],
            markerDiameter,optional_mp["RightTibialTorsion"],optional_mp["RightShankRotation"])

    # KAD - and Kadmed
    if dcm["Left Knee"] == enums.JointCalibrationMethod.KAD:
        logging.warning("CASE FOUND ===> Left Side = Knee-KAD")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left")
        if  dcm["Left Ankle"] == enums.JointCalibrationMethod.Medial:
            logging.warning("CASE FOUND ===> Left Side = Ankle-Med")
            modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")

    if dcm["Right Knee"] == enums.JointCalibrationMethod.KAD:
        logging.warning("CASE FOUND ===> Right Side = Knee-KAD")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right")
        if  dcm["Right Ankle"] == enums.JointCalibrationMethod.Medial:
            logging.warning("CASE FOUND ===> Right Side = Ankle-Med")
            modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")

    if not cgm1only:

        #Kad-like (KneeMed)
        if dcm["Left Knee"] == enums.JointCalibrationMethod.Medial and dcm["Left Ankle"] == enums.JointCalibrationMethod.Basic:
            modelDecorator.KneeCalibrationDecorator(model).midCondyles_KAD(acqStatic, markerDiameter=markerDiameter, side="left")
        if dcm["Right Knee"] == enums.JointCalibrationMethod.Medial and dcm["Right Ankle"] == enums.JointCalibrationMethod.Basic:
            modelDecorator.KneeCalibrationDecorator(model).midCondyles_KAD(acqStatic, markerDiameter=markerDiameter, side="right")

        #knee and ankle Med
        if dcm["Left Knee"] == enums.JointCalibrationMethod.Medial and dcm["Left Ankle"] == enums.JointCalibrationMethod.Medial:
            modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="left")
            modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")

        if dcm["Right Knee"] == enums.JointCalibrationMethod.Medial and dcm["Right Ankle"] == enums.JointCalibrationMethod.Medial:
            modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="right")
            modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")


def applyHJCDecorators(model,method):

    if method["Left"] == "Hara":
        logging.info("[pyCGM2] Left HJC : Hara")
        modelDecorator.HipJointCenterDecorator(model).hara(side = "left")
    elif len(method["Left"]) == 3:
        logging.info("[pyCGM2] Left HJC : Custom")
        logging.warning(method["Left"])
        modelDecorator.HipJointCenterDecorator(model).custom(position_Left =  np.array(method["Left"]), methodDesc = "custom",side="left")

    if method["Right"] == "Hara":
        logging.info("[pyCGM2] Right HJC : Hara")
        modelDecorator.HipJointCenterDecorator(model).hara(side = "right")
    elif len(method["Right"]) == 3:
        logging.info("[pyCGM2] Right HJC : Custom")
        logging.warning(method["Right"])
        modelDecorator.HipJointCenterDecorator(model).custom(position_Right =  np.array(method["Right"]), methodDesc = "custom",side="right")
