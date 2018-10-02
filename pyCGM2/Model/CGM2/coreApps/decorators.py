# -*- coding: utf-8 -*-
import numpy as np
import logging
from pyCGM2 import enums
from pyCGM2.Model import modelFilters, modelDecorator

def applyDecorators_CGM1(smc, model,acqStatic,optional_mp,markerDiameter):

    # native but thighRotation altered in mp
    if smc["left"] == enums.CgmStaticMarkerConfig.Native and optional_mp["LeftThighRotation"] !=0:
        logging.warning("CASE FOUND ===> Left Side = NATIVE CGM1 + manual Thigh  ")
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",optional_mp["LeftThighRotation"],
            markerDiameter, optional_mp["LeftTibialTorsion"], optional_mp["LeftShankRotation"])

    if smc["right"] == enums.CgmStaticMarkerConfig.Native and optional_mp["RightThighRotation"] !=0:
        logging.warning("CASE FOUND ===> Right Side = NATIVE CGM1 + manual Thigh  ")
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",optional_mp["RightThighRotation"],
            markerDiameter,optional_mp["RightTibialTorsion"],optional_mp["RightShankRotation"])

    # KAD
    if smc["left"] == enums.CgmStaticMarkerConfig.KAD:
        logging.warning("CASE FOUND ===> Left Side = KAD")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left")

    if smc["right"] == enums.CgmStaticMarkerConfig.KAD:
        logging.warning("CASE FOUND ===> Right Side = KAD")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right")

    # KADmed
    if smc["left"] == enums.CgmStaticMarkerConfig.KADmed:
        logging.warning("CASE FOUND ===> Left Side = KAD+med")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")

    if smc["right"] == enums.CgmStaticMarkerConfig.KADmed:
        logging.warning("CASE FOUND ===> Right Side = KAD+med")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")

def applyDecorators_CGM(smc, model,acqStatic,optional_mp,markerDiameter):

    # native but thighRotation altered in mp
    if smc["left"] == enums.CgmStaticMarkerConfig.Native and optional_mp["LeftThighRotation"] !=0:
        logging.warning("CASE FOUND ===> Left Side = NATIVE CGM1 + manual Thigh  ")
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",optional_mp["LeftThighRotation"],
            markerDiameter, optional_mp["LeftTibialTorsion"], optional_mp["LeftShankRotation"])

    if smc["right"] == enums.CgmStaticMarkerConfig.Native and optional_mp["RightThighRotation"] !=0:
        logging.warning("CASE FOUND ===> Right Side = NATIVE CGM1 + manual Thigh  ")
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",optional_mp["RightThighRotation"],
            markerDiameter,optional_mp["RightTibialTorsion"],optional_mp["RightShankRotation"])

    # KAD
    if smc["left"] == enums.CgmStaticMarkerConfig.KAD:
        logging.warning("CASE FOUND ===> Left Side = KAD")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left")

    if smc["right"] == enums.CgmStaticMarkerConfig.KAD:
        logging.warning("CASE FOUND ===> Right Side = KAD")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right")

    # KADmed
    if smc["left"] == enums.CgmStaticMarkerConfig.KADmed:
        logging.warning("CASE FOUND ===> Left Side = KAD+med")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")

    if smc["right"] == enums.CgmStaticMarkerConfig.KADmed:
        logging.warning("CASE FOUND ===> Right Side = KAD+med")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")

    # KneeMed
    if smc["left"] == enums.CgmStaticMarkerConfig.KneeMed:
        logging.warning("CASE FOUND ===> Left Side = KneeMed -  ankle for KJC")
        modelDecorator.KneeCalibrationDecorator(model).midCondyles_KAD(acqStatic, markerDiameter=markerDiameter, side="left")

    if smc["right"] == enums.CgmStaticMarkerConfig.KneeMed:
        logging.warning("CASE FOUND ===> Right Side = KneeMed -  ankle for KJC")
        modelDecorator.KneeCalibrationDecorator(model).midCondyles_KAD(acqStatic, markerDiameter=markerDiameter, side="right")

    # KneeAnkleMed
    if smc["left"] == enums.CgmStaticMarkerConfig.KneeAnkleMed:
        logging.warning("CASE FOUND ===> Left Side = Knee and Ankle Medial")
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="left")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")

    if smc["right"] == enums.CgmStaticMarkerConfig.KneeAnkleMed:
        logging.warning("CASE FOUND ===> Right Side = Knee and Ankle Medial")
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="right")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")

    # AnkleMed
    if smc["left"] == enums.CgmStaticMarkerConfig.AnkleMed:
        logging.warning("CASE FOUND ===> Left Side = Ankle Medial")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")

    if smc["right"] == enums.CgmStaticMarkerConfig.AnkleMed:
        logging.warning("CASE FOUND ===> Right Side = Ankle Medial")
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
