# -*- coding: utf-8 -*-

import logging
from pyCGM2 import enums
from pyCGM2.Model import modelFilters, modelDecorator


def get_markerLabelForPiGStatic(smc):

    useLeftKJCmarkerLabel = "LKJC"
    useLeftAJCmarkerLabel = "LAJC"
    useRightKJCmarkerLabel = "RKJC"
    useRightAJCmarkerLabel = "RAJC"


    # KAD
    if smc["left"] == enums.CgmStaticMarkerConfig.KAD:
        useLeftKJCmarkerLabel = "LKJC_KAD"
        useLeftAJCmarkerLabel = "LAJC_KAD"

    if smc["right"] == enums.CgmStaticMarkerConfig.KAD:
        useRightKJCmarkerLabel = "RKJC_KAD"
        useRightAJCmarkerLabel = "RAJC_KAD"

    # KADmed
    if smc["left"] == enums.CgmStaticMarkerConfig.KADmed:
        useLeftKJCmarkerLabel = "LKJC_KAD"
        useLeftAJCmarkerLabel = "LAJC_MID"

    if smc["right"] == enums.CgmStaticMarkerConfig.KADmed:
        useRightKJCmarkerLabel = "RKJC_KAD"
        useRightAJCmarkerLabel = "RAJC_MID"

    return [useLeftKJCmarkerLabel,useLeftAJCmarkerLabel,useRightKJCmarkerLabel,useRightAJCmarkerLabel]



class argsManager_cgm(object):
    def __init__(self, settings, args):
        self.settings = settings
        self.args = args

    def getLeftFlatFoot(self):
        if self.args.leftFlatFoot is not None:
            logging.warning("Left flat foot forces : %s"%(str(bool(self.args.leftFlatFoot))))
            return  bool(self.args.leftFlatFoot)
        else:
            return bool(self.settings["Calibration"]["Left flat foot"])

    def getRightFlatFoot(self):
        if self.args.rightFlatFoot is not None:
            logging.warning("Right flat foot forces : %s"%(str(bool(self.args.rightFlatFoot))))
            return bool(self.args.rightFlatFoot)
        else:
            return  bool(self.settings["Calibration"]["Right flat foot"])


    def getMarkerDiameter(self):
        if self.args.markerDiameter is not None:
            logging.warning("marker diameter forced : %s", str(float(self.args.markerDiameter)))
            return float(self.args.markerDiameter)
        else:
            return float(self.settings["Global"]["Marker diameter"])

    def getPointSuffix(self,checkValue):
        if self.args.check:
            return checkValue
        else:
            if self.args.pointSuffix is not None:
                return self.args.pointSuffix
            else:
                return self.settings["Global"]["Point suffix"]


    def getMomentProjection(self):
        if self.args.proj is not None:
            if self.args.proj == "Distal":
                return  enums.MomentProjection.Distal
            elif self.args.proj == "Proximal":
                return  enums.MomentProjection.Proximal
            elif self.args.proj == "Global":
                return  enums.MomentProjection.Global
            elif args.proj == "JCS":
                return pyCGM2Enums.MomentProjection.JCS
            else:
                raise Exception("[pyCGM2] Moment projection doesn t recognise in your settings. choice is Proximal, Distal or Global")

        else:
            if self.settings["Fitting"]["Moment Projection"] == "Distal":
                return  enums.MomentProjection.Distal
            elif self.settings["Fitting"]["Moment Projection"] == "Proximal":
                return  enums.MomentProjection.Proximal
            elif self.settings["Fitting"]["Moment Projection"] == "Global":
                return  enums.MomentProjection.Global
            elif settings["Fitting"]["Moment Projection"] == "JCS":
                return enums.MomentProjection.JCS

            else:
                raise Exception("[pyCGM2] Moment projection doesn t recognise in your settings. choice is Proximal, Distal or Global")

    def getManualForcePlateAssign(self):
        return self.args.mfpa

class argsManager_cgm1(argsManager_cgm):
    def __init__(self, settings, args):
        super(argsManager_cgm1, self).__init__(settings, args)

    def getMomentProjection(self):
        if self.args.proj is not None:
            if self.args.proj == "Distal":
                return  enums.MomentProjection.Distal
            elif self.args.proj == "Proximal":
                return  enums.MomentProjection.Proximal
            elif self.args.proj == "Global":
                return  enums.MomentProjection.Global
            else:
                raise Exception("[pyCGM2] Moment projection doesn t recognise in your settings. choice is Proximal, Distal or Global")

        else:
            if self.settings["Fitting"]["Moment Projection"] == "Distal":
                return  enums.MomentProjection.Distal
            elif self.settings["Fitting"]["Moment Projection"] == "Proximal":
                return  enums.MomentProjection.Proximal
            elif self.settings["Fitting"]["Moment Projection"] == "Global":
                return  enums.MomentProjection.Global

            else:
                raise Exception("[pyCGM2] Moment projection doesn t recognise in your settings. choice is Proximal, Distal or Global")



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
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left")

    # KADmed
    if smc["left"] == enums.CgmStaticMarkerConfig.KADmed:
        logging.warning("CASE FOUND ===> Right Side = KAD+med")
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
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left")

    # KADmed
    if smc["left"] == enums.CgmStaticMarkerConfig.KADmed:
        logging.warning("CASE FOUND ===> Right Side = KAD+med")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")

    if smc["right"] == enums.CgmStaticMarkerConfig.KADmed:
        logging.warning("CASE FOUND ===> Right Side = KAD+med")
        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")

    # KneeMed
    if smc["left"] == enums.CgmStaticMarkerConfig.KneeMed:
        logging.warning("CASE FOUND ===> Right Side = KneeMed -  ankle for KJC")
        modelDecorator.KneeCalibrationDecorator(model).midCondyles_KAD(acqStatic, markerDiameter=markerDiameter, side="left")

    if smc["right"] == enums.CgmStaticMarkerConfig.KneeMed:
        logging.warning("CASE FOUND ===> Right Side = KneeMed -  ankle for KJC")
        modelDecorator.KneeCalibrationDecorator(model).midCondyles_KAD(acqStatic, markerDiameter=markerDiameter, side="right")

    # KneeAnkleMed
    if smc["left"] == enums.CgmStaticMarkerConfig.KneeAnkleMed:
        logging.warning("CASE FOUND ===> Right Side = Knee and Ankle Medial")
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="left")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")

    if smc["right"] == enums.CgmStaticMarkerConfig.KneeAnkleMed:
        logging.warning("CASE FOUND ===> Right Side = Knee and Ankle Medial")
        modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="right")
        modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")

def applyHJCDecorators(model,method,side="both"):

    if method == "Hara":
        modelDecorator.HipJointCenterDecorator(model).hara(side = side)
