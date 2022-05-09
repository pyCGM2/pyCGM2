# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Model/CGM2
#APIDOC["Draft"]=False
#--end--
"""
this module gathers decorators specific to the CGM
"""
import numpy as np
import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2 import enums
from pyCGM2.Model import modelDecorator



def applyBasicDecorators(dcm, model,acqStatic,optional_mp,markerDiameter,cgm1only=False):
    """Apply decorators from detected calibration method

    Args:
        dcm (dict): dictionary returned from the function `detectCalibrationMethods`
        model (pyCGM2.Model.CGM2.cgm): a CGM model instance
        acqStatic (btk.acquisition): a btk acquisition instance of a static file
        optional_mp (dict): optional anthropometric parameters of the CGM
        markerDiameter (float): marker diameter
        cgm1only (bool, optional[False]): enable computation for the CGM1 only

    """

    if model.getBodyPart() != enums.BodyPart.UpperLimb:
        # native but thighRotation altered in mp
        if dcm["Left Knee"] == enums.JointCalibrationMethod.Basic and  dcm["Left Ankle"] == enums.JointCalibrationMethod.Basic and optional_mp["LeftThighRotation"] !=0:
            LOGGER.logger.debug("CASE FOUND ===> Left Side = NATIVE CGM1 + manual Thigh  ")
            modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",optional_mp["LeftThighRotation"],
                markerDiameter, optional_mp["LeftTibialTorsion"], optional_mp["LeftShankRotation"])

        if dcm["Right Knee"] == enums.JointCalibrationMethod.Basic and  dcm["Right Ankle"] == enums.JointCalibrationMethod.Basic and optional_mp["RightThighRotation"] !=0:
            LOGGER.logger.debug("CASE FOUND ===> Right Side = NATIVE CGM1 + manual Thigh  ")
            modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",optional_mp["RightThighRotation"],
                markerDiameter,optional_mp["RightTibialTorsion"],optional_mp["RightShankRotation"])

        # KAD - and Kadmed
        if dcm["Left Knee"] == enums.JointCalibrationMethod.KAD:
            LOGGER.logger.debug("CASE FOUND ===> Left Side = Knee-KAD")
            modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left")
            if  dcm["Left Ankle"] == enums.JointCalibrationMethod.Medial:
                LOGGER.logger.debug("CASE FOUND ===> Left Side = Ankle-Med")
                modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")

        if dcm["Right Knee"] == enums.JointCalibrationMethod.KAD:
            LOGGER.logger.debug("CASE FOUND ===> Right Side = Knee-KAD")
            modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right")
            if  dcm["Right Ankle"] == enums.JointCalibrationMethod.Medial:
                LOGGER.logger.debug("CASE FOUND ===> Right Side = Ankle-Med")
                modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")

    if not cgm1only:

        if model.getBodyPart() != enums.BodyPart.UpperLimb:
            #Kad-like (KneeMed)
            if dcm["Left Knee"] == enums.JointCalibrationMethod.Medial and dcm["Left Ankle"] == enums.JointCalibrationMethod.Basic:
                modelDecorator.KneeCalibrationDecorator(model).midCondyles_KAD(acqStatic, markerDiameter=markerDiameter, side="left")
            if dcm["Right Knee"] == enums.JointCalibrationMethod.Medial and dcm["Right Ankle"] == enums.JointCalibrationMethod.Basic:
                modelDecorator.KneeCalibrationDecorator(model).midCondyles_KAD(acqStatic, markerDiameter=markerDiameter, side="right")

            #knee and ankle Med
            if dcm["Left Knee"] == enums.JointCalibrationMethod.Medial and dcm["Left Ankle"] == enums.JointCalibrationMethod.Medial:
                LOGGER.logger.info("[pyCGM2] Left Knee : Medial - Left Ankle : Medial")
                modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="left")
                modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")

            if dcm["Right Knee"] == enums.JointCalibrationMethod.Medial and dcm["Right Ankle"] == enums.JointCalibrationMethod.Medial:
                LOGGER.logger.info("[pyCGM2] Right Knee : Medial - Right Ankle : Medial")
                modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="right")
                modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")


def applyHJCDecorators(model, method):
    """apply hip joint centre decorators

    Args:
        model (pyCGM2.Model.CGM2.cgm): a CGM model instance
        method (dict): dict indicating HJC method to use

    Example:

    ```python
    method = {"Left": "Hara", "Right": [1,3,4]} # a string or a list (ie. position in the pelvic coordinate system)
    applyHJCDecorators(model, method)
    ```

    """
    if model.getBodyPart() != enums.BodyPart.UpperLimb:
        if method["Left"] == "Hara":
            LOGGER.logger.info("[pyCGM2] Left HJC : Hara")
            modelDecorator.HipJointCenterDecorator(model).hara(side = "left")
        elif len(method["Left"]) == 3:
            LOGGER.logger.info("[pyCGM2] Left HJC : Custom")
            LOGGER.logger.debug(method["Left"])
            modelDecorator.HipJointCenterDecorator(model).custom(position_Left =  np.array(method["Left"]), methodDesc = "custom",side="left")

        if method["Right"] == "Hara":
            LOGGER.logger.info("[pyCGM2] Right HJC : Hara")
            modelDecorator.HipJointCenterDecorator(model).hara(side = "right")
        elif len(method["Right"]) == 3:
            LOGGER.logger.info("[pyCGM2] Right HJC : Custom")
            LOGGER.logger.debug(method["Right"])
            modelDecorator.HipJointCenterDecorator(model).custom(position_Right =  np.array(method["Right"]), methodDesc = "custom",side="right")
