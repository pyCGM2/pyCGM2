"""
this module gathers decorators specific to the CGM
"""
import numpy as np
from typing import List, Tuple, Dict, Optional,Union,Any

import pyCGM2; LOGGER = pyCGM2.LOGGER
from pyCGM2 import enums
from pyCGM2.Model import modelDecorator

from pyCGM2.Model.CGM2.cgm import CGM
import btk


def applyBasicDecorators(dcm: dict, model: CGM, acqStatic: btk.btkAcquisition, 
                         optional_mp: dict, markerDiameter: float, cgm1only: bool = False) -> None:
    """
    Apply decorators from detected calibration method for a CGM model.

    Args:
        dcm (dict): Dictionary returned from the function `detectCalibrationMethods`.
        model (CGM): A CGM model instance.
        acqStatic (btk.btkAcquisition): A BTK acquisition instance of a static file.
        optional_mp (dict): Optional anthropometric parameters of the CGM.
        markerDiameter (float): Diameter of the marker.
        cgm1only (bool, optional): Enable computation for the CGM1 only. Default is False.
    """


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
    
    # medial ankle only
    if dcm["Left Knee"] == enums.JointCalibrationMethod.Basic:
        LOGGER.logger.debug("CASE FOUND ===> Left Side = Knee-Lateral marker")
        if  dcm["Left Ankle"] == enums.JointCalibrationMethod.Medial:
            LOGGER.logger.debug("CASE FOUND ===> Left Side = Ankle-Med")
            modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")

    if dcm["Right Knee"] == enums.JointCalibrationMethod.Basic:
        LOGGER.logger.debug("CASE FOUND ===> Right Side = Knee-lateral marker")
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

def applyKJC_AJCDecorators(dcm: dict, model: CGM, acqStatic: btk.btkAcquisition, 
                           optional_mp: dict, markerDiameter: float, cgm1only: bool = False, 
                           forceMP: bool = False) -> None:
    """
    Apply decorators from detected calibration method with specific focus on KJC and AJC.

    Args:
        dcm (dict): Dictionary returned from the function `detectCalibrationMethods`.
        model (CGM): A CGM model instance.
        acqStatic (btk.btkAcquisition): A BTK acquisition instance of a static file.
        optional_mp (dict): Optional anthropometric parameters of the CGM.
        markerDiameter (float): Diameter of the marker.
        cgm1only (bool, optional): Enable computation for the CGM1 only. Default is False.
        forceMP (bool): Force the use of mp offset to compute KJC and AJC. Default is False.
    """

    if forceMP:
        LOGGER.logger.info("[pyCGM2] KJC and AJC computed from MP offset values")
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",optional_mp["LeftThighRotation"],
            markerDiameter, optional_mp["LeftTibialTorsion"], optional_mp["LeftShankRotation"])
        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",optional_mp["RightThighRotation"],
            markerDiameter,optional_mp["RightTibialTorsion"],optional_mp["RightShankRotation"])
        
    else:
        if not cgm1only:
            if dcm["Left Knee"] == enums.JointCalibrationMethod.Medial:
                if  dcm["Left Ankle"] == enums.JointCalibrationMethod.Medial:
                    LOGGER.logger.info("[pyCGM2] scenario Left : Medial Knee - Medial ankle")
                    modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="left")
                    modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")
                if dcm["Left Ankle"] == enums.JointCalibrationMethod.Basic:
                    LOGGER.logger.info("[pyCGM2] scenario Left : Medial knee - lateral shank marker (=>Kad-like) ")
                    modelDecorator.KneeCalibrationDecorator(model).midCondyles_KAD(acqStatic, markerDiameter=markerDiameter, side="left")
                    #no ankle decorator


        if dcm["Left Knee"] == enums.JointCalibrationMethod.KAD:
            if  dcm["Left Ankle"] == enums.JointCalibrationMethod.Medial:
                LOGGER.logger.info("[pyCGM2] scenario Left : KAD - Medial ankle marker")
                modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left")
                modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")
            elif  dcm["Left Ankle"] == enums.JointCalibrationMethod.Basic:
                LOGGER.logger.info("[pyCGM2] scenario Left : KAD - lateral shank marker")
                modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left")


        if dcm["Left Knee"] == enums.JointCalibrationMethod.Basic:
            if  dcm["Left Ankle"] == enums.JointCalibrationMethod.Medial:
                LOGGER.logger.info("[pyCGM2] scenario Left : lateral thigh marker - Medial ankle")
                modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")
            if  dcm["Left Ankle"] == enums.JointCalibrationMethod.Basic:
                LOGGER.logger.info("[pyCGM2] scenario Left : lateral thigh marker - lateral shank marker")
                 #no ankle decorator


        if not cgm1only:
            if dcm["Right Knee"] == enums.JointCalibrationMethod.Medial:
                if  dcm["Right Ankle"] == enums.JointCalibrationMethod.Medial:
                    LOGGER.logger.info("[pyCGM2] scenario right : Medial Knee - Medial ankle")
                    modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="right")
                    modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")
                if dcm["Right Ankle"] == enums.JointCalibrationMethod.Basic:
                    LOGGER.logger.info("[pyCGM2] scenario right : Medial knee - lateral shank marker (=>Kad-like) ")
                    modelDecorator.KneeCalibrationDecorator(model).midCondyles_KAD(acqStatic, markerDiameter=markerDiameter, side="right")
                    #no ankle decorator


        if dcm["Right Knee"] == enums.JointCalibrationMethod.KAD:
            if  dcm["Right Ankle"] == enums.JointCalibrationMethod.Medial:
                LOGGER.logger.info("[pyCGM2] scenario right : KAD - Medial ankle marker")
                modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right")
                modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")
            elif  dcm["Right Ankle"] == enums.JointCalibrationMethod.Basic:
                LOGGER.logger.info("[pyCGM2] scenario Left : KAD - lateral shank marker")
                modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right")
        

        if dcm["Right Knee"] == enums.JointCalibrationMethod.Basic:
            if  dcm["Right Ankle"] == enums.JointCalibrationMethod.Medial:
                LOGGER.logger.info("[pyCGM2] scenario right : lateral thigh marker - Medial ankle")
                modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")
            if  dcm["Right Ankle"] == enums.JointCalibrationMethod.Basic:
                LOGGER.logger.info("[pyCGM2] scenario right : lateral thigh marker - lateral shank marker")
                 #no ankle decorator
        



def applyHJCDecorators(model: CGM, method: dict) -> None:
    """
    Apply hip joint centre decorators to the CGM model.

    Args:
        model (CGM): A CGM model instance.
        method (dict): Dictionary indicating HJC method to use. Keys are 'Left' and 'Right' with values
                       being either "Hara" or a list of three numbers (position in the pelvic coordinate system).

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
