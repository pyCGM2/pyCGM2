# -*- coding: utf-8 -*-
import numpy as np
import logging
import copy

import pyCGM2

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2.Model.CGM2 import cgm,cgm2
from pyCGM2.Model import  modelFilters, modelDecorator
from pyCGM2 import enums
from pyCGM2.Signal import signal_processing



def detectSide(acq,left_markerLabel,right_markerLabel):

    flag,vff,vlf = btkTools.findValidFrames(acq,[left_markerLabel,right_markerLabel])

    left = acq.GetPoint(left_markerLabel).GetValues()[vff:vlf,2]
    right = acq.GetPoint(right_markerLabel).GetValues()[vff:vlf,2]

    side = "Left" if np.max(left)>np.max(right) else "Right"

    return side

def calibration2Dof(model,
    DATA_PATH,reconstructFilenameLabelled,translators,
    side,beginFrame,endFrame,jointRange,**kwargs):

    # --- btk acquisition ----
    if "forceBtkAcq" in kwargs.keys():
        acqFunc = kwargs["forceBtkAcq"]
    else:
        acqFunc = btkTools.smartReader((DATA_PATH + reconstructFilenameLabelled))

    btkTools.checkMultipleSubject(acqFunc)
    acqFunc =  btkTools.applyTranslators(acqFunc,translators)

    # filtering
    # -----------------------
    if "fc_lowPass_marker" in kwargs.keys() :
        fc = kwargs["fc_lowPass_marker"]
        order = 4
        if "order_lowPass_marker" in kwargs.keys(): order = kwargs["order_lowPass_marker"]
        signal_processing.markerFiltering(acqFunc,order=order, fc =fc)

    if "fc_lowPass_forcePlate" in kwargs.keys() :
        fc = kwargs["fc_lowPass_forcePlate"]
        order = 4
        if "order_lowPass_forcePlate" in kwargs.keys(): order = kwargs["order_lowPass_forcePlate"]
        signal_processing.forcePlateFiltering(acqFunc,order=order, fc =fc)

    #---get frame range of interest---
    ff = acqFunc.GetFirstFrame()
    lf = acqFunc.GetLastFrame()

    # motion
    if side is None:
        side = detectSide(acqFunc,"LANK","RANK")
        logging.info("Detected motion side : %s" %(side) )

    start,end = btkTools.getStartEndEvents(acqFunc,side)

    if start is not None:
        logging.info("Start event detected")
        initFrame=start
    else:
        initFrame = beginFrame if beginFrame is not None else ff

    if end is not None:
        logging.info("End event detected")
        endFrame=end
    else:
        endFrame = endFrame if endFrame is not None else lf

    iff=initFrame-ff
    ilf=endFrame-ff



    if model.version in  ["CGM1.0","CGM1.1","CGM2.1","CGM2.2"]:
        validFrames,vff,vlf = btkTools.findValidFrames(acqFunc,cgm.CGM1.LOWERLIMB_TRACKING_MARKERS)

    # --------------------------RESET OF THE STATIC File---------
    # load btkAcq from static file
    staticFilename = model.m_staticFilename
    acqStatic = btkTools.smartReader((DATA_PATH+staticFilename))
    btkTools.checkMultipleSubject(acqStatic)
    acqStatic =  btkTools.applyTranslators(acqStatic,translators)

    # initial calibration ( i.e calibration from Calibration operation)
    leftFlatFoot = model.m_properties["CalibrationParameters"]["leftFlatFoot"]
    rightFlatFoot = model.m_properties["CalibrationParameters"]["rightFlatFoot"]
    headFlat = model.m_properties["CalibrationParameters"]["headFlat"]

    markerDiameter = model.m_properties["CalibrationParameters"]["markerDiameter"]

    if side == "Left":
        # remove other functional calibration
        model.mp_computed["LeftKneeFuncCalibrationOffset"] = 0

    if side == "Right":
        # remove other functional calibration
        model.mp_computed["RightKneeFuncCalibrationOffset"] = 0

    # no rotation on both thigh - re init anatonical frame
    scp=modelFilters.StaticCalibrationProcedure(model)
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                           headFlat = headFlat,
                           markerDiameter=markerDiameter).compute()


    if model.version in  ["CGM1.0","CGM1.1","CGM2.1","CGM2.2"]:

        modMotion=modelFilters.ModelMotionFilter(scp,acqFunc,model,enums.motionMethod.Determinist)
        modMotion.compute()

    elif model.version in  ["CGM2.3","CGM2.4","CGM2.5"]:
        if side == "Left":
            thigh_markers = model.getSegment("Left Thigh").m_tracking_markers
            shank_markers = model.getSegment("Left Shank").m_tracking_markers

        elif side == "Right":
            thigh_markers = model.getSegment("Right Thigh").m_tracking_markers
            shank_markers = model.getSegment("Right Shank").m_tracking_markers

        validFrames,vff,vlf = btkTools.findValidFrames(acqFunc,thigh_markers+shank_markers)

        proximalSegmentLabel=(side+" Thigh")
        distalSegmentLabel=(side+" Shank")

        # Motion
        modMotion=modelFilters.ModelMotionFilter(scp,acqFunc,model,enums.motionMethod.Sodervisk)
        modMotion.segmentalCompute([proximalSegmentLabel,distalSegmentLabel])


    # calibration decorators
    modelDecorator.KneeCalibrationDecorator(model).calibrate2dof(side,
                                                        indexFirstFrame = iff,
                                                        indexLastFrame = ilf,
                                                        jointRange =  jointRange)



    # --------------------------FINAL CALIBRATION OF THE STATIC File---------

    # ----  Calibration
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                       leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                       headFlat = headFlat,
                       markerDiameter=markerDiameter).compute()

    return model,acqFunc,side


def sara(model,
    DATA_PATH,reconstructFilenameLabelled,translators,
    side,beginFrame,endFrame,**kwargs):

    # --- btk acquisition ----
    if "forceBtkAcq" in kwargs.keys():
        acqFunc = kwargs["forceBtkAcq"]
    else:
        acqFunc = btkTools.smartReader((DATA_PATH + reconstructFilenameLabelled))

    btkTools.checkMultipleSubject(acqFunc)
    acqFunc =  btkTools.applyTranslators(acqFunc,translators)

    # filtering
    # -----------------------
    if "fc_lowPass_marker" in kwargs.keys() :
        fc = kwargs["fc_lowPass_marker"]
        order = 4
        if "order_lowPass_marker" in kwargs.keys(): order = kwargs["order_lowPass_marker"]
        signal_processing.markerFiltering(acqFunc,order=order, fc =fc)

    if "fc_lowPass_forcePlate" in kwargs.keys() :
        fc = kwargs["fc_lowPass_forcePlate"]
        order = 4
        if "order_lowPass_forcePlate" in kwargs.keys(): order = kwargs["order_lowPass_forcePlate"]
        signal_processing.forcePlateFiltering(acqFunc,order=order, fc =fc)

    #---get frame range of interest---
    ff = acqFunc.GetFirstFrame()
    lf = acqFunc.GetLastFrame()

    #---motion side of the lower limb---
    if side is None:
        side = detectSide(acqFunc,"LANK","RANK")
        logging.info("Detected motion side : %s" %(side) )

    start,end = btkTools.getStartEndEvents(acqFunc,side)

    if start is not None:
        logging.info("Start event detected")
        initFrame=start
    else:
        initFrame = beginFrame if beginFrame is not None else ff

    if end is not None:
        logging.info("End event detected")
        endFrame=end
    else:
        endFrame = endFrame if endFrame is not None else lf

    iff=initFrame-ff
    ilf=endFrame-ff




    # --------------------------RESET OF THE STATIC File---------

    # load btkAcq from static file
    staticFilename = model.m_staticFilename
    acqStatic = btkTools.smartReader((DATA_PATH+staticFilename))
    btkTools.checkMultipleSubject(acqStatic)
    acqStatic =  btkTools.applyTranslators(acqStatic,translators)

    # initial calibration ( i.e calibration from Calibration operation)
    leftFlatFoot = model.m_properties["CalibrationParameters"]["leftFlatFoot"]
    rightFlatFoot = model.m_properties["CalibrationParameters"]["rightFlatFoot"]
    markerDiameter = model.m_properties["CalibrationParameters"]["markerDiameter"]
    headFlat = model.m_properties["CalibrationParameters"]["headFlat"]

    if side == "Left":
        model.mp_computed["LeftKneeFuncCalibrationOffset"] = 0
    if side == "Right":
        model.mp_computed["RightKneeFuncCalibrationOffset"] = 0



    # initial calibration ( zero previous KneeFunc offset on considered side )
    scp=modelFilters.StaticCalibrationProcedure(model)
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                           headFlat = headFlat,
                           markerDiameter=markerDiameter).compute()


    if model.version in  ["CGM2.3","CGM2.4","CGM2.5"]:
        if side == "Left":
            thigh_markers = model.getSegment("Left Thigh").m_tracking_markers
            shank_markers = model.getSegment("Left Shank").m_tracking_markers

        elif side == "Right":
            thigh_markers = model.getSegment("Right Thigh").m_tracking_markers
            shank_markers = model.getSegment("Right Shank").m_tracking_markers

        validFrames,vff,vlf = btkTools.findValidFrames(acqFunc,thigh_markers+shank_markers)

        proximalSegmentLabel=(side+" Thigh")
        distalSegmentLabel=(side+" Shank")

        # segment Motion
        modMotion=modelFilters.ModelMotionFilter(scp,acqFunc,model,enums.motionMethod.Sodervisk)
        modMotion.segmentalCompute([proximalSegmentLabel,distalSegmentLabel])

        # decorator

        modelDecorator.KneeCalibrationDecorator(model).sara(side,
                                                            indexFirstFrame = iff,
                                                            indexLastFrame = ilf )

        # --------------------------FINAL CALIBRATION OF THE STATIC File---------

        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                           headFlat = headFlat,
                           markerDiameter=markerDiameter).compute()

    return model,acqFunc,side
