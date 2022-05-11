# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Functions/CGM
#APIDOC["Draft"]=False
#--end--

import pyCGM2; LOGGER = pyCGM2.LOGGER

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2 import enums

from pyCGM2.Model import modelFilters
from pyCGM2.Model import bodySegmentParameters
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model.CGM2 import decorators
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Processing.ProgressionFrame import progressionFrameFilters
from pyCGM2.Processing.ProgressionFrame import progressionFrameProcedures
from pyCGM2.Signal import signal_processing
from pyCGM2.Anomaly import anomalyFilters
from pyCGM2.Anomaly import anomalyDetectionProcedures
from pyCGM2.Inspector import inspectorFilters
from pyCGM2.Inspector import inspectorProcedures

def calibrate(DATA_PATH,calibrateFilenameLabelled,translators,
              required_mp,optional_mp,
              leftFlatFoot,rightFlatFoot,headFlat,markerDiameter,
              pointSuffix,**kwargs):

    """
    CGM1.1 calibration.

    Args:
        DATA_PATH (str): folder path.
        calibrateFilenameLabelled (str): filename of your static file.
        translators (dict): marker translators.
        required_mp (dict):  required anthropometric parameter.
        optional_mp (dict): optional anthropometric parameter..
        leftFlatFoot (bool): flat foot option.
        rightFlatFoot (bool): flat foot option.
        headFlat (bool): flat head option.
        markerDiameter (float): marker diameter
        pointSuffix (str): suffix to add to ouputs

    Keyword Arguments:
        anomalyException (bool): raise exception if anomaly detected
        forceBtkAcq (btk.Acquisition): use a btkAcquisition instance instead of building the btkAcquisition from the static filename
        displayCoordinateSystem (bool): return virtual markers for visualisation of the anatomical refentials
        noKinematicsCalculation (bool) : disable computation of joint kinematics

    Returns:
        model (pyCGM2.Model): the calibrated Model
        acqStatic (Btk.Acquisition): static btkAcquisition instance with model ouputs
        detectAnomaly  (bool): presence of anomaly

    """

    detectAnomaly = False

    if "anomalyException" in kwargs.keys():
        anomalyException = kwargs["anomalyException"]
    else:
        anomalyException=False
    # --------------------------ACQUISITION ------------------------------------

    # ---btk acquisition---
    if "forceBtkAcq" in kwargs.keys():
        acqStatic = kwargs["forceBtkAcq"]
    else:
        acqStatic = btkTools.smartReader((DATA_PATH+calibrateFilenameLabelled))

    btkTools.checkMultipleSubject(acqStatic)
    if btkTools.isPointExist(acqStatic,"SACR"):
        translators["LPSI"] = "SACR"
        translators["RPSI"] = "SACR"
        LOGGER.logger.info("[pyCGM2] Sacrum marker detected")

    acqStatic =  btkTools.applyTranslators(acqStatic,translators)

    trackingMarkers = cgm.CGM1.LOWERLIMB_TRACKING_MARKERS + cgm.CGM1.THORAX_TRACKING_MARKERS+ cgm.CGM1.UPPERLIMB_TRACKING_MARKERS
    actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqStatic, trackingMarkers)

    vff = acqStatic.GetFirstFrame()
    vlf = acqStatic.GetLastFrame()
    # vff,vlf = btkTools.getFrameBoundaries(acqStatic,actual_trackingMarkers)
    flag = btkTools.getValidFrames(acqStatic,actual_trackingMarkers,frameBounds=[vff,vlf])

    gapFlag = btkTools.checkGap(acqStatic,actual_trackingMarkers,frameBounds=[vff,vlf])
    if gapFlag:
        raise Exception("[pyCGM2] Calibration aborted. Gap find during interval [%i-%i]. Crop your c3d " %(vff,vlf))

    # --------------------ANOMALY------------------------------
    # --Check MP
    adap = anomalyDetectionProcedures.AnthropoDataAnomalyProcedure( required_mp)
    adf = anomalyFilters.AnomalyDetectionFilter(None,None,adap)
    mp_anomaly = adf.run()
    if mp_anomaly["ErrorState"]: detectAnomaly = True

    # --marker presence
    markersets = [cgm.CGM1.LOWERLIMB_TRACKING_MARKERS, cgm.CGM1.THORAX_TRACKING_MARKERS, cgm.CGM1.UPPERLIMB_TRACKING_MARKERS]
    for markerset in markersets:
        ipdp = inspectorProcedures.MarkerPresenceDetectionProcedure( markerset)
        idf = inspectorFilters.InspectorFilter(acqStatic,calibrateFilenameLabelled,ipdp)
        inspector = idf.run()

        # # --marker outliers
        if inspector["In"] !=[]:
            madp = anomalyDetectionProcedures.MarkerAnomalyDetectionRollingProcedure(inspector["In"], plot=False, window=4,threshold = 3)
            adf = anomalyFilters.AnomalyDetectionFilter(acqStatic,calibrateFilenameLabelled,madp)
            anomaly = adf.run()
            anomalyIndexes = anomaly["Output"]
            if anomaly["ErrorState"]: detectAnomaly = True


    if detectAnomaly and anomalyException:
        raise Exception ("Anomalies has been detected - Check Warning message of the log file")


    # --------------------MODELLING------------------------------

    # ---detectedCalibrationMethods----
    dcm= cgm.CGM.detectCalibrationMethods(acqStatic)

    # ---definition---
    model=cgm.CGM1()
    model.setVersion("CGM1.1")
    model.configure(detectedCalibrationMethods=dcm)
    model.addAnthropoInputParameters(required_mp,optional=optional_mp)

    if dcm["Left Knee"] == enums.JointCalibrationMethod.KAD: actual_trackingMarkers.append("LKNE")
    if dcm["Right Knee"] == enums.JointCalibrationMethod.KAD: actual_trackingMarkers.append("RKNE")
    model.setStaticTrackingMarkers(actual_trackingMarkers)

    # --store calibration parameters--
    model.setStaticFilename(calibrateFilenameLabelled)
    model.setCalibrationProperty("leftFlatFoot",leftFlatFoot)
    model.setCalibrationProperty("rightFlatFoot",rightFlatFoot)
    model.setCalibrationProperty("headFlat",headFlat)
    model.setCalibrationProperty("markerDiameter",markerDiameter)



    # --------------------------STATIC CALBRATION--------------------------
    scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

    # ---initial calibration filter----
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                        leftFlatFoot = leftFlatFoot,
                                        rightFlatFoot = rightFlatFoot,
                                        headFlat= headFlat,
                                        markerDiameter = markerDiameter,
                                        ).compute()
    # ---- Decorators -----
    decorators.applyBasicDecorators(dcm, model,acqStatic,optional_mp,markerDiameter)

    # ----Final Calibration filter if model previously decorated -----
    if model.decoratedModel:
        # initial static filter
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                           headFlat= headFlat,
                           markerDiameter=markerDiameter).compute()


    # ----------------------CGM MODELLING----------------------------------
    # ----motion filter----
    # notice : viconCGM1compatible option duplicate error on Construction of the foot coordinate system

    modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,enums.motionMethod.Determinist,
                                              markerDiameter=markerDiameter)
    modMotion.compute()

    # ----progression Frame----
    progressionFlag = False
    if btkTools.isPointsExist(acqStatic, ['LASI', 'RASI', 'RPSI', 'LPSI'],ignorePhantom=False):
        LOGGER.logger.info("[pyCGM2] - progression axis detected from Pelvic markers ")
        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure()
        pff = progressionFrameFilters.ProgressionFrameFilter(acqStatic,pfp)
        pff.compute()
        progressionAxis = pff.outputs["progressionAxis"]
        globalFrame = pff.outputs["globalFrame"]
        forwardProgression = pff.outputs["forwardProgression"]
        progressionFlag = True
    elif btkTools.isPointsExist(acqStatic, ['C7', 'T10', 'CLAV', 'STRN'],ignorePhantom=False) and not progressionFlag:
        LOGGER.logger.info("[pyCGM2] - progression axis detected from Thoracic markers ")
        pfp = progressionFrameProcedures.ThoraxProgressionFrameProcedure()
        pff = progressionFrameFilters.ProgressionFrameFilter(acqStatic,pfp)
        pff.compute()
        progressionAxis = pff.outputs["progressionAxis"]
        globalFrame = pff.outputs["globalFrame"]
        forwardProgression = pff.outputs["forwardProgression"]

    else:
        globalFrame = "XYZ"
        progressionAxis = "X"
        forwardProgression = True
        LOGGER.logger.error("[pyCGM2] - impossible to detect progression axis - neither pelvic nor thoracic markers are present. Progression set to +X by default ")


    if "displayCoordinateSystem" in kwargs.keys() and kwargs["displayCoordinateSystem"]:
        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqStatic)
        csdf.setStatic(False)
        csdf.display()

    if "noKinematicsCalculation" in kwargs.keys() and kwargs["noKinematicsCalculation"]:
        LOGGER.logger.warning("[pyCGM2] No Kinematic calculation done for the static file")
        return model, acqStatic
    else:
        #---- Joint kinematics----
        # relative angles
        modelFilters.ModelJCSFilter(model,acqStatic).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

        modelFilters.ModelAbsoluteAnglesFilter(model,acqStatic,
                                               segmentLabels=["Left Foot","Right Foot","Pelvis","Thorax","Head"],
                                                angleLabels=["LFootProgress", "RFootProgress","Pelvis","Thorax", "Head"],
                                                eulerSequences=["TOR","TOR", "ROT","YXZ","TOR"],
                                                globalFrameOrientation = globalFrame,
                                                forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)

        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()

        modelFilters.CentreOfMassFilter(model,acqStatic).compute(pointLabelSuffix=pointSuffix)

        btkTools.cleanAcq(acqStatic)
        if detectAnomaly and not anomalyException:
            LOGGER.logger.error("Anomalies has been detected - Check Warning messages of the log file")

        return model, acqStatic,detectAnomaly



def fitting(model,DATA_PATH, reconstructFilenameLabelled,
    translators,
    markerDiameter,
    pointSuffix,
    mfpa,
    momentProjection,**kwargs):

    """
    CGM1.1 Fitting.

    Args:
        DATA_PATH (str): folder path.
        reconstructFilenameLabelled (str): filename of your gait trial.
        translators (dict): marker translators.
        markerDiameter (float): marker diameter
        pointSuffix (str): suffix to add to ouputs
        mfpa (str): force plate assignment
        momentProjection (str) : referential for projection of joint moment

    Keyword Arguments:
        anomalyException (bool): raise exception if anomaly detected
        forceBtkAcq (btk.Acquisition): use a btkAcquisition instance instead of building the btkAcquisition from the static filename
        frameInit (int):  frame index.
        frameEnd (int):  frame index
        fc_lowPass_marker (float): low-pass fiter cutoff frequency applied on marker trajectories
        order_lowPass_marker (int): order of the low-pass filter applied on marker trajectories
        fc_lowPass_forcePlate (float): low-pass fiter cutoff frequency applied on force plate measurements
        order_lowPass_forcePlate: order fiter cutoff frequency applied on force plate measurements
        displayCoordinateSystem (bool): return virtual markers for visualisation of the anatomical refentials
        noKinematicsCalculation (bool) : disable computation of joint kinematics

    Returns:
        acqGait (Btk.Acquisition): static btkAcquisition instance with model ouputs
        detectAnomaly  (bool): presence of anomaly

    """

    detectAnomaly = False

    if "anomalyException" in kwargs.keys():
        anomalyException = kwargs["anomalyException"]
    else:
        anomalyException=False


    # --------------------------ACQUISITION ------------------------------------

    # --- btk acquisition ----
    if "forceBtkAcq" in kwargs.keys():
        acqGait = kwargs["forceBtkAcq"]
    else:
        acqGait = btkTools.smartReader((DATA_PATH + reconstructFilenameLabelled))

    btkTools.checkMultipleSubject(acqGait)
    if btkTools.isPointExist(acqGait,"SACR"):
        translators["LPSI"] = "SACR"
        translators["RPSI"] = "SACR"
        LOGGER.logger.info("[pyCGM2] Sacrum marker detected")

    acqGait =  btkTools.applyTranslators(acqGait,translators)

    trackingMarkers = cgm.CGM1.LOWERLIMB_TRACKING_MARKERS + cgm.CGM1.THORAX_TRACKING_MARKERS+ cgm.CGM1.UPPERLIMB_TRACKING_MARKERS
    actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqGait, trackingMarkers)
    vff,vlf = btkTools.getFrameBoundaries(acqGait,actual_trackingMarkers)
    if "frameInit" in kwargs.keys() and kwargs["frameInit"] is not None:
        vff = kwargs["frameInit"]
        LOGGER.logger.info("[pyCGM2]  first frame forced to frame [%s]"%(vff))
    if "frameEnd" in kwargs.keys() and kwargs["frameEnd"] is not None:
        vlf = kwargs["frameEnd"]
        LOGGER.logger.info("[pyCGM2]  end frame forced to frame [%s]"%(vlf))
    flag = btkTools.getValidFrames(acqGait,actual_trackingMarkers,frameBounds=[vff,vlf])

    # --------------------ANOMALY------------------------------
    for marker in actual_trackingMarkers:
        if marker not in model.getStaticTrackingMarkers():
            LOGGER.logger.warning("[pyCGM2-Anomaly]  marker [%s] - not used during static calibration - wrong kinematic for the segment attached to this marker. "%(marker))

    # --marker presence
    markersets = [cgm.CGM1.LOWERLIMB_TRACKING_MARKERS, cgm.CGM1.THORAX_TRACKING_MARKERS, cgm.CGM1.UPPERLIMB_TRACKING_MARKERS]
    for markerset in markersets:
        ipdp = inspectorProcedures.MarkerPresenceDetectionProcedure( markerset)
        idf = inspectorFilters.InspectorFilter(acqGait,reconstructFilenameLabelled,ipdp)
        inspector = idf.run()

        # --marker outliers
        if inspector["In"] !=[]:
            madp = anomalyDetectionProcedures.MarkerAnomalyDetectionRollingProcedure( inspector["In"], plot=False, window=5,threshold = 3)
            adf = anomalyFilters.AnomalyDetectionFilter(acqGait,reconstructFilenameLabelled,madp, frameRange=[vff,vlf])
            anomaly = adf.run()
            anomalyIndexes = anomaly["Output"]
            if anomaly["ErrorState"]: detectAnomaly = True


    if btkTools.checkForcePlateExist(acqGait):
        afpp = anomalyDetectionProcedures.ForcePlateAnomalyProcedure()
        adf = anomalyFilters.AnomalyDetectionFilter(acqGait,reconstructFilenameLabelled,afpp, frameRange=[vff,vlf])
        anomaly = adf.run()
        if anomaly["ErrorState"]: detectAnomaly = True

    if detectAnomaly and anomalyException:
        raise Exception ("Anomalies has been detected - Check Warning message of the log file")

   # --------------------MODELLING------------------------------


    # filtering
    # -----------------------
    if "fc_lowPass_marker" in kwargs.keys() and kwargs["fc_lowPass_marker"]!=0 :
        fc = kwargs["fc_lowPass_marker"]
        order = 4
        if "order_lowPass_marker" in kwargs.keys():
            order = kwargs["order_lowPass_marker"]
        signal_processing.markerFiltering(acqGait,trackingMarkers,order=order, fc =fc)

    if "fc_lowPass_forcePlate" in kwargs.keys() and kwargs["fc_lowPass_forcePlate"]!=0 :
        fc = kwargs["fc_lowPass_forcePlate"]
        order = 4
        if "order_lowPass_forcePlate" in kwargs.keys():
            order = kwargs["order_lowPass_forcePlate"]
        signal_processing.forcePlateFiltering(acqGait,order=order, fc =fc)


    scp=modelFilters.StaticCalibrationProcedure(model) # procedure

    # ---Motion filter----
    modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist,
                                              markerDiameter=markerDiameter)

    modMotion.compute()

    progressionFlag = False
    if btkTools.isPointExist(acqGait, 'LHEE',ignorePhantom=False) or btkTools.isPointExist(acqGait, 'RHEE',ignorePhantom=False):

        pfp = progressionFrameProcedures.PointProgressionFrameProcedure(marker="LHEE") \
            if btkTools.isPointExist(acqGait, 'LHEE',ignorePhantom=False) \
            else  progressionFrameProcedures.PointProgressionFrameProcedure(marker="RHEE")

        pff = progressionFrameFilters.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        progressionAxis = pff.outputs["progressionAxis"]
        globalFrame = pff.outputs["globalFrame"]
        forwardProgression = pff.outputs["forwardProgression"]
        progressionFlag = True

    elif btkTools.isPointsExist(acqGait, ['LASI', 'RASI', 'RPSI', 'LPSI'],ignorePhantom=False) and not progressionFlag:
        LOGGER.logger.info("[pyCGM2] - progression axis detected from Pelvic markers ")
        pfp = progressionFrameProcedures.PelvisProgressionFrameProcedure()
        pff = progressionFrameFilters.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        forwardProgression = pff.outputs["forwardProgression"]

        progressionFlag = True
    elif btkTools.isPointsExist(acqGait, ['C7', 'T10', 'CLAV', 'STRN'],ignorePhantom=False) and not progressionFlag:
        LOGGER.logger.info("[pyCGM2] - progression axis detected from Thoracic markers ")
        pfp = progressionFrameProcedures.ThoraxProgressionFrameProcedure()
        pff = progressionFrameFilters.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        progressionAxis = pff.outputs["progressionAxis"]
        globalFrame = pff.outputs["globalFrame"]
        forwardProgression = pff.outputs["forwardProgression"]

    else:
        globalFrame = "XYZ"
        progressionAxis = "X"
        forwardProgression = True
        LOGGER.logger.error("[pyCGM2] - impossible to detect progression axis - neither pelvic nor thoracic markers are present. Progression set to +X by default ")

    if "displayCoordinateSystem" in kwargs.keys() and kwargs["displayCoordinateSystem"]:
        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,acqGait)
        csdf.setStatic(False)
        csdf.display()

    if "NaimKneeCorrection" in kwargs.keys() and kwargs["NaimKneeCorrection"]:

        # Apply Naim 2019 method
        if type(kwargs["NaimKneeCorrection"]) is float:
            nmacp = modelFilters.Naim2019ThighMisaligmentCorrectionProcedure(model,"Both",threshold=(kwargs["NaimKneeCorrection"]))
        else:
            nmacp = modelFilters.Naim2019ThighMisaligmentCorrectionProcedure(model,"Both")
        mmcf = modelFilters.ModelMotionCorrectionFilter(nmacp)
        mmcf.correct()

        # btkTools.smartAppendPoint(acqGait,"LNaim",mmcf.m_procedure.m_virtual["Left"])
        # btkTools.smartAppendPoint(acqGait,"RNaim",mmcf.m_procedure.m_virtual["Right"])


    #---- Joint kinematics----
    # relative angles
    modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix=pointSuffix)


    modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
                                           segmentLabels=["Left Foot","Right Foot","Pelvis","Thorax","Head"],
                                            angleLabels=["LFootProgress", "RFootProgress","Pelvis","Thorax", "Head"],
                                            eulerSequences=["TOR","TOR", "ROT","YXZ","TOR"],
                                            globalFrameOrientation = globalFrame,
                                            forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)

    #---- Body segment parameters----
    bspModel = bodySegmentParameters.Bsp(model)
    bspModel.compute()


    modelFilters.CentreOfMassFilter(model,acqGait).compute(pointLabelSuffix=pointSuffix)

    # Inverse dynamics
    if btkTools.checkForcePlateExist(acqGait):
        if model.m_bodypart != enums.BodyPart.UpperLimb:
            # --- force plate handling----
            # find foot  in contact
            mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait,mfpa=mfpa)
            forceplates.addForcePlateGeneralEvents(acqGait,mappedForcePlate)
            LOGGER.logger.info("Manual Force plate assignment : %s" %mappedForcePlate)

            # assembly foot and force plate
            modelFilters.ForcePlateAssemblyFilter(model,acqGait,mappedForcePlate,
                                     leftSegmentLabel="Left Foot",
                                     rightSegmentLabel="Right Foot").compute(pointLabelSuffix=pointSuffix)

            #---- Joint kinetics----
            idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
            modelFilters.InverseDynamicFilter(model,
                                 acqGait,
                                 procedure = idp,
                                 projection = momentProjection,
                                 globalFrameOrientation = globalFrame,
                                 forwardProgression = forwardProgression
                                 ).compute(pointLabelSuffix=pointSuffix)


            #---- Joint energetics----
            modelFilters.JointPowerFilter(model,acqGait).compute(pointLabelSuffix=pointSuffix)

    btkTools.cleanAcq(acqGait)
    btkTools.applyOnValidFrames(acqGait,flag)
    #---- zero unvalid frames ---
    # btkTools.applyValidFramesOnOutput(acqGait,validFrames)
    if detectAnomaly and not anomalyException:
        LOGGER.logger.error("Anomalies has been detected - Check Warning messages of the log file")

    return acqGait,detectAnomaly
