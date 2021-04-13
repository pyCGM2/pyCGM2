# -*- coding: utf-8 -*-
#import ipdb
import pyCGM2; LOGGER = pyCGM2.LOGGER

# pyCGM2 settings
import pyCGM2

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2 import enums

from pyCGM2.Model import modelFilters, bodySegmentParameters
from pyCGM2.Model.CGM2 import cgm,cgm2
from pyCGM2.Model.CGM2 import decorators
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Model.Opensim import opensimFilters
from pyCGM2.Processing import progressionFrame
from pyCGM2.Signal import signal_processing
from pyCGM2.Anomaly import AnomalyFilter, AnomalyDetectionProcedure
from pyCGM2.Inspector import InspectorFilter, InspectorProcedure

def calibrate(DATA_PATH,calibrateFilenameLabelled,translators,weights,
              required_mp,optional_mp,
              ik_flag,leftFlatFoot,rightFlatFoot,headFlat,
              markerDiameter,hjcMethod,
              pointSuffix,**kwargs):
    """
    Calibration of the CGM2.5

    :param DATA_PATH [str]: path to your data
    :param calibrateFilenameLabelled [str]: c3d file
    :param translators [dict]:  translators to apply
    :param required_mp [dict]: required anthropometric data
    :param optional_mp [dict]: optional anthropometric data (ex: LThighOffset,...)
    :param ik_flag [bool]: enable the inverse kinematic solver
    :param leftFlatFoot [bool]: enable of the flat foot option for the left foot
    :param rightFlatFoot [bool]: enable of the flat foot option for the right foot
    :param headFlat [bool]: enable of the head flat  option
    :param markerDiameter [double]: marker diameter (mm)
    :param hjcMethod [str or list of 3 float]: method for locating the hip joint centre
    :param pointSuffix [str]: suffix to add to model outputs

    """
    detectAnomaly = False


    if "anomalyException" in kwargs.keys():
        anomalyException = kwargs["anomalyException"]
    else:
        anomalyException=False

    if "Fitting" in weights.keys():
        weights  = weights["Fitting"]["Weight"]

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

    trackingMarkers = cgm2.CGM2_5.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_5.THORAX_TRACKING_MARKERS+ cgm2.CGM2_5.UPPERLIMB_TRACKING_MARKERS
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
    adap = AnomalyDetectionProcedure.AnthropoDataAnomalyProcedure( required_mp)
    adf = AnomalyFilter.AnomalyDetectionFilter(None,None,adap)
    mp_anomaly = adf.run()
    if mp_anomaly["ErrorState"]: detectAnomaly = True

    # --marker presence
    markersets = [cgm2.CGM2_5.LOWERLIMB_TRACKING_MARKERS, cgm2.CGM2_5.THORAX_TRACKING_MARKERS, cgm2.CGM2_5.UPPERLIMB_TRACKING_MARKERS]
    for markerset in markersets:
        ipdp = InspectorProcedure.MarkerPresenceDetectionProcedure( markerset)
        idf = InspectorFilter.InspectorFilter(acqStatic,calibrateFilenameLabelled,ipdp)
        inspector = idf.run()

        # # --marker outliers
        if inspector["In"] !=[]:
            madp = AnomalyDetectionProcedure.MarkerAnomalyDetectionRollingProcedure(inspector["In"], plot=False, window=4,threshold = 3)
            adf = AnomalyFilter.AnomalyDetectionFilter(acqStatic,calibrateFilenameLabelled,madp)
            anomaly = adf.run()
            anomalyIndexes = anomaly["Output"]
            if anomaly["ErrorState"]: detectAnomaly = True


    if detectAnomaly and anomalyException:
        raise Exception ("Anomalies has been detected - Check Warning message of the log file")


    # --------------------MODELLING------------------------------

    # ---check marker set used----
    dcm = cgm.CGM.detectCalibrationMethods(acqStatic)

    # --------------------------MODEL--------------------------------------
    # ---definition---
    model=cgm2.CGM2_5()
    model.configure(acq=acqStatic,detectedCalibrationMethods=dcm)
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
    # use if all optional mp are zero
    modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                        leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                                        headFlat= headFlat,
                                        markerDiameter=markerDiameter,
                                        ).compute()

    # ---- Decorators -----
    decorators.applyBasicDecorators(dcm, model,acqStatic,optional_mp,markerDiameter)
    decorators.applyHJCDecorators(model,hjcMethod)

    # ----Final Calibration filter if model previously decorated -----
    if model.decoratedModel:
        # initial static filter
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                           headFlat= headFlat,
                           markerDiameter=markerDiameter).compute()


    # ----------------------CGM MODELLING----------------------------------
    # ----motion filter----
    modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,enums.motionMethod.Sodervisk,
                                              markerDiameter=markerDiameter)

    modMotion.compute()


    # ----progression Frame----
    progressionFlag = False
    if btkTools.isPointsExist(acqStatic, ['LASI', 'RASI', 'RPSI', 'LPSI'],ignorePhantom=False):
        LOGGER.logger.info("[pyCGM2] - progression axis detected from Pelvic markers ")
        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqStatic,pfp)
        pff.compute()
        progressionAxis = pff.outputs["progressionAxis"]
        globalFrame = pff.outputs["globalFrame"]
        forwardProgression = pff.outputs["forwardProgression"]
        progressionFlag = True
    elif btkTools.isPointsExist(acqStatic, ['C7', 'T10', 'CLAV', 'STRN'],ignorePhantom=False) and not progressionFlag:
        LOGGER.logger.info("[pyCGM2] - progression axis detected from Thoracic markers ")
        pfp = progressionFrame.ThoraxProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqStatic,pfp)
        pff.compute()
        progressionAxis = pff.outputs["progressionAxis"]
        globalFrame = pff.outputs["globalFrame"]
        forwardProgression = pff.outputs["forwardProgression"]

    else:
        globalFrame = "XYZ"
        progressionAxis = "X"
        forwardProgression = True
        LOGGER.logger.error("[pyCGM2] - impossible to detect progression axis - neither pelvic nor thoracic markers are present. Progression set to +X by default ")


    # ----manage IK Targets----
    ikTargets = list()
    for target in weights.keys():
        if target not in actual_trackingMarkers:
            weights[target] = 0
            LOGGER.logger.warning("[pyCGM2] - the IK targeted marker [%s] is not labelled in the acquisition [%s]"%(target,calibrateFilenameLabelled))
        else:
            ikTargets.append(target)
    model.setStaticIkTargets(ikTargets)

    if "noKinematicsCalculation" in kwargs.keys() and kwargs["noKinematicsCalculation"]:
        LOGGER.logger.warning("[pyCGM2] No Kinematic calculation done for the static file")
        return model, acqStatic,detectAnomaly
    else:

        if ik_flag:
            #                        ---OPENSIM IK---

            # --- opensim calibration Filter ---
            osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile
            markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-markerset.xml" # markerset
            cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure

            oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                    model,
                                                    cgmCalibrationprocedure,
                                                    DATA_PATH )
            oscf.addMarkerSet(markersetFile)
            scalingOsim = oscf.build()


            # --- opensim Fitting Filter ---
            iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-ikSetUp_template.xml" # ik tool file

            cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model) # procedure
            cgmFittingProcedure.updateMarkerWeight("LASI",weights["LASI"])
            cgmFittingProcedure.updateMarkerWeight("RASI",weights["RASI"])
            cgmFittingProcedure.updateMarkerWeight("LPSI",weights["LPSI"])
            cgmFittingProcedure.updateMarkerWeight("RPSI",weights["RPSI"])
            cgmFittingProcedure.updateMarkerWeight("RTHI",weights["RTHI"])
            cgmFittingProcedure.updateMarkerWeight("RKNE",weights["RKNE"])
            cgmFittingProcedure.updateMarkerWeight("RTIB",weights["RTIB"])
            cgmFittingProcedure.updateMarkerWeight("RANK",weights["RANK"])
            cgmFittingProcedure.updateMarkerWeight("RHEE",weights["RHEE"])
            cgmFittingProcedure.updateMarkerWeight("RTOE",weights["RTOE"])

            cgmFittingProcedure.updateMarkerWeight("LTHI",weights["LTHI"])
            cgmFittingProcedure.updateMarkerWeight("LKNE",weights["LKNE"])
            cgmFittingProcedure.updateMarkerWeight("LTIB",weights["LTIB"])
            cgmFittingProcedure.updateMarkerWeight("LANK",weights["LANK"])
            cgmFittingProcedure.updateMarkerWeight("LHEE",weights["LHEE"])
            cgmFittingProcedure.updateMarkerWeight("LTOE",weights["LTOE"])

            cgmFittingProcedure.updateMarkerWeight("LTHAP",weights["LTHAP"])
            cgmFittingProcedure.updateMarkerWeight("LTHAD",weights["LTHAD"])
            cgmFittingProcedure.updateMarkerWeight("LTIAP",weights["LTIAP"])
            cgmFittingProcedure.updateMarkerWeight("LTIAD",weights["LTIAD"])
            cgmFittingProcedure.updateMarkerWeight("RTHAP",weights["RTHAP"])
            cgmFittingProcedure.updateMarkerWeight("RTHAD",weights["RTHAD"])
            cgmFittingProcedure.updateMarkerWeight("RTIAP",weights["RTIAP"])
            cgmFittingProcedure.updateMarkerWeight("RTIAD",weights["RTIAD"])

            cgmFittingProcedure.updateMarkerWeight("LSMH",weights["LSMH"])
            cgmFittingProcedure.updateMarkerWeight("LFMH",weights["LFMH"])
            cgmFittingProcedure.updateMarkerWeight("LVMH",weights["LVMH"])

            cgmFittingProcedure.updateMarkerWeight("RSMH",weights["RSMH"])
            cgmFittingProcedure.updateMarkerWeight("RFMH",weights["RFMH"])
            cgmFittingProcedure.updateMarkerWeight("RVMH",weights["RVMH"])

    #            cgmFittingProcedure.updateMarkerWeight("LTHL",weights["LTHL"])
    #            cgmFittingProcedure.updateMarkerWeight("LTHLD",weights["LTHLD"])
    #            cgmFittingProcedure.updateMarkerWeight("LPAT",weights["LPAT"])
    #            cgmFittingProcedure.updateMarkerWeight("LTIBL",weights["LTIBL"])
    #            cgmFittingProcedure.updateMarkerWeight("RTHL",weights["RTHL"])
    #            cgmFittingProcedure.updateMarkerWeight("RTHLD",weights["RTHLD"])
    #            cgmFittingProcedure.updateMarkerWeight("RPAT",weights["RPAT"])
    #            cgmFittingProcedure.updateMarkerWeight("RTIBL",weights["RTIBL"])


            osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                              scalingOsim,
                                                              cgmFittingProcedure,
                                                              DATA_PATH,
                                                              acqStatic,
                                                               accuracy = 1e-5)

            LOGGER.logger.info("-------INVERSE KINEMATICS IN PROGRESS----------")
            try:
                acqStaticIK = osrf.run(DATA_PATH + calibrateFilenameLabelled,
                             progressionAxis = progressionAxis ,
                             forwardProgression = forwardProgression)
                LOGGER.logger.info("[pyCGM2] - IK solver complete")
            except:
                LOGGER.logger.error("[pyCGM2] - IK solver fails")
                acqStaticIK = acqStatic
            LOGGER.logger.info("-----------------------------------------------")



        # eventual static acquisition to consider for joint kinematics
        finalAcqStatic = acqStaticIK if ik_flag else acqStatic

        # --- final pyCGM2 model motion Filter ---
        # use fitted markers
        modMotionFitted=modelFilters.ModelMotionFilter(scp,finalAcqStatic,model,enums.motionMethod.Sodervisk)
        modMotionFitted.compute()

        if "displayCoordinateSystem" in kwargs.keys() and kwargs["displayCoordinateSystem"]:
            csp = modelFilters.ModelCoordinateSystemProcedure(model)
            csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,finalAcqStatic)
            csdf.setStatic(False)
            csdf.display()

        #---- Joint kinematics----
        # relative angles
        modelFilters.ModelJCSFilter(model,finalAcqStatic).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

        modelFilters.ModelAbsoluteAnglesFilter(model,finalAcqStatic,
                                               segmentLabels=["Left Foot","Right Foot","Pelvis","Thorax","Head"],
                                                angleLabels=["LFootProgress", "RFootProgress","Pelvis","Thorax", "Head"],
                                                eulerSequences=["TOR","TOR", "ROT","YXZ","TOR"],
                                                globalFrameOrientation = globalFrame,
                                                forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)
        # BSP model
        bspModel = bodySegmentParameters.Bsp(model)
        bspModel.compute()


        modelFilters.CentreOfMassFilter(model,finalAcqStatic).compute(pointLabelSuffix=pointSuffix)


        btkTools.cleanAcq(finalAcqStatic)
        if detectAnomaly and not anomalyException:
            LOGGER.logger.error("Anomalies has been detected - Check Warning messages of the log file")


        return model, finalAcqStatic,detectAnomaly


def fitting(model,DATA_PATH, reconstructFilenameLabelled,
    translators,weights,
    ik_flag,markerDiameter,
    pointSuffix,
    mfpa,
    momentProjection,**kwargs):

    """
    Fitting of the CGM2.5

    :param model [str]: pyCGM2 model previously calibrated
    :param DATA_PATH [str]: path to your data
    :param reconstructFilenameLabelled [string list]: c3d files
    :param translators [dict]:  translators to apply
    :param ik_flag [bool]: enable the inverse kinematic solver
    :param mfpa [str]: manual force plate assignement
    :param markerDiameter [double]: marker diameter (mm)
    :param pointSuffix [str]: suffix to add to model outputs
    :param momentProjection [str]: Coordinate system in which joint moment is expressed
    """

    detectAnomaly = False


    if "anomalyException" in kwargs.keys():
        anomalyException = kwargs["anomalyException"]
    else:
        anomalyException=False

    if "forceFoot6DoF" in kwargs.keys() and kwargs["forceFoot6DoF"]:
        forceFoot6DoF_flag = True
    else:
        forceFoot6DoF_flag=False

    if "Fitting" in weights.keys():
        weights  = weights["Fitting"]["Weight"]

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
    trackingMarkers = cgm2.CGM2_5.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_5.THORAX_TRACKING_MARKERS+ cgm2.CGM2_5.UPPERLIMB_TRACKING_MARKERS
    actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqGait, trackingMarkers)
    vff,vlf = btkTools.getFrameBoundaries(acqGait,actual_trackingMarkers)
    flag = btkTools.getValidFrames(acqGait,actual_trackingMarkers,frameBounds=[vff,vlf])

    LOGGER.logger.info("[pyCGM2]  Computation from frame [%s] to frame [%s]"%(vff,vlf))
    # --------------------ANOMALY------------------------------
    for marker in actual_trackingMarkers:
        if marker not in model.getStaticTrackingMarkers():
            LOGGER.logger.warning("[pyCGM2-Anomaly]  marker [%s] - not used during static calibration - wrong kinematic for the segment attached to this marker. "%(marker))

    # --marker presence
    markersets = [cgm2.CGM2_5.LOWERLIMB_TRACKING_MARKERS, cgm2.CGM2_5.THORAX_TRACKING_MARKERS, cgm2.CGM2_5.UPPERLIMB_TRACKING_MARKERS]
    for markerset in markersets:
        ipdp = InspectorProcedure.MarkerPresenceDetectionProcedure( markerset)
        idf = InspectorFilter.InspectorFilter(acqGait,reconstructFilenameLabelled,ipdp)
        inspector = idf.run()

        # --marker outliers
        if inspector["In"] !=[]:
            madp = AnomalyDetectionProcedure.MarkerAnomalyDetectionRollingProcedure( inspector["In"], plot=False, window=5,threshold = 3)
            adf = AnomalyFilter.AnomalyDetectionFilter(acqGait,reconstructFilenameLabelled,madp, frameRange=[vff,vlf])
            anomaly = adf.run()
            anomalyIndexes = anomaly["Output"]
            if anomaly["ErrorState"]: detectAnomaly = True


    if btkTools.checkForcePlateExist(acqGait):
        afpp = AnomalyDetectionProcedure.ForcePlateAnomalyProcedure()
        adf = AnomalyFilter.AnomalyDetectionFilter(acqGait,reconstructFilenameLabelled,afpp, frameRange=[vff,vlf])
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

    # --- initial motion Filter ---
    scp=modelFilters.StaticCalibrationProcedure(model)
    modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Sodervisk)
    modMotion.compute()

    progressionFlag = False
    if btkTools.isPointExist(acqGait, 'LHEE',ignorePhantom=False) or btkTools.isPointExist(acqGait, 'RHEE',ignorePhantom=False):

        pfp = progressionFrame.PointProgressionFrameProcedure(marker="LHEE") \
            if btkTools.isPointExist(acqGait, 'LHEE',ignorePhantom=False) \
            else  progressionFrame.PointProgressionFrameProcedure(marker="RHEE")


        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        progressionAxis = pff.outputs["progressionAxis"]
        globalFrame = pff.outputs["globalFrame"]
        forwardProgression = pff.outputs["forwardProgression"]
        progressionFlag = True

    elif btkTools.isPointsExist(acqGait, ['LASI', 'RASI', 'RPSI', 'LPSI'],ignorePhantom=False) and not progressionFlag:
        LOGGER.logger.info("[pyCGM2] - progression axis detected from Pelvic markers ")
        pfp = progressionFrame.PelvisProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        globalFrame = pff.outputs["globalFrame"]
        forwardProgression = pff.outputs["forwardProgression"]

        progressionFlag = True
    elif btkTools.isPointsExist(acqGait, ['C7', 'T10', 'CLAV', 'STRN'],ignorePhantom=False) and not progressionFlag:
        LOGGER.logger.info("[pyCGM2] - progression axis detected from Thoracic markers ")
        pfp = progressionFrame.ThoraxProgressionFrameProcedure()
        pff = progressionFrame.ProgressionFrameFilter(acqGait,pfp)
        pff.compute()
        progressionAxis = pff.outputs["progressionAxis"]
        globalFrame = pff.outputs["globalFrame"]
        forwardProgression = pff.outputs["forwardProgression"]

    else:
        globalFrame = "XYZ"
        progressionAxis = "X"
        forwardProgression = True
        LOGGER.logger.error("[pyCGM2] - impossible to detect progression axis - neither pelvic nor thoracic markers are present. Progression set to +X by default ")


    for target in weights.keys():
        if target not in actual_trackingMarkers or target not in model.getStaticIkTargets():
            weights[target] = 0
            LOGGER.logger.warning("[pyCGM2] - the IK targeted marker [%s] is not labelled in the acquisition [%s]"%(target,reconstructFilenameLabelled))


    if ik_flag:
        #                        ---OPENSIM IK---

        # --- opensim calibration Filter ---
        osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile
        markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-markerset.xml" # markerset
        cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure

        oscf = opensimFilters.opensimCalibrationFilter(osimfile,
                                                model,
                                                cgmCalibrationprocedure,
                                                DATA_PATH )
        oscf.addMarkerSet(markersetFile)
        scalingOsim = oscf.build()


        # --- opensim Fitting Filter ---
        iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_4\\cgm2_4-ikSetUp_template.xml" # ik tl file

        cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model) # procedure
        cgmFittingProcedure.updateMarkerWeight("LASI",weights["LASI"])
        cgmFittingProcedure.updateMarkerWeight("RASI",weights["RASI"])
        cgmFittingProcedure.updateMarkerWeight("LPSI",weights["LPSI"])
        cgmFittingProcedure.updateMarkerWeight("RPSI",weights["RPSI"])
        cgmFittingProcedure.updateMarkerWeight("RTHI",weights["RTHI"])
        cgmFittingProcedure.updateMarkerWeight("RKNE",weights["RKNE"])
        cgmFittingProcedure.updateMarkerWeight("RTIB",weights["RTIB"])
        cgmFittingProcedure.updateMarkerWeight("RANK",weights["RANK"])
        cgmFittingProcedure.updateMarkerWeight("RHEE",weights["RHEE"])
        cgmFittingProcedure.updateMarkerWeight("RTOE",weights["RTOE"])

        cgmFittingProcedure.updateMarkerWeight("LTHI",weights["LTHI"])
        cgmFittingProcedure.updateMarkerWeight("LKNE",weights["LKNE"])
        cgmFittingProcedure.updateMarkerWeight("LTIB",weights["LTIB"])
        cgmFittingProcedure.updateMarkerWeight("LANK",weights["LANK"])
        cgmFittingProcedure.updateMarkerWeight("LHEE",weights["LHEE"])
        cgmFittingProcedure.updateMarkerWeight("LTOE",weights["LTOE"])


        cgmFittingProcedure.updateMarkerWeight("LTHAP",weights["LTHAP"])
        cgmFittingProcedure.updateMarkerWeight("LTHAD",weights["LTHAD"])
        cgmFittingProcedure.updateMarkerWeight("LTIAP",weights["LTIAP"])
        cgmFittingProcedure.updateMarkerWeight("LTIAD",weights["LTIAD"])
        cgmFittingProcedure.updateMarkerWeight("RTHAP",weights["RTHAP"])
        cgmFittingProcedure.updateMarkerWeight("RTHAD",weights["RTHAD"])
        cgmFittingProcedure.updateMarkerWeight("RTIAP",weights["RTIAP"])
        cgmFittingProcedure.updateMarkerWeight("RTIAD",weights["RTIAD"])

        cgmFittingProcedure.updateMarkerWeight("LSMH",weights["LSMH"])
        cgmFittingProcedure.updateMarkerWeight("LFMH",weights["LFMH"])
        cgmFittingProcedure.updateMarkerWeight("LVMH",weights["LVMH"])

        cgmFittingProcedure.updateMarkerWeight("RSMH",weights["RSMH"])
        cgmFittingProcedure.updateMarkerWeight("RFMH",weights["RFMH"])
        cgmFittingProcedure.updateMarkerWeight("RVMH",weights["RVMH"])


#       cgmFittingProcedure.updateMarkerWeight("LTHL",weights["LTHL"])
#       cgmFittingProcedure.updateMarkerWeight("LTHLD",weights["LTHLD"])
#       cgmFittingProcedure.updateMarkerWeight("LPAT",weights["LPAT"])
#       cgmFittingProcedure.updateMarkerWeight("LTIBL",weights["LTIBL"])
#       cgmFittingProcedure.updateMarkerWeight("RTHL",weights["RTHL"])
#       cgmFittingProcedure.updateMarkerWeight("RTHLD",weights["RTHLD"])
#       cgmFittingProcedure.updateMarkerWeight("RPAT",weights["RPAT"])
#       cgmFittingProcedure.updateMarkerWeight("RTIBL",weights["RTIBL"])


        osrf = opensimFilters.opensimFittingFilter(iksetupFile,
                                                          scalingOsim,
                                                          cgmFittingProcedure,
                                                          DATA_PATH,
                                                          acqGait )
        osrf.setTimeRange(acqGait,beginFrame = vff, lastFrame=vlf)
        if "ikAccuracy" in kwargs.keys():
            osrf.setAccuracy(kwargs["ikAccuracy"])

        LOGGER.logger.info("-------INVERSE KINEMATICS IN PROGRESS----------")
        try:
            acqIK = osrf.run(DATA_PATH + reconstructFilenameLabelled,
                            progressionAxis = progressionAxis ,
                            forwardProgression = forwardProgression)
            LOGGER.logger.info("[pyCGM2] - IK solver complete")
        except:
            LOGGER.logger.error("[pyCGM2] - IK solver fails")
            acqIK = acqGait
        LOGGER.logger.info("---------------------------------------------------")




    # eventual gait acquisition to consider for joint kinematics
    finalAcqGait = acqIK if ik_flag else acqGait

    if "displayCoordinateSystem" in kwargs.keys() and kwargs["displayCoordinateSystem"]:
        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,finalAcqGait)
        csdf.setStatic(False)
        csdf.display()

    # --- final pyCGM2 model motion Filter ---
    # use fitted markers
    modMotionFitted=modelFilters.ModelMotionFilter(scp,finalAcqGait,model,enums.motionMethod.Sodervisk ,
                                              markerDiameter=markerDiameter,
                                              forceFoot6DoF=forceFoot6DoF_flag)

    modMotionFitted.compute()


    #---- Joint kinematics----
    # relative angles
    modelFilters.ModelJCSFilter(model,finalAcqGait).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

    modelFilters.ModelAbsoluteAnglesFilter(model,finalAcqGait,
                                           segmentLabels=["Left Foot","Right Foot","Pelvis","Thorax","Head"],
                                            angleLabels=["LFootProgress", "RFootProgress","Pelvis","Thorax", "Head"],
                                            eulerSequences=["TOR","TOR", "ROT","YXZ","TOR"],
                                            globalFrameOrientation = globalFrame,
                                            forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)

    #---- Body segment parameters----
    bspModel = bodySegmentParameters.Bsp(model)
    bspModel.compute()

    modelFilters.CentreOfMassFilter(model,finalAcqGait).compute(pointLabelSuffix=pointSuffix)


    # Inverse dynamics
    if btkTools.checkForcePlateExist(acqGait):
        # --- force plate handling----
        # find foot  in contact
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(finalAcqGait,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(finalAcqGait,mappedForcePlate)
        LOGGER.logger.warning("Manual Force plate assignment : %s" %mappedForcePlate)

        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,finalAcqGait,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix=pointSuffix)

        #---- Joint kinetics----
        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
        modelFilters.InverseDynamicFilter(model,
                             finalAcqGait,
                             procedure = idp,
                             projection = momentProjection,
                             globalFrameOrientation = globalFrame,
                             forwardProgression = forwardProgression
                             ).compute(pointLabelSuffix=pointSuffix)


        #---- Joint energetics----
        modelFilters.JointPowerFilter(model,finalAcqGait).compute(pointLabelSuffix=pointSuffix)

    #---- zero unvalid frames ---
    btkTools.cleanAcq(finalAcqGait)
    btkTools.applyOnValidFrames(finalAcqGait,flag)

    if detectAnomaly and not anomalyException:
        LOGGER.logger.error("Anomalies has been detected - Check Warning messages of the log file")

    return finalAcqGait,detectAnomaly
