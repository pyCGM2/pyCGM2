# -*- coding: utf-8 -*-

import pyCGM2; LOGGER = pyCGM2.LOGGER

# pyCGM2 settings
import pyCGM2

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2 import enums
from pyCGM2.Utils import files

from pyCGM2.Model import modelFilters,bodySegmentParameters
from pyCGM2.Model.CGM2 import cgm,cgm2
from pyCGM2.Model.CGM2 import decorators
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Processing import progressionFrame
from pyCGM2.Signal import signal_processing
from pyCGM2.Anomaly import AnomalyFilter, AnomalyDetectionProcedure
from pyCGM2.Inspector import InspectorFilter, InspectorProcedure

from pyCGM2.Model.Opensim import opensimInterfaceFilters
from pyCGM2.Model.Opensim import opensimScalingInterfaceProcedure
from pyCGM2.Model.Opensim import opensimInverseKinematicsInterfaceProcedure
from pyCGM2.Model.Opensim import opensimAnalysesInterfaceProcedure
from pyCGM2.Model.Opensim import opensimIO
from pyCGM2.Lib import processing

def calibrate(DATA_PATH,calibrateFilenameLabelled,translators,weights,
              required_mp,optional_mp,
              ik_flag,leftFlatFoot,rightFlatFoot,headFlat,
              markerDiameter,hjcMethod,
              pointSuffix,**kwargs):
    """
    CGM23 calibration.

    Args:
        DATA_PATH (str): folder path.
        calibrateFilenameLabelled (str): filename of your static file.
        translators (dict): marker translators.
        required_mp (dict):  required anthropometric parameter.
        optional_mp (dict): optional anthropometric parameter.
        ik_flag (bool): enable or disable inverse kinematics
        leftFlatFoot (bool): flat foot option.
        rightFlatFoot (bool): flat foot option.
        headFlat (bool): flat head option.
        markerDiameter (float): marker diameter
        hjcMethod (dict): hip joint centre regressions
        pointSuffix (str): suffix to add to ouputs

    Low-level Keyword Args :
        anomalyException (bool): raise exception if anomaly detected
        forceBtkAcq (btk.Acquisition): use a btkAcquisition instance instead of building the btkAcquisition from the static filename
        displayCoordinateSystem (bool): return virtual markers for visualisation of the anatomical refentials
        noKinematicsCalculation (bool) : disable computation of joint kinematics

    Returns:
        pyCGM2.Model: the calibrated Model
        Btk.Acquisition : static btkAcquisition instance with model ouputs
        bool: presence of deteced anomalies

    """
    detectAnomaly = False


    if "anomalyException" in kwargs.keys():
        anomalyException = kwargs["anomalyException"]
    else:
        anomalyException=False

    # --------------------------STATIC FILE WITH TRANSLATORS --------------------------------------

    if "Fitting" in weights.keys():
        weights  = weights["Fitting"]["Weight"]


    # --- btk acquisition ----
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
    trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
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
    markersets = [cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS, cgm2.CGM2_3.THORAX_TRACKING_MARKERS, cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS]
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
    model=cgm2.CGM2_3()
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
    progressionAxis, forwardProgression, globalFrame =processing.detectProgressionFrame(acqStatic,staticFlag=True)


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

            modelVersion = "CGM2.3"

            # --- osim builder ---
            markersetTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "markerset\\CGM23-markerset.xml"
            osimTemplateFullFile =pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "osim\\pycgm2-gait2354_simbody.osim"
            # osimConverterSettings = files.openFile(pyCGM2.OPENSIM_PREBUILD_MODEL_PATH,"setup\\CGM23\\OsimToC3dConverter.settings")

            # scaling
            scaleToolFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23_scaleSetup_template.xml"

            proc = opensimScalingInterfaceProcedure.highLevelScalingProcedure(DATA_PATH,modelVersion,osimTemplateFullFile,markersetTemplateFullFile,scaleToolFullFile)
            proc.preProcess( acqStatic, calibrateFilenameLabelled[:-4])
            proc.setAnthropometry(required_mp["Bodymass"],required_mp["Height"])
            oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
            oisf.run()
            scaledOsim = oisf.getOsim()
            scaledOsimName = oisf.getOsimName()

            # --- IK ---
            # ikWeights = settings["Fitting"]["Weight"]
            # --- IK ---
            ikTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-ikSetUp_template.xml"

            procIK = opensimInverseKinematicsInterfaceProcedure.highLevelInverseKinematicsProcedure(DATA_PATH,scaledOsimName,modelVersion,ikTemplateFullFile)
            procIK.setProgression(progressionAxis,forwardProgression)
            procIK.preProcess(acqStatic,calibrateFilenameLabelled[:-4])
            procIK.setAccuracy(1e-5)
            procIK.setWeights(weights)
            procIK.setTimeRange()
            # procIK.setResultsDirname("verif")
            oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
            oiikf.run()
            # oiikf.motToC3d(osimConverterSettings)
            acqStaticIK =oiikf.getAcq()




            # #                        ---OPENSIM IK---
            #
            # # --- opensim calibration Filter ---
            # osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile
            # markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-markerset.xml" # markerset
            # cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure
            #
            # oscf = opensimFilters.opensimCalibrationFilter(osimfile,
            #                                         model,
            #                                         cgmCalibrationprocedure,
            #                                         DATA_PATH )
            # oscf.addMarkerSet(markersetFile)
            # scalingOsim = oscf.build()
            #
            #
            # # --- opensim Fitting Filter ---
            # iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-ikSetUp_template.xml" # ik tool file
            #
            #
            # cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model) # procedure
            # cgmFittingProcedure.updateMarkerWeight("LASI",weights["LASI"])
            # cgmFittingProcedure.updateMarkerWeight("RASI",weights["RASI"])
            # cgmFittingProcedure.updateMarkerWeight("LPSI",weights["LPSI"])
            # cgmFittingProcedure.updateMarkerWeight("RPSI",weights["RPSI"])
            # cgmFittingProcedure.updateMarkerWeight("RTHI",weights["RTHI"])
            # cgmFittingProcedure.updateMarkerWeight("RKNE",weights["RKNE"])
            # cgmFittingProcedure.updateMarkerWeight("RTIB",weights["RTIB"])
            # cgmFittingProcedure.updateMarkerWeight("RANK",weights["RANK"])
            # cgmFittingProcedure.updateMarkerWeight("RHEE",weights["RHEE"])
            # cgmFittingProcedure.updateMarkerWeight("RTOE",weights["RTOE"])
            # cgmFittingProcedure.updateMarkerWeight("LTHI",weights["LTHI"])
            # cgmFittingProcedure.updateMarkerWeight("LKNE",weights["LKNE"])
            # cgmFittingProcedure.updateMarkerWeight("LTIB",weights["LTIB"])
            # cgmFittingProcedure.updateMarkerWeight("LANK",weights["LANK"])
            # cgmFittingProcedure.updateMarkerWeight("LHEE",weights["LHEE"])
            # cgmFittingProcedure.updateMarkerWeight("LTOE",weights["LTOE"])
            #
            # cgmFittingProcedure.updateMarkerWeight("LTHAP",weights["LTHAP"])
            # cgmFittingProcedure.updateMarkerWeight("LTHAD",weights["LTHAD"])
            # cgmFittingProcedure.updateMarkerWeight("LTIAP",weights["LTIAP"])
            # cgmFittingProcedure.updateMarkerWeight("LTIAD",weights["LTIAD"])
            # cgmFittingProcedure.updateMarkerWeight("RTHAP",weights["RTHAP"])
            # cgmFittingProcedure.updateMarkerWeight("RTHAD",weights["RTHAD"])
            # cgmFittingProcedure.updateMarkerWeight("RTIAP",weights["RTIAP"])
            # cgmFittingProcedure.updateMarkerWeight("RTIAD",weights["RTIAD"])
            #
            # osrf = opensimFilters.opensimFittingFilter(iksetupFile,
            #                                                   scalingOsim,
            #                                                   cgmFittingProcedure,
            #                                                   DATA_PATH,
            #                                                   acqStatic,
            #                                                   accuracy = 1e-5)
            # osrf.setTimeRange(acqStatic,beginFrame = vff, lastFrame=vlf)
            # LOGGER.logger.info("-------INVERSE KINEMATICS IN PROGRESS----------")
            # try:
            #     acqStaticIK = osrf.run(DATA_PATH + calibrateFilenameLabelled,
            #                 progressionAxis = progressionAxis ,
            #                 forwardProgression = forwardProgression)
            #     LOGGER.logger.info("[pyCGM2] - IK solver complete")
            # except:
            #     LOGGER.logger.error("[pyCGM2] - IK solver fails")
            #     acqStaticIK = acqStatic
            #     detectAnomaly = True
            # LOGGER.logger.info("-----------------------------------------------")



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
    CGM23 Fitting.

    Args:
        DATA_PATH (str): folder path.
        reconstructFilenameLabelled (str): filename of your gait trial.
        translators (dict): marker translators.
        weights (dict): marker weights
        ik_flag (bool): enable/disable inverse kinematics
        markerDiameter (float): marker diameter
        pointSuffix (str): suffix to add to ouputs
        mfpa (str): force plate assignment
        momentProjection (str) : referential for projection of joint moment

    Low-level Keyword Args :
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
        Btk.Acquisition :  btkAcquisition instance with model ouputs
        bool: presence of deteced anomalies

    """

    detectAnomaly = False

    # --------------------ACQUISITION------------------------------

    if "anomalyException" in kwargs.keys():
        anomalyException = kwargs["anomalyException"]
    else:
        anomalyException=False

    if "Fitting" in weights.keys():
        weights  = weights["Fitting"]["Weight"]

    if "forceFoot6DoF" in kwargs.keys() and kwargs["forceFoot6DoF"]:
        forceFoot6DoF_flag = True
    else:
        forceFoot6DoF_flag=False
    # --------------------------ACQ WITH TRANSLATORS --------------------------------------

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

    trackingMarkers = cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS + cgm2.CGM2_3.THORAX_TRACKING_MARKERS+ cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS
    actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqGait, trackingMarkers)
    vff,vlf = btkTools.getFrameBoundaries(acqGait,actual_trackingMarkers)
    if "frameInit" in kwargs.keys() and kwargs["frameInit"] is not None:
        vff = kwargs["frameInit"]
        LOGGER.logger.info("[pyCGM2]  first frame forced to frame [%s]"%(vff))
    if "frameEnd" in kwargs.keys() and kwargs["frameEnd"] is not None:
        vlf = kwargs["frameEnd"]
        LOGGER.logger.info("[pyCGM2]  end frame forced to frame [%s]"%(vlf))
    flag = btkTools.getValidFrames(acqGait,actual_trackingMarkers,frameBounds=[vff,vlf])

    LOGGER.logger.info("[pyCGM2]  Computation from frame [%s] to frame [%s]"%(vff,vlf))
    # --------------------ANOMALY------------------------------
    for marker in actual_trackingMarkers:
        if marker not in model.getStaticTrackingMarkers():
            LOGGER.logger.warning("[pyCGM2-Anomaly]  marker [%s] - not used during static calibration - wrong kinematic for the segment attached to this marker. "%(marker))

    # --marker presence
    markersets = [cgm2.CGM2_3.LOWERLIMB_TRACKING_MARKERS, cgm2.CGM2_3.THORAX_TRACKING_MARKERS, cgm2.CGM2_3.UPPERLIMB_TRACKING_MARKERS]
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
    # section to remove : - copy motion of ProximalShank from Shank with Sodervisk
    modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Sodervisk)
    modMotion.compute()
    # /section to remove

    progressionAxis, forwardProgression, globalFrame =processing.detectProgressionFrame(acqGait)


    for target in weights.keys():
        if target not in actual_trackingMarkers or target not in model.getStaticIkTargets():
            weights[target] = 0
            LOGGER.logger.warning("[pyCGM2] - the IK targeted marker [%s] is not labelled in the acquisition [%s]"%(target,reconstructFilenameLabelled))

    if ik_flag:

        #                        ---OPENSIM IK---

        # --- new ---
        modelVersion="CGM2.3"
        scaledOsimName = "CGM23-ScaledModel.osim"

        accuracy = kwargs["ikAccuracy"] if "ikAccuracy" in kwargs.keys() else 1e-8



        ikTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-ikSetUp_template.xml"

        procIK = opensimInverseKinematicsInterfaceProcedure.highLevelInverseKinematicsProcedure(DATA_PATH,scaledOsimName,modelVersion,ikTemplateFullFile)
        procIK.setProgression(progressionAxis,forwardProgression)
        procIK.preProcess(acqGait,reconstructFilenameLabelled[:-4])
        procIK.setAccuracy(accuracy)
        procIK.setWeights(weights)
        procIK.setTimeRange()
        # procIK.setResultsDirname("verif")
        oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
        oiikf.run()
        # oiikf.motToC3d(osimConverterSettings)
        acqIK =oiikf.getAcq()

        # # --- old ---
        #
        # # --- opensim calibration Filter ---
        # osimfile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\osim\\lowerLimb_ballsJoints.osim"    # osimfile
        # markersetFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-markerset.xml" # markerset
        # cgmCalibrationprocedure = opensimFilters.CgmOpensimCalibrationProcedures(model) # procedure
        #
        # oscf = opensimFilters.opensimCalibrationFilter(osimfile,
        #                                         model,
        #                                         cgmCalibrationprocedure,
        #                                         DATA_PATH )
        # oscf.addMarkerSet(markersetFile)
        # scalingOsim = oscf.build()
        #
        #
        # # --- opensim Fitting Filter ---
        # iksetupFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "models\\settings\\cgm2_3\\cgm2_3-ikSetUp_template.xml" # ik tl file
        #
        # cgmFittingProcedure = opensimFilters.CgmOpensimFittingProcedure(model) # procedure
        # cgmFittingProcedure.updateMarkerWeight("LASI",weights["LASI"])
        # cgmFittingProcedure.updateMarkerWeight("RASI",weights["RASI"])
        # cgmFittingProcedure.updateMarkerWeight("LPSI",weights["LPSI"])
        # cgmFittingProcedure.updateMarkerWeight("RPSI",weights["RPSI"])
        # cgmFittingProcedure.updateMarkerWeight("RTHI",weights["RTHI"])
        # cgmFittingProcedure.updateMarkerWeight("RKNE",weights["RKNE"])
        # cgmFittingProcedure.updateMarkerWeight("RTIB",weights["RTIB"])
        # cgmFittingProcedure.updateMarkerWeight("RANK",weights["RANK"])
        # cgmFittingProcedure.updateMarkerWeight("RHEE",weights["RHEE"])
        # cgmFittingProcedure.updateMarkerWeight("RTOE",weights["RTOE"])
        # cgmFittingProcedure.updateMarkerWeight("LTHI",weights["LTHI"])
        # cgmFittingProcedure.updateMarkerWeight("LKNE",weights["LKNE"])
        # cgmFittingProcedure.updateMarkerWeight("LTIB",weights["LTIB"])
        # cgmFittingProcedure.updateMarkerWeight("LANK",weights["LANK"])
        # cgmFittingProcedure.updateMarkerWeight("LHEE",weights["LHEE"])
        # cgmFittingProcedure.updateMarkerWeight("LTOE",weights["LTOE"])
        #
        # cgmFittingProcedure.updateMarkerWeight("LTHAP",weights["LTHAP"])
        # cgmFittingProcedure.updateMarkerWeight("LTHAD",weights["LTHAD"])
        # cgmFittingProcedure.updateMarkerWeight("LTIAP",weights["LTIAP"])
        # cgmFittingProcedure.updateMarkerWeight("LTIAD",weights["LTIAD"])
        # cgmFittingProcedure.updateMarkerWeight("RTHAP",weights["RTHAP"])
        # cgmFittingProcedure.updateMarkerWeight("RTHAD",weights["RTHAD"])
        # cgmFittingProcedure.updateMarkerWeight("RTIAP",weights["RTIAP"])
        # cgmFittingProcedure.updateMarkerWeight("RTIAD",weights["RTIAD"])
        #
        # osrf = opensimFilters.opensimFittingFilter(iksetupFile,
        #                                                   scalingOsim,
        #                                                   cgmFittingProcedure,
        #                                                   DATA_PATH,
        #                                                   acqGait)
        # osrf.setTimeRange(acqGait,beginFrame = vff, lastFrame=vlf)
        # if "ikAccuracy" in kwargs.keys():
        #     osrf.setAccuracy(kwargs["ikAccuracy"])
        #
        # LOGGER.logger.info("-------INVERSE KINEMATICS IN PROGRESS----------")
        # try:
        #     acqIK = osrf.run(DATA_PATH + reconstructFilenameLabelled,
        #                     progressionAxis = progressionAxis ,
        #                     forwardProgression = forwardProgression)
        #     LOGGER.logger.info("[pyCGM2] - IK solver complete")
        # except:
        #     LOGGER.logger.error("[pyCGM2] - IK solver fails")
        #     acqIK = acqGait
        #     detectAnomaly = True
        # LOGGER.logger.info("---------------------------------------------------")

    # eventual gait acquisition to consider for joint kinematics
    finalAcqGait = acqIK if ik_flag else acqGait

    # --- final pyCGM2 model motion Filter ---
    # use fitted markers
    modMotionFitted=modelFilters.ModelMotionFilter(scp,finalAcqGait,model,enums.motionMethod.Sodervisk ,
                                              markerDiameter=markerDiameter,
                                              forceFoot6DoF=forceFoot6DoF_flag)

    modMotionFitted.compute()

    if "displayCoordinateSystem" in kwargs.keys() and kwargs["displayCoordinateSystem"]:
        csp = modelFilters.ModelCoordinateSystemProcedure(model)
        csdf = modelFilters.CoordinateSystemDisplayFilter(csp,model,finalAcqGait)
        csdf.setStatic(False)
        csdf.display()

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



    # --- Analyses ------
    if "muscleLength" in kwargs.keys() and kwargs["muscleLength"]:

        #correct the ankle angles
        motDataframe = opensimIO.OpensimDataFrame(DATA_PATH,reconstructFilenameLabelled[:-4]+".mot")
        motDataframe.getDataFrame()["ankle_flexion_r"] = finalAcqGait.GetPoint("RAnkleAngles").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_r"] = finalAcqGait.GetPoint("RAnkleAngles").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_r"] = finalAcqGait.GetPoint("RAnkleAngles").GetValues()[:,2]
        motDataframe.getDataFrame()["ankle_flexion_l"] = finalAcqGait.GetPoint("LAnkleAngles").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_l"] = finalAcqGait.GetPoint("LAnkleAngles").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_l"] = finalAcqGait.GetPoint("LAnkleAngles").GetValues()[:,2]
        motDataframe.save()

        anaTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\CGM23\\CGM23-analysisSetup_template.xml"
        externalLoadTemplateFullFile = pyCGM2.OPENSIM_PREBUILD_MODEL_PATH + "setup\\walk_grf.xml"
        procAna = opensimAnalysesInterfaceProcedure.highLevelAnalysesProcedure(DATA_PATH,scaledOsimName,modelVersion,anaTemplateFullFile,externalLoadTemplateFullFile)
        procAna.setProgression(progressionAxis,forwardProgression)
        procAna.preProcess(finalAcqGait,reconstructFilenameLabelled[:-4])
        # procAna.setResultsDirname("verif")
        procAna.setTimeRange()
        oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAna)
        oiamf.run()
        oiamf.stoToC3d()

    #---- zero unvalid frames ---
    btkTools.cleanAcq(finalAcqGait)
    btkTools.applyOnValidFrames(finalAcqGait,flag)

    if detectAnomaly and not anomalyException:
        LOGGER.logger.error("Anomalies has been detected - Check Warning messages of the log file")

    return finalAcqGait,detectAnomaly
