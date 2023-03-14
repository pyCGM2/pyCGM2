# -*- coding: utf-8 -*-
import os

import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Tools import btkTools
from pyCGM2 import enums
from pyCGM2.Utils import utils
from pyCGM2.Utils import files
from pyCGM2.Model import modelFilters
from pyCGM2.Model import bodySegmentParameters
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.Model.CGM2 import cgm2
from pyCGM2.Model.CGM2 import decorators
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Model.Opensim import opensimFilters
from pyCGM2.Processing.ProgressionFrame import progressionFrameFilters
from pyCGM2.Processing.ProgressionFrame import progressionFrameProcedures
from pyCGM2.Signal import signal_processing
from pyCGM2.Anomaly import anomalyFilters
from pyCGM2.Anomaly import anomalyDetectionProcedures
from pyCGM2.Inspector import inspectorFilters
from pyCGM2.Inspector import inspectorProcedures
from pyCGM2.Lib.Processing import progression

from pyCGM2.Model.Opensim.interface import opensimInterfaceFilters
from pyCGM2.Model.Opensim.interface.procedures.scaling import opensimScalingInterfaceProcedure
from pyCGM2.Model.Opensim.interface.procedures.inverseKinematics import opensimInverseKinematicsInterfaceProcedure
from pyCGM2.Model.Opensim.interface.procedures.inverseDynamics import opensimInverseDynamicsInterfaceProcedure
from pyCGM2.Model.Opensim.interface.procedures.analysisReport import opensimAnalysesInterfaceProcedure

from pyCGM2.Model.Opensim import opensimIO


def calibrate(DATA_PATH,calibrateFilenameLabelled,translators,weights,
              required_mp,optional_mp,
              ik_flag,leftFlatFoot,rightFlatFoot,headFlat,
              markerDiameter,hjcMethod,
              pointSuffix,*argv,**kwargs):
    """
    CGM22 calibration.

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
        hjcMethod (dict): hip joint centre regressions
        pointSuffix (str): suffix to add to ouputs

    Keyword Arguments:
        anomalyException (bool): raise exception if anomaly detected
        forceBtkAcq (btk.Acquisition): use a btkAcquisition instance instead of building the btkAcquisition from the static filename
        displayCoordinateSystem (bool): return virtual markers for visualisation of the anatomical refentials
        noKinematicsCalculation (bool) : disable computation of joint kinematics
        forceMP (bool) : force the use of mp offset to compute the knee and ankle joint centres

    Returns:
        model (pyCGM2.Model): the calibrated Model
        acqStatic (Btk.Acquisition): static btkAcquisition instance with model ouputs
        detectAnomaly  (bool): presence of anomaly

    """
    utils.homogeneizeArguments(argv,kwargs)

    detectAnomaly = False

    # --------------------ACQUISITION------------------------------

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

    trackingMarkers = cgm.CGM1.LOWERLIMB_TRACKING_MARKERS + cgm.CGM1.THORAX_TRACKING_MARKERS+ cgm.CGM1.UPPERLIMB_TRACKING_MARKERS
    actual_trackingMarkers,phatoms_trackingMarkers = btkTools.createPhantoms(acqStatic, trackingMarkers)

    vff = acqStatic.GetFirstFrame()
    vlf = acqStatic.GetLastFrame()
    # vff,vlf = btkTools.getFrameBoundaries(acqStatic,actual_trackingMarkers)
    flag = btkTools.getValidFrames(acqStatic,actual_trackingMarkers,frameBounds=[vff,vlf]) #not used

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



    # ---check marker set used----
    dcm= cgm.CGM.detectCalibrationMethods(acqStatic)

    # ---definition---
    model=cgm2.CGM2_2()
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
                                        markerDiameter=markerDiameter
                                        ).compute()

    # ---- Decorators -----
    forceMP = False if not "forceMP" in kwargs else kwargs["forceMP"]
    decorators.applyKJC_AJCDecorators(dcm, model,acqStatic,optional_mp,markerDiameter,forceMP=forceMP)
    decorators.applyHJCDecorators(model,hjcMethod)

    # ----Final Calibration filter if model previously decorated -----
    if model.decoratedModel:
        # initial static filter
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                           leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                           headFlat= headFlat,
                           markerDiameter=markerDiameter,
                           ).compute()

    # ----------------------CGM MODELLING----------------------------------
    # ----motion filter----
    modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,enums.motionMethod.Determinist,
                                              markerDiameter=markerDiameter)

    modMotion.compute()

    # ----progression Frame----
    progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqStatic,staticFlag=True)

    # ----manage IK Targets----
    ikTargets = list()
    for target in weights.keys():
        if target not in actual_trackingMarkers:
            weights[target] = 0
            LOGGER.logger.info("[pyCGM2] - the IK targeted marker [%s] is not labelled in the acquisition [%s]"%(target,calibrateFilenameLabelled))
        else:
            ikTargets.append(target)
    model.setStaticIkTargets(ikTargets)

    # scaling
    proc = opensimScalingInterfaceProcedure.ScalingXmlCgmProcedure(DATA_PATH,"CGM2.2")
    proc.setStaticTrial( acqStatic, calibrateFilenameLabelled[:-4])
    proc.setAnthropometry(required_mp["Bodymass"],required_mp["Height"])
    proc.prepareXml()    
    
    oisf = opensimInterfaceFilters.opensimInterfaceScalingFilter(proc)
    oisf.run()
    scaledOsim = oisf.getOsim()
    scaledOsimName = oisf.getOsimName()
        
    model.m_properties["scaledOsimName"] = scaledOsimName


    # virtual standstill
    procAnaDriven = opensimAnalysesInterfaceProcedure.AnalysesXmlCgmDrivenModelProcedure(DATA_PATH,scaledOsimName,"musculoskeletal_modelling/pose_standstill","CGM2.3")
    procAnaDriven.setPose("standstill")
    procAnaDriven.prepareXml()
    oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAnaDriven)
    oiamf.run()


    if "noKinematicsCalculation" in kwargs.keys() and kwargs["noKinematicsCalculation"]:
        LOGGER.logger.warning("[pyCGM2] No Kinematic calculation done for the static file")
        return model, acqStatic,detectAnomaly
    else:

        if ik_flag:

            procIK = opensimInverseKinematicsInterfaceProcedure.InverseKinematicXmlCgmProcedure(DATA_PATH,scaledOsimName,"musculoskeletal_modelling","CGM2.3")
            procIK.setProgression(progressionAxis,forwardProgression)
            procIK.prepareDynamicTrial(acqStatic,calibrateFilenameLabelled[:-4])
            procIK.setAccuracy(1e-5)
            procIK.setWeights(weights)
            procIK.setTimeRange()
            procIK.prepareXml()

            oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
            oiikf.run()
            oiikf.pushFittedMarkersIntoAcquisition()
            #oiikf.pushMotToAcq(osimConverterSettings)
            acqStaticIK =oiikf.getAcq()


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
    ik_flag,
    markerDiameter,
    pointSuffix,
    mfpa,
    momentProjection,*argv, **kwargs):

    """
    CGM22 Fitting.

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
    modelVersion="CGM2.2"

    utils.homogeneizeArguments(argv,kwargs)
    
    detectAnomaly = False

    if "anomalyException" in kwargs.keys():
        anomalyException = kwargs["anomalyException"]
    else:
        anomalyException=False

    if "Fitting" in weights.keys():
        weights  = weights["Fitting"]["Weight"]

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

    LOGGER.logger.info("[pyCGM2]  Computation from frame [%s] to frame [%s]"%(vff,vlf))
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

    # --- initial motion Filter ---

    scp=modelFilters.StaticCalibrationProcedure(model)
    modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,enums.motionMethod.Determinist)
    modMotion.compute()

    progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGait)
    

    for target in weights.keys():
        if target not in actual_trackingMarkers or target not in model.getStaticIkTargets():
            weights[target] = 0
            LOGGER.logger.warning("[pyCGM2] - the IK targeted marker [%s] is not labelled in the acquisition [%s]"%(target,reconstructFilenameLabelled))

    if ik_flag:
        #                        ---OPENSIM IK---

        scaledOsimName = model.m_properties["scaledOsimName"]#"CGM23-ScaledModel.osim"
        accuracy = kwargs["ikAccuracy"] if "ikAccuracy" in kwargs.keys() else 1e-8

        
        # --- IK ---
        progressionAxis, forwardProgression, globalFrame =progression.detectProgressionFrame(acqGait)

        procIK = opensimInverseKinematicsInterfaceProcedure.InverseKinematicXmlCgmProcedure(DATA_PATH,scaledOsimName,"musculoskeletal_modelling","CGM2.2")
        procIK.setProgression(progressionAxis,forwardProgression)
        procIK.prepareDynamicTrial(acqGait,reconstructFilenameLabelled[:-4])
        procIK.setAccuracy(accuracy)
        procIK.setWeights(weights)
        procIK.setTimeRange()
        procIK.prepareXml()

        oiikf = opensimInterfaceFilters.opensimInterfaceInverseKinematicsFilter(procIK)
        oiikf.run()
        oiikf.pushFittedMarkersIntoAcquisition()
        #oiikf.pushMotToAcq(osimConverterSettings)
        acqIK =oiikf.getAcq()


    # eventual gait acquisition to consider for joint kinematics
    finalAcqGait = acqIK if ik_flag else acqGait

    # --- final pyCGM2 model motion Filter ---
    # use fitted markers
    modMotionFitted=modelFilters.ModelMotionFilter(scp,finalAcqGait,model,enums.motionMethod.Determinist ,
                                              markerDiameter=markerDiameter)

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
    
    if os.path.isfile(DATA_PATH+"mp.settings"):
        custom_mp = files.openFile(DATA_PATH,"mp.settings")
        bodySegmentParameters.updateFromcustomMp(model,custom_mp)

    modelFilters.CentreOfMassFilter(model,finalAcqGait).compute(pointLabelSuffix=pointSuffix)

    # Inverse dynamics
    if btkTools.checkForcePlateExist(acqGait):
        # --- force plate handling----
        # find foot  in contact
        mappedForcePlate = forceplates.matchingFootSideOnForceplate(finalAcqGait,mfpa=mfpa)
        forceplates.addForcePlateGeneralEvents(finalAcqGait,mappedForcePlate)
        LOGGER.logger.info("Manual Force plate assignment : %s" %mappedForcePlate)

        # assembly foot and force plate
        modelFilters.ForcePlateAssemblyFilter(model,finalAcqGait,mappedForcePlate,
                                 leftSegmentLabel="Left Foot",
                                 rightSegmentLabel="Right Foot").compute(pointLabelSuffix=pointSuffix)

        # standardize grf
        cgrff = modelFilters.GroundReactionForceAdapterFilter(acqGait,globalFrameOrientation=globalFrame, forwardProgression=forwardProgression)
        cgrff.compute(pointLabelSuffix=pointSuffix)

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
        motDataframe = opensimIO.OpensimDataFrame(DATA_PATH+"musculoskeletal_modelling"+"\\",reconstructFilenameLabelled[:-4]+".mot")
        motDataframe.getDataFrame()["ankle_flexion_r"] = finalAcqGait.GetPoint("RAnkleAngles").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_r"] = finalAcqGait.GetPoint("RAnkleAngles").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_r"] = finalAcqGait.GetPoint("RAnkleAngles").GetValues()[:,2]
        motDataframe.getDataFrame()["ankle_flexion_l"] = finalAcqGait.GetPoint("LAnkleAngles").GetValues()[:,0]
        motDataframe.getDataFrame()["ankle_adduction_l"] = finalAcqGait.GetPoint("LAnkleAngles").GetValues()[:,1]
        motDataframe.getDataFrame()["ankle_rotation_l"] = finalAcqGait.GetPoint("LAnkleAngles").GetValues()[:,2]
        motDataframe.save()

        procAna = opensimAnalysesInterfaceProcedure.AnalysesXmlCgmProcedure(DATA_PATH,scaledOsimName,"musculoskeletal_modelling","CGM2.2")
        procAna.setProgression(progressionAxis,forwardProgression)
        procAna.prepareDynamicTrial(finalAcqGait,reconstructFilenameLabelled[:-4],mappedForcePlate)
        procAna.setTimeRange()
        procAna.prepareXml()

        oiamf = opensimInterfaceFilters.opensimInterfaceAnalysesFilter(procAna)
        oiamf.run()
        oiamf.pushStoToAcq()

    btkTools.cleanAcq(finalAcqGait)
    btkTools.applyOnValidFrames(finalAcqGait,flag)

    if detectAnomaly and not anomalyException:
        LOGGER.logger.error("Anomalies has been detected - Check Warning messages of the log file")

    return finalAcqGait,detectAnomaly
