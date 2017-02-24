# -*- coding: utf-8 -*-
import logging
import json
import pdb

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Model.CGM2 import cgm, modelFilters, modelDecorator, forceplates,bodySegmentParameters
from pyCGM2 import  smartFunctions 


def checkCGM1_StaticMarkerConfig(acqStatic):

    out = dict()

    # medial ankle markers
    out["leftMedialAnkleFlag"] = True if btkTools.isPointsExist(acqStatic,["LMED","LANK"]) else False
    out["rightMedialAnkleFlag"] = True if btkTools.isPointsExist(acqStatic,["RMED","RANK"]) else False

    # medial knee markers
    out["leftMedialKneeFlag"] = True if btkTools.isPointsExist(acqStatic,["LMEPI","LKNE"]) else False
    out["rightMedialKneeFlag"] = True if btkTools.isPointsExist(acqStatic,["RMEPI","RKNE"]) else False


    # kad
    out["leftKadFlag"] = True if btkTools.isPointsExist(acqStatic,["LKAX","LKD1","LKD2"]) else False
    out["rightKadFlag"] = True if btkTools.isPointsExist(acqStatic,["RKAX","RKD1","RKD2"]) else False

    return out



if __name__ == "__main__":

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if NEXUS_PYTHON_CONNECTED: # run Operation

        DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\CGM1\\PIG standard\\basic\\"

        # ---- load pyCGM2 inputs    
        INPUTS = json.loads(open(str(DATA_PATH +'pyCGM2.inputs')).read())
        
        # --------------------------DATA--------------------------------------
        
        staticTrialName = INPUTS["Static trial"]
        dynamicTrialNames = INPUTS["Dynamic trials"]
        
        # --------------------------STATIC CALBRATION--------------------------                
        staticTrialNameNoExt = staticTrialName[:-4]
        
        NEXUS.OpenTrial( str(DATA_PATH+staticTrialNameNoExt), 30 )
        NEXUS.RunPipeline( 'pyCGM2-CGM1-Calibration', 'Private', 45 )
        NEXUS.SaveTrial( 30 )





#    # ---btk acquisition---
#    acqStatic = btkTools.smartReader(str(DATA_PATH+staticTrialName))
#
#    # ---check marker set used----
#    staticMarkerConfiguration= checkCGM1_StaticMarkerConfig(acqStatic)
#
#
#        
#    # ---calibration parameters---
#    flag_leftFlatFoot =  bool(INPUTS["Calibration"]["Left flat foot"])
#    flag_rightFlatFoot =  bool(INPUTS["Calibration"]["Right flat foot"])
#    markerDiameter = float(INPUTS["Calibration"]["Marker diameter"])
#    pointSuffix = INPUTS["Calibration"]["Point suffix"]
#
#    #---Calibration Filter---
#    scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure
#           
#    modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
#                                            leftFlatFoot = flag_leftFlatFoot, rightFlatFoot = flag_rightFlatFoot,
#                                            markerDiameter=markerDiameter,
#                                            ).compute()
#
#    # ---- Decorators -----
#
#    staticMarkerConfiguration= checkCGM1_StaticMarkerConfig(acqStatic) 
#
#    # initialisation of node label 
#    useLeftKJCnodeLabel = "LKJC_chord"
#    useLeftAJCnodeLabel = "LAJC_chord"
#    useRightKJCnodeLabel = "RKJC_chord"
#    useRightAJCnodeLabel = "RAJC_chord"
#
#    # case 1 : NO kad, NO medial ankle BUT thighRotation different from zero ( mean manual modification or new calibration from a previous one )
#    #   This 
#    if not staticMarkerConfiguration["leftKadFlag"]  and not staticMarkerConfiguration["leftMedialAnkleFlag"] and not staticMarkerConfiguration["leftMedialKneeFlag"] and optional_mp["LeftThighRotation"] !=0:
#        logging.warning("CASE FOUND ===> Left Side - CGM1 - Origine - manual offsets")            
#        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",optional_mp["LeftThighRotation"],markerDiameter,optional_mp["LeftTibialTorsion"],optional_mp["LeftShankRotation"])
#        useLeftKJCnodeLabel = "LKJC_mo"
#        useLeftAJCnodeLabel = "LAJC_mo"
#   
#
#    if not staticMarkerConfiguration["rightKadFlag"]  and not staticMarkerConfiguration["rightMedialAnkleFlag"] and not staticMarkerConfiguration["rightMedialKneeFlag"] and optional_mp["RightThighRotation"] !=0:
#        logging.warning("CASE FOUND ===> Right Side - CGM1 - Origine - manual offsets")            
#        modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",optional_mp["RightThighRotation"],markerDiameter,optional_mp["RightTibialTorsion"],optional_mp["RightShankRotation"])
#        useRightKJCnodeLabel = "RKJC_mo"
#        useRightAJCnodeLabel = "RAJC_mo"
#
#    # case 2 : kad FOUND and NO medial Ankle 
#    if staticMarkerConfiguration["leftKadFlag"]:
#        logging.warning("CASE FOUND ===> Left Side - CGM1 - KAD variant")
#        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left", displayMarkers = False)
#        useLeftKJCnodeLabel = "LKJC_kad"
#        useLeftAJCnodeLabel = "LAJC_kad"
#    if staticMarkerConfiguration["rightKadFlag"]:
#        logging.warning("CASE FOUND ===> Right Side - CGM1 - KAD variant")
#        modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right", displayMarkers = False)
#        useRightKJCnodeLabel = "RKJC_kad"
#        useRightAJCnodeLabel = "RAJC_kad"
#    
#    # case 3 : both kad and medial ankle FOUND 
#    if staticMarkerConfiguration["leftKadFlag"]:
#        if staticMarkerConfiguration["leftMedialAnkleFlag"]:
#            logging.warning("CASE FOUND ===> Left Side - CGM1 - KAD + medial ankle ")
#            modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")
#            useLeftAJCnodeLabel = "LAJC_mid"
#
#    if staticMarkerConfiguration["rightKadFlag"]:
#        if staticMarkerConfiguration["rightMedialAnkleFlag"]:
#            logging.warning("CASE FOUND ===> Right Side - CGM1 - KAD + medial ankle ")
#            modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")
#            useRightAJCnodeLabel = "RAJC_mid"
#
#    # ----Final Calibration filter if model previously decorated ----- 
#    if model.decoratedModel:
#        # initial static filter
#        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
#                           useLeftKJCnode=useLeftKJCnodeLabel, useLeftAJCnode=useLeftAJCnodeLabel,
#                           useRightKJCnode=useRightKJCnodeLabel, useRightAJCnode=useRightAJCnodeLabel,
#                           markerDiameter=markerDiameter).compute()
#
#                 
#    # --- save modelled static trial                           
#    #btkTools.smartWriter(acqStatic,str(DATA_PATH + staticTrialName[:-4] + "_CGM1.c3d"))
#    
#    # --------------------------FITTING--------------------------
#    for dynamicTrialName in dynamicTrialNames:
#        # --- btk acquisition ----
#        acqGait = btkTools.smartReader(str(DATA_PATH + dynamicTrialName))
#    
#            
#        # ---Motion filter----    
#        modMotion=modelFilters.ModelMotionFilter(scp,acqGait,model,pyCGM2Enums.motionMethod.Native,
#                                                  markerDiameter=markerDiameter).compute()
#    
#        #---- Joint kinematics----
#        # relative angles
#        modelFilters.ModelJCSFilter(model,acqGait).compute(description="vectoriel", pointLabelSuffix=pointSuffix)
#    
#        # detection of traveling axis
#        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionFromPoints(acqGait,"SACR","midASIS","LPSI")
#        # absolute angles        
#        modelFilters.ModelAbsoluteAnglesFilter(model,acqGait,
#                                               segmentLabels=["Left Foot","Right Foot","Pelvis"],
#                                                angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
#                                                eulerSequences=["TOR","TOR", "TOR"],
#                                                globalFrameOrientation = globalFrame,
#                                                forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)
#    
#        #---- Body segment parameters----
#        bspModel = bodySegmentParameters.Bsp(model)
#        bspModel.compute()
#    
#        # --- force plate handling----
#        # find foot  in contact        
#        mappedForcePlate = forceplates.matchingFootSideOnForceplate(acqGait)
#        # assembly foot and force plate        
#        modelFilters.ForcePlateAssemblyFilter(model,acqGait,mappedForcePlate,
#                                 leftSegmentLabel="Left Foot",
#                                 rightSegmentLabel="Right Foot").compute()
#    
#        #---- Joint kinetics----
#        idp = modelFilters.CGMLowerlimbInverseDynamicProcedure()
#        modelFilters.InverseDynamicFilter(model,
#                             acqGait,
#                             procedure = idp,
#                             projection = pyCGM2Enums.MomentProjection.Distal
#                             ).compute(pointLabelSuffix=pointSuffix)
#    
#        #---- Joint energetics----
#        modelFilters.JointPowerFilter(model,acqGait).compute(pointLabelSuffix=pointSuffix)
#    
#        # --- save modelled dynamic trial    
#        btkTools.smartWriter(acqGait,str(DATA_PATH + dynamicTrialName[:-4] + "_CGM1.c3d"))
#
#    # --------------------------GAIT PROCESSING--------------------------
#
#    # -----infos--------     
#    model = None if  INPUTS["Model"]=={} else INPUTS["Model"]  
#    subject = None if INPUTS["Subject"]=={} else INPUTS["Subject"] 
#    experimental = None if INPUTS["Experimental conditions"]=={} else INPUTS["Experimental conditions"] 
#
#    normativeData = INPUTS["Normative data"]
#
#    # --------------------------PROCESSING --------------------------------
#    # pycgm2-filter pipeline are gathered in a single function
#    modelledDynamicTrials = [str(trial[:-4]+"_CGM1.c3d") for trial in dynamicTrialNames]
#    
#    smartFunctions.gaitProcessing_cgm1 (modelledDynamicTrials, DATA_PATH,
#                           model,  subject, experimental,
#                           pointLabelSuffix = pointSuffix,
#                           plotFlag= True, 
#                           exportBasicSpreadSheetFlag = False,
#                           exportAdvancedSpreadSheetFlag = False,
#                           exportAnalysisC3dFlag = False,
#                           consistencyOnly = True,
#                           normativeDataDict = normativeData)