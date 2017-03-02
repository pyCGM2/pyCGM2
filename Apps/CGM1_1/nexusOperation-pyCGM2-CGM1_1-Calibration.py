# -*- coding: utf-8 -*-
import os
import logging
import matplotlib.pyplot as plt
import json
import pdb
import cPickle
import json
from shutil import copyfile

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)


# vicon nexus
import ViconNexus

# openMA
#import ma.io
#import ma.body

#btk
import btk
import numpy as np



# pyCGM2 libraries
from pyCGM2.Tools import btkTools,nexusTools
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Model.CGM2 import cgm, modelFilters, modelDecorator

from pyCGM2 import viconInterface

def updateNexusSubjectMp(NEXUS,model,subjectName):
    th_l = 0 if np.abs(model.getViconThighOffset("Left")) < 0.000001 else model.getViconThighOffset("Left")
    sh_l = 0 if np.abs(model.getViconShankOffset("Left"))< 0.000001 else model.getViconShankOffset("Left")
    tt_l = 0 if np.abs(model.getViconTibialTorsion("Left")) < 0.000001 else model.getViconTibialTorsion("Left")

    th_r = 0 if np.abs(model.getViconThighOffset("Right")) < 0.000001 else model.getViconThighOffset("Right")
    sh_r = 0 if np.abs(model.getViconShankOffset("Right")) < 0.000001 else model.getViconShankOffset("Right")
    tt_r = 0 if np.abs(model.getViconTibialTorsion("Right")) < 0.000001 else model.getViconTibialTorsion("Right")


    spf_l,sro_l = model.getViconFootOffset("Left")
    spf_r,sro_r = model.getViconFootOffset("Right")

    abdAdd_l = 0  if np.abs(model.getViconAnkleAbAddOffset("Left")) < 0.000001 else model.getViconAnkleAbAddOffset("Left") 
    abdAdd_r = 0  if np.abs(model.getViconAnkleAbAddOffset("Right")) < 0.000001 else model.getViconAnkleAbAddOffset("Right") 



    NEXUS.SetSubjectParam( subjectName, "InterAsisDistance",model.mp_computed["InterAsisDistance"])
    NEXUS.SetSubjectParam( subjectName, "LeftAsisTrocanterDistance",model.mp_computed["LeftAsisTrocanterDistance"])
    NEXUS.SetSubjectParam( subjectName, "LeftThighRotation",th_l)
    NEXUS.SetSubjectParam( subjectName, "LeftShankRotation",sh_l)
    NEXUS.SetSubjectParam( subjectName, "LeftTibialTorsion",tt_l)


    NEXUS.SetSubjectParam( subjectName, "RightAsisTrocanterDistance",model.mp_computed["RightAsisTrocanterDistance"])
    NEXUS.SetSubjectParam( subjectName, "RightThighRotation",th_r)
    NEXUS.SetSubjectParam( subjectName, "RightShankRotation",sh_r)
    NEXUS.SetSubjectParam( subjectName, "RightTibialTorsion",tt_r)


    NEXUS.SetSubjectParam( subjectName, "LeftStaticPlantFlex",spf_l)
    NEXUS.SetSubjectParam( subjectName, "LeftStaticRotOff",sro_l)
    NEXUS.SetSubjectParam( subjectName, "LeftAnkleAbAdd",abdAdd_l)

    NEXUS.SetSubjectParam( subjectName, "RightStaticPlantFlex",spf_r)
    NEXUS.SetSubjectParam( subjectName, "RightStaticRotOff",sro_r)
    NEXUS.SetSubjectParam( subjectName, "RightAnkleAbAdd",abdAdd_r)


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

    plt.close("all")
    DEBUG = False

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------------INPUTS ------------------------------------

        # --- acquisition file and path----
        if DEBUG:
            DATA_PATH = "C:\\Users\\AAA34169\\Documents\\VICON DATA\\pyCGM2-Data\\CGM1\\CGM1-NexusPlugin\\CGM1-Calibration\\"
            calibrateFilenameLabelledNoExt = "test" #"static Cal 01-noKAD-noAnkleMed" #
            NEXUS.OpenTrial( str(DATA_PATH+calibrateFilenameLabelledNoExt), 30 )

        else:
            DATA_PATH, calibrateFilenameLabelledNoExt = NEXUS.GetTrialName()

        calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ calibrateFilenameLabelled)

        # ---btk acquisition---
        acqStatic = btkTools.smartReader(str(DATA_PATH+calibrateFilenameLabelled))

        # check if acq was saved with only one  activated subject
        if acqStatic.GetPoint(0).GetLabel().count(":"):
            raise Exception("[pyCGM2] Your input static c3d was saved with two activate subject. Re-save it with only one before pyCGM2 calculation") 

        # ---relabel PIG output if processing previously---
        cgm.CGM.reLabelPigOutputs(acqStatic) 


        # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.ckeckActivatedSubject(NEXUS,subjects,"LASI")        
        Parameters = NEXUS.GetSubjectParamNames(subject)


        # --------------------pyCGM2 INPUT FILES ------------------------------
        if not os.path.isfile( DATA_PATH + subject+"-pyCGM2.inputs"):
            copyfile(str(pyCGM2.CONFIG.PYCGM2_SETTINGS_FOLDER+"pyCGM2.inputs"), str(DATA_PATH + subject+"-pyCGM2.inputs"))
            logging.warning("Copy of pyCGM2.inputs from pyCGM2 settings")
            inputs = json.loads(open(DATA_PATH +subject+'-pyCGM2.inputs').read())
        else:
            inputs = json.loads(open(DATA_PATH +subject+'-pyCGM2.inputs').read())


        # ---- configuration parameters ----
        flag_leftFlatFoot =  bool(inputs["Calibration"]["Left flat foot"])
        flag_rightFlatFoot =  bool(inputs["Calibration"]["Right flat foot"])
        markerDiameter = float(inputs["Calibration"]["Marker diameter"])
        pointSuffix = inputs["Calibration"]["Point suffix"]



        # --- mp data ----
        required_mp={
        'Bodymass'   : NEXUS.GetSubjectParamDetails( subject, "Bodymass")[0],#71.0,
        'LeftLegLength' : NEXUS.GetSubjectParamDetails( subject, "LeftLegLength")[0],#860.0,
        'RightLegLength' : NEXUS.GetSubjectParamDetails( subject, "RightLegLength")[0],#865.0 ,
        'LeftKneeWidth' : NEXUS.GetSubjectParamDetails( subject, "LeftKneeWidth")[0],#102.0,
        'RightKneeWidth' : NEXUS.GetSubjectParamDetails( subject, "RightKneeWidth")[0],#103.4,
        'LeftAnkleWidth' : NEXUS.GetSubjectParamDetails( subject, "LeftAnkleWidth")[0],#75.3,
        'RightAnkleWidth' : NEXUS.GetSubjectParamDetails( subject, "RightAnkleWidth")[0],#72.9,
        'LeftSoleDelta' : NEXUS.GetSubjectParamDetails( subject, "LeftSoleDelta")[0],#75.3,
        'RightSoleDelta' : NEXUS.GetSubjectParamDetails( subject, "RightSoleDelta")[0],#72.9,        
        }

        optional_mp={
        'InterAsisDistance'   : NEXUS.GetSubjectParamDetails( subject, "InterAsisDistance")[0],#0,
        'LeftAsisTrocanterDistance' : NEXUS.GetSubjectParamDetails( subject, "LeftAsisTrocanterDistance")[0],#0,
        'LeftTibialTorsion' : NEXUS.GetSubjectParamDetails( subject, "LeftTibialTorsion")[0],#0 ,
        'LeftThighRotation' : NEXUS.GetSubjectParamDetails( subject, "LeftThighRotation")[0],#0,
        'LeftShankRotation' : NEXUS.GetSubjectParamDetails( subject, "LeftShankRotation")[0],#0,
        'RightAsisTrocanterDistance' : NEXUS.GetSubjectParamDetails( subject, "RightAsisTrocanterDistance")[0],#0,
        'RightTibialTorsion' : NEXUS.GetSubjectParamDetails( subject, "RightTibialTorsion")[0],#0 ,
        'RightThighRotation' : NEXUS.GetSubjectParamDetails( subject, "RightThighRotation")[0],#0,
        'RightShankRotation' : NEXUS.GetSubjectParamDetails( subject, "RightShankRotation")[0],#0,
        }

        # --------------------------MODEL--------------------------------------
        # ---definition---
        model=cgm.CGM1LowerLimbs()
        model.configure()
        model.addAnthropoInputParameters(required_mp,optional=optional_mp)

        # ---check marker set used----
        staticMarkerConfiguration= checkCGM1_StaticMarkerConfig(acqStatic)


        # --------------------------STATIC CALBRATION--------------------------                
        scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure
        
        # ---initial calibration filter----
        # use if all optional mp are zero    
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = flag_leftFlatFoot, rightFlatFoot = flag_rightFlatFoot,
                                            markerDiameter=markerDiameter,
                                            ).compute()

        # ---- Decorators -----
        # Goal = modified calibration according the identified marker set or if offsets manually set   
 


        # initialisation of node label 
        useLeftKJCnodeLabel = "LKJC_chord"
        useLeftAJCnodeLabel = "LAJC_chord"
        useRightKJCnodeLabel = "RKJC_chord"
        useRightAJCnodeLabel = "RAJC_chord"

        # case 1 : NO kad, NO medial ankle BUT thighRotation different from zero ( mean manual modification or new calibration from a previous one )
        if not staticMarkerConfiguration["leftKadFlag"]  and not staticMarkerConfiguration["leftMedialAnkleFlag"] and not staticMarkerConfiguration["leftMedialKneeFlag"] and optional_mp["LeftThighRotation"] !=0:
            logging.warning("CASE FOUND ===> Left Side - CGM1 - Origine - manual offsets")            
            modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",optional_mp["LeftThighRotation"],markerDiameter,optional_mp["LeftTibialTorsion"],optional_mp["LeftShankRotation"])
            useLeftKJCnodeLabel = "LKJC_mo"
            useLeftAJCnodeLabel = "LAJC_mo"
        if not staticMarkerConfiguration["rightKadFlag"]  and not staticMarkerConfiguration["rightMedialAnkleFlag"] and not staticMarkerConfiguration["rightMedialKneeFlag"] and optional_mp["RightThighRotation"] !=0:
            logging.warning("CASE FOUND ===> Right Side - CGM1 - Origine - manual offsets")            
            modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",optional_mp["RightThighRotation"],markerDiameter,optional_mp["RightTibialTorsion"],optional_mp["RightShankRotation"])
            useRightKJCnodeLabel = "RKJC_mo"
            useRightAJCnodeLabel = "RAJC_mo"

        # case 2 : kad FOUND and NO medial Ankle 
        if staticMarkerConfiguration["leftKadFlag"]:
            logging.warning("CASE FOUND ===> Left Side - CGM1 - KAD variant")
            modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left", displayMarkers = False)
            useLeftKJCnodeLabel = "LKJC_kad"
            useLeftAJCnodeLabel = "LAJC_kad"
        if staticMarkerConfiguration["rightKadFlag"]:
            logging.warning("CASE FOUND ===> Right Side - CGM1 - KAD variant")
            modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right", displayMarkers = False)
            useRightKJCnodeLabel = "RKJC_kad"
            useRightAJCnodeLabel = "RAJC_kad"


        # case 3 : medial knee FOUND 
        # notice: cgm1Behaviour is enable mean effect on AJC location 
        if staticMarkerConfiguration["leftMedialKneeFlag"]:
            logging.warning("CASE FOUND ===> Left Side - CGM1 - medial knee ")
            modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="left",cgm1Behaviour=True)
            useLeftKJCnodeLabel = "LKJC_mid"
            useLeftAJCnodeLabel = "LAJC_midKnee"
        if staticMarkerConfiguration["rightMedialKneeFlag"]:
            logging.warning("CASE FOUND ===> Right Side - CGM1 - medial knee ")
            modelDecorator.KneeCalibrationDecorator(model).midCondyles(acqStatic, markerDiameter=markerDiameter, side="right",cgm1Behaviour=True)
            useRightKJCnodeLabel = "RKJC_mid"
            useRightAJCnodeLabel = "RAJC_midKnee"


        # case 4 : medial ankle FOUND 
        if staticMarkerConfiguration["leftMedialAnkleFlag"]:
            logging.warning("CASE FOUND ===> Left Side - CGM1 - medial ankle ")
            modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")
            useLeftAJCnodeLabel = "LAJC_mid"
        if staticMarkerConfiguration["rightMedialAnkleFlag"]:
            logging.warning("CASE FOUND ===> Right Side - CGM1 - medial ankle ")
            modelDecorator.AnkleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="right")
            useRightAJCnodeLabel = "RAJC_mid"

        # ----Final Calibration filter if model previously decorated ----- 
        if model.decoratedModel:
            # initial static filter
            modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                               useLeftKJCnode=useLeftKJCnodeLabel, useLeftAJCnode=useLeftAJCnodeLabel,
                               useRightKJCnode=useRightKJCnodeLabel, useRightAJCnode=useRightAJCnodeLabel,
                               markerDiameter=markerDiameter).compute()

        
        #----update subject mp----
        updateNexusSubjectMp(NEXUS,model,subject)



        # ----------------------CGM MODELLING----------------------------------
        # ----motion filter----
        # notice : viconCGM1compatible option duplicate error on Construction of the foot coordinate system         
    
        modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,pyCGM2Enums.motionMethod.Native,
                                                  markerDiameter=markerDiameter)

        modMotion.compute()


        #---- Joint kinematics----
        # relative angles
        modelFilters.ModelJCSFilter(model,acqStatic).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

        # detection of traveling axis
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionFromPoints(acqStatic,"SACR","midASIS","LPSI")

        # absolute angles
        modelFilters.ModelAbsoluteAnglesFilter(model,acqStatic,
                                               segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                                angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                                eulerSequences=["TOR","TOR", "ROT"],
                                                globalFrameOrientation = globalFrame,
                                                forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)

        

        # ----------------------SAVE-------------------------------------------
        modelFile = open(DATA_PATH + subject+"-pyCGM2.model", "w")
        cPickle.dump(model, modelFile)
        modelFile.close()

        # ----------------------DISPLAY ON VICON-------------------------------
        viconInterface.ViconInterface(NEXUS,
                                      model,acqStatic,subject,
                                      staticProcessing=True).run()

        # ========END of the nexus OPERATION if run from Nexus  =========


        if DEBUG:
            NEXUS.SaveTrial(30)

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
