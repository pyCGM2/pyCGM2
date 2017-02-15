# -*- coding: utf-8 -*-
import os
import logging
import matplotlib.pyplot as plt
import json
import pdb
import cPickle
import json

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



# pyCGM2 libraries
from pyCGM2.Tools import btkTools
import pyCGM2.enums as pyCGM2Enums
from pyCGM2.Model.CGM2 import cgm, modelFilters, modelDecorator

from pyCGM2 import viconInterface

def updateNexusSubjectMp(NEXUS,model):
    th_l = model.getViconThighOffset("Left")
    sh_l = model.getViconShankOffset("Left")
    tt_l = model.getViconTibialTorsion("Left")

    th_r = model.getViconThighOffset("Right")
    sh_r = model.getViconShankOffset("Right")
    tt_r = model.getViconTibialTorsion("Right")

    spf_l,sro_l = model.getViconFootOffset("Left")
    spf_r,sro_r = model.getViconFootOffset("Right")

    abdAdd_l = model.getViconAnkleAbAddOffset("Left")
    abdAdd_r = model.getViconAnkleAbAddOffset("Right")


    NEXUS.SetSubjectParam( subject, "InterAsisDistance",model.mp_computed["InterAsisDistance"])
    NEXUS.SetSubjectParam( subject, "LeftAsisTrocanterDistance",model.mp_computed["LeftAsisTrocanterDistance"])
    NEXUS.SetSubjectParam( subject, "LeftThighRotation",th_l)
    NEXUS.SetSubjectParam( subject, "LeftShankRotation",sh_l)
    NEXUS.SetSubjectParam( subject, "LeftTibialTorsion",tt_l)


    NEXUS.SetSubjectParam( subject, "RightAsisTrocanterDistance",model.mp_computed["RightAsisTrocanterDistance"])
    NEXUS.SetSubjectParam( subject, "RightThighRotation",th_r)
    NEXUS.SetSubjectParam( subject, "RightShankRotation",sh_r)
    NEXUS.SetSubjectParam( subject, "RightTibialTorsion",tt_r)


    NEXUS.SetSubjectParam( subject, "LeftStaticPlantFlex",spf_l)
    NEXUS.SetSubjectParam( subject, "LeftStaticRotOff",sro_l)
    NEXUS.SetSubjectParam( subject, "LeftAnkleAbAdd",abdAdd_l)

    NEXUS.SetSubjectParam( subject, "RightStaticPlantFlex",spf_r)
    NEXUS.SetSubjectParam( subject, "RightStaticRotOff",sro_r)
    NEXUS.SetSubjectParam( subject, "RightAnkleAbAdd",abdAdd_r)


def checkCGM1_StaticMarkerConfig(acqStatic):

    out = dict()

    # medial ankle markers
    out["leftMedialAnkleFlag"] = True if btkTools.isPointExist(acqStatic,"LMED") else False
    out["rightMedialAnkleFlag"] = True if btkTools.isPointExist(acqStatic,"RMED") else False

    # medial ankle markers
    out["leftMedialKneeFlag"] = True if btkTools.isPointExist(acqStatic,"LMEPI") else False
    out["rightMedialKneeFlag"] = True if btkTools.isPointExist(acqStatic,"RMEPI") else False


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
            calibrateFilenameLabelledNoExt = "static Cal 01-noKAD-noAnkleMed" #"static Cal 01-noKAD-noAnkleMed" #
            da,pa = NEXUS.GetTrialName()
            NEXUS.OpenTrial( str(DATA_PATH+calibrateFilenameLabelledNoExt), 30 )

        else:
            DATA_PATH, calibrateFilenameLabelledNoExt = NEXUS.GetTrialName()

        calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ calibrateFilenameLabelled)


        # ---- pyCGM2 files ----
        if not os.path.isfile( DATA_PATH + "pyCGM2.inputs"):
            raise Exception ("pyCGM2.inputs file doesn't exist")
        else:
            inputs = json.loads(open(DATA_PATH +'pyCGM2.inputs').read())

        # ---- configuration parameters ----
        flag_leftFlatFoot =  bool(inputs["Calibration"]["Left flat foot"])
        flag_rightFlatFoot =  bool(inputs["Calibration"]["Right flat foot"])
        markerDiameter = float(inputs["Calibration"]["Marker diameter"])
        pointSuffix = inputs["Calibration"]["Point suffix"]


        # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject =   subjects[0]
        logging.info(  "Subject name : " + subject  )
        Parameters = NEXUS.GetSubjectParamNames(subject)


        # --- mp data ----
        required_mp={
        'Bodymass'   : NEXUS.GetSubjectParamDetails( subject, "Bodymass")[0],#71.0,
        'LeftLegLength' : NEXUS.GetSubjectParamDetails( subject, "LeftLegLength")[0],#860.0,
        'RightLegLength' : NEXUS.GetSubjectParamDetails( subject, "RightLegLength")[0],#865.0 ,
        'LeftKneeWidth' : NEXUS.GetSubjectParamDetails( subject, "LeftKneeWidth")[0],#102.0,
        'RightKneeWidth' : NEXUS.GetSubjectParamDetails( subject, "RightKneeWidth")[0],#103.4,
        'LeftAnkleWidth' : NEXUS.GetSubjectParamDetails( subject, "LeftAnkleWidth")[0],#75.3,
        'RightAnkleWidth' : NEXUS.GetSubjectParamDetails( subject, "RightAnkleWidth")[0],#72.9,
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

        # ---btk acquisition---
        acqStatic = btkTools.smartReader(str(DATA_PATH+calibrateFilenameLabelled))

        # ---relabel PIG output if processing previously---
        cgm.CGM.reLabelPigOutputs(acqStatic) 


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
        #   This 
        if not staticMarkerConfiguration["leftKadFlag"]  and not staticMarkerConfiguration["leftMedialAnkleFlag"] and not staticMarkerConfiguration["leftMedialKneeFlag"] and optional_mp["LeftThighRotation"] !=0:
            logging.info("Left Side - CGM1 - Origine - manual offsets")            
            modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"left",optional_mp["LeftThighRotation"],markerDiameter,optional_mp["LeftTibialTorsion"],optional_mp["LeftShankRotation"])
            useLeftKJCnodeLabel = "LKJC_mo"
            useLeftAJCnodeLabel = "LAJC_mo"
       

        if not staticMarkerConfiguration["rightKadFlag"]  and not staticMarkerConfiguration["rightMedialAnkleFlag"] and not staticMarkerConfiguration["rightMedialKneeFlag"] and optional_mp["RightThighRotation"] !=0:
            logging.info("Right Side - CGM1 - Origine - manual offsets")            
            modelDecorator.Cgm1ManualOffsets(model).compute(acqStatic,"right",optional_mp["RightThighRotation"],markerDiameter,optional_mp["RightTibialTorsion"],optional_mp["RightShankRotation"])
            useRightKJCnodeLabel = "RKJC_mo"
            useRightAJCnodeLabel = "RAJC_mo"

        # case 2 : kad FOUND and NO medial Ankle 
        if staticMarkerConfiguration["leftKadFlag"]:
            logging.info("Left Side - CGM1 - KAD variant")
            modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="left", displayMarkers = False)
            useLeftKJCnodeLabel = "LKJC_kad"
            useLeftAJCnodeLabel = "LAJC_kad"
        if staticMarkerConfiguration["rightKadFlag"]:
            logging.info("Right Side - CGM1 - KAD variant")
            modelDecorator.Kad(model,acqStatic).compute(markerDiameter=markerDiameter, side="right", displayMarkers = False)
            useRightKJCnodeLabel = "RKJC_kad"
            useRightAJCnodeLabel = "RAJC_kad"
        
        # case 3 : both kad and medial ankle FOUND 
        if staticMarkerConfiguration["leftKadFlag"]:
            if staticMarkerConfiguration["leftMedialAnkleFlag"]:
                logging.info("Left Side - CGM1 - KAD + medial ankle ")
                modelDecorator.AnkstaticConfigurationleCalibrationDecorator(model).midMaleolus(acqStatic, markerDiameter=markerDiameter, side="left")
                useLeftAJCnodeLabel = "LAJC_mid"

        if staticMarkerConfiguration["rightKadFlag"]:

            if staticMarkerConfiguration["rightMedialAnkleFlag"]:
                logging.info("Right Side - CGM1 - KAD + medial ankle ")
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
        updateNexusSubjectMp(NEXUS,model)



        # ----------------------CGM MODELLING----------------------------------
        # ----motion filter----
        # notice : viconCGM1compatible option duplicate error on Construction of the foot coordinate system         
    
        modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,pyCGM2Enums.motionMethod.Native,
                                                  markerDiameter=markerDiameter,
                                                  viconCGM1compatible=True)

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
                                                globalFrameOrientation = globalFrame,
                                                forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)

        

        # ----------------------SAVE-------------------------------------------
        modelFile = open(DATA_PATH + "pyCGM2.model", "w")
        cPickle.dump(model, modelFile)
        modelFile.close()

        # ----------------------DISPLAY ON VICON-------------------------------
        viconInterface.ViconInterface(NEXUS,
                                      model,acqStatic,subject,
                                      staticProcessing=True).run()

        # ========END of the nexus OPERATION if run from Nexus  =========


        if DEBUG:
            NEXUS.SaveTrial(30)

            # code below is similar to operation "nexusOperation_pyCGM2-CGM1-metadata.py"        
            # add metadata
            acqStatic2= btkTools.smartReader(str(DATA_PATH + calibrateFilenameLabelled))
            md_Model = btk.btkMetaData('MODEL') # create main metadata
            btk.btkMetaDataCreateChild(md_Model, "NAME", "CGM1")
            btk.btkMetaDataCreateChild(md_Model, "PROCESSOR", "pyCGM2")
            acqStatic2.GetMetaData().AppendChild(md_Model)
    
            # save
            btkTools.smartWriter(acqStatic2,str(DATA_PATH + calibrateFilenameLabelled[:-4] + ".c3d"))
            logging.info( "[pyCGM2] : file ( %s) reconstructed in pyCGM2-model path " % (calibrateFilenameLabelled))


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
