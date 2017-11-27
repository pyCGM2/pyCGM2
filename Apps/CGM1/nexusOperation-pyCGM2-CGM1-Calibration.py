# -*- coding: utf-8 -*-
#import ipdb
import logging
import matplotlib.pyplot as plt
import argparse


# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# vicon nexus
import ViconNexus

# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2 import enums

from pyCGM2.Model import modelFilters, modelDecorator
from pyCGM2.Model.CGM2 import cgm
from pyCGM2.apps import cgmUtils

from pyCGM2.Utils import files
from pyCGM2.Nexus import nexusFilters, nexusUtils,nexusTools



if __name__ == "__main__":

    plt.close("all")
    DEBUG = False

    parser = argparse.ArgumentParser(description='CGM1 Calibration')
    parser.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix' )
    parser.add_argument('--resetMP', action='store_false', help='reset optional mass parameters')
    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------------GLOBAL SETTINGS ------------------------------------
        # global setting ( in user/AppData)
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM1-pyCGM2.settings")

        # --------------------------LOADING ------------------------------------
        if DEBUG:
            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\CGM1\\kad-med\\"
            calibrateFilenameLabelledNoExt = "Static Cal 01-onlyLeft" #"static Cal 01-noKAD-noAnkleMed" #
            NEXUS.OpenTrial( str(DATA_PATH+calibrateFilenameLabelledNoExt), 30 )

        else:
            DATA_PATH, calibrateFilenameLabelledNoExt = NEXUS.GetTrialName()

        calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ calibrateFilenameLabelled)

        # --------------------------SUBJECT ------------------------------------
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.checkActivatedSubject(NEXUS,subjects)
        Parameters = NEXUS.GetSubjectParamNames(subject)

        required_mp,optional_mp = nexusUtils.getNexusSubjectMp(NEXUS,subject,resetFlag=args.resetMP)

        # --------------------------SESSION INFOS ------------------------------------
        # info file
        info = files.manage_pycgm2SessionInfos(DATA_PATH,subject)

        #  translators management
        translators = files.manage_pycgm2Translators(DATA_PATH,"CGM1.translators")
        if not translators:
           translators = settings["Translators"]

        # --------------------------CONFIG ------------------------------------
        argsManager = cgmUtils.argsManager_cgm1(settings,args)
        leftFlatFoot = argsManager.getLeftFlatFoot()
        rightFlatFoot = argsManager.getRightFlatFoot()
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm1")


        # --------------------------ACQUISITION ------------------------------------

        # ---btk acquisition---
        acqStatic = btkTools.smartReader(str(DATA_PATH+calibrateFilenameLabelled))
        btkTools.checkMultipleSubject(acqStatic)

        acqStatic =  btkTools.applyTranslators(acqStatic,translators)

        # ---definition---
        model=cgm.CGM1LowerLimbs()
        model.configure()
        model.addAnthropoInputParameters(required_mp,optional=optional_mp)

        # --store calibration parameters--
        model.setStaticFilename(calibrateFilenameLabelled)
        model.setCalibrationProperty("LeftFlatFoot",leftFlatFoot)
        model.setCalibrationProperty("rightFlatFoot",rightFlatFoot)
        model.setCalibrationProperty("markerDiameter",markerDiameter)



        # ---check marker set used----
        smc= cgm.CGM.checkCGM1_StaticMarkerConfig(acqStatic)
        # --------------------------STATIC CALBRATION--------------------------
        scp=modelFilters.StaticCalibrationProcedure(model) # load calibration procedure

        # ---initial calibration filter----
        modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                                            leftFlatFoot = leftFlatFoot,
                                            rightFlatFoot = rightFlatFoot,
                                            markerDiameter = markerDiameter,
                                            ).compute()
        # ---- Decorators -----
        cgmUtils.applyDecorators_CGM1(smc, model,acqStatic,optional_mp,markerDiameter)
        pigStaticMarkers = cgmUtils.get_markerLabelForPiGStatic(smc)

        # ----Final Calibration filter if model previously decorated -----
        if model.decoratedModel:
            # initial static filter
            modelFilters.ModelCalibrationFilter(scp,acqStatic,model,
                               leftFlatFoot = leftFlatFoot, rightFlatFoot = rightFlatFoot,
                               markerDiameter=markerDiameter).compute()


        # ----------------------CGM MODELLING----------------------------------
        # ----motion filter----
        # notice : viconCGM1compatible option duplicate error on Construction of the foot coordinate system

        modMotion=modelFilters.ModelMotionFilter(scp,acqStatic,model,enums.motionMethod.Determinist,
                                                  markerDiameter=markerDiameter,
                                                  viconCGM1compatible=False,
                                                  pigStatic=True,
                                                  useLeftKJCmarker=pigStaticMarkers[0], useLeftAJCmarker=pigStaticMarkers[1],
                                                  useRightKJCmarker=pigStaticMarkers[2], useRightAJCmarker=pigStaticMarkers[3])
        modMotion.compute()


        #---- Joint kinematics----
        # relative angles
        modelFilters.ModelJCSFilter(model,acqStatic).compute(description="vectoriel", pointLabelSuffix=pointSuffix)

        # detection of traveling axis
        longitudinalAxis,forwardProgression,globalFrame = btkTools.findProgressionAxisFromPelvicMarkers(acqStatic,["LASI","RASI","RPSI","LPSI"])


        # absolute angles
        modelFilters.ModelAbsoluteAnglesFilter(model,acqStatic,
                                               segmentLabels=["Left Foot","Right Foot","Pelvis"],
                                                angleLabels=["LFootProgress", "RFootProgress","Pelvis"],
                                                eulerSequences=["TOR","TOR", "TOR"],
                                                globalFrameOrientation = globalFrame,
                                                forwardProgression = forwardProgression).compute(pointLabelSuffix=pointSuffix)

        # ----------------------SAVE-------------------------------------------
        #pyCGM2.model
        files.saveModel(model,DATA_PATH,subject)

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusUtils.updateNexusSubjectMp(NEXUS,model,subject)
        nexusFilters.NexusModelFilter(NEXUS,
                                      model,acqStatic,subject,
                                      pointSuffix,
                                      staticProcessing=True).run()

        # ========END of the nexus OPERATION if run from Nexus  =========

        if DEBUG:
            NEXUS.SaveTrial(30)
    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
