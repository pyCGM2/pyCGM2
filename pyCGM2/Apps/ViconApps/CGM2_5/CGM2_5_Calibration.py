# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/CGM2.5
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--
import os
import pyCGM2; LOGGER = pyCGM2.LOGGER
import argparse
import warnings
warnings.filterwarnings("ignore")

# pyCGM2 settings
import pyCGM2



# vicon nexus
from viconnexusapi import ViconNexus


# pyCGM2 libraries
from pyCGM2.Utils import files
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusUtils
from pyCGM2.Nexus import nexusTools
from pyCGM2.Apps.ViconApps import CgmArgsManager
from pyCGM2.Lib.CGM import  cgm2_5



def main():


    parser = argparse.ArgumentParser(description='CGM2.5 Calibration')
    parser.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser.add_argument('-hf','--headFlat',type=int,  help='head flat option')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    parser.add_argument('--resetMP', action='store_true', help='reset optional anthropometric parameters')
    parser.add_argument('--forceLHJC', nargs='+')
    parser.add_argument('--forceRHJC', nargs='+')
    parser.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')


    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False


    if NEXUS_PYTHON_CONNECTED: # run Operation
        args = parser.parse_args()
        # --------------------GLOBAL SETTINGS ------------------------------

        # --------------------------LOADING ------------------------------------
        DATA_PATH, calibrateFilenameLabelledNoExt = NEXUS.GetTrialName()

        calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"

        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.set_file_handler(DATA_PATH+"pyCGM2-Calibration.log")
        LOGGER.logger.info( "calibration file: "+ calibrateFilenameLabelled)

        # --------------------------GLOBAL SETTINGS ------------------------------------
        settings = files.loadModelSettings(DATA_PATH,"CGM2_5-pyCGM2.settings")


        argsManager = CgmArgsManager.argsManager_cgm(settings,args)
        leftFlatFoot = argsManager.getLeftFlatFoot()
        rightFlatFoot = argsManager.getRightFlatFoot()
        headFlat = argsManager.getHeadFlat()
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm2.5")
        ik_flag = argsManager.enableIKflag()

        hjcMethod = settings["Calibration"]["HJC"]
        lhjc = argsManager.forceHjc("left")
        rhjc = argsManager.forceHjc("right")
        if  lhjc is not None:
            hjcMethod["Left"] = lhjc
        if  rhjc is not None:
            hjcMethod["Right"] = rhjc



        # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)
        Parameters = NEXUS.GetSubjectParamNames(subject)

        required_mp,optional_mp = nexusUtils.getNexusSubjectMp(NEXUS,subject,resetFlag=args.resetMP)

        # --------------------------SESSION INFOS -----------------------------
         # --------------------------SESSIONS INFOS -----------------------------------
        mpInfo,mpFilename = files.getMpFileContent(DATA_PATH,"mp.pyCGM2",subject)

        #  translators management
        translators = files.getTranslators(DATA_PATH,"CGM2_5.translators")
        if not translators:  translators = settings["Translators"]

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,calibrateFilenameLabelledNoExt,subject)
        acq = nacf.build()

        # --------------------------CONFIG ------------------------------------
        model,finalAcqStatic,detectAnomaly = cgm2_5.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                      required_mp,optional_mp,
                      ik_flag,leftFlatFoot,rightFlatFoot,headFlat,
                      markerDiameter,
                      hjcMethod,
                      pointSuffix,forceBtkAcq=acq, anomalyException=args.anomalyException)



        # ----------------------SAVE-------------------------------------------
        files.saveModel(model,DATA_PATH,subject)

        # save mp
        files.saveMp(mpInfo,model,DATA_PATH,mpFilename)

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusUtils.updateNexusSubjectMp(NEXUS,model,subject)
        nexusFilters.NexusModelFilter(NEXUS,
                                      model,finalAcqStatic,subject,
                                      pointSuffix,
                                      staticProcessing=True).run()

        # ========END of the nexus OPERATION if run from Nexus  =========



    else:
        return parser

if __name__ == "__main__":

    main()
