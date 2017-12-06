# -*- coding: utf-8 -*-
#import ipdb
import logging
import argparse
import matplotlib.pyplot as plt

# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)


# vicon nexus
import ViconNexus


# pyCGM2 libraries
from pyCGM2.Model.CGM2.coreApps import cgmUtils, cgm2_3
from pyCGM2.Utils import files
from pyCGM2.Nexus import nexusFilters, nexusUtils,nexusTools


if __name__ == "__main__":

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM2.3 Calibration')
    parser.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    parser.add_argument('--resetMP', action='store_false', help='reset optional mass parameters')

    parser.add_argument('--DEBUG', action='store_true', help='debug model. load file into nexus externally')

    args = parser.parse_args()


    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------GLOBAL SETTINGS ------------------------------
        # ( in user/AppData)
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_3-pyCGM2.settings")

        # --------------------------CONFIG ------------------------------------
        argsManager = cgmUtils.argsManager_cgm(settings,args)
        leftFlatFoot = argsManager.getLeftFlatFoot()
        rightFlatFoot = argsManager.getRightFlatFoot()
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm2.3")

        hjcMethod = settings["Calibration"]["HJC regression"]
        ik_flag = False if args.noIk else True

        # --------------------------LOADING------------------------------

        # --- acquisition file and path----
        if args.DEBUG:

            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.3\\medial\\"
            calibrateFilenameLabelledNoExt = "static"

            NEXUS.OpenTrial( str(DATA_PATH+calibrateFilenameLabelledNoExt), 30 )
            ik_flag=True

        else:
            DATA_PATH, calibrateFilenameLabelledNoExt = NEXUS.GetTrialName()

        calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ calibrateFilenameLabelled)


        # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.checkActivatedSubject(NEXUS,subjects)
        Parameters = NEXUS.GetSubjectParamNames(subject)

        required_mp,optional_mp = nexusUtils.getNexusSubjectMp(NEXUS,subject,resetFlag=args.resetMP)


        # -------------------------- INFOS ------------------------------------
        mpInfo,mpFilename = files.getJsonFileContent(DATA_PATH,"mp.pyCGM2",subject)

        #  translators management
        translators = files.getTranslators(DATA_PATH,"CGM2_3.translators")
        if not translators:  translators = settings["Translators"]

        # --------------------------MODELLING PROCESSING -----------------------
        model,finalAcqStatic = cgm2_3.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                              required_mp,optional_mp,
                              ik_flag,leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethod,
                              pointSuffix)


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


        if args.DEBUG:
            NEXUS.SaveTrial(30)

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
