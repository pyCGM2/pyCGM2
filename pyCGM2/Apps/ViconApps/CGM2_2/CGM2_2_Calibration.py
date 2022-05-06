# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/CGM2.2
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--
import os
import traceback
import argparse
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# pyCGM2 settings
import pyCGM2
LOGGER = pyCGM2.LOGGER

# from pyCGM2 import log; #log.setloggerLevel(LOGGER.logger.INFO)
# logger = log.get_logger(__name__)

# vicon nexus
from viconnexusapi import ViconNexus


# pyCGM2 libraries
from pyCGM2.Utils import files
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusUtils
from pyCGM2.Nexus import nexusTools
from pyCGM2.Apps.ViconApps import CgmArgsManager
from pyCGM2.Lib.CGM import  cgm2_2


def main():
    """ run the CGM2.2 calibration operation from Nexus.

    Usage:

    ```bash
        nexus_CGM22_Calibration.exe -l  1 --md 24 -ps "withSuffix"
        nexus_CGM22_Calibration.exe --leftFlatFoot  1 --markerDiameter 24 --pointSuffix "withSuffix"
        nexus_CGM22_Calibration.exe --forceLHJC 0.3 0.25 1.2
        nexus_CGM22_Calibration.exe --noIk
    ```

    Args:
        [-l, --leftFlatFoot] (int) : set the left longitudinal foot axis parallel to the ground
        [-r, --rightFlatFoot] (int) : set the right longitudinal foot axis parallel to the ground
        [-hf, --headFlat] (int) : set the  longitudinal head axis parallel to the ground
        [-md, --markerDiameter] (int) : marker diameter
        [-ps, --pointSuffix] (str) : suffix of the model ouputs
        [--check] (bool) :force _cgm1  as model output suffix
        [--resetMP] (bool) : reset optional mass parameters
        [--forceLHJC] (array) : force the left hip joint centre position in the pelvic coordinate system
        [--forceRHJC] (array) : force the right hip joint centre position in the pelvic coordinate system
        [-ae,--anomalyException] (bool) : return exception if one anomaly detected ')
        [--noIk] (bool): disable inverse kinematics

    Note:
        Marker diameter is used for locating joint centre from an origin ( eg LKNE) by an offset along a direction.
        respect the same marker diameter for the following markers :
        L(R)KNE - L(R)ANK - L(R)ASI - L(R)PSI

    """

    parser = argparse.ArgumentParser(description='CGM2.2 Calibration')
    parser.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser.add_argument('-hf','--headFlat',type=int,  help='head flat option')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    parser.add_argument('--forceLHJC', nargs='+')
    parser.add_argument('--forceRHJC', nargs='+')
    parser.add_argument('--resetMP', action='store_true', help='reset optional mass parameters')
    parser.add_argument('-ae','--anomalyException', action='store_true', help='stop if anomaly detected ')
    args = parser.parse_args()


    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()


    if NEXUS_PYTHON_CONNECTED: # run Operation


        # --------------------------LOADING ------------------------------------
        DATA_PATH, calibrateFilenameLabelledNoExt = NEXUS.GetTrialName()

        calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"

        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.set_file_handler(DATA_PATH+"pyCGM2-Calibration.log")
        LOGGER.logger.info( "calibration file: "+ calibrateFilenameLabelled)

        # --------------------------GLOBAL SETTINGS ------------------------------------
        settings = files.loadModelSettings(DATA_PATH,"CGM2_2-pyCGM2.settings")


        # --------------------------CONFIG ------------------------------------
        argsManager = CgmArgsManager.argsManager_cgm(settings,args)
        leftFlatFoot = argsManager.getLeftFlatFoot()
        rightFlatFoot = argsManager.getRightFlatFoot()
        headFlat = argsManager.getHeadFlat()
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm2.2")

        hjcMethod = settings["Calibration"]["HJC"]
        lhjc = argsManager.forceHjc("left")
        rhjc = argsManager.forceHjc("right")
        if  lhjc is not None:
            hjcMethod["Left"] = lhjc
        if  rhjc is not None:
            hjcMethod["Right"] = rhjc


        ik_flag = False if args.noIk else True


        # --------------------------SUBJECT -----------------------------------

        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)
        Parameters = NEXUS.GetSubjectParamNames(subject)

        required_mp,optional_mp = nexusUtils.getNexusSubjectMp(NEXUS,subject,resetFlag=args.resetMP)

        # --------------------------SESSIONS INFOS -----------------------------------
        mpInfo,mpFilename = files.getMpFileContent(DATA_PATH,"mp.pyCGM2",subject)

        #  translators management
        translators = files.getTranslators(DATA_PATH,"CGM2_2.translators")
        if not translators:  translators = settings["Translators"]

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,calibrateFilenameLabelledNoExt,subject)
        acq = nacf.build()

        model,finalAcqStatic,detectAnomaly = cgm2_2.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                      required_mp,optional_mp,
                      ik_flag,leftFlatFoot,rightFlatFoot,headFlat,
                      markerDiameter,
                      hjcMethod,
                      pointSuffix,forceBtkAcq=acq,
                      anomalyException=args.anomalyException)



        # ----------------------SAVE-------------------------------------------
        #pyCGM2.model
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
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":

    main()
