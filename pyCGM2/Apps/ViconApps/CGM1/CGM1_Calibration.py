# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/CGM1
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--
from pyCGM2.Nexus import nexusFilters, nexusUtils, nexusTools
import warnings
import argparse
from pyCGM2.Lib.CGM import cgm1
from pyCGM2.Configurator import CgmArgsManager
from pyCGM2.Utils import files
from viconnexusapi import ViconNexus
import pyCGM2
import os
import pyCGM2
LOGGER = pyCGM2.LOGGER
warnings.filterwarnings("ignore")

# pyCGM2 settings

# vicon nexus

# pyCGM2 libraries


def main():
    """ run the CGM1 calibration operation from Nexus

    Usage:

    ```bash
        nexus_CGM1_Calibration.exe -l  1 --md 24 -ps "withSuffix"
        nexus_CGM1_Calibration.exe --leftFlatFoot  1 --markerDiameter 24 --pointSuffix "withSuffix"
    ```

    Args:
        [-l, --leftFlatFoot] (int) : set the left longitudinal foot axis parallel to the ground
        [-r, --rightFlatFoot] (int) : set the right longitudinal foot axis parallel to the ground
        [-hf, --headFlat] (int) : set the  longitudinal head axis parallel to the ground
        [-md, --markerDiameter] (int) : marker diameter
        [-ps, --pointSuffix] (str) : suffix of the model ouputs
        [--check] (bool) :force _cgm1  as model output suffix
        [--resetMP] (bool) : reset optional mass parameters
        [-ae,--anomalyException] (bool) : return exception if one anomaly detected ')

    Note:
        Marker diameter is used for locating joint centre from an origin ( eg LKNE) by an offset along a direction.
        respect the same marker diameter for the following markers :
        L(R)KNE - L(R)ANK - L(R)ASI - L(R)PSI

    """

    parser = argparse.ArgumentParser(description='CGM1 Calibration')
    parser.add_argument('-l', '--leftFlatFoot', type=int,
                        help='left flat foot option')
    parser.add_argument('-r', '--rightFlatFoot', type=int,
                        help='right flat foot option')
    parser.add_argument('-hf', '--headFlat', type=int,
                        help='head flat option')
    parser.add_argument('-md', '--markerDiameter',
                        type=float, help='marker diameter')
    parser.add_argument('-ps', '--pointSuffix', type=str,
                        help='suffix of model outputs')
    parser.add_argument('--check', action='store_true',
                        help='force model output suffix')
    parser.add_argument('--resetMP', action='store_true',
                        help='reset optional mass parameters')
    parser.add_argument('-ae', '--anomalyException',
                        action='store_true', help='stop if anomaly detected ')

    args = parser.parse_args()

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    if NEXUS_PYTHON_CONNECTED:  # run Operation

        DATA_PATH, calibrateFilenameLabelledNoExt = NEXUS.GetTrialName()

        calibrateFilenameLabelled = calibrateFilenameLabelledNoExt+".c3d"

        LOGGER.logger.info("data Path: " + DATA_PATH)
        LOGGER.set_file_handler(DATA_PATH+"pyCGM2-Calibration.log")
        LOGGER.logger.info("calibration file: " + calibrateFilenameLabelled)

        # --------------------------GLOBAL SETTINGS ------------------------------------
        settings = files.loadModelSettings(DATA_PATH, "CGM1-pyCGM2.settings")

        # --------------------------CONFIG ------------------------------------
        argsManager = CgmArgsManager.argsManager_cgm1(settings, args)
        leftFlatFoot = argsManager.getLeftFlatFoot()
        rightFlatFoot = argsManager.getRightFlatFoot()
        headFlat = argsManager.getHeadFlat()
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm1")

        # --------------------------SUBJECT ------------------------------------
        subjects = NEXUS.GetSubjectNames()
        # checkActivatedSubject(NEXUS,subjects)
        subject = nexusTools.getActiveSubject(NEXUS)
        Parameters = NEXUS.GetSubjectParamNames(subject)

        required_mp, optional_mp = nexusUtils.getNexusSubjectMp(
            NEXUS, subject, resetFlag=args.resetMP)

        # -------------------------- INFOS ------------------------------------
        mpInfo, mpFilename = files.getMpFileContent(
            DATA_PATH, "mp.pyCGM2", subject)

        #  translators management
        translators = files.getTranslators(DATA_PATH, "CGM1.translators")
        if not translators:
            translators = settings["Translators"]

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(
            DATA_PATH, calibrateFilenameLabelledNoExt, subject)
        acq = nacf.build()

        # --------------------------MODELLING PROCESSING -----------------------
        model, acqStatic, detectAnomaly = cgm1.calibrate(DATA_PATH, calibrateFilenameLabelled, translators,
                                                         required_mp, optional_mp,
                                                         leftFlatFoot, rightFlatFoot, headFlat, markerDiameter,
                                                         pointSuffix, forceBtkAcq=acq, anomalyException=args.anomalyException)

        # ----------------------SAVE-------------------------------------------
        #pyCGM2.model
        files.saveModel(model, DATA_PATH, subject)

        # save mp
        files.saveMp(mpInfo, model, DATA_PATH, mpFilename)

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusUtils.updateNexusSubjectMp(NEXUS, model, subject)
        nexusFilters.NexusModelFilter(NEXUS,
                                      model, acqStatic, subject,
                                      pointSuffix,
                                      staticProcessing=True).run()

        # ========END of the nexus OPERATION if run from Nexus  =========


if __name__ == "__main__":

    main()
