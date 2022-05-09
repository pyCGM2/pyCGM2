# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/CGM1.1
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--

from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusUtils
from pyCGM2.Nexus import nexusTools
import warnings
import argparse
from pyCGM2.Lib.CGM import cgm1_1
from pyCGM2.Apps.ViconApps import CgmArgsManager
from pyCGM2.Utils import files
from viconnexusapi import ViconNexus
import pyCGM2
import os
import pyCGM2
LOGGER = pyCGM2.LOGGER
warnings.filterwarnings("ignore")



def main():

    parser = argparse.ArgumentParser(description='CGM1.1 Calibration')
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
                        help='force ggm1.1 as model output suffix')
    parser.add_argument('--resetMP', action='store_true',
                        help='reset optional anthropometric parameters')
    parser.add_argument('-ae', '--anomalyException',
                        action='store_true', help='raise an exception if an anomaly is detected')



    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False

    if NEXUS_PYTHON_CONNECTED:  # run Operation
        args = parser.parse_args()
        DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"
        LOGGER.logger.info("data Path: " + DATA_PATH)
        LOGGER.set_file_handler(DATA_PATH+"pyCGM2-Fitting.log")
        LOGGER.logger.info("calibration file: " + reconstructFilenameLabelled)

        # --------------------------GLOBAL SETTINGS ------------------------------------
        settings = files.loadModelSettings(DATA_PATH, "CGM1_1-pyCGM2.settings")

        # --------------------------CONFIG ------------------------------------
        argsManager = CgmArgsManager.argsManager_cgm1(settings, args)
        leftFlatFoot = argsManager.getLeftFlatFoot()
        rightFlatFoot = argsManager.getRightFlatFoot()
        headFlat = argsManager.getHeadFlat()
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm1_1")

        # --------------------------SUBJECT ------------------------------------
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)
        Parameters = NEXUS.GetSubjectParamNames(subject)

        required_mp, optional_mp = nexusUtils.getNexusSubjectMp(
            NEXUS, subject, resetFlag=args.resetMP)
        # -------------------------- INFOS ------------------------------------
        mpInfo, mpFilename = files.getMpFileContent(
            DATA_PATH, "mp.pyCGM2", subject)

        #  translators management
        translators = files.getTranslators(DATA_PATH, "CGM1_1.translators")
        if not translators:
            translators = settings["Translators"]

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(
            DATA_PATH, reconstructFilenameLabelledNoExt, subject)
        acq = nacf.build()

        # --------------------------MODELLING PROCESSING -----------------------
        model, acqStatic, detectAnomaly = cgm1_1.calibrate(DATA_PATH, reconstructFilenameLabelledNoExt, translators,
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

    else:
        return parser


if __name__ == "__main__":

    main()
