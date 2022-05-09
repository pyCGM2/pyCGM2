# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/CGM1.1
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--

import warnings
import argparse
from pyCGM2.Nexus import nexusFilters
from pyCGM2.Nexus import nexusUtils
from pyCGM2.Nexus import nexusTools
from pyCGM2.Utils import files
from pyCGM2.Lib.CGM import cgm1_1
from pyCGM2.Apps.ViconApps import CgmArgsManager
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

    parser = argparse.ArgumentParser(description='CGM1.1 Fitting')
    parser.add_argument(
        '--proj', type=str, help='Referential to project joint moment. Choice : Distal, Proximal, Global')
    parser.add_argument('-md', '--markerDiameter',
                        type=float, help='marker diameter')
    parser.add_argument('-ps', '--pointSuffix', type=str,
                        help='suffix of model outputs')
    parser.add_argument('--check', action='store_true',
                        help='force model output suffix')
    parser.add_argument('-ae', '--anomalyException',
                        action='store_true', help='raise an exception if an anomaly is detected')
    parser.add_argument('-fi', '--frameInit', type=int,
                        help='first frame to process')
    parser.add_argument('-fe', '--frameEnd', type=int,
                        help='last frame to process')

    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False


    if NEXUS_PYTHON_CONNECTED: # run Operation
        args = parser.parse_args()
        DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"
        LOGGER.logger.info("data Path: " + DATA_PATH)
        LOGGER.set_file_handler(DATA_PATH+"pyCGM2-Fitting.log")
        LOGGER.logger.info("calibration file: " + reconstructFilenameLabelled)

        # --------------------------GLOBAL SETTINGS ------------------------------------
        settings = files.loadModelSettings(DATA_PATH, "CGM1_1-pyCGM2.settings")

        # --------------------------CONFIG ------------------------------------
        argsManager = CgmArgsManager.argsManager_cgm(settings, args)
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm1_1")
        momentProjection = argsManager.getMomentProjection()

        # --------------------------SUBJECT ------------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)
        LOGGER.logger.info("Subject name : " + subject)

        # --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH, subject)

        # -------------------------- MP ------------------------------------
        # allow alteration of thigh offset
        model.mp_computed["LeftThighRotationOffset"] = NEXUS.GetSubjectParamDetails(
            subject, "LeftThighRotation")[0]
        model.mp_computed["RightThighRotationOffset"] = NEXUS.GetSubjectParamDetails(
            subject, "RightThighRotation")[0]

        # --------------------------CHECKING -----------------------------------
        # check model is the CGM1
        LOGGER.logger.info("loaded model : %s" % (model.version))
        if model.version != "CGM1.1":
            raise Exception(
                "%s-pyCGM2.model file was not calibrated from the CGM1.1 calibration pipeline" % model.version)

        # --------------------------SESSION INFOS ------------------------------------

        #  translators management
        translators = files.getTranslators(DATA_PATH, "CGM1_1.translators")
        if not translators:
            translators = settings["Translators"]

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(
            DATA_PATH, reconstructFilenameLabelledNoExt, subject)
        acq = nacf.build()

        #force plate assignement from Nexus
        mfpa = nexusTools.getForcePlateAssignment(NEXUS)

        # --------------------------MODELLING PROCESSING -----------------------
        acqGait, detectAnomaly = cgm1_1.fitting(model, DATA_PATH, reconstructFilenameLabelled,
                                                translators,
                                                markerDiameter,
                                                pointSuffix,
                                                mfpa, momentProjection,
                                                forceBtkAcq=acq,
                                                anomalyException=args.anomalyException,
                                                frameInit=args.frameInit, frameEnd=args.frameEnd)

        # ----------------------SAVE-------------------------------------------
        # Todo: pyCGM2 model :  cpickle doesn t work. Incompatibility with Swig. ( see about BTK wrench)

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusFilters.NexusModelFilter(
            NEXUS, model, acqGait, subject, pointSuffix).run()
        nexusTools.createGeneralEvents(
            NEXUS, subject, acqGait, ["Left-FP", "Right-FP"])

        # ========END of the nexus OPERATION if run from Nexus  =========

    else:
        return parser


if __name__ == "__main__":

    main()
