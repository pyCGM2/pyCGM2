# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/CGM1
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--

from pyCGM2.Nexus import nexusFilters, nexusTools, nexusUtils
import warnings
import argparse
from pyCGM2.Utils import files
from pyCGM2.Lib.CGM import cgm1
from pyCGM2.Configurator import CgmArgsManager
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
    """ run the CGM1 fitting operation from Nexus

    Usage:

    ```bash
        nexus_CGM1_Fitting.exe -fi  100 -fe 150 --md 24 -ps "withSuffix"
        nexus_CGM1_Fitting.exe --frameInit  100 --frameEnd 150 --markerDiameter 24 --pointSuffix "withSuffix"
    ```

    Args:
        [--proj] (str) : segmental coordinate system to project the joint moment (Choice : Distal, Proximal, Global')
        [-md, --markerDiameter] (int) : marker diameter
        [-ps, --pointSuffix] (str) : suffix of the model ouputs
        [--check] (bool) :force _cgm1  as model output suffix
        [-ae,--anomalyException] (bool) : return exception if one anomaly detected ')
        [-fi,--frameInit] (int) : first frame to process
        [-fe,--frameEnd] (int) : last frame to process

    Note:
        Marker diameter is used for locating joint centre from an origin ( eg LKNE) by an offset along a direction.
        respect the same marker diameter for the following markers :
        L(R)KNE - L(R)ANK - L(R)ASI - L(R)PSI

    """

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    parser = argparse.ArgumentParser(description='CGM1 Fitting')
    parser.add_argument(
        '--proj', type=str, help='Moment Projection. Choice : Distal, Proximal, Global')
    parser.add_argument('-md', '--markerDiameter',
                        type=float, help='marker diameter')
    parser.add_argument('-ps', '--pointSuffix', type=str,
                        help='suffix of model outputs')
    parser.add_argument('--check', action='store_true',
                        help='force model output suffix')
    parser.add_argument('-ae', '--anomalyException',
                        action='store_true', help='stop if anomaly detected ')
    parser.add_argument('-fi', '--frameInit', type=int,
                        help='first frame to process')
    parser.add_argument('-fe', '--frameEnd', type=int,
                        help='last frame to process')

    args = parser.parse_args()

    if NEXUS_PYTHON_CONNECTED:  # run Operation
        DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"
        LOGGER.logger.info("data Path: " + DATA_PATH)
        LOGGER.set_file_handler(DATA_PATH+"pyCGM2-Fitting.log")
        LOGGER.logger.info("calibration file: " + reconstructFilenameLabelled)

        # --------------------------GLOBAL SETTINGS ------------------------------------
        settings = files.loadModelSettings(DATA_PATH, "CGM1-pyCGM2.settings")

        # --------------------------CONFIG ------------------------------------
        argsManager = CgmArgsManager.argsManager_cgm1(settings, args)
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm1")
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
        if model.version != "CGM1.0":
            raise Exception(
                "%s-pyCGM2.model file was not calibrated from the CGM1.0 calibration pipeline" % model.version)

        # --------------------------SESSION INFOS ------------------------------------

        #  translators management
        translators = files.getTranslators(DATA_PATH, "CGM1.translators")
        if not translators:
            translators = settings["Translators"]

        #force plate assignement from Nexus
        mfpa = nexusTools.getForcePlateAssignment(NEXUS)

        # btkAcq builder
        nacf = nexusFilters.NexusConstructAcquisitionFilter(
            DATA_PATH, reconstructFilenameLabelledNoExt, subject)
        acq = nacf.build()
        # --------------------------MODELLING PROCESSING -----------------------
        acqGait, detectAnomaly = cgm1.fitting(model, DATA_PATH, reconstructFilenameLabelled,
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

    else:
        raise Exception("NO Nexus connection. Turn on Nexus")


if __name__ == "__main__":

    main()
