# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/CGM1.1
#APIDOC["Import"]=False
#APIDOC["Draft"]=False
#--end--

import warnings
import argparse
from pyCGM2.Utils import files
from pyCGM2.Lib.CGM import cgm1_1
from pyCGM2.Apps.ViconApps import CgmArgsManager
import pyCGM2
import os
import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Tools import btkTools

warnings.filterwarnings("ignore")

# pyCGM2 settings


# vicon nexus

# pyCGM2 libraries


def main(args=None):

    if args is None:
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
        parser.add_argument('--offline', nargs= 3, help=' subject name - dynamic c3d file - mfpa', required=False)

        args = parser.parse_args()

    NEXUS_PYTHON_CONNECTED = False
    OFFLINE_MODE = False if args.offline is None else True

    if not OFFLINE_MODE:
        try:
            from viconnexusapi import ViconNexus
            from pyCGM2.Nexus import nexusFilters
            from pyCGM2.Nexus import nexusUtils
            from pyCGM2.Nexus import nexusTools
            NEXUS = ViconNexus.ViconNexus()
            NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
        except:
            LOGGER.logger.error("Vicon nexus not connected")
    else:
        LOGGER.logger.info("You are working in offlinemode")


    if NEXUS_PYTHON_CONNECTED or OFFLINE_MODE: # run Operation

        # --------------------------LOADING ------------------------------------
        if NEXUS_PYTHON_CONNECTED:        
            DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()
            reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"
        else:
            DATA_PATH = os.getcwd()+"\\"
            reconstructFilenameLabelled = args.offline[1]
            if not os.path.exists(DATA_PATH+reconstructFilenameLabelled):
                raise Exception("[pyCGM2]  file [%s] not found in the folder"%(reconstructFilenameLabelled))
            
        LOGGER.logger.info("data Path: " + DATA_PATH)
        LOGGER.set_file_handler(DATA_PATH+"pyCGM2-Fitting.log")
        LOGGER.logger.info("Fitting file: " + reconstructFilenameLabelled)

        # --------------------------GLOBAL SETTINGS ------------------------------------
        settings = files.loadModelSettings(DATA_PATH, "CGM1_1-pyCGM2.settings")

        # --------------------------CONFIG ------------------------------------
        argsManager = CgmArgsManager.argsManager_cgm(settings, args)
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm1_1")
        momentProjection = argsManager.getMomentProjection()

        # --------------------------SUBJECT ------------------------------------
        if NEXUS_PYTHON_CONNECTED:
            subjects = NEXUS.GetSubjectNames()
            subject = nexusTools.getActiveSubject(NEXUS)
            LOGGER.logger.info(  "Subject name : " + subject  )
        else:
            subject = args.offline[0]
            if not os.path.exists(DATA_PATH+subject+"-mp.pyCGM2"):
                raise Exception("[pyCGM2]  the mp file [%s] not found in the folder"%(subject+"-mp.pyCGM2"))
            
            mpFilename = subject+"-mp.pyCGM2"
            mpInfo = files.openFile(DATA_PATH,mpFilename)
            
            required_mp = mpInfo["MP"]["Required"].copy()
            optional_mp = mpInfo["MP"]["Optional"].copy()

        # --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH, subject)

        # -------------------------- MP ------------------------------------
        # allow alteration of thigh offset
        if NEXUS_PYTHON_CONNECTED:
            model.mp_computed["LeftThighRotationOffset"] =   NEXUS.GetSubjectParamDetails( subject, "LeftThighRotation")[0]
            model.mp_computed["RightThighRotationOffset"] =   NEXUS.GetSubjectParamDetails( subject, "RightThighRotation")[0]
        else:
            model.mp_computed["LeftThighRotationOffset"] =   optional_mp["LeftThighRotation"]
            model.mp_computed["RightThighRotationOffset"] =  optional_mp["RightThighRotation"]

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

        if NEXUS_PYTHON_CONNECTED:
            mfpa = nexusTools.getForcePlateAssignment(NEXUS)
        else:
            mfpa =  args.offline[2]

        # btkAcq builder
        if NEXUS_PYTHON_CONNECTED:
            nacf = nexusFilters.NexusConstructAcquisitionFilter(NEXUS,DATA_PATH,reconstructFilenameLabelledNoExt,subject)
            acq = nacf.build()
        else: 
            acq=btkTools.smartReader(DATA_PATH+reconstructFilenameLabelled)

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
        if NEXUS_PYTHON_CONNECTED:
            nexusFilters.NexusModelFilter(
                NEXUS, model, acqGait, subject, pointSuffix).run()
            nexusTools.createGeneralEvents(
                NEXUS, subject, acqGait, ["Left-FP", "Right-FP"])
            
            for label in ["LTHIplanarAngle", "RTHIplanarAngle","LTIBplanarAngle", "RTIBplanarAngle"]:
                nexusTools.appendBtkScalarFromAcq(NEXUS,subject,"Quality",label,"Angle",acqGait) 
            # ========END of the nexus OPERATION if run from Nexus  =========
        else: 
            btkTools.smartWriter(acqGait, DATA_PATH+reconstructFilenameLabelled[:-4]+"-offlineProcessed.c3d")

    else:
        return 0


if __name__ == "__main__":

    main(args=None)
