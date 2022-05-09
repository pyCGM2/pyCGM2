# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Executable Apps/Vicon/CGM2.4
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
from pyCGM2.Lib.CGM import  cgm2_4


def main():

    parser = argparse.ArgumentParser(description='CGM2-4 Fitting')
    parser.add_argument('--proj', type=str, help='Referential to project joint moment. Choice : Distal, Proximal, Global')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    parser.add_argument('-a','--accuracy', type=float, help='Inverse Kinematics accuracy')
    parser.add_argument('-ae','--anomalyException', action='store_true', help='raise an exception if an anomaly is detected')
    parser.add_argument('-fi','--frameInit',type=int,  help='first frame to process')
    parser.add_argument('-fe','--frameEnd',type=int,  help='last frame to process')

    try:
        NEXUS = ViconNexus.ViconNexus()
        NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()
    except:
        LOGGER.logger.error("Vicon nexus not connected")
        NEXUS_PYTHON_CONNECTED = False


    if NEXUS_PYTHON_CONNECTED: # run Operation
        args = parser.parse_args()

        # --------------------------LOADING ------------------------------------
        DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"

        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.set_file_handler(DATA_PATH+"pyCGM2-Fitting.log")
        LOGGER.logger.info( "calibration file: "+ reconstructFilenameLabelled)

        # --------------------------GLOBAL SETTINGS ------------------------------------
        settings = files.loadModelSettings(DATA_PATH,"CGM2_4-pyCGM2.settings")


        # --------------------------CONFIG ------------------------------------
        argsManager = CgmArgsManager.argsManager_cgm(settings,args)
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm2.4")
        momentProjection =  argsManager.getMomentProjection()
        ik_flag = argsManager.enableIKflag()
        ikAccuracy = argsManager.getIkAccuracy()


        # --------------------------SUBJECT -----------------------------------
        # Notice : Work with ONE subject by session
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.getActiveSubject(NEXUS)
        LOGGER.logger.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH,subject)

        # -------------------------- MP ------------------------------------
        # allow alteration of thigh offset
        model.mp_computed["LeftThighRotationOffset"] =   NEXUS.GetSubjectParamDetails( subject, "LeftThighRotation")[0]
        model.mp_computed["RightThighRotationOffset"] =   NEXUS.GetSubjectParamDetails( subject, "RightThighRotation")[0]



        # check model
        LOGGER.logger.info("loaded model : %s" %(model.version))
        if model.version != "CGM2.4":
            raise Exception ("%s-pyCGM2.model file was not calibrated from the CGM2.4e calibration pipeline"%subject)

        # --------------------------SESSION INFOS -----------------------------
        #  translators management
        translators = files.getTranslators(DATA_PATH,"CGM2_4.translators")
        if not translators: translators = settings["Translators"]

        #  ikweight
        ikWeight = files.getIKweightSet(DATA_PATH,"CGM2_4.ikw")
        if not ikWeight: ikWeight = settings["Fitting"]["Weight"]

        #force plate assignement from Nexus
        mfpa = nexusTools.getForcePlateAssignment(NEXUS)

        # btkAcquisition
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,reconstructFilenameLabelledNoExt,subject)
        acq = nacf.build()

        # --------------------------MODELLING PROCESSING -----------------------
        finalAcqGait,detectAnomaly = cgm2_4.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            translators,settings,
            ik_flag,markerDiameter,
            pointSuffix,
            mfpa,
            momentProjection,forceBtkAcq=acq,
            anomalyException=args.anomalyException,
            ikAccuracy = ikAccuracy,
            frameInit= args.frameInit, frameEnd= args.frameEnd )

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusFilters.NexusModelFilter(NEXUS,model,finalAcqGait,subject,pointSuffix).run()
        nexusTools.createGeneralEvents(NEXUS,subject,finalAcqGait,["Left-FP","Right-FP"])
        # ========END of the nexus OPERATION if run from Nexus  =========

    else:
        return parser

if __name__ == "__main__":


    main()
