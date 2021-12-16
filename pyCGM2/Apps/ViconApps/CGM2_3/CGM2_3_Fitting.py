# -*- coding: utf-8 -*-
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
from pyCGM2.Nexus import nexusFilters,nexusTools,nexusUtils

from pyCGM2.Configurator import CgmArgsManager
from pyCGM2.Lib.CGM import  cgm2_3
from pyCGM2.Tools import  btkTools

def main():

    parser = argparse.ArgumentParser(description='CGM2-3 Fitting')
    parser.add_argument('--proj', type=str, help='Moment Projection. Choice : Distal, Proximal, Global')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('--noIk', action='store_true', help='cancel inverse kinematic')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('-a','--accuracy', type=float, help='Inverse Kinematics accuracy')
    parser.add_argument('-ae','--anomalyException', action='store_true', help='stop if anomaly detected ')
    parser.add_argument('-fi','--frameInit',type=int,  help='first frame to process')
    parser.add_argument('-fe','--frameEnd',type=int,  help='last frame to process')
    parser.add_argument('--muscleLength', action='store_true', help='enable muscle length calculation')
    args = parser.parse_args()


    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()



    if NEXUS_PYTHON_CONNECTED: # run Operation

        # --------------------------LOADING ------------------------------------
        DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()

        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"

        LOGGER.logger.info( "data Path: "+ DATA_PATH )
        LOGGER.set_file_handler(DATA_PATH+"pyCGM2-Fitting.log")
        LOGGER.logger.info( "calibration file: "+ reconstructFilenameLabelled)

        # --------------------------GLOBAL SETTINGS ------------------------------------
        settings = files.loadModelSettings(DATA_PATH,"CGM2_3-pyCGM2.settings")



        # --------------------------CONFIG ------------------------------------
        argsManager = CgmArgsManager.argsManager_cgm(settings,args)
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm2.3")
        momentProjection =  argsManager.getMomentProjection()
        ik_flag = argsManager.enableIKflag
        ikAccuracy = argsManager.getIkAccuracy()

        muscleLengthFlag = args.muscleLength

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




        # --------------------------CHECKING -----------------------------------
        # check model
        LOGGER.logger.info("loaded model : %s" %(model.version))
        if model.version != "CGM2.3":
            raise Exception ("%s-pyCGM2.model file was not calibrated from the CGM2.3 calibration pipeline"%subject)

        # --------------------------SESSION INFOS ------------------------------------
        #  translators management
        translators = files.getTranslators(DATA_PATH,"CGM2_3.translators")
        if not translators: translators = settings["Translators"]

        #  ikweight
        ikWeight = files.getIKweightSet(DATA_PATH,"CGM2_3.ikw")
        if not ikWeight: ikWeight = settings["Fitting"]["Weight"]

        #force plate assignement from Nexus
        mfpa = nexusTools.getForcePlateAssignment(NEXUS)

        # btkAcquisition
        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,reconstructFilenameLabelledNoExt,subject)
        acq = nacf.build()

        # --------------------------MODELLING PROCESSING -----------------------
        finalAcqGait,detectAnomaly = cgm2_3.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            translators,settings,
            ik_flag,markerDiameter,
            pointSuffix,
            mfpa,
            momentProjection,
            forceBtkAcq=acq,
            ikAccuracy = ikAccuracy,
            anomalyException=args.anomalyException,
            frameInit= args.frameInit, frameEnd= args.frameEnd,
            muscleLength = muscleLengthFlag )


        # ----------------------DISPLAY ON VICON-------------------------------
        nexusFilters.NexusModelFilter(NEXUS,model,finalAcqGait,subject,pointSuffix).run()
        nexusTools.createGeneralEvents(NEXUS,subject,finalAcqGait,["Left-FP","Right-FP"])

        if muscleLengthFlag:
            muscleLabels = btkTools.getLabelsFromScalar(finalAcqGait,"MuscleLength")
            for label in muscleLabels:
                nexusTools.appendBtkScalarFromAcq(NEXUS,subject,"MuscleLength",label,"Length",finalAcqGait)


        # ========END of the nexus OPERATION if run from Nexus  =========



    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":


    # ---- main script -----
    main()
