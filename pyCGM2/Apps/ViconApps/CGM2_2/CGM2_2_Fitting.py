# -*- coding: utf-8 -*-
#APIDOC: /Apps/Vicon/CGM2

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
from pyCGM2.Lib.CGM import  cgm2_2



def main():
    """ run the CGM2.2 fitting operation from Nexus

    Usage:

    ```bash
        python CGM2_2_Fitting.py -fi  100 -fe 150 --md 24 -ps "withSuffix"
        python CGM2_2_Fitting.py --frameInit  100 --frameEnd 150 --markerDiameter 24 --pointSuffix "withSuffix"
        python CGM2_2_Fitting --accuracy 1e-5
    ```

    Args:
        [--proj] (str) : segmental coordinate system selected to project the joint moment (Choice : Distal, Proximal, Global,JCS"
        [-md, --markerDiameter] (int) : marker diameter
        [-ps, --pointSuffix] (str) : suffix of the model ouputs
        [--check] (bool) :force _cgm1  as model output suffix
        [-a,--accuracy] (float) : accuracy of the inverse kinematic solver (default: 1e-8)
        [-ae,--anomalyException] (bool) : return exception if one anomaly detected ')
        [-fi,--frameInit] (int) : first frame to process
        [-fe,--frameEnd] (int) : last frame to process

    Note:
        Marker diameter is used for locating joint centre from an origin ( eg LKNE) by an offset along a direction.
        respect the same marker diameter for the following markers :
        L(R)KNE - L(R)ANK - L(R)ASI - L(R)PSI

    """
    parser = argparse.ArgumentParser(description='CGM2-2 Fitting')
    parser.add_argument('--proj', type=str, help='Moment Projection. Choice : Distal, Proximal, Global')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('-a','--accuracy', type=float, help='Inverse Kinematics accuracy')
    parser.add_argument('-ae','--anomalyException', action='store_true', help='stop if anomaly detected ')
    parser.add_argument('-fi','--frameInit',type=int,  help='first frame to process')
    parser.add_argument('-fe','--frameEnd',type=int,  help='last frame to process')
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
        settings = files.loadModelSettings(DATA_PATH,"CGM2_2-pyCGM2.settings")


        # --------------------------CONFIG ------------------------------------
        argsManager = CgmArgsManager.argsManager_cgm(settings,args)
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm2.2")
        momentProjection =  argsManager.getMomentProjection()
        ikAccuracy = argsManager.getIkAccuracy()


        # --------------------------SUBJECT ------------------------------------
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
        if model.version != "CGM2.2":
            raise Exception ("%s-pyCGM2.model file was not calibrated from the CGM2.2 calibration pipeline"%subject)

        # --------------------------SESSION INFOS ------------------------------------
        translators = files.getTranslators(DATA_PATH,"CGM2_2.translators")
        if not translators: translators = settings["Translators"]

        #  ikweight
        ikWeight = files.getIKweightSet(DATA_PATH,"CGM2_2.ikw")
        if not ikWeight:  ikWeight = settings["Fitting"]["Weight"]

        #force plate assignement from Nexus
        mfpa = nexusTools.getForcePlateAssignment(NEXUS)

        nacf = nexusFilters.NexusConstructAcquisitionFilter(DATA_PATH,reconstructFilenameLabelledNoExt,subject)
        acq = nacf.build()

        # --------------------------MODELLING PROCESSING -----------------------
        acqIK,detectAnomaly = cgm2_2.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            translators,ikWeight,
            True,
            markerDiameter,
            pointSuffix,
            mfpa,
            momentProjection,
            forceBtkAcq=acq,
            ikAccuracy = ikAccuracy,
            anomalyException=args.anomalyException,
            frameInit= args.frameInit, frameEnd= args.frameEnd )



        # ----------------------DISPLAY ON VICON-------------------------------
        nexusFilters.NexusModelFilter(NEXUS,model,acqIK,subject,pointSuffix).run()
        nexusTools.createGeneralEvents(NEXUS,subject,acqIK,["Left-FP","Right-FP"])
        # ========END of the nexus OPERATION if run from Nexus  =========


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")

if __name__ == "__main__":

    main()
