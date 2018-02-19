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
from pyCGM2.Utils import files
from pyCGM2.Nexus import nexusFilters, nexusUtils,nexusTools

from pyCGM2.Model.CGM2.coreApps import cgmUtils, cgm2_2



if __name__ == "__main__":

    NEXUS = ViconNexus.ViconNexus()
    NEXUS_PYTHON_CONNECTED = NEXUS.Client.IsConnected()

    parser = argparse.ArgumentParser(description='CGM2-2 Fitting')
    parser.add_argument('--proj', type=str, help='Moment Projection. Choice : Distal, Proximal, Global')
    parser.add_argument('-mfpa',type=str,  help='manual assignment of force plates')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('--DEBUG', action='store_true', help='debug model. load file into nexus externally')

    args = parser.parse_args()



    if NEXUS_PYTHON_CONNECTED: # run Operation


        # --------------------------GLOBAL SETTINGS ------------------------------------
        # global setting ( in user/AppData)
        settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM2_2-pyCGM2.settings")

        # --------------------------CONFIG ------------------------------------
        argsManager = cgmUtils.argsManager_cgm(settings,args)
        markerDiameter = argsManager.getMarkerDiameter()
        pointSuffix = argsManager.getPointSuffix("cgm2.2")
        momentProjection =  argsManager.getMomentProjection()
        mfpa = argsManager.getManualForcePlateAssign()

        # --------------------------LOADING ------------------------------------
        if args.DEBUG:
            #DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.2\\native\\"
            #reconstructFilenameLabelledNoExt = "gait trial"

            DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH +"CGM3\\Salford_healthy_DataCollection\\PN01OP01S01\\"
            reconstructFilenameLabelledNoExt = "PN01OP01S01STAT-copy"

            NEXUS.OpenTrial( str(DATA_PATH+reconstructFilenameLabelledNoExt), 10 )

        else:
            DATA_PATH, reconstructFilenameLabelledNoExt = NEXUS.GetTrialName()


        reconstructFilenameLabelled = reconstructFilenameLabelledNoExt+".c3d"

        logging.info( "data Path: "+ DATA_PATH )
        logging.info( "calibration file: "+ reconstructFilenameLabelled)


        # --------------------------SUBJECT ------------------------------------
        subjects = NEXUS.GetSubjectNames()
        subject = nexusTools.checkActivatedSubject(NEXUS,subjects)
        logging.info(  "Subject name : " + subject  )

        # --------------------pyCGM2 MODEL ------------------------------
        model = files.loadModel(DATA_PATH,subject)

        # check model
        logging.info("loaded model : %s" %(model.version))
        if model.version != "CGM2.2":
            raise Exception ("%s-pyCGM2.model file was not calibrated from the CGM2.2 calibration pipeline"%subject)

        # --------------------------SESSION INFOS ------------------------------------
        translators = files.getTranslators(DATA_PATH,"CGM2_2.translators")
        if not translators: translators = settings["Translators"]

        #  ikweight
        ikWeight = files.getIKweightSet(DATA_PATH,"CGM2_2.ikw")
        if not ikWeight: translators = settings["Fitting"]["Weight"]=ikWeight["Weight"]


        # --------------------------MODELLING PROCESSING -----------------------
        acqIK = cgm2_2.fitting(model,DATA_PATH, reconstructFilenameLabelled,
            translators,settings,
            markerDiameter,
            pointSuffix,
            mfpa,
            momentProjection)

        # ----------------------DISPLAY ON VICON-------------------------------
        nexusFilters.NexusModelFilter(NEXUS,model,acqIK,subject,pointSuffix).run()
        nexusTools.createGeneralEvents(NEXUS,subject,acqIK,["Left-FP","Right-FP"])
        # ========END of the nexus OPERATION if run from Nexus  =========

        if args.DEBUG:

            NEXUS.SaveTrial(30)


    else:
        raise Exception("NO Nexus connection. Turn on Nexus")
