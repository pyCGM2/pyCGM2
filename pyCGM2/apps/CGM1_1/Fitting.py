# -*- coding: utf-8 -*-
import os
import logging
import argparse
import matplotlib.pyplot as plt

# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)



# pyCGM2 libraries
from pyCGM2.Tools import btkTools
from pyCGM2.Model.CGM2.coreApps import cgmUtils, cgm1_1
from pyCGM2.Utils import files

if __name__ == "__main__":
    DEBUG = False

    parser = argparse.ArgumentParser(description='CGM1.1 Fitting')
    parser.add_argument('--trial', type=str,  required=True, help='static c3d')
    parser.add_argument('--subject',type=str, required=True,  help='subject')
    parser.add_argument('--proj', type=str, help='Moment Projection. Choice : Distal, Proximal, Global')
    parser.add_argument('-mfpa',type=str,  help='manual assignment of force plates')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix')
    parser.add_argument('-fs','--fileSuffix', type=str, help='suffix of output file')
    parser.add_argument('--DEBUG', action='store_true', help='debug model')
    args = parser.parse_args()

    # --------------------------GLOBAL SETTINGS ------------------------------------
    # global setting ( in user/AppData)
    settings = files.openJson(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1_1-pyCGM2.settings")

    # --------------------------CONFIG ------------------------------------
    subject = args.subject

    argsManager = cgmUtils.argsManager_cgm(settings,args)
    markerDiameter = argsManager.getMarkerDiameter()
    pointSuffix = argsManager.getPointSuffix("cgm1_1")
    momentProjection =  argsManager.getMomentProjection()
    mfpa = argsManager.getManualForcePlateAssign()

    # --------------------------LOADING ------------------------------------
    if args.DEBUG:
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1\\native\\"
        reconstructFilenameLabelled = "gait Trial.c3d"
        args.fileSuffix="cgm1_1"

    else:
        DATA_PATH =os.getcwd()+"\\"
        reconstructFilenameLabelled = args.trial

    logging.info( "data Path: "+ DATA_PATH )
    logging.info( "calibration file: "+ reconstructFilenameLabelled)

    # --------------------pyCGM2 MODEL ------------------------------
    model = files.loadModel(DATA_PATH,subject)

    # --------------------------CHECKING -----------------------------------
    # check model is the CGM1
    logging.info("loaded model : %s" %(model.version ))
    if model.version != "CGM1.1":
        raise Exception ("%s-pyCGM2.model file was not calibrated from the CGM1.1 calibration pipeline"%model.version)

        # --------------------------SESSION INFOS ------------------------------------

    #  translators management
    translators = files.getTranslators(DATA_PATH,"CGM1_1.translators")
    if not translators:  translators = settings["Translators"]

    # --------------------------MODELLING PROCESSING -----------------------
    acqGait = cgm1_1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
        translators,
        markerDiameter,
        pointSuffix,
        mfpa,momentProjection)

    # ----------------------SAVE-------------------------------------------
    # Todo: pyCGM2 model :  cpickle doesn t work. Incompatibility with Swig. ( see about BTK wrench)

    # new static file
    if args.fileSuffix is not None:
        btkTools.smartWriter(acqGait, str(DATA_PATH+reconstructFilenameLabelled[:-4]+"-modelled-"+args.fileSuffix+".c3d"))
    else:
        btkTools.smartWriter(acqGait, str(DATA_PATH+reconstructFilenameLabelled[:-4]+"-modelled.c3d"))
