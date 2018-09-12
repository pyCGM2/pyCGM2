# -*- coding: utf-8 -*-
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np

# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)


# pyCGM2 libraries
from pyCGM2 import enums
from pyCGM2.Utils import files
from pyCGM2.Tools import btkTools
from pyCGM2.Model.CGM2.coreApps import cgmUtils, kneeCalibration
from pyCGM2.Model import  modelFilters


if __name__ == "__main__":

    plt.close("all")

    parser = argparse.ArgumentParser(description='2Dof Knee Calibration')
    parser.add_argument('--trial', type=str,   help='functional c3d')
    parser.add_argument('--subject',type=str,  help='subject (vsk Name)')

    parser.add_argument('-s','--side', type=str, help="Side : Left or Right")
    parser.add_argument('-b','--beginFrame', type=int, help="begin frame")
    parser.add_argument('-e','--endFrame', type=int, help="end frame")

    parser.add_argument('-fs','--fileSuffix', type=str, help='suffix of output file')
    parser.add_argument('--DEBUG', action='store_true', help='debug model. load file into nexus externally')
    args = parser.parse_args()

    subject = args.subject

    if args.DEBUG:
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "CGM2\\knee calibration\\CGM2.3-calibration2Dof\\"
        reconstructFilenameLabelled = "Left Knee.c3d"
        subject = "MRI-US-01"
        args.fileSuffix="2dof"

    else:
        DATA_PATH =os.getcwd()+"\\"
        reconstructFilenameLabelled = args.trial

    logging.info( "data Path: "+ DATA_PATH )
    logging.info( "reconstructed file: "+ reconstructFilenameLabelled)



    # --------------------pyCGM2 MODEL - INIT ------------------------------
    model = files.loadModel(DATA_PATH,subject)

    logging.info("loaded model : %s" %(model.version ))

    if model.version == "CGM1.0":
        settings = files.openJson(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1-pyCGM2.settings")
    elif model.version == "CGM1.1":
        settings = files.openJson(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1_1-pyCGM2.settings")
    elif model.version == "CGM2.1":
        settings = files.openJson(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_1-pyCGM2.settings")
    elif model.version == "CGM2.2":
        settings = files.openJson(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_2-pyCGM2.settings")
    elif model.version == "CGM2.2e":
        settings = files.openJson(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_2-Expert-pyCGM2.settings")
    elif model.version == "CGM2.3":
        settings = files.openJson(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_3-pyCGM2.settings")
    elif model.version == "CGM2.3e":
        settings = files.openJson(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_3-Expert-pyCGM2.settings")
    elif model.version == "CGM2.4":
        settings = files.openJson(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_4-pyCGM2.settings")
    elif model.version == "CGM2.4e":
        settings = files.openJson(pyCGM2.PYCGM2_APPDATA_PATH,"CGM2_4-Expert-pyCGM2.settings")
    else:
        raise Exception ("model version not found [contact admin]")

    # --------------------------SESSION INFOS ------------------------------------
    mpInfo,mpFilename = files.getJsonFileContent(DATA_PATH,"mp.pyCGM2",subject)

    #  translators management
    if model.version in  ["CGM1.0"]:
        translators = files.getTranslators(DATA_PATH,"CGM1.translators")
    elif model.version in  ["CGM1.1"]:
        translators = files.getTranslators(DATA_PATH,"CGM1_1.translators")
    elif model.version in  ["CGM2.1"]:
        translators = files.getTranslators(DATA_PATH,"CGM2_1.translators")
    elif model.version in  ["CGM2.2","CGM2.2e"]:
        translators = files.getTranslators(DATA_PATH,"CGM2_2.translators")
    elif model.version in  ["CGM2.3","CGM2.3e"]:
        translators = files.getTranslators(DATA_PATH,"CGM2_3.translators")
    elif model.version in  ["CGM2.4","CGM2.4e"]:
        translators = files.getTranslators(DATA_PATH,"CGM2_4.translators")

    if not translators:
       translators = settings["Translators"]

    # --------------------------MODEL PROCESSING----------------------------
    model,acqFunc,side = kneeCalibration.calibration2Dof(model,
        DATA_PATH,reconstructFilenameLabelled,translators,
        args.side,args.beginFrame,args.endFrame)

    if side == "Left":
        logging.warning("offset %s" %(str(model.mp_computed["LeftKneeFuncCalibrationOffset"] )))
    elif side == "Right":
        logging.warning("offset %s" %(str(model.mp_computed["RightKneeFuncCalibrationOffset"] )))


    # ----------------------SAVE-------------------------------------------
    files.saveModel(model,DATA_PATH,subject)
    logging.warning("model updated with a  %s knee calibrated with 2Dof method" %(side))

    # save mp
    files.saveMp(mpInfo,model,DATA_PATH,mpFilename)


    if args.fileSuffix is not None:
        btkTools.smartWriter(acqFunc, str(DATA_PATH+reconstructFilenameLabelled[:-4]+"-modelled-2DOF-"+args.fileSuffix+".c3d"))
    else:
        btkTools.smartWriter(acqFunc, str(DATA_PATH+reconstructFilenameLabelled[:-4]+"-modelled-2DOF.c3d"))
