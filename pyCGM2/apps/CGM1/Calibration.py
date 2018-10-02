# -*- coding: utf-8 -*-
import os
import logging
import matplotlib.pyplot as plt
import argparse


# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)


# pyCGM2 libraries
from pyCGM2.Eclipse import vskTools
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files
from pyCGM2.Model.CGM2.coreApps import CgmArgsManager, cgm1


if __name__ == "__main__":

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM1 Calibration')

    parser.add_argument('--trial', type=str, required=True,  help='static c3d')
    parser.add_argument('--subject',type=str, required=True, help='subject (vsk Name)')

    parser.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix' )
    parser.add_argument('--resetMP', action='store_true', help='reset optional mass parameters')

    parser.add_argument('-fs','--fileSuffix', type=str, help='suffix of output file')

    parser.add_argument('--DEBUG', action='store_true', help='debug model')
    args = parser.parse_args()

    # --------------------------GLOBAL SETTINGS ------------------------------------
    # global setting ( in user/AppData)
    settings = files.openJson(pyCGM2.PYCGM2_APPDATA_PATH,"CGM1-pyCGM2.settings")

    # --------------------------CONFIG ------------------------------------
    subject = args.subject

    argsManager = CgmArgsManager.argsManager_cgm1(settings,args)
    leftFlatFoot = argsManager.getLeftFlatFoot()
    rightFlatFoot = argsManager.getRightFlatFoot()
    markerDiameter = argsManager.getMarkerDiameter()
    pointSuffix = argsManager.getPointSuffix("cgm1")

    # --------------------------LOADING ------------------------------------
    if args.DEBUG:
        DATA_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1\\native\\"
        calibrateFilenameLabelled = "static.c3d"
        subject = "MRI-US-01 - Pig"
        args.fileSuffix="cgm1"

    else:
        DATA_PATH =os.getcwd()+"\\"
        calibrateFilenameLabelled = args.trial

    logging.info( "data Path: "+ DATA_PATH )
    logging.info( "calibration file: "+ calibrateFilenameLabelled)


    # --------------------------SUBJECT ------------------------------------
    #  translators management
    translators = files.getTranslators(DATA_PATH,"CGM1.translators")
    if not translators: translators = settings["Translators"]

    # --------------------------SUBJECT ------------------------------------
    if os.path.isfile(DATA_PATH+subject+".vsk"):
        logging.info("vsk file found")
        vsk = vskTools.Vsk(str(DATA_PATH + subject+".vsk"))
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=args.resetMP)
        mpInfo,mpFilename = files.getJsonFileContent(DATA_PATH,"mp.pyCGM2",subject)
    else:
        logging.info("no vsk found")
        mpInfo,mpFilename = files.getJsonFileContent(DATA_PATH,"mp.pyCGM2",subject)
        required_mp,optional_mp = files.getMp(mpInfo, resetFlag=args.resetMP)

    #---------------------------------------------------------------------------
    # --------------------------MODELLING PROCESSING ---------------------------
    model,acqStatic = cgm1.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,required_mp,optional_mp,
                  leftFlatFoot,rightFlatFoot,markerDiameter,
                  pointSuffix)

    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    # ----------------------SAVE-------------------------------------------
    #pyCGM2.model
    files.saveModel(model,DATA_PATH,subject)

    # save mp
    files.saveMp(mpInfo,model,DATA_PATH,mpFilename)

    # ----------------------DISPLAY ON VICON---------------------
    if args.fileSuffix is not None:
        btkTools.smartWriter(acqStatic, str(DATA_PATH+calibrateFilenameLabelled[:-4]+"-modelled-"+args.fileSuffix+".c3d"))
    else:
        btkTools.smartWriter(acqStatic, str(DATA_PATH+calibrateFilenameLabelled[:-4]+"-modelled.c3d"))
