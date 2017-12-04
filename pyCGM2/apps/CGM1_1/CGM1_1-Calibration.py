# -*- coding: utf-8 -*-
import logging
import matplotlib.pyplot as plt
import argparse


# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)


# pyCGM2 libraries
from pyCGM2.Eclipse import vskTools
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files
from pyCGM2.Model.CGM2.coreApps import cgmUtils, cgm1_1


if __name__ == "__main__":

    plt.close("all")
    DEBUG = False

    parser = argparse.ArgumentParser(description='CGM1.1 Calibration')
    parser.add_argument('--static', type=str, help='static c3d')
    parser.add_argument('-l','--leftFlatFoot', type=int, help='left flat foot option')
    parser.add_argument('-r','--rightFlatFoot',type=int,  help='right flat foot option')
    parser.add_argument('-md','--markerDiameter', type=float, help='marker diameter')
    parser.add_argument('-ps','--pointSuffix', type=str, help='suffix of model outputs')
    parser.add_argument('--check', action='store_true', help='force model output suffix' )
    parser.add_argument('--resetMP', action='store_false', help='reset optional mass parameters')
    parser.add_argument('-fs','--fileSuffix', type=str, help='suffix of output file')
    parser.add_argument('--vsk', type=str, help='vicon skeleton filename')
    args = parser.parse_args()

    # --------------------------GLOBAL SETTINGS ------------------------------------
    # global setting ( in user/AppData)
    settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,"CGM1_1-pyCGM2.settings")

    # --------------------------CONFIG ------------------------------------
    argsManager = cgmUtils.argsManager_cgm1(settings,args)
    leftFlatFoot = argsManager.getLeftFlatFoot()
    rightFlatFoot = argsManager.getRightFlatFoot()
    markerDiameter = argsManager.getMarkerDiameter()
    pointSuffix = argsManager.getPointSuffix("cgm1_1")

    # --------------------------LOADING ------------------------------------
    if DEBUG:
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "Datasets Tests\\fraser\\New Session\\"
        calibrateFilenameLabelled = "15KUFC01_Trial03.c3d"
        args.vsk = "15KUFC01.vsk"
        args.fileSuffix="TEST"

    else:
        DATA_PATH =os.getcwd()+"\\"
        calibrateFilenameLabelled = args.static

    logging.info( "data Path: "+ DATA_PATH )
    logging.info( "calibration file: "+ calibrateFilenameLabelled)


    # --------------------------SUBJECT ------------------------------------
    #  translators management
    translators = files.getTranslators(DATA_PATH,"CGM1_1.translators")
    if not translators: translators = settings["Translators"]

    # --------------------------SUBJECT ------------------------------------
    if args.vsk is None:
        mpInfo = files.openJson("pyCGM2-mp.json")
        required_mp,optional_mp = files.getMp(mpInfo, resetFlag=args.resetMP)
    else:
        vsk = vskTools.Vsk(str(DATA_PATH + args.vsk))
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=args.resetMP)
        mpInfo = files.getMpFile(DATA_PATH,"pyCGM2-mp.json")

    #---------------------------------------------------------------------------
    # --------------------------MODELLING PROCESSING ---------------------------
    model,acqStatic = cgm1_1.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,required_mp,optional_mp,
                  leftFlatFoot,rightFlatFoot,markerDiameter,
                  pointSuffix)
    #---------------------------------------------------------------------------
    #---------------------------------------------------------------------------

    # ----------------------SAVE-------------------------------------------
    #pyCGM2.model
    files.saveModel(model,DATA_PATH,None)

    # save mp
    files.saveMp(mpInfo,model,DATA_PATH,str("pyCGM2-mp.json"))

    # ----------------------DISPLAY ON VICON---------------------    # new static file
    if args.fileSuffix is not None:
        btkTools.smartWriter(acqStatic, str(DATA_PATH+calibrateFilenameLabelled[:-4]+"-modelled-"+args.fileSuffix+".c3d"))
    else:
        btkTools.smartWriter(acqStatic, str(DATA_PATH+calibrateFilenameLabelled[:-4]+"-modelled.c3d"))
