# -*- coding: utf-8 -*-
import os
import logging
import argparse
import matplotlib.pyplot as plt


# pyCGM2 settings
import pyCGM2
pyCGM2.CONFIG.setLoggingLevel(logging.INFO)

# pyCGM2 libraries
from pyCGM2 import enums
from pyCGM2.Model.CGM2.coreApps import cgmProcessing, kneeCalibration
from pyCGM2.Model.CGM2.coreApps import cgm1,cgm1_1,cgm2_1,cgm2_2,cgm2_2e,cgm2_3,cgm2_3e,cgm2_4,cgm2_4e
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files

from  manager import *

if __name__ == "__main__":

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM Pipeline')
    parser.add_argument('-f','--file', type=str, help='pipeline file', default="pipeline.pyCGM2")
    parser.add_argument('--export', action='store_true', help='xls export')
    parser.add_argument('--DEBUG', action='store_true', help='debug mode')

    args = parser.parse_args()
    xlsExport_flag = args.export

    pipelineFile = args.file

    #args.DEBUG = True
    if args.DEBUG:
        #DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\CGM1\\pipeline\\"
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.3\\medialPipeline\\"
    else:
        DATA_PATH = os.getcwd()+"\\"

    manager = pipelineFileManager(DATA_PATH,pipelineFile)
    modelVersion = manager.getCGMVersion()

    # --------------------------MODELLING ------------------------------------

    # manage global settings and translators
    if modelVersion == "CGM1.0":
        translatorFiles=  "CGM1.translators"
        globalPyCGM2settingFile = "CGM1-pyCGM2.settings"
    elif modelVersion == "CGM1.1":
        translatorFiles=  "CGM1_1.translators"
        globalPyCGM2settingFile = "CGM1_1-pyCGM2.settings"
    elif modelVersion == "CGM2.1":
        translatorFiles=  "CGM2_1.translators"
        globalPyCGM2settingFile = "CGM2_1-pyCGM2.settings"
    elif modelVersion == "CGM2.2":
        translatorFiles=  "CGM2_2.translators"
        globalPyCGM2settingFile = "CGM2_2-pyCGM2.settings"
    elif modelVersion == "CGM2.2e":
        translatorFiles=  "CGM2_2.translators"
        globalPyCGM2settingFile = "CGM2_2-Expert-pyCGM2.settings"
    elif modelVersion == "CGM2.3":
        translatorFiles=  "CGM2_3.translators"
        globalPyCGM2settingFile = "CGM2_3-pyCGM2.settings"
    elif modelVersion == "CGM2.3e":
        translatorFiles=  "CGM2_3.translators"
        globalPyCGM2settingFile = "CGM2_3-Expert-pyCGM2.settings"
    elif modelVersion == "CGM2.4":
        translatorFiles=  "CGM2_4.translators"
        globalPyCGM2settingFile = "CGM2_4-pyCGM2.settings"
    elif modelVersion == "CGM2.4e":
        translatorFiles=  "CGM2_4.translators"
        globalPyCGM2settingFile = "CGM2_4-Expert-pyCGM2.settings"

    else:
        raise Exception( "model version not known")

    settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,globalPyCGM2settingFile)
    translators = files.getTranslators(DATA_PATH,translatorFiles)
    if not translators: translators = settings["Translators"]

    # mp file
    required_mp,optional_mp = manager.getMP()

    fileSuffix = manager.getFileSuffix()
    ik_flag = manager.isIkFitting()

    # calibration
    leftFlatFoot = manager.getLeftFlatFoot()
    rightFlatFoot = manager.getRightFlatFoot()
    markerDiameter = manager.getMarkerDiameter()
    pointSuffix = manager.getPointSuffix()
    calibrateFilenameLabelled = manager.getStaticTial()

    if modelVersion == "CGM1.0":
        model,acqStatic = cgm1.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,required_mp,optional_mp,
               leftFlatFoot,rightFlatFoot,markerDiameter,
               pointSuffix)
    elif modelVersion == "CGM1.1":
        model,acqStatic = cgm1_1.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,required_mp,optional_mp,
               leftFlatFoot,rightFlatFoot,markerDiameter,
               pointSuffix)
    elif modelVersion == "CGM2.1":
        hjcMethod = manager.getHJCmethod()
        model,acqStatic = cgm2_1.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,required_mp,optional_mp,
                      leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethod,
                      pointSuffix)

    elif modelVersion == "CGM2.2":
        hjcMethod = manager.getHJCmethod()
        model,finalAcqStatic = cgm2_2.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                          required_mp,optional_mp,
                          True,leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethod,
                          pointSuffix)

    elif modelVersion == "CGM2.2e":
        hjcMethod = manager.getHJCmethod()
        model,finalAcqStatic = cgm2_2e.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                          required_mp,optional_mp,
                          True,leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethod,
                          pointSuffix)

    elif modelVersion == "CGM2.3":
        hjcMethod = manager.getHJCmethod()
        model,finalAcqStatic = cgm2_3.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                                  required_mp,optional_mp,
                                  ik_flag,leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethod,
                                  pointSuffix)

    elif modelVersion == "CGM2.3e":
        hjcMethod = manager.getHJCmethod()
        model,finalAcqStatic = cgm2_3e.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                                  required_mp,optional_mp,
                                  ik_flag,leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethod,
                                  pointSuffix)

    elif modelVersion == "CGM2.4":
        hjcMethod = manager.getHJCmethod()
        model,finalAcqStatic = cgm2_4.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                                  required_mp,optional_mp,
                                  ik_flag,leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethod,
                                  pointSuffix)

    elif modelVersion == "CGM2.4e":
        hjcMethod = manager.getHJCmethod()
        model,finalAcqStatic = cgm2_4e.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                                  required_mp,optional_mp,
                                  ik_flag,leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethod,
                                  pointSuffix)


    # knee calibration
    leftEnable = manager.isKneeCalibrationEnable("Left")
    rightEnable = manager.isKneeCalibrationEnable("Right")

    if leftEnable:
        method, trial,begin,end = manager.getKneeCalibration("Left")
        if method == "Calibration2Dof":
            model,acqFunc,side = kneeCalibration.calibration2Dof(model,
                DATA_PATH,trial,translators,
                "Left",begin,end)
        elif method == "SARA":
            model,acqFunc,side = kneeCalibration.sara(model,
                DATA_PATH,trial,translators,
                "Left",begin,end)
    if rightEnable:
        method, trial,begin,end = manager.getKneeCalibration("Right")
        if method == "Calibration2Dof":
            model,acqFunc,side = kneeCalibration.calibration2Dof(model,
                DATA_PATH,trial,translators,
                "Right",begin,end)
        elif method == "SARA":
            model,acqFunc,side = kneeCalibration.sara(model,
                DATA_PATH,trial,translators,
                "Right",begin,end)

    # Fitting
    trials = manager.getFittingTrials()

    momentProjection = manager.getMomentProjection()

    for trial in trials:
        mfpa = None if trial["Mfpa"] == "Auto" else trial["Mfpa"]

        reconstructFilenameLabelled = trial["File"]

        if modelVersion == "CGM1.0":
            acqGait = cgm1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
                translators,
                markerDiameter,
                pointSuffix,
                mfpa,momentProjection)
        elif modelVersion == "CGM1.1":
            acqGait = cgm1_1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
                translators,
                markerDiameter,
                pointSuffix,
                mfpa,momentProjection)

        elif modelVersion == "CGM2.1":
            acqGait = cgm2_1.fitting(model,DATA_PATH, reconstructFilenameLabelled,
                translators,
                markerDiameter,
                pointSuffix,
                mfpa,momentProjection)

        elif modelVersion == "CGM2.2":
            acqGait = cgm2_2.fitting(model,DATA_PATH, reconstructFilenameLabelled,
                    translators,settings,
                    markerDiameter,
                    pointSuffix,
                    mfpa,
                    momentProjection)

        elif modelVersion == "CGM2.2e":
            acqGait = cgm2_2e.fitting(model,DATA_PATH, reconstructFilenameLabelled,
                    translators,settings,
                    markerDiameter,
                    pointSuffix,
                    mfpa,
                    momentProjection)

        elif modelVersion == "CGM2.3":
            acqGait = cgm2_3.fitting(model,DATA_PATH, reconstructFilenameLabelled,
                    translators,settings,
                    ik_flag,markerDiameter,
                    pointSuffix,
                    mfpa,
                    momentProjection)

        elif modelVersion == "CGM2.3e":
            acqGait = cgm2_3e.fitting(model,DATA_PATH, reconstructFilenameLabelled,
                    translators,settings,
                    ik_flag,markerDiameter,
                    pointSuffix,
                    mfpa,
                    momentProjection)

        elif modelVersion == "CGM2.4":
            acqGait = cgm2_4.fitting(model,DATA_PATH, reconstructFilenameLabelled,
                    translators,settings,
                    ik_flag,markerDiameter,
                    pointSuffix,
                    mfpa,
                    momentProjection)

        elif modelVersion == "CGM2.4e":
            acqGait = cgm2_4e.fitting(model,DATA_PATH, reconstructFilenameLabelled,
                    translators,settings,
                    ik_flag,markerDiameter,
                    pointSuffix,
                    mfpa,
                    momentProjection)


        if fileSuffix is not None:
            btkTools.smartWriter(acqGait, str(DATA_PATH+reconstructFilenameLabelled[:-4]+"-modelled-"+fileSuffix+".c3d"))
        else:
            btkTools.smartWriter(acqGait, str(DATA_PATH+reconstructFilenameLabelled[:-4]+"-modelled.c3d"))

    # Processing
    modelInfo = manager.getModelInfo()
    subjectInfo = manager.getSubjectInfo()
    experimentalInfo = manager.getSubjectInfo()

    tasks = manager.getProcessingTasks()

    for task in tasks:
        experimentalInfo["Type"] = task["Type"]
        experimentalInfo.update(task["Conditions"])

        normativeData = task["Normative data"]

        modelledFilenames= task["Trials"]
        if fileSuffix is not None:
            modelledFilenames = [str(x[:-4]+"-modelled-"+fileSuffix+".c3d") for x in modelledFilenames]
        else:
            modelledFilenames = [str(x[:-4]+"-modelled.c3d") for x in modelledFilenames]

        outputFilenameNoExt = task["outputFilenameNoExt"]

        # --------------------------PROCESSING --------------------------------
        cgmProcessing.gaitprocessing(DATA_PATH,modelledFilenames,modelVersion,
             modelInfo, subjectInfo, experimentalInfo,
             normativeData,
             pointSuffix,
             outputFilename = outputFilenameNoExt,
             exportXls=xlsExport_flag)
