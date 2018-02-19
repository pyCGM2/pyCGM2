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
from pyCGM2.Eclipse import vskTools

from  manager import *

if __name__ == "__main__":

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM Pipeline')
    parser.add_argument('-f','--file', type=str, help='pipeline file', default="pipeline.pyCGM2")
    parser.add_argument('--vsk', type=str, help='vicon skeleton filename')
    parser.add_argument('--export', action='store_true', help='xls export')
    parser.add_argument('--plot', action='store_true', help='enable Gait Plot')
    parser.add_argument('-dm','--disableModelling', action='store_true', help='disable  modelling')
    parser.add_argument('-dp','--disableProcessing', action='store_true', help='disable  processing')
    parser.add_argument('--DEBUG', action='store_true', help='debug mode')

    args = parser.parse_args()
    xlsExport_flag = args.export
    pipelineFile = args.file
    modellingFlag = True if not args.disableModelling else False
    processingFlag = True if not args.disableProcessing else False
    plotFlag = True if  args.plot else False


    args.DEBUG = False
    if args.DEBUG:
        #DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\CGM1\\pipeline\\"
        #DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM2\\cgm2.3\\medialPipeline\\"
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\Programming\\API\\pyCGM2\\pyCGM2-Analyses\\CGM3-dataCollection\\CGM24_preAnalysis_3DMA\\dataS01OP1\\"
        pipelineFile = "pipeline2_4.pyCGM2"
        xlsExport_flag =  True
        plotFlag= True

    else:
        wd = os.getcwd()+"\\"

    manager = pipelineFileManager(wd,pipelineFile)
    modelVersion = manager.getCGMVersion()
    logging.info("model version : %s" %(modelVersion))

    DATA_PATH = wd if manager.getDataPath() is None else manager.getDataPath()

    DATA_PATH_OUT = DATA_PATH if manager.getOutDataPath() is None else manager.getOutDataPath()
    if manager.getOutDataPath() is not None:
        files.createDir(DATA_PATH_OUT)



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
    if args.vsk is None:
        logging.info("mp from pipeline file")
        required_mp,optional_mp = manager.getMP()
    else:
        logging.warning("mp from vsk file")
        vsk = vskTools.Vsk(str(DATA_PATH + args.vsk))
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)


    fileSuffix = manager.getFileSuffix()
    pointSuffix = manager.getPointSuffix()
    ik_flag = manager.isIkFitting()

    if modellingFlag:
        #------calibration--------
        leftFlatFoot = manager.getLeftFlatFoot()
        rightFlatFoot = manager.getRightFlatFoot()
        markerDiameter = manager.getMarkerDiameter()
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

        logging.info("Static Calibration -----> Done")

        # knee calibration
        leftEnable = manager.isKneeCalibrationEnable("Left")
        rightEnable = manager.isKneeCalibrationEnable("Right")

        if leftEnable:
            method, trial,begin,end = manager.getKneeCalibration("Left")
            if method == "Calibration2Dof":
                model,acqFunc,side = kneeCalibration.calibration2Dof(model,
                    DATA_PATH,trial,translators,
                    "Left",begin,end)
                logging.info("Left knee Calibration (Calibration2Dof) -----> Done")
            elif method == "SARA":
                model,acqFunc,side = kneeCalibration.sara(model,
                    DATA_PATH,trial,translators,
                    "Left",begin,end)
                logging.info("Left knee Calibration (SARA) -----> Done")
        if rightEnable:
            method, trial,begin,end = manager.getKneeCalibration("Right")
            if method == "Calibration2Dof":
                model,acqFunc,side = kneeCalibration.calibration2Dof(model,
                    DATA_PATH,trial,translators,
                    "Right",begin,end)
                logging.info("Right knee Calibration (Calibration2Dof) -----> Done")
            elif method == "SARA":
                model,acqFunc,side = kneeCalibration.sara(model,
                    DATA_PATH,trial,translators,
                    "Right",begin,end)
                logging.info("Right knee Calibration (SARA) -----> Done")

        # update mp
        manager.updateMp(model)
        # save settings
        manager.save(DATA_PATH,str(pipelineFile+"-saved"))
        logging.info("pipeline file -----> Save")

        # Fitting
        trials = manager.getFittingTrials()
        momentProjection = manager.getMomentProjection()

        if modelVersion not in ["CGM1.0", "CGM1.1", "CGM2.1"]:
            ikwf = manager.getIkWeightFile()
            if ikwf is not None:
                ikWeight = files.openJson(DATA_PATH,ikwf)
                settings["Fitting"]["Weight"]=ikWeight["Weight"]


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

            logging.info("Fitting -----> Done")


            if fileSuffix is not None:
                c3dFilename = str(reconstructFilenameLabelled[:-4]+"-modelled-"+fileSuffix+".c3d")
            else:
                c3dFilename = str(reconstructFilenameLabelled[:-4]+"-modelled.c3d")

            btkTools.smartWriter(acqGait, str(DATA_PATH_OUT+c3dFilename))
            logging.info("c3d file (%s) generated" %(c3dFilename) )

    #----------------Processing -----------------------
    if processingFlag:

        modelInfo = manager.getModelInfo()
        subjectInfo = manager.getSubjectInfo()
        experimentalInfo = manager.getExpInfo()

        tasks = manager.getProcessingTasks()

        for task in tasks:
            logging.info(" Processing ----- Task : %s ------------" %(task["TaskTitle"]))

            analyseType = str(task["AnalysisType"])

            experimentalInfo["TaskTitle"] = task["TaskTitle"]
            experimentalInfo.update(task["Conditions"])

            normativeData = task["Normative data"]

            modelledFilenames= task["Trials"]
            if fileSuffix is not None:
                modelledFilenames = [str(x[:-4]+"-modelled-"+fileSuffix+".c3d") for x in modelledFilenames]
            else:
                modelledFilenames = [str(x[:-4]+"-modelled.c3d") for x in modelledFilenames]

            outputFilenameNoExt = task["outputFilenameNoExt"]

            # --------------------------PROCESSING --------------------------------
            if analyseType == "Gait":
                cgmProcessing.gaitProcessing(DATA_PATH_OUT,modelledFilenames,modelVersion,
                     modelInfo, subjectInfo, experimentalInfo,
                     normativeData,
                     pointSuffix,
                     outputPath=DATA_PATH_OUT,
                     outputFilename = outputFilenameNoExt,
                     exportXls=xlsExport_flag,
                     plot=plotFlag)
            else:
                cgmProcessing.standardProcessing(DATA_PATH_OUT,modelledFilenames,modelVersion,
                     modelInfo, subjectInfo, experimentalInfo,
                     pointSuffix,
                     outputPath=DATA_PATH_OUT,
                     outputFilename = outputFilenameNoExt,
                     exportXls=xlsExport_flag)

            logging.info("Task : %s -----> processed" %(task["TaskTitle"]))
