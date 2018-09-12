# -*- coding: utf-8 -*-
import os
import logging
import argparse
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# pyCGM2 settings
import pyCGM2
from pyCGM2 import log; log.setLoggingLevel(logging.INFO)

# pyCGM2 libraries
from pyCGM2 import enums
from pyCGM2.Model.CGM2.coreApps import cgmProcessing, kneeCalibration
from pyCGM2.Model.CGM2.coreApps import cgm1,cgm1_1,cgm2_1,cgm2_2,cgm2_2e,cgm2_3,cgm2_3e,cgm2_4,cgm2_4e
from pyCGM2.Tools import btkTools,trialTools
from pyCGM2.Utils import files
from pyCGM2.Eclipse import vskTools

from pyCGM2.apps.CGM_Pipeline import pipManager

def modelling(manager,DATA_PATH,DATA_PATH_OUT,vskFile=None):
    modelVersion = manager.getModelVersion()
    logging.info("model version : %s" %(modelVersion))

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

    settings = files.openJson(pyCGM2.PYCGM2_APPDATA_PATH,globalPyCGM2settingFile)
    translators = files.getTranslators(DATA_PATH,translatorFiles)
    if not translators: translators = settings["Translators"]

    # mp file
    if vskFile is None:
        logging.info("mp from pipeline file")
        required_mp,optional_mp = manager.getMP()
    else:
        logging.warning("mp from vsk file")
        vsk = vskTools.Vsk(str(DATA_PATH + vskFile))
        required_mp,optional_mp = vskTools.getFromVskSubjectMp(vsk, resetFlag=True)


    fileSuffix = manager.getFileSuffix()
    pointSuffix = manager.getPointSuffix()
    ik_flag = manager.isIkFitting()

    #------calibration--------
    leftFlatFoot = manager.getLeftFlatFoot()
    rightFlatFoot = manager.getRightFlatFoot()
    markerDiameter = manager.getMarkerDiameter()
    calibrateFilenameLabelled = manager.getStaticTial()

    if modelVersion == "CGM1.0":
        model,finalAcqStatic = cgm1.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,required_mp,optional_mp,
               leftFlatFoot,rightFlatFoot,markerDiameter,
               pointSuffix)
    elif modelVersion == "CGM1.1":
        model,finalAcqStatic = cgm1_1.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,required_mp,optional_mp,
               leftFlatFoot,rightFlatFoot,markerDiameter,
               pointSuffix)
    elif modelVersion == "CGM2.1":

        hjcMethods = manager.getHJC()
        hjcMethods["Left"] = "Hara" if hjcMethods["Left"] == [] else hjcMethods["Left"]
        hjcMethods["Right"] = "Hara" if hjcMethods["Right"] == [] else hjcMethods["Right"]
        model,finalAcqStatic = cgm2_1.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,required_mp,optional_mp,
                      leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethods,
                      pointSuffix)

    elif modelVersion == "CGM2.2":
        hjcMethods = manager.getHJC()
        hjcMethods["Left"] = "Hara" if hjcMethods["Left"] == [] else hjcMethods["Left"]
        hjcMethods["Right"] = "Hara" if hjcMethods["Right"] == [] else hjcMethods["Right"]
        model,finalAcqStatic = cgm2_2.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                          required_mp,optional_mp,
                          True,leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethods,
                          pointSuffix)

    elif modelVersion == "CGM2.2e":
        hjcMethods =  manager.getHJC()
        hjcMethods["Left"] = "Hara" if hjcMethods["Left"] == [] else hjcMethods["Left"]
        hjcMethods["Right"] = "Hara" if hjcMethods["Right"] == [] else hjcMethods["Right"]
        model,finalAcqStatic = cgm2_2e.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                          required_mp,optional_mp,
                          True,leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethods,
                          pointSuffix)

    elif modelVersion == "CGM2.3":
        hjcMethods = manager.getHJC()
        hjcMethods["Left"] = "Hara" if hjcMethods["Left"] == [] else hjcMethods["Left"]
        hjcMethods["Right"] = "Hara" if hjcMethods["Right"] == [] else hjcMethods["Right"]
        model,finalAcqStatic = cgm2_3.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                                  required_mp,optional_mp,
                                  ik_flag,leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethods,
                                  pointSuffix)

    elif modelVersion == "CGM2.3e":
        hjcMethods = manager.getHJC()
        hjcMethods["Left"] = "Hara" if hjcMethods["Left"] == [] else hjcMethods["Left"]
        hjcMethods["Right"] = "Hara" if hjcMethods["Right"] == [] else hjcMethods["Right"]
        model,finalAcqStatic = cgm2_3e.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                                  required_mp,optional_mp,
                                  ik_flag,leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethods,
                                  pointSuffix)

    elif modelVersion == "CGM2.4":
        hjcMethods = manager.getHJC()
        hjcMethods["Left"] = "Hara" if hjcMethods["Left"] == [] else hjcMethods["Left"]
        hjcMethods["Right"] = "Hara" if hjcMethods["Right"] == [] else hjcMethods["Right"]
        model,finalAcqStatic = cgm2_4.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                                  required_mp,optional_mp,
                                  ik_flag,leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethods,
                                  pointSuffix)

    elif modelVersion == "CGM2.4e":
        hjcMethods = manager.getHJC()
        hjcMethods["Left"] = "Hara" if hjcMethods["Left"] == [] else hjcMethods["Left"]
        hjcMethods["Right"] = "Hara" if hjcMethods["Right"] == [] else hjcMethods["Right"]
        model,finalAcqStatic = cgm2_4e.calibrate(DATA_PATH,calibrateFilenameLabelled,translators,settings,
                                  required_mp,optional_mp,
                                  ik_flag,leftFlatFoot,rightFlatFoot,markerDiameter,hjcMethods,
                                  pointSuffix)

    btkTools.smartWriter(finalAcqStatic, str(DATA_PATH_OUT+"calibrated.c3d"))
    logging.info("Static Calibration -----> Done")

    # knee calibration
    leftEnable = manager.isKneeCalibrationEnable("Left")
    rightEnable = manager.isKneeCalibrationEnable("Right")

    if leftEnable:
        method, trial,begin,end,jointRange = manager.getKneeCalibration("Left")
        if method == "Calibration2Dof":
            model,acqFunc,side = kneeCalibration.calibration2Dof(model,
                DATA_PATH,trial,translators,
                "Left",begin,end,jointRange)

            logging.info("Left knee Calibration (Calibration2Dof) -----> Done")
        elif method == "SARA":
            model,acqFunc,side = kneeCalibration.sara(model,
                DATA_PATH,trial,translators,
                "Left",begin,end)
            logging.info("Left knee Calibration (SARA) -----> Done")
    if rightEnable:
        method, trial,begin,end,jointRange = manager.getKneeCalibration("Right")
        if method == "Calibration2Dof":
            model,acqFunc,side = kneeCalibration.calibration2Dof(model,
                DATA_PATH,trial,translators,
                "Right",begin,end,jointRange)
            logging.info("Right knee Calibration (Calibration2Dof) -----> Done")
        elif method == "SARA":
            model,acqFunc,side = kneeCalibration.sara(model,
                DATA_PATH,trial,translators,
                "Right",begin,end)
            logging.info("Right knee Calibration (SARA) -----> Done")

    # update mp
    manager.updateMp(model)
    # save settings
    if manager.m_pipelineFile is not None:
        manager.save(DATA_PATH,str( manager.m_pipelineFile+"-saved"))
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


def processing(manager,DATA_PATH,DATA_PATH_OUT,plotFlag=True, exportFlag = True):
    fileSuffix = manager.getFileSuffix()
    pointSuffix = manager.getPointSuffix()
    modelVersion = manager.getModelVersion()

    modelInfo = manager.getModelInfo()
    subjectInfo = manager.getSubjectInfo()
    experimentalInfo = manager.getExpInfo()

    analyses = manager.getProcessingAnalyses()

    for analysisIt in analyses:
        logging.info(" Processing ----- Analysis : %s ------------" %(analysisIt["AnalysisTitle"]))

        taskType = str(analysisIt["TaskType"])

        experimentalInfo["AnalysesTitle"] = analysisIt["AnalysisTitle"]
        experimentalInfo.update(analysisIt["Conditions"])

        normativeData = analysisIt["Normative data"]

        modelledFilenames= analysisIt["Trials"]
        if fileSuffix is not None:
            modelledFilenames = [str(x[:-4]+"-modelled-"+fileSuffix+".c3d") for x in modelledFilenames]
        else:
            modelledFilenames = [str(x[:-4]+"-modelled.c3d") for x in modelledFilenames]

        outputFilenameNoExt = analysisIt["outputFilenameNoExt"]

        # --------------------------PROCESSING --------------------------------
        if taskType == "Gait":
            cgmProcessing.gaitProcessing(DATA_PATH_OUT,modelledFilenames,modelVersion,
                 modelInfo, subjectInfo, experimentalInfo,
                 normativeData,
                 pointSuffix,
                 outputPath=DATA_PATH_OUT,
                 outputFilename = outputFilenameNoExt,
                 exportXls=exportFlag,
                 plot=plotFlag)
        else:
            cgmProcessing.standardProcessing(DATA_PATH_OUT,modelledFilenames,modelVersion,
                 modelInfo, subjectInfo, experimentalInfo,
                 pointSuffix,
                 outputPath=DATA_PATH_OUT,
                 outputFilename = outputFilenameNoExt,
                 exportXls=exportFlag)

        logging.info("Analysis : %s -----> processed" %(analysisIt["AnalysisTitle"]))


if __name__ == "__main__":

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM Pipeline')
    parser.add_argument('-f','--file', type=str, help='pipeline file', default="pipeline.pyCGM2")
    parser.add_argument('--vsk', type=str, help='vicon skeleton filename')
    parser.add_argument('-dm','--disableModelling', action='store_true', help='disable  modelling')
    parser.add_argument('-dp','--disableProcessing', action='store_true', help='disable  processing')
    parser.add_argument('--noExport', action='store_true', help='xls export')
    parser.add_argument('--noPlot', action='store_true', help='enable Gait Plot')
    parser.add_argument('--DEBUG', action='store_true', help='debug mode')

    args = parser.parse_args()
    pipelineFile = args.file
    modellingFlag = True if not args.disableModelling else False
    vskFile = args.vsk
    processingFlag = True if not args.disableProcessing else False
    xlsExport_flag = False if  args.noExport else True
    plotFlag = False if  args.noPlot else True


    args.DEBUG = False
    if args.DEBUG:
        #DATA_PATH = pyCGM2.TEST_DATA_PATH + "CGM1\\CGM1\\pipeline\\"
        #DATA_PATH = pyCGM2.TEST_DATA_PATH + "CGM2\\cgm2.3\\medialPipeline\\"
        DATA_PATH = "C:\\Users\\HLS501\\Documents\\VICON DATA\\pyCGM2-Data\\Datasets Tests\\didier\\08_02_18_Vincent Pere\\"
        pipelineFile = "pipeline.pyCGM2"
        xlsExport_flag =  True
        plotFlag= True
        wd= DATA_PATH
    else:
        wd = os.getcwd()+"\\"


    # ----------------setting manager----------------

    manager = pipManager.PipelineFileManager(wd,pipelineFile)

    # data path configurations
    if not args.DEBUG:
        DATA_PATH = wd if manager.getDataPath() is None else manager.getDataPath()
        print  DATA_PATH

    DATA_PATH_OUT = DATA_PATH if manager.getOutDataPath() is None else manager.getOutDataPath()
    if manager.getOutDataPath() is not None:
        files.createDir(DATA_PATH_OUT)

    #----------------Modelling -----------------------
    if modellingFlag:
        modelling(manager,DATA_PATH,DATA_PATH_OUT,vskFile=vskFile)
    #----------------Processing -----------------------
    if processingFlag:
        processing(manager,DATA_PATH,DATA_PATH_OUT,plotFlag=plotFlag, exportFlag = xlsExport_flag)
