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
from pyCGM2.Model.CGM2.coreApps import cgm1
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files

class pipelineFileManager(object):
    def __init__(self,DATA_PATH,pipelineFile):
        self.pipSettings = files.openJson(DATA_PATH,pipelineFile)

    def getCGMVersion(self):
        return self.pipSettings["Modelling"]["Model"]["CGM"]

    def getMP(self):
        required_mp,optional_mp = files.getMp(self.pipSettings["Modelling"])
        return required_mp,optional_mp

    def getPointSuffix(self):
        return self.pipSettings["Modelling"]["pointSuffix"]

    def getFileSuffix(self):
        return None if self.pipSettings["Modelling"]["fileSuffix"]=="" else self.pipSettings["Modelling"]["fileSuffix"]

    def getMarkerDiameter(self):
        return self.pipSettings["Modelling"]["MarkerDiameter"]

    # calibration
    def getLeftFlatFoot(self):
        return bool(self.pipSettings["Modelling"]["Calibration"]["LeftFlatFoot"])

    def getRightFlatFoot(self):
        return bool(self.pipSettings["Modelling"]["Calibration"]["RightFlatFoot"])

    def getStaticTial(self):
        return str(self.pipSettings["Modelling"]["Calibration"]["Trial"])

    #KneeCalibrationTrials
    def isKneeCalibrationEnable(self,side):
        kc_dict = self.pipSettings["Modelling"]["KneeCalibrationTrials"][side]
        method = kc_dict["Method"]
        if method == "Calibration2Dof" or method == "SARA":
            return True
        else:
            return False

    def getKneeCalibration(self,side):
        kc_dict = self.pipSettings["Modelling"]["KneeCalibrationTrials"][side]
        method = kc_dict["Method"]
        trial = str(kc_dict["Trial"])
        begin = None if kc_dict["BeginFrame"]==0 else kc_dict["BeginFrame"]
        end = None if kc_dict["EndFrame"]==0 else kc_dict["EndFrame"]

        return method, trial,begin,end

    # fitting
    def getFittingTrials(self):
        return self.pipSettings["Modelling"]["Fitting"]["Trials"]

    def getMomentProjection(self):
        if self.pipSettings["Modelling"]["Fitting"]["Projection"] == "Distal":
            return  enums.MomentProjection.Distal
        elif self.pipSettings["Modelling"]["Fitting"]["Projection"] == "Proximal":
            return  enums.MomentProjection.Proximal
        elif self.pipSettings["Modelling"]["Fitting"]["Projection"] == "Global":
            return  enums.MomentProjection.Global
        elif self.pipSettings["Modelling"]["Fitting"]["Projection"] == "JCS":
            return enums.MomentProjection.JCS

    def isIkFitting(self):
        return bool(self.pipSettings["Modelling"]["Fitting"]["NoIK"])

    # processing
    def getSubjectInfo(self):
        return self.pipSettings["Subject"]
    def getExpInfo(self):
        return self.pipSettings["ExperimentalContext"]
    def getModelInfo(self):
        return self.pipSettings["Modelling"]["Model"]
    def getProcessingTasks(self):
        return self.pipSettings["Processing"]["Tasks"]



if __name__ == "__main__":

    plt.close("all")

    parser = argparse.ArgumentParser(description='CGM Pipeline')
    parser.add_argument('-f','--file', type=str, help='pipeline file', default="pipeline.pyCGM2")
    parser.add_argument('--DEBUG', action='store_true', help='debug mode')

    args = parser.parse_args()

    pipelineFile = args.file

    if args.DEBUG:
        DATA_PATH = pyCGM2.CONFIG.TEST_DATA_PATH + "CGM1\\CGM1\\pipeline\\"
    else:
        DATA_PATH = os.getcwd()+"\\"

    manager = pipelineFileManager(DATA_PATH,pipelineFile)
    modelVersion = manager.getCGMVersion()

    # --------------------------MODELLING ------------------------------------

    # manage global settings and translators
    if modelVersion == "CGM1.0":
        translatorFiles=  "CGM1.translators"
        globalPyCGM2settingFile = "CGM1-pyCGM2.settings"

    settings = files.openJson(pyCGM2.CONFIG.PYCGM2_APPDATA_PATH,globalPyCGM2settingFile)
    translators = files.getTranslators(DATA_PATH,translatorFiles)
    if not translators: translators = settings["Translators"]

    # mp file
    required_mp,optional_mp = manager.getMP()

    fileSuffix = manager.getFileSuffix()

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
    ik_flag = manager.isIkFitting()
    momentProjection = manager.getMomentProjection()

    for trial in trials:
        mfpa = None if trial["Mfpa"] == "Auto" else trial["Mfpa"]

        if modelVersion == "CGM1.0":
            acqGait = cgm1.fitting(model,DATA_PATH, trial["File"],
                translators,
                markerDiameter,
                pointSuffix,
                mfpa,momentProjection)

        if fileSuffix is not None:
            btkTools.smartWriter(acqGait, str(DATA_PATH+trial["File"][:-4]+"-modelled-"+fileSuffix+".c3d"))
        else:
            btkTools.smartWriter(acqGait, str(DATA_PATH+trial["File"][:-4]+"-modelled.c3d"))

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

        pdfFilename = task["outputFilenameNoExt"]+ ".pdf"

        # --------------------------PROCESSING --------------------------------
        cgmProcessing.gaitprocessing(DATA_PATH,modelledFilenames,modelVersion,
             modelInfo, subjectInfo, experimentalInfo,
             normativeData,
             pointSuffix,
             pdfname = pdfFilename)
