# -*- coding: utf-8 -*-

# pyCGM2 libraries
from pyCGM2 import enums
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
    def getHJCmethod(self):
        return self.pipSettings["Modelling"]["Calibration"]["HJC_method"]

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
        return bool(self.pipSettings["Modelling"]["NoIK"])

    # processing
    def getSubjectInfo(self):
        return self.pipSettings["Subject"]
    def getExpInfo(self):
        return self.pipSettings["ExperimentalContext"]
    def getModelInfo(self):
        return self.pipSettings["Modelling"]["Model"]
    def getProcessingTasks(self):
        return self.pipSettings["Processing"]["Tasks"]
