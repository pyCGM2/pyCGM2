# -*- coding: utf-8 -*-

# pyCGM2 libraries
from pyCGM2 import enums
from pyCGM2.Utils import files

class pipelineFileManager(object):
    def __init__(self,DATA_PATH,pipelineFile):
        self.pipSettings = files.openJson(DATA_PATH,pipelineFile)

    def getPipelineSetttings(self):
        return self.pipSettings

    def getDataPath(self):
        out_path = str(self.pipSettings["DATA_PATH"])
        return out_path if out_path!="" else None

    def getOutDataPath(self):
        out_path = str(self.pipSettings["OutDataPath"])
        return out_path if out_path!="" else None

    def getIkWeightFile(self):
        ikwf = str(self.pipSettings["Modelling"]["Fitting"]["IkweightFile"])
        return ikwf if ikwf !="" else None

    def getCGMVersion(self):
        return str(self.pipSettings["Modelling"]["Model"]["CGM"])

    def getMP(self):
        required_mp,optional_mp = files.getMp(self.pipSettings["Modelling"],resetFlag=True)
        return required_mp,optional_mp

    def getPointSuffix(self):
        return str(self.pipSettings["Modelling"]["pointSuffix"])

    def getFileSuffix(self):
        return None if self.pipSettings["Modelling"]["fileSuffix"]=="" else str(self.pipSettings["Modelling"]["fileSuffix"])

    def getMarkerDiameter(self):
        return self.pipSettings["Modelling"]["MarkerDiameter"]

    # calibration
    def getHJCmethod(self):
        return str(self.pipSettings["Modelling"]["Calibration"]["HJC_method"])

    def getLeftFlatFoot(self):
        return bool(self.pipSettings["Modelling"]["Calibration"]["LeftFlatFoot"])

    def getRightFlatFoot(self):
        return bool(self.pipSettings["Modelling"]["Calibration"]["RightFlatFoot"])

    def getStaticTial(self):
        return str(self.pipSettings["Modelling"]["Calibration"]["Trial"])

    #KneeCalibrationTrials
    def isKneeCalibrationEnable(self,side):
        kc_dict = self.pipSettings["Modelling"]["KneeCalibrationTrials"][side]
        method = str(kc_dict["Method"])
        if method == "Calibration2Dof" or method == "SARA":
            return True
        else:
            return False

    def getKneeCalibration(self,side):
        kc_dict = self.pipSettings["Modelling"]["KneeCalibrationTrials"][side]
        method = str(kc_dict["Method"])
        trial = str(kc_dict["Trial"])
        begin = None if kc_dict["BeginFrame"]==0 else kc_dict["BeginFrame"]
        end = None if kc_dict["EndFrame"]==0 else kc_dict["EndFrame"]

        return method, trial,begin,end

    # fitting
    def getFittingTrials(self):
        return self.pipSettings["Modelling"]["Fitting"]["Trials"]

    def getMomentProjection(self):
        if str(self.pipSettings["Modelling"]["Fitting"]["Projection"]) == "Distal":
            return  enums.MomentProjection.Distal
        elif str(self.pipSettings["Modelling"]["Fitting"]["Projection"]) == "Proximal":
            return  enums.MomentProjection.Proximal
        elif str(self.pipSettings["Modelling"]["Fitting"]["Projection"]) == "Global":
            return  enums.MomentProjection.Global
        elif str(self.pipSettings["Modelling"]["Fitting"]["Projection"]) == "JCS":
            return enums.MomentProjection.JCS

    def isIkFitting(self):
        return True if not bool(self.pipSettings["Modelling"]["NoIK"]) else False

    # processing
    def getSubjectInfo(self):
        return self.pipSettings["Subject"]
    def getExpInfo(self):
        return self.pipSettings["ExperimentalContext"]
    def getModelInfo(self):
        return self.pipSettings["Modelling"]["Model"]
    def getProcessingTasks(self):
        return self.pipSettings["Processing"]["Tasks"]


    def updateMp(self,model):

        self.pipSettings["Modelling"]["MP"]["Required"][ "Bodymass"] = model.mp["Bodymass"]
        self.pipSettings["Modelling"]["MP"]["Required"][ "Height"] = model.mp["Height"]
        self.pipSettings["Modelling"]["MP"]["Required"][ "LeftLegLength"] = model.mp["LeftLegLength"]
        self.pipSettings["Modelling"]["MP"]["Required"][ "LeftKneeeWidth"] = model.mp["LeftKneeWidth"]
        self.pipSettings["Modelling"]["MP"]["Required"][ "LeftAnkleWidth"] = model.mp["LeftAnkleWidth"]
        self.pipSettings["Modelling"]["MP"]["Required"][ "LeftSoleDelta"] = model.mp["LeftSoleDelta"]
        self.pipSettings["Modelling"]["MP"]["Required"][ "RightLegLength"] = model.mp["RightLegLength"]
        self.pipSettings["Modelling"]["MP"]["Required"][ "RightKneeeWidth"] = model.mp["RightKneeWidth"]
        self.pipSettings["Modelling"]["MP"]["Required"][ "RightAnkleWidth"] = model.mp["RightAnkleWidth"]
        self.pipSettings["Modelling"]["MP"]["Required"][ "RightSoleDelta"] = model.mp["RightSoleDelta"]

        # update optional mp and save a new info file
        self.pipSettings["Modelling"]["MP"]["Optional"][ "InterAsisDistance"] = model.mp_computed["InterAsisDistance"]
        self.pipSettings["Modelling"]["MP"]["Optional"][ "LeftAsisTrocanterDistance"] = model.mp_computed["LeftAsisTrocanterDistance"]
        self.pipSettings["Modelling"]["MP"]["Optional"][ "LeftTibialTorsion"] = model.mp_computed["LeftTibialTorsionOffset"]
        self.pipSettings["Modelling"]["MP"]["Optional"][ "LeftThighRotation"] = model.mp_computed["LeftThighRotationOffset"]
        self.pipSettings["Modelling"]["MP"]["Optional"][ "LeftShankRotation"] = model.mp_computed["LeftShankRotationOffset"]

        self.pipSettings["Modelling"]["MP"]["Optional"][ "RightAsisTrocanterDistance"] = model.mp_computed["RightAsisTrocanterDistance"]
        self.pipSettings["Modelling"]["MP"]["Optional"][ "RightTibialTorsion"] = model.mp_computed["RightTibialTorsionOffset"]
        self.pipSettings["Modelling"]["MP"]["Optional"][ "RightThighRotation"] = model.mp_computed["RightThighRotationOffset"]
        self.pipSettings["Modelling"]["MP"]["Optional"][ "RightShankRotation"] = model.mp_computed["RightShankRotationOffset"]

        self.pipSettings["Modelling"]["MP"]["Optional"][ "LeftKneeFuncCalibrationOffset"] = model.mp_computed["LeftKneeFuncCalibrationOffset"]
        self.pipSettings["Modelling"]["MP"]["Optional"][ "RightKneeFuncCalibrationOffset"] = model.mp_computed["RightKneeFuncCalibrationOffset"]

    def save(self,DATA_PATH,pipelineFilename):
        files.saveJson(DATA_PATH, pipelineFilename, self.pipSettings)
