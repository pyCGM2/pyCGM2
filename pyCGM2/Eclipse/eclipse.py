# -*- coding: utf-8 -*-
import logging
import os
import ConfigParser

import pyCGM2
from pyCGM2 import enums

from pyCGM2.ForcePlates import forceplates
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files


def getEnfFiles(path, type):
    path = path[:-1] if path[-1:]=="\\" else path

    enfFiles = files.getFiles(str(path+"\\"),type.value)

    if type == enums.EclipseType.Session:
        if len(enfFiles)>1:
            raise Exception ("Vicon Eclipse badly configured. Two session enf found")
        else:
            return  enfFiles[0]

    elif type == enums.EclipseType.Patient:
        if len(enfFiles)>1:
            raise Exception ("Vicon Eclipse badly configured. Two Patient enf found")
        else:
            return  enfFiles[0]
    elif type == enums.EclipseType.Trial:
        return enfFiles
    else:
        raise Exception ("eclipse file type not recognize. Shoud be an item of enums.eClipseType")


def findCalibration(path):
    enfs = getEnfFiles(path,enums.EclipseType.Trial)


    detected = list()
    for enf in enfs:
        enfTrial = TrialEnfReader(path,enf)
        if enfTrial.isCalibrationTrial() and enfTrial.isActivate() :
            detected.append(enf)

    if len(detected)>1:
        raise Exception("You should have only one activated calibration c3d")
    else:
        return detected[0]


def findMotions(path):
    enfs = getEnfFiles(path,enums.EclipseType.Trial)

    detected = list()
    for enf in enfs:
        enfTrial = TrialEnfReader(path,enf)
        if enfTrial.isMotionTrial() and enfTrial.isActivate():
            detected.append(enf)

    if detected ==[]:
        return None
    else:
        return detected


def cleanEnf(path,enf):
    src = open(path+enf,"r")
    filteredContent = ""
    for line in src:
        if line[0] != "=":
           filteredContent += line
    src.close()

    dst = open(path+enf, 'w')
    dst.write(filteredContent)
    dst.close()
    return filteredContent

class EnfReader(object):

    def __init__(self, path,enfFile):

        config = ConfigParser.ConfigParser()
        config.optionxform=str # keep letter case

        try:
            config.read(path+enfFile)
        except ConfigParser.ParsingError:
            print "enf cleaned"
            cleanEnf(path,enfFile)
            config.read(path+enfFile)


        self.m_path =  path
        self.m_file =  enfFile
        self.m_config = config

    def getSection(self,section):
        dict1 = {}
        options = self.m_config.options(section)
        for option in options:
            try:
                dict1[option] = self.m_config.get(section, option)
                if dict1[option] == -1:
                    DebugPrint("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict1[option] = None
        return dict1


class SessionEnfReader(EnfReader):

    def __init__(self, path,enfFile):
        super(SessionEnfReader, self).__init__(path,enfFile)
        self.m_sessionInfos = super(SessionEnfReader, self).getSection("SESSION_INFO")

        for key in self.m_sessionInfos:
            if self.m_sessionInfos[key] == "":
                self.m_sessionInfos[key] = None
            elif self.m_sessionInfos[key].lower() == "true":
                self.m_sessionInfos[key]=True
            elif self.m_sessionInfos[key].lower() == "false":
                self.m_sessionInfos[key]=False
            else:
                pass

    def get(self,label):
        if label in self.m_sessionInfos.keys():
            return self.m_sessionInfos[label]

    def getSessionInfos(self):
        return self.m_sessionInfos


class TrialEnfReader(EnfReader):

    def __init__(self, path,enfFile):
        super(TrialEnfReader, self).__init__(path,enfFile)
        self.m_trialInfos = super(TrialEnfReader, self).getSection("TRIAL_INFO")

        for key in self.m_trialInfos:
            if self.m_trialInfos[key] == "":
                self.m_trialInfos[key] = None
            elif self.m_trialInfos[key].lower() == "true":
                self.m_trialInfos[key]=True
            elif self.m_trialInfos[key].lower() == "false":
                self.m_trialInfos[key]=False
            else:
                pass


    def getTrialInfos(self):
        return self.m_trialInfos

    def get(self,label):
        if label in self.m_trialInfos.keys():
            return self.m_trialInfos[label]


    def isActivate(self):
        flag = False
        if "Activate" in self.m_trialInfos.keys():
            if self.m_trialInfos["Activate"] == "Selected" :
                flag = True
        return flag



    def isCalibrationTrial(self):
        flag = False
        if "PROCESSING" in self.m_trialInfos.keys() and "TRIAL_TYPE" in self.m_trialInfos.keys():
            if self.m_trialInfos["PROCESSING"] == "Ready" and self.m_trialInfos["TRIAL_TYPE"] == "Static":
                flag = True
        return flag

    def isKneeCalibrationTrial(self):
        flag = False
        if "PROCESSING" in self.m_trialInfos.keys() and "TRIAL_TYPE" in self.m_trialInfos.keys():
            if self.m_trialInfos["PROCESSING"] == "Ready" and self.m_trialInfos["TRIAL_TYPE"] == "KneeCalibration":
                flag = True
        return flag

    def isC3dExist(self):

        return os.path.isfile(self.m_path + self.m_file.replace(".Trial.enf",".c3d"))



    def isMotionTrial(self):
        flag = False
        if "PROCESSING" in self.m_trialInfos.keys() and "TRIAL_TYPE" in self.m_trialInfos.keys():
            if self.m_trialInfos["PROCESSING"] == "Ready" and self.m_trialInfos["TRIAL_TYPE"] == "Motion":
                flag =  True
        return flag

    def getForcePlateAssigment(self):

        c3dFilename = self.m_file.replace(".Trial.enf",".c3d")
        acq = btkTools.smartReader(str(self.m_path + c3dFilename))
        nfp = btkTools.getNumberOfForcePlate(acq)

        mfpa = ""
        for i in range(1,nfp+1):
            if self.m_trialInfos["FP"+str(i)]=="Left": mfpa = mfpa +"L"
            if self.m_trialInfos["FP"+str(i)]=="Right": mfpa = mfpa +"R"
            if self.m_trialInfos["FP"+str(i)]=="Invalid": mfpa = mfpa +"X"
            if self.m_trialInfos["FP"+str(i)]=="Auto": mfpa = mfpa +"A"

        return mfpa

    def getMarkerDiameter(self):
        if "MarkerDiameter" in self.m_trialInfos.keys():
            return float(self.m_trialInfos["MarkerDiameter"]) if self.m_trialInfos["MarkerDiameter"] is not None else 14.0

    def getFlatFootOptions(self):
        if "LeftFlatFoot" in self.m_trialInfos.keys():
             leftFlatFoot =  self.m_trialInfos["LeftFlatFoot"]

        if "RightFlatFoot" in self.m_trialInfos.keys():
            RightFlatFoot  = self.m_trialInfos["LeftFlatFoot"]

            return leftFlatFoot, RightFlatFoot
