# -*- coding: utf-8 -*-
# from __future__ import print_function
import os
import configparser

import pyCGM2
from pyCGM2 import enums

from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files
from bs4 import BeautifulSoup



DEFAULT_SUBSESSION = {"Task":"","Shoes":"","ProthesisOrthosis":"","ExternalAid":"","PersonalAid":""}


def generateEmptyENF(path):
    c3ds = files.getFiles(path,"c3d")
    for c3d in c3ds:
        enfName = c3d[:-4]+".Trial.enf"
        if enfName not in os.listdir(path):
            open((path+enfName), 'a').close()


def getCurrentMarkedEnfs():
    currentMarkedNodesFile = os.getenv("PUBLIC")+"\\Documents\\Vicon\\Eclipse\\CurrentlyMarkedNodes.xml"

    infile = open(currentMarkedNodesFile,"r")
    soup = BeautifulSoup(infile.read(),'xml')

    out=list()
    nodes = soup.find_all("MarkedNode")
    for node in nodes:
        fullFilename = node.get("MarkedNodePath")
        out.append(fullFilename.split("\\")[-1])

    return out if out!=[] else None


def getEnfFiles(path, type):
    path = path[:-1] if path[-1:]=="\\" else path

    enfFiles = files.getFiles(path+"\\",type.value)

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


def findCalibration(path, ignoreSelect=False):
    enfs = getEnfFiles(path,enums.EclipseType.Trial)


    detected = list()
    for enf in enfs:
        enfTrial = TrialEnfReader(path,enf)
        if not ignoreSelect:
            if enfTrial.isCalibrationTrial() and enfTrial.isSelected() :
                detected.append(enf)
        else:
            if enfTrial.isCalibrationTrial() and enfTrial.get("Processing")=="Ready":
                detected.append(enf)

    if len(detected)>1:
        raise Exception("You should have only one activated calibration c3d")
    else:
        if detected ==[] : raise Exception("No static file detected")
        return detected[0]


def findMotions(path,ignoreSelect=False):
    enfs = getEnfFiles(path,enums.EclipseType.Trial)

    detected = list()
    for enf in enfs:
        enfTrial = TrialEnfReader(path,enf)
        if not ignoreSelect:
            if enfTrial.isMotionTrial() and enfTrial.isSelected():
                detected.append(enf)
        else:
            if enfTrial.isMotionTrial() and enfTrial.get("Processing")=="Ready":
                detected.append(enf)

    if detected ==[]:
        raise Exception("No motion files detected")
    else:
        return detected

def findKneeMotions(path,ignoreSelect=False):
    enfs = getEnfFiles(path,enums.EclipseType.Trial)

    detected = list()
    for enf in enfs:
        enfTrial = TrialEnfReader(path,enf)
        if not ignoreSelect:
            if enfTrial.isKneeCalibrationTrial() and enfTrial.isSelected():
                detected.append(enf)
        else:
            if enfTrial.isKneeCalibrationTrial() and enfTrial.get("Processing")=="Ready":
                detected.append(enf)

    if detected ==[]:
        return None
    else:
        return detected


def cleanEnf(path,enf):
    src = open((path+enf),"r")
    filteredContent = ""
    for line in src:
        if line[0] != "=":
           filteredContent += line
    src.close()

    dst = open((path+enf), 'w')
    dst.write(filteredContent)
    dst.close()
    return filteredContent


def classifyEnfMotions(path,ignoreSelect=False,
    criteria=["Task","Shoes","ProthesisOrthosis","ExternalAid","PersonalAid"]):
    """


    """
    enfs = getEnfFiles(path,enums.EclipseType.Trial)
    subSessions = list()

    # 1 : detect task

    for enf in enfs:
        subSession =dict(DEFAULT_SUBSESSION)
        enfTrial = TrialEnfReader(path,enf)
        if not ignoreSelect:
            if enfTrial.isMotionTrial() and enfTrial.isSelected():
                for it in criteria:
                    subSession[it] = enfTrial.get(it)
                if subSession not in subSessions:
                    subSessions.append(subSession)
        else:
            if enfTrial.isMotionTrial() and enfTrial.get("Processing")=="Ready":
                for it in criteria:
                    subSession[it] = enfTrial.get(it)
                if subSession not in subSessions:   subSessions.append(subSession)


    #2 : match c3d
    c3ds = list()
    for subSessionIt in subSessions:
        c3d_bySubSession = list()
        for enf in enfs:
            subSession =dict(DEFAULT_SUBSESSION)
            enfTrial = TrialEnfReader(path,enf)
            if not ignoreSelect:
                if enfTrial.isMotionTrial() and enfTrial.isSelected():
                    for it in criteria:
                        subSession[it] = enfTrial.get(it)
                    if subSession == subSessionIt:
                        c3d_bySubSession.append(enfTrial.getFile())
            else:
                if enfTrial.isMotionTrial() and enfTrial.get("Processing")=="Ready":
                    for it in criteria:
                        subSession[it] = enfTrial.get(it)
                    if subSession == subSessionIt:
                        c3d_bySubSession.append(enfTrial.getFile())
        c3ds.append(c3d_bySubSession)

    shortNames = list()
    for subSessionIt in subSessions:
        shortNames.append(subSessionIt["Task"][0] +"_"+ subSessionIt["Shoes"][0]+ subSessionIt["ProthesisOrthosis"][0]+ subSessionIt["ExternalAid"][0]+ subSessionIt["PersonalAid"][0])


    # zipped and retrun list of ClassifiedEnf object
    zipped = zip(subSessions,c3ds,shortNames )
    out = list()
    for it in zipped:
        out.append(ClassifiedEnf(it[0],it[1],it[2]))

    return out







class EnfReader(object):

    def __init__(self, path,enfFile):

        config = configparser.ConfigParser()
        config.optionxform=str # keep letter case

        if not os.path.isfile((path+enfFile)):
            raise Exception ("[pyCGM2] : enf file (%s) not find"%(path+enfFile))

        try:
            config.read((path+enfFile))
        except configparser.ParsingError:
            cleanEnf(path,enfFile)
            config.read((path+enfFile))


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
                    print("skip: %s" % option)
            except:
                print("exception on %s!" % option)
                dict1[option] = None
        return dict1

    def getFile(self):
        return self.m_file

    def getPath(self):
        return self.m_path



class PatientEnfReader(EnfReader):

    def __init__(self, path,enfFile):
        super(PatientEnfReader, self).__init__(path,enfFile)
        self.m_patientInfos = super(PatientEnfReader, self).getSection("SUBJECT_INFO")

        for key in self.m_patientInfos:
            if self.m_patientInfos[key] == "":
                self.m_patientInfos[key] = None
            elif self.m_patientInfos[key].lower() == "true":
                self.m_patientInfos[key]=True
            elif self.m_patientInfos[key].lower() == "false":
                self.m_patientInfos[key]=False
            else:
                pass

    def get(self,label):
        if label in self.m_patientInfos.keys():
            return self.m_patientInfos[label]

    def getPatientInfos(self):
        return self.m_patientInfos



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
            if self.m_trialInfos[key] == "" or self.m_trialInfos[key] == "None":
                self.m_trialInfos[key] = None
            elif self.m_trialInfos[key].lower() == "true":
                self.m_trialInfos[key]=True
            elif self.m_trialInfos[key].lower() == "false":
                self.m_trialInfos[key]=False
            else:
                pass

    def set(self,label,value):
        self.m_trialInfos[label] = value
        self.m_config.set('TRIAL_INFO', label, value)


    def save(self):
        with open((self.m_path + self.m_file), 'w') as configfile:
            self.m_config.write(configfile)



    def getTrialInfos(self):
        return self.m_trialInfos

    def get(self,label):
        if label in self.m_trialInfos.keys():
            return self.m_trialInfos[label]

    def getC3d(self):

        return self.m_file.replace(".Trial.enf",".c3d")


    def isSelected(self):
        flag = False
        if "Selected" in self.m_trialInfos.keys():
            if self.m_trialInfos["Selected"] == "Selected" :
                flag = True
        return flag



    def isCalibrationTrial(self):
        flag = False
        if "TrialType" in self.m_trialInfos.keys() and self.m_trialInfos["TrialType"] == "Static":
            flag = True
        return flag

    def isKneeCalibrationTrial(self):
        flag = False
        if "TrialType" in self.m_trialInfos.keys() and self.m_trialInfos["TrialType"] == "Knee Calibration":
            flag = True
        return flag


    def isC3dExist(self):
        return os.path.isfile(self.m_path + self.m_file.replace(".Trial.enf",".c3d"))



    def isMotionTrial(self):
        flag = False
        if "TrialType" in self.m_trialInfos.keys() and self.m_trialInfos["TrialType"] == "Motion":
            flag =  True
        return flag

    def getForcePlateAssigment(self):

        c3dFilename = self.m_file.replace(".Trial.enf",".c3d")
        acq = btkTools.smartReader((self.m_path + c3dFilename))
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

    def isMarked(self):
        markedFiles = getCurrentMarkedEnfs()
        return True if self.m_file in markedFiles else false



class ClassifiedEnf(object):

    def __init__(self,criteria,enfFiles,shortName):
        self.__criteria = criteria
        self.__enfFiles = enfFiles
        self.__shortName = shortName




    def getEnfFiles(self):
        return self.__enfFiles

    def getCriteria(self):
        return self.__criteria

    def getshortName(self):
        return self.__shortName
