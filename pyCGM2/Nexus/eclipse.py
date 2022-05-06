# -*- coding: utf-8 -*-
#APIDOC["Path"]=/Core/Eclipse
#APIDOC["Draft"]=False
#--end--

""" This module contains convenient classes and functions for dealing with the enf files associated with vicon Eclipse

check out the script : *\Tests\test_eclipse.py* for examples

"""

from bs4 import BeautifulSoup
from pyCGM2.Utils import files
from pyCGM2.Tools import btkTools
from pyCGM2 import enums
import os
import configparser
import pyCGM2
LOGGER = pyCGM2.LOGGER


def generateEmptyENF(path):
    """ generate empty enf files of a folder containing c3d files only

    Args:
        path (str): path to a folder with c3d and no enf

    """

    c3ds = files.getFiles(path, "c3d")
    for c3d in c3ds:
        enfName = c3d[:-4]+".Trial.enf"
        if enfName not in os.listdir(path):
            open((path+enfName), 'a').close()


def getCurrentMarkedEnfs():
    """
    Get marked enf files
    """
    currentMarkedNodesFile = os.getenv(
        "PUBLIC")+"\\Documents\\Vicon\\Eclipse\\CurrentlyMarkedNodes.xml"

    infile = open(currentMarkedNodesFile, "r")
    soup = BeautifulSoup(infile.read(), 'xml')

    out = list()
    nodes = soup.find_all("MarkedNode")
    for node in nodes:
        fullFilename = node.get("MarkedNodePath")
        out.append(fullFilename.split("\\")[-1])

    return out if out != [] else None


def getCurrentMarkedNodes(fileType="c3d"):
    """Get current marked node from the eclipse interface.

    the argument `fileType` is set by default to c3d to return marked c3d files

    Args:
        fileType (str,Optional[c3d]): file extension

    """
    currentMarkedNodesFile = os.getenv(
        "PUBLIC")+"\\Documents\\Vicon\\Eclipse\\CurrentlyMarkedNodes.xml"

    infile = open(currentMarkedNodesFile, "r")
    soup = BeautifulSoup(infile.read(), 'xml')

    path = list()
    outFiles = list()
    nodes = soup.find_all("MarkedNode")

    for node in nodes:
        fullFilename = node.get("MarkedNodePath")
        nodepath = fullFilename[0:fullFilename.rfind("\\")+1]

        if fileType == "c3d":
            fullFilename = fullFilename.replace(".Trial.enf", "."+fileType)
        outFiles.append(fullFilename.split("\\")[-1])
        if nodepath not in path:
            path.append(nodepath)

    if outFiles == []:
        return None
    else:
        if len(path) == 1:
            path = path[0]
        return path, outFiles


def getEnfFiles(path, type):
    """return the list of enf files found in a folder

    Args:
        path (str): Description of parameter `path`.
        type (pyCGM2.enums.EclipseType): type of enf file (Session,Patient or Trial)

    Returns:
        list: enf files


    """
    path = path[:-1] if path[-1:] == "\\" else path

    enfFiles = files.getFiles(path+"\\", type.value)

    if type == enums.EclipseType.Session:
        if len(enfFiles) > 1:
            raise Exception(
                "Vicon Eclipse badly configured. Two session enf found")
        else:
            return enfFiles[0]

    elif type == enums.EclipseType.Patient:
        if len(enfFiles) > 1:
            raise Exception(
                "Vicon Eclipse badly configured. Two Patient enf found")
        else:
            return enfFiles[0]
    elif type == enums.EclipseType.Trial:
        return enfFiles
    else:
        raise Exception(
            "eclipse file type not recognize. Shoud be an item of enums.eClipseType")


def cleanEnf(path, enf):
    src = open((path+enf), "r")
    filteredContent = ""
    for line in src:
        if line[0] != "=":
           filteredContent += line
    src.close()

    dst = open((path+enf), 'w')
    dst.write(filteredContent)
    dst.close()
    return filteredContent


class EnfReader(object):
    """ class for handling a generic enf file

    Args:
        path (str): folder path
        enfFile (str): enf filename
    """

    def __init__(self, path, enfFile):

        config = configparser.ConfigParser()
        config.optionxform = str  # keep letter case

        if not os.path.isfile((path+enfFile)):
            raise Exception("[pyCGM2] : enf file (%s) not find" %
                            (path+enfFile))

        try:
            config.read((path+enfFile))
        except configparser.ParsingError:
            cleanEnf(path, enfFile)
            config.read((path+enfFile))

        self.m_path = path
        self.m_file = enfFile
        self.m_config = config

    def getSection(self, section):
        """ return content of a section

        Args:
            section (str): section name

        """

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
        """ get the filename"""
        return self.m_file

    def getPath(self):
        """ get the folder path"""
        return self.m_path


class PatientEnfReader(EnfReader):
    """ Class for handling the Patient.enf file created by Vicon Nexus

    Args:
        path (str): folder path
        enfFile (str): enf filename
    """

    def __init__(self, path, enfFile):
        super(PatientEnfReader, self).__init__(path, enfFile)
        self.m_patientInfos = super(
            PatientEnfReader, self).getSection("SUBJECT_INFO")

        for key in self.m_patientInfos:
            if self.m_patientInfos[key] == "":
                self.m_patientInfos[key] = None
            elif self.m_patientInfos[key].lower() == "true":
                self.m_patientInfos[key] = True
            elif self.m_patientInfos[key].lower() == "false":
                self.m_patientInfos[key] = False
            else:
                pass

    def get(self, label):
        """get value of a given parameter ( ie column name)

        Args:
            label (str): name of the parameter
        """

        if label in self.m_patientInfos.keys():
            return self.m_patientInfos[label]

    def getPatientInfos(self):
        """ get the patient section"""
        return self.m_patientInfos

    def set(self, label, value):
        """set value of a given parameter ( ie column name)

        Args:
            label (str): name of the parameter
            value (str): value
        """
        self.m_patientInfos[label] = value
        self.m_config.set('SUBJECT_INFO', label, value)

    def save(self):
        """ save the enf file"""
        with open((self.m_path + self.m_file), 'w') as configfile:
            self.m_config.write(configfile)


class SessionEnfReader(EnfReader):
    """ Class for handling the Session.enf file created by Vicon Eclipse

    Args:
        path (str): folder path
        enfFile (str): enf filename
    """

    def __init__(self, path, enfFile):

        super(SessionEnfReader, self).__init__(path, enfFile)
        self.m_sessionInfos = super(
            SessionEnfReader, self).getSection("SESSION_INFO")

        for key in self.m_sessionInfos:
            if self.m_sessionInfos[key] == "":
                self.m_sessionInfos[key] = None
            elif self.m_sessionInfos[key].lower() == "true":
                self.m_sessionInfos[key] = True
            elif self.m_sessionInfos[key].lower() == "false":
                self.m_sessionInfos[key] = False
            else:
                pass

    def get(self, label):
        """get value of a given parameter ( ie column name)

        Args:
            label (str): name of the parameter
        """
        if label in self.m_sessionInfos.keys():
            return self.m_sessionInfos[label]

    def getSessionInfos(self):
        """ return the session section as a dict"""
        return self.m_sessionInfos

    def set(self, label, value):
        """set value of a given parameter ( ie column name)

        Args:
            label (str): name of the parameter
            value (str): value
        """
        self.m_sessionInfos[label] = value
        self.m_config.set('SESSION_INFO', label, value)

    def save(self):
        """ save the enf file"""
        with open((self.m_path + self.m_file), 'w') as configfile:
            self.m_config.write(configfile)


class TrialEnfReader(EnfReader):
    """Class for handing the Trial.enf file created by Vicon Eclipse

    Args:
        path (str): folder path
        enfFile (str): enf filename
    """

    def __init__(self, path, enfFile):

        super(TrialEnfReader, self).__init__(path, enfFile)
        self.m_trialInfos = super(
            TrialEnfReader, self).getSection("TRIAL_INFO")

        for key in self.m_trialInfos:
            if self.m_trialInfos[key] == "" or self.m_trialInfos[key] == "None":
                self.m_trialInfos[key] = None
            elif self.m_trialInfos[key].lower() == "true":
                self.m_trialInfos[key] = True
            elif self.m_trialInfos[key].lower() == "false":
                self.m_trialInfos[key] = False
            else:
                pass

    def set(self, label, value):
        """set value of a given parameter ( ie column name)

        Args:
            label (str): name of the parameter
            value (str): value
        """
        self.m_trialInfos[label] = value
        self.m_config.set('TRIAL_INFO', label, value)

    def save(self):
        """ save the enf file"""
        with open((self.m_path + self.m_file), 'w') as configfile:
            self.m_config.write(configfile)

    def getTrialInfos(self):
        """ return the trial section as a dict"""
        return self.m_trialInfos

    def get(self, label):
        """get value of a given parameter ( ie column name)

        Args:
            label (str): name of the parameter
        """
        if label in self.m_trialInfos.keys():
            return self.m_trialInfos[label]

    def getC3d(self):
        """ return the c3d name"""
        return self.m_file.split(".")[0]+".c3d"
        # return self.m_file.replace(".Trial.enf",".c3d")

    def isC3dExist(self):
        """check if c3d matches  the enf  """
        return os.path.isfile(self.m_path + self.m_file.replace(".Trial.enf", ".c3d"))


    def getForcePlateAssigment(self):
        """ return the first letter of force plates

        """

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

    def setForcePlates(self,mappedForcePlateCharacters):

        index = 1
        for character in  mappedForcePlateCharacters:
            if character == "L":
                self.set("FP"+str(index), "Left")
            elif character == "R":
                self.set("FP"+str(index), "Right")
            elif character == "X":
                self.set("FP"+str(index), "Invalid")
            elif character == "A":
                self.set("FP"+str(index), "Auto")
            else:
                LOGGER.logger.error("character of your mapped force plate characters not known (L,R,X,A only) ")
                raise Exception()

            index+=1
