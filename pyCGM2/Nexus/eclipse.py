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

from typing import List, Tuple, Dict, Optional,Union


def generateEmptyENF(path:str):
    """
    Generates empty .enf files in a folder containing only .c3d files.

    Args:
        path (str): Path to a folder containing .c3d files and no .enf files.
    """

    c3ds = files.getFiles(path, "c3d")
    for c3d in c3ds:
        enfName = c3d[:-4]+".Trial.enf"
        if enfName not in os.listdir(path):
            open((path+enfName), 'a').close()


def getCurrentMarkedEnfs():
    """
    Retrieves currently marked .enf files from the Vicon Eclipse environment.

    Returns:
        List[str]: A list of marked .enf filenames, or None if none are found.
    """
    currentMarkedNodesFile = os.getenv(
        "PUBLIC")+"\\Documents\\Vicon\\Eclipse\\CurrentlyMarkedNodes.xml"

    infile = open(currentMarkedNodesFile, "r")
    soup = BeautifulSoup(infile.read(), 'xml')

    out = []
    nodes = soup.find_all("MarkedNode")
    for node in nodes:
        fullFilename = node.get("MarkedNodePath")
        out.append(fullFilename.split("\\")[-1])

    return out if out != [] else None


def getCurrentMarkedNodes(fileType:str="c3d"):
    """
    Retrieves currently marked nodes from the Vicon Eclipse interface.

    Args:
        fileType (str, optional): The file extension to filter marked nodes. Defaults to "c3d".

    Returns:
        Tuple[str, List[str]]: The path and a list of marked files with the specified file type.
    """
    currentMarkedNodesFile = os.getenv(
        "PUBLIC")+"\\Documents\\Vicon\\Eclipse\\CurrentlyMarkedNodes.xml"

    infile = open(currentMarkedNodesFile, "r")
    soup = BeautifulSoup(infile.read(), 'xml')

    path = []
    outFiles = []
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


def getEnfFiles(path:str, type:enums.EclipseType):
    """
    Retrieves .enf files from a specified folder based on the enf type.

    Args:
        path (str): The folder path.
        type (enums.EclipseType): The type of .enf file (Session, Patient, or Trial).

    Returns:
        List[str] or str: List of .enf filenames or a single filename, depending on the type.
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


def cleanEnf(path:str, enf:str):
    """
    Cleans the content of an .enf file, removing any non-standard lines.

    Args:
        path (str): The path to the folder containing the .enf file.
        enf (str): The .enf filename.

    Returns:
        str: The cleaned content of the .enf file.
    """
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
    """
    A class for handling a generic .enf file.

    Attributes:
        m_path (str): The folder path.
        m_file (str): The .enf filename.
        m_config (configparser.ConfigParser): The configuration parser object.
    """

    def __init__(self, path:str, enfFile:str):
        """Initializes the EnfReader object."""

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

    def getSection(self, section:str):
        """
        Returns the content of a specified section from the .enf file.

        Args:
            section (str): The name of the section.

        Returns:
            dict: A dictionary containing key-value pairs of the section's contents.
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
        """
        Retrieves the filename of the .enf file.

        Returns:
            str: The .enf filename.
        """
        return self.m_file

    def getPath(self):
        """
        Retrieves the path of the folder containing the .enf file.

        Returns:
            str: The folder path.
        """
        return self.m_path


class PatientEnfReader(EnfReader):
    """
    A class for handling a Patient.enf file created by Vicon Nexus.

    Inherits from EnfReader.
    """

    def __init__(self, path:str, enfFile:str):
        """Initializes the PatientEnfReader object."""
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

    def get(self, label:str):
        """
        Retrieves the value of a specified parameter from the Patient.enf file.

        Args:
            label (str): The name of the parameter.

        Returns:
            Any: The value of the specified parameter.
        """

        if label in self.m_patientInfos.keys():
            return self.m_patientInfos[label]

    def getPatientInfos(self):
        """
        Retrieves the patient information section from the Patient.enf file.

        Returns:
            dict: A dictionary containing patient information.
        """
        return self.m_patientInfos

    def set(self, label:str, value:str):
        """
        Sets the value of a specified parameter in the Patient.enf file.

        Args:
            label (str): The name of the parameter.
            value (str): The value to be set.
        """
        self.m_patientInfos[label] = value
        self.m_config.set('SUBJECT_INFO', label, value)

    def save(self):
        """
        Saves the changes made to the Patient.enf file.
        """
        with open((self.m_path + self.m_file), 'w') as configfile:
            self.m_config.write(configfile)


class SessionEnfReader(EnfReader):
    """
    A class for handling a Session.enf file created by Vicon Eclipse.

    Inherits from EnfReader.
    """

    def __init__(self, path:str, enfFile:str):
        """Initializes the SessionEnfReader object."""


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

    def get(self, label:str):
        """
        Retrieves the value of a specified parameter from the Session.enf file.

        Args:
            label (str): The name of the parameter.

        Returns:
            Any: The value of the specified parameter.
        """
        if label in self.m_sessionInfos.keys():
            return self.m_sessionInfos[label]

    def getSessionInfos(self):
        """
        Retrieves the session information section from the Session.enf file.

        Returns:
            dict: A dictionary containing session information.
        """
        return self.m_sessionInfos

    def set(self, label:str, value:str):
        """
        Sets the value of a specified parameter in the Session.enf file.

        Args:
            label (str): The name of the parameter.
            value (str): The value to be set.
        """
        self.m_sessionInfos[label] = value
        self.m_config.set('SESSION_INFO', label, value)

    def save(self):
        """
        Saves the changes made to the Session.enf file.
        """
        with open((self.m_path + self.m_file), 'w') as configfile:
            self.m_config.write(configfile)


class TrialEnfReader(EnfReader):
    """
    A class for handling a Trial.enf file created by Vicon Eclipse.

    Inherits from EnfReader.
    """

    def __init__(self, path:str, enfFile:str):
        """Initializes the SessionEnfReader object."""

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

    def set(self, label:str, value:str):
        """
        Sets the value of a specified parameter in the Trial.enf file.

        Args:
            label (str): The name of the parameter.
            value (str): The value to be set.
        """
        self.m_trialInfos[label] = value
        self.m_config.set('TRIAL_INFO', label, value)

    def save(self):
        """"
        Saves the changes made to the Trial.enf file.
        """
        with open((self.m_path + self.m_file), 'w') as configfile:
            self.m_config.write(configfile)

    def getTrialInfos(self):
        """
        Retrieves the trial information section from the Trial.enf file.

        Returns:
            dict: A dictionary containing trial information.
        """
        return self.m_trialInfos

    def get(self, label:str):
        """
        Retrieves the value of a specified parameter from the Trial.enf file.

        Args:
            label (str): The name of the parameter.

        Returns:
            Any: The value of the specified parameter.
        """
        if label in self.m_trialInfos.keys():
            return self.m_trialInfos[label]

    def getC3d(self):
        """
        Retrieves the corresponding .c3d filename for the Trial.enf file.

        Returns:
            str: The .c3d filename.
        """
        return self.m_file.split(".")[0]+".c3d"
        # return self.m_file.replace(".Trial.enf",".c3d")

    def isC3dExist(self):
        """
        Checks if the corresponding .c3d file for the Trial.enf exists.

        Returns:
            bool: True if the .c3d file exists, False otherwise.
        """
        return os.path.isfile(self.m_path + self.m_file.replace(".Trial.enf", ".c3d"))


    def getForcePlateAssigment(self):
        """
        Retrieves the force plate assignment from the Trial.enf file.

        Returns:
            str: A string indicating the assignment of force plates.
        """

        c3dFilename = self.m_file.replace(".Trial.enf",".c3d")
        acq = btkTools.smartReader((self.m_path + c3dFilename))
        nfp = btkTools.getNumberOfForcePlate(acq)

        mfpa = ""
        for i in range(1,nfp+1):
            if "FP"+str(i) in self.m_trialInfos:
                if self.m_trialInfos["FP"+str(i)]=="Left": mfpa = mfpa +"L"
                if self.m_trialInfos["FP"+str(i)]=="Right": mfpa = mfpa +"R"
                if self.m_trialInfos["FP"+str(i)]=="Invalid": mfpa = mfpa +"X"
                if self.m_trialInfos["FP"+str(i)]=="Auto": mfpa = mfpa +"A"
            else: 
                mfpa = mfpa +"X"

        return mfpa

    def setForcePlates(self,mappedForcePlateCharacters:str):
        """
        Assigns force plates based on the provided mapping characters.

        Args:
            mappedForcePlateCharacters (str): Characters indicating the side assigned to each force plate.
        """

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
