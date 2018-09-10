# -*- coding: utf-8 -*-
import logging
import os
import ConfigParser

import pyCGM2
from pyCGM2.ForcePlates import forceplates
from pyCGM2.Tools import btkTools
from pyCGM2.Utils import files


# Module compatible with the pyCGM2 eclipse scheme

# ---- low levels-----
def ConfigSectionMap(config,section):
    dict1 = {}
    options = config.options(section)
    for option in options:
        try:
            dict1[option] = config.get(section, option)
            if dict1[option] == -1:
                DebugPrint("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1


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

def InitConfigParser(path,enf):
    Config = ConfigParser.ConfigParser()
    try:
        Config.read(path+enf)
    except ConfigParser.ParsingError:
        print "pb et nettoyage"
        cleanEnf(path,enf)
        Config.read(path+enf)
    return Config

# get files---------

def getEnfTrials(path):
    path = path[:-1] if path[-1:]=="\\" else path
    enfFiles = files.getFiles(path+"\\","Trial.enf")
    return enfFiles

def getEnfSession(path):
    path = path[:-1] if path[-1:]=="\\" else path
    enfFile = files.getFiles(path+"\\","Session.enf")
    if len(enfFile)>1:
        raise Exception ("Vicon Eclipse badly configured. Two session enf found")
    return enfFile[0]

# ---- High Levels -----
def getSessionInfos(path,sessionEnf):
    Config = InitConfigParser(path,sessionEnf)
    nodeinfos =  ConfigSectionMap(Config,"SESSION_INFO")
    return nodeinfos

def getTrialInfos(path,trialEnf):
    Config = InitConfigParser(path,trialEnf)
    nodeinfos =  ConfigSectionMap(Config,"TRIAL_INFO")
    return nodeinfos


def findCalibrationFromEnfs(path, enfs):

    for enf in enfs:
        Config = InitConfigParser(path,enf)

        sections = Config.sections()
        nodeinfos =  ConfigSectionMap(Config,"TRIAL_INFO")
        if "processing" in nodeinfos.keys() and "trial_type" in nodeinfos.keys():
            if nodeinfos["processing"] == "Ready" and nodeinfos["trial_type"] == "Static":
                return enf.replace(".Trial.enf",".c3d")
            else:
                logging.error("No calibration enf found")


def findMotionFromEnfs(path, enfs):

    outList = list()
    for enf in enfs:
        Config = InitConfigParser(path,enf)

        sections = Config.sections()
        nodeinfos =  ConfigSectionMap(Config,"TRIAL_INFO")

        if "processing" in nodeinfos.keys() and "trial_type" in nodeinfos.keys():
            if nodeinfos["processing"] == "Ready" and nodeinfos["trial_type"] == "Motion":
                outList.append( enf.replace(".Trial.enf",".c3d"))

    if outList ==[]:
        return None
    else:
        return outList

def getForcePlateAssignementFrom(path,c3dFile):

    enf = c3dFile.replace(".c3d",".Trial.enf")
    Config = InitConfigParser(path,enf)

    sections = Config.sections()
    nodeinfos =  ConfigSectionMap(Config,"TRIAL_INFO")

    acq = btkTools.smartReader(path+c3dFile)

    nfp = btkTools.getNumberOfForcePlate(acq)

    mfpa = ""
    for i in range(1,nfp+1):
        if nodeinfos["fp"+str(i)]=="Left":
            mfpa = mfpa + "L"
        elif nodeinfos["fp"+str(i)]=="Right":
                mfpa = mfpa + "R"
        else:
                mfpa = mfpa + "X"


    return mfpa



def enfForcePlateAssignment(path,c3dFilename,mappedForcePlate):
    """
        Add Force plate assignement in the enf file

        :Parameters:
            - `c3dFullFilename` (str) - filename with path of the c3d
    """

    acqGait = btkTools.smartReader(str(path + c3dFilename))
    enfFile = str(path + c3dFilename[:-4]+".Trial.enf")

    if not os.path.isfile(enfFile):
        raise Exception ("[pyCGM2] - No enf file associated with the c3d")
    else:
        # --------------------Modify ENF --------------------------------------
        Config = InitConfigParser(path,enfFile)
        configEnf = Configparser.ConfigParser()
        configEnf.optionxform = str
        configEnf.read(enfFile)


        indexFP=1
        for letter in mappedForcePlate:

            if letter =="L": configEnf["TRIAL_INFO"]["FP"+str(indexFP)]="Left"
            if letter =="R": configEnf["TRIAL_INFO"]["FP"+str(indexFP)]="Right"
            if letter =="X": configEnf["TRIAL_INFO"]["FP"+str(indexFP)]="Invalid"

            indexFP+=1

        tmpFile =str(c3dFullFilename[:-4]+".Trial.enf-tmp")
        with open(tmpFile, 'w') as configfile:
            configEnf.write(configfile)

        os.remove(enfFile)
        os.rename(tmpFile,enfFile)
        logging.warning("Enf file updated with Force plate assignment")
