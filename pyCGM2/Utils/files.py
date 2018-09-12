# -*- coding: utf-8 -*-
import cPickle
import logging
import json
import os
from shutil import copyfile
from collections import OrderedDict
import shutil
import yaml

import pyCGM2

def loadModel(path,FilenameNoExt):
    if FilenameNoExt is not None:
        filename = FilenameNoExt + "-pyCGM2.model"
    else:
        filename = "pyCGM2.model"

    # --------------------pyCGM2 MODEL ------------------------------
    if not os.path.isfile(path + filename):
        raise Exception ("%s-pyCGM2.model file doesn't exist. Run CGM Calibration operation"%filename)
    else:
        f = open(path + filename, 'r')
        model = cPickle.load(f)
        f.close()

        return model

def saveModel(model,path,FilenameNoExt):

    if FilenameNoExt is not None:
        filename = FilenameNoExt + "-pyCGM2.model"
    else:
        filename = "pyCGM2.model"

    #pyCGM2.model
    if os.path.isfile(path + filename):
        logging.warning("previous model removed")
        os.remove(path + filename)

    modelFile = open(path + filename, "w")
    cPickle.dump(model, modelFile)
    modelFile.close()


def loadAnalysis(path,FilenameNoExt):
    if FilenameNoExt is not None:
        filename = FilenameNoExt + "-pyCGM2.analysis"
    else:
        filename = "pyCGM2.analysis"

    # --------------------pyCGM2 MODEL ------------------------------
    if not os.path.isfile(path + filename):
        raise Exception ("%s-pyCGM2.analysis file doesn't exist"%filename)
    else:
        f = open(path + filename, 'r')
        analysis = cPickle.load(f)
        f.close()

        return analysis

def saveAnalysis(analysisInstance,path,FilenameNoExt):

    if FilenameNoExt is not None:
        filename = FilenameNoExt + "-pyCGM2.analysis"
    else:
        filename = "pyCGM2.analysis"

    #pyCGM2.model
    if os.path.isfile(path + filename):
        logging.warning("previous analysis removed")
        os.remove(path + filename)

    analysisFile = open(path + filename, "w")
    cPickle.dump(analysisInstance, analysisFile)
    analysisFile.close()


def openJson(path,filename,stringContent=None):

    if stringContent is None:
        try:
            if path is None:
                jsonStuct= json.loads(open(str(filename)).read(),object_pairs_hook=OrderedDict)
            else:
                jsonStuct= json.loads(open(str(path+filename)).read(),object_pairs_hook=OrderedDict)
            return jsonStuct
        except :
            raise Exception ("[pyCGM2] : json syntax of file (%s) is incorrect. check it" %(filename))
    else:
        jsonStuct = json.loads(stringContent,object_pairs_hook=OrderedDict)
        return jsonStuct

def saveJson(path, filename, content):
    with open(str(path+filename), 'w') as outfile:
        json.dump(content, outfile,indent=4)

def openYaml(path,filename,stringContent=None):
    if stringContent is None:
        try:
            if path is None:
                struct = yaml.load(open(str(filename)).read())
            else:
                struct= yaml.load(open(str(path+filename)).read())
            return struct
        except :
            raise Exception ("[pyCGM2] : yaml syntax of file (%s) is incorrect. check it" %(filename))
    else:
        struct = yaml.load(stringContent)
        return struct

def openPipelineFile(path,filename,stringContent=None):
    if stringContent is None:
        try:
            if path is None:
                content = open(str(filename)).read()
            else:
                content = open(str(path+filename)).read()
        except "IOError":
            print "Don t find your pipeline file"
    else:
        content = stringContent

    jsonFlag = is_json(content)

    if jsonFlag:
        logging.info("your config file matches json syntax")
        struct = openJson(None,None,stringContent=content)
    else:
        yamlFlag = is_yaml(content)

        if yamlFlag:
            logging.info("your config file matches yaml syntax")
            struct = openYaml(path,filename,stringContent=content)
        else:
            raise Exception("[pYCGM2]: pipeline config file is neither a json file nor a yaml file")

    return struct


def prettyJsonDisplay(parsedContent):
    print json.dumps(parsedContent, indent=4, sort_keys=True)

def openTranslators(DATA_PATH, translatorsFilename):
    filename = openJson(DATA_PATH, translatorsFilename)
    return filename["Translators"]


def getJsonFileContent(DATA_PATH,jsonfile,subject):

    if subject is not None:
        outJson = subject+"-" + jsonfile
    else:
        outJson = jsonfile


    if not os.path.isfile( DATA_PATH + outJson):
        copyfile(str(pyCGM2.PYCGM2_SESSION_SETTINGS_FOLDER+jsonfile), str(DATA_PATH + outJson))
        logging.warning("Copy of %s from pyCGM2 Settings folder"%(jsonfile))

    content = openJson(DATA_PATH,outJson)


    return content,outJson





def getSessioninfoFile(DATA_PATH,subject):

    if subject is not None:
        infoJsonFile = subject+"-pyCGM2.info"
    else:
        infoJsonFile = "pyCGM2.info"


    if not os.path.isfile( DATA_PATH + infoJsonFile):
        copyfile(str(pyCGM2.PYCGM2_SESSION_SETTINGS_FOLDER+"pyCGM2.info"), str(DATA_PATH + infoJsonFile))
        logging.warning("Copy of pyCGM2.info from pyCGM2 Settings folder")

    infoSettings = openJson(DATA_PATH,infoJsonFile)

    return infoSettings

def getTranslators(DATA_PATH, translatorType = "CGM1.translators"):
    #  translators management
    if os.path.isfile( DATA_PATH + translatorType):
       logging.warning("local translator found")
       sessionTranslators = openJson(DATA_PATH,translatorType)
       translators = sessionTranslators["Translators"]
       return translators
    else:
       return False

def getIKweightSet(DATA_PATH, ikwf):
    #  translators management
    if os.path.isfile( DATA_PATH + ikwf):
       logging.warning("local ik weightSet file found")
       ikWeight = files.openJson(DATA_PATH,ikwf)
       return ikWeight
    else:
       return False




def getMp(mpInfo,resetFlag=True):

    required_mp={
    'Bodymass'   : mpInfo["MP"]["Required"]["Bodymass"],
    'Height'   : mpInfo["MP"]["Required"]["Height"],
    'LeftLegLength' :mpInfo["MP"]["Required"]["LeftLegLength"],
    'RightLegLength' : mpInfo["MP"]["Required"][ "RightLegLength"],
    'LeftKneeWidth' : mpInfo["MP"]["Required"][ "LeftKneeWidth"],
    'RightKneeWidth' : mpInfo["MP"]["Required"][ "RightKneeWidth"],
    'LeftAnkleWidth' : mpInfo["MP"]["Required"][ "LeftAnkleWidth"],
    'RightAnkleWidth' : mpInfo["MP"]["Required"][ "RightAnkleWidth"],
    'LeftSoleDelta' : mpInfo["MP"]["Required"][ "LeftSoleDelta"],
    'RightSoleDelta' : mpInfo["MP"]["Required"]["RightSoleDelta"]
    }

    if resetFlag:
        optional_mp={
        'InterAsisDistance'   : 0,
        'LeftAsisTrocanterDistance' : 0,
        'LeftTibialTorsion' : 0 ,
        'LeftThighRotation' : 0,
        'LeftShankRotation' : 0,
        'RightAsisTrocanterDistance' : 0,
        'RightTibialTorsion' :0 ,
        'RightThighRotation' : 0,
        'RightShankRotation' : 0
        }
    else:
        optional_mp={
        'InterAsisDistance'   : mpInfo["MP"]["Optional"][ "InterAsisDistance"],#0,
        'LeftAsisTrocanterDistance' : mpInfo["MP"]["Optional"][ "LeftAsisTrocanterDistance"],#0,
        'LeftTibialTorsion' : mpInfo["MP"]["Optional"][ "LeftTibialTorsion"],#0 ,
        'LeftThighRotation' : mpInfo["MP"]["Optional"][ "LeftThighRotation"],#0,
        'LeftShankRotation' : mpInfo["MP"]["Optional"][ "LeftShankRotation"],#0,
        'RightAsisTrocanterDistance' : mpInfo["MP"]["Optional"][ "RightAsisTrocanterDistance"],#0,
        'RightTibialTorsion' : mpInfo["MP"]["Optional"][ "RightTibialTorsion"],#0 ,
        'RightThighRotation' : mpInfo["MP"]["Optional"][ "RightThighRotation"],#0,
        'RightShankRotation' : mpInfo["MP"]["Optional"][ "RightShankRotation"],#0,
        }

    return required_mp,optional_mp



def saveMp(mpInfo,model,DATA_PATH,mpFilename):

    # update optional mp and save a new info file
    mpInfo["MP"]["Required"][ "Bodymass"] = model.mp["Bodymass"]
    mpInfo["MP"]["Required"][ "LeftLegLength"] = model.mp["LeftLegLength"]
    mpInfo["MP"]["Required"][ "RightLegLength"] = model.mp["RightLegLength"]
    mpInfo["MP"]["Required"][ "LeftKneeWidth"] = model.mp["LeftKneeWidth"]
    mpInfo["MP"]["Required"][ "RightKneeWidth"] = model.mp["RightKneeWidth"]
    mpInfo["MP"]["Required"][ "LeftAnkleWidth"] = model.mp["LeftAnkleWidth"]
    mpInfo["MP"]["Required"][ "RightAnkleWidth"] = model.mp["RightAnkleWidth"]
    mpInfo["MP"]["Required"][ "LeftSoleDelta"] = model.mp["LeftSoleDelta"]
    mpInfo["MP"]["Required"][ "RightSoleDelta"] = model.mp["RightSoleDelta"]

    mpInfo["MP"]["Optional"][ "InterAsisDistance"] = model.mp_computed["InterAsisDistance"]
    mpInfo["MP"]["Optional"][ "LeftAsisTrocanterDistance"] = model.mp_computed["LeftAsisTrocanterDistance"]
    mpInfo["MP"]["Optional"][ "LeftTibialTorsion"] = model.mp_computed["LeftTibialTorsionOffset"]
    mpInfo["MP"]["Optional"][ "LeftThighRotation"] = model.mp_computed["LeftThighRotationOffset"]
    mpInfo["MP"]["Optional"][ "LeftShankRotation"] = model.mp_computed["LeftShankRotationOffset"]

    mpInfo["MP"]["Optional"][ "RightAsisTrocanterDistance"] = model.mp_computed["RightAsisTrocanterDistance"]
    mpInfo["MP"]["Optional"][ "RightTibialTorsion"] = model.mp_computed["RightTibialTorsionOffset"]
    mpInfo["MP"]["Optional"][ "RightThighRotation"] = model.mp_computed["RightThighRotationOffset"]
    mpInfo["MP"]["Optional"][ "RightShankRotation"] = model.mp_computed["RightShankRotationOffset"]

    mpInfo["MP"]["Optional"][ "LeftKneeFuncCalibrationOffset"] = model.mp_computed["LeftKneeFuncCalibrationOffset"]
    mpInfo["MP"]["Optional"][ "RightKneeFuncCalibrationOffset"] = model.mp_computed["RightKneeFuncCalibrationOffset"]

    saveJson(DATA_PATH, mpFilename, mpInfo)



def getFiles(path, extension, ignore=None):

    out=list()
    for file in os.listdir(path):
        if ignore is None:
            if file.endswith(extension):
                out.append(file)
        else:
            if file.endswith(extension) and ignore not in file:
                out.append(file)

    return out


def getC3dFiles(path, text="", ignore=None ):

    out=list()
    for file in os.listdir(path):
       if ignore is None:
           if file.endswith(".c3d"):
               if text in file:  out.append(file)
       else:
           if file.endswith(".c3d") and ignore not in file:
               if text in file:  out.append(file)

    return out

def copySessionFolder(folderPath, folder2copy, newFolder, selectedFiles=None):

    if not os.path.isdir(str(folderPath+"\\"+newFolder)):
        os.makedirs(str(folderPath+"\\"+newFolder))


    for file in os.listdir(folderPath+"\\"+folder2copy):
        if file.endswith(".Session.enf"):

            src = folderPath+"\\"+folder2copy+"\\" +file
            dst = folderPath+"\\"+newFolder+"\\" +newFolder+".Session.enf"

            shutil.copyfile(src, dst)
        else:
            if selectedFiles is None:
                fileToCopy = file

                src = folderPath+"\\"+folder2copy+"\\" +fileToCopy
                dst = folderPath+"\\"+newFolder+"\\" + fileToCopy

                shutil.copyfile(src, dst)


            else:
                if file in selectedFiles:
                    fileToCopy = file

                    src = folderPath+"\\"+folder2copy+"\\" +fileToCopy
                    dst = folderPath+"\\"+newFolder+"\\" + fileToCopy

                    shutil.copyfile(src, dst)

def createDir(fullPathName):
    pathOut = fullPathName[:-1] if fullPathName[-1:]=="\\" else fullPathName
    if not os.path.isdir(str(pathOut)):
        os.makedirs(str(pathOut))
    else:
        logging.warning("directory already exists")

def getDirs(folderPath):
    pathOut = folderPath[:-1] if folderPath[-1:]=="\\" else folderPath
    dirs = [ name for name in os.listdir(pathOut) if os.path.isdir(os.path.join(pathOut, name)) ]
    return ( dirs)

def try_as(loader, s, on_error):
    try:
        loader(s)
        return True
    except on_error:
        return False

def is_json(s):
    return try_as(json.loads, s, ValueError)

def is_yaml(s):
    return try_as(yaml.safe_load, s, yaml.scanner.ScannerError)
