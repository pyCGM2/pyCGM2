# coding: utf-8
import pickle
import logging
import json
import os
from shutil import copyfile
from collections import OrderedDict
import shutil
import yaml
import yamlordereddictloader
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
import pyCGM2


def openFile(path,filename):
    """

    """
    if os.path.isfile( (path + filename)):
        content = open((path+filename)).read()

        jsonFlag = is_json(content)
        yamlFlag = is_yaml(content)
        if jsonFlag:
            logging.debug("your file (%s) matches json syntax"%filename)
            struct = openJson(path ,(filename))

        if yamlFlag:
            logging.debug("your file (%s) matches yaml syntax"%filename)
            struct = openYaml(path,filename)

        if not yamlFlag and not yamlFlag:
            raise Exception ("%s is neither a Yaml or a json file"%filename)

        return struct
    else:
        return False

def readContent(stringContent):

    jsonFlag = is_json(stringContent)
    yamlFlag = is_yaml(stringContent)
    if jsonFlag:
        logging.debug("your content  matches json syntax")
        struct = json.loads(stringContent,object_pairs_hook=OrderedDict)

    if yamlFlag:
        logging.debug("your content  matches yaml syntax")
        struct =  yaml.load(stringContent,Loader=yamlordereddictloader.Loader)

    if not yamlFlag and not yamlFlag:
        raise Exception ("content is neither a Yaml or a json")

    return struct



def loadModel(path,FilenameNoExt):
    if FilenameNoExt is not None:
        filename = FilenameNoExt + "-pyCGM2.model"
    else:
        filename = "pyCGM2.model"

    # --------------------pyCGM2 MODEL ------------------------------
    if not os.path.isfile((path + filename)):
        raise Exception ("%s-pyCGM2.model file doesn't exist. Run CGM Calibration operation"%filename)
    else:
        f = open((path+filename), 'r')
        model = pickle.load(f)
        f.close()

        return model

def saveModel(model,path,FilenameNoExt):

    if FilenameNoExt is not None:
        filename = FilenameNoExt + "-pyCGM2.model"
    else:
        filename = "pyCGM2.model"

    #pyCGM2.model
    if os.path.isfile((path + filename)):
        logging.warning("previous model removed")
        os.remove((path + filename))

    modelFile = open((path+filename), "w")
    pickle.dump(model, modelFile)
    modelFile.close()


def loadAnalysis(path,FilenameNoExt):
    if FilenameNoExt is not None:
        filename = FilenameNoExt + "-pyCGM2.analysis"
    else:
        filename = "pyCGM2.analysis"

    # --------------------pyCGM2 MODEL ------------------------------
    if not os.path.isfile((path + filename)):
        raise Exception ("%s-pyCGM2.analysis file doesn't exist"%filename)
    else:
        f = open((path+filename), 'r')
        analysis = pickle.load(f)
        f.close()

        return analysis

def saveAnalysis(analysisInstance,path,FilenameNoExt):

    if FilenameNoExt is not None:
        filename = FilenameNoExt + "-pyCGM2.analysis"
    else:
        filename = "pyCGM2.analysis"

    #pyCGM2.model
    if os.path.isfile((path + filename)):
        logging.warning("previous analysis removed")
        os.remove((path + filename))

    analysisFile = open((path+filename), "w")
    pickle.dump(analysisInstance, analysisFile)
    analysisFile.close()


def openJson(path,filename):

    if path is not None: path = path
    filename = filename

    try:
        if path is None:
            jsonStuct= json.loads(open((filename)).read(),object_pairs_hook=OrderedDict)
        else:
            jsonStuct= json.loads(open((path+filename)).read(),object_pairs_hook=OrderedDict)
        return jsonStuct
    except :
        raise Exception ("[pyCGM2] : json syntax of file (%s) is incorrect. check it" %(filename))

def saveJson(path, filename, content):
    if path is not None: path = path
    filename = filename
    if path is None:
        with open((filename), 'w') as outfile:
            json.dump(content, outfile,indent=4)
    else:
        with open((path+filename), 'w') as outfile:
            json.dump(content, outfile,indent=4)


def prettyDictPrint(parsedContent):
    print (json.dumps(parsedContent, indent=4, sort_keys=True))



def openYaml(path,filename):
    if path is not None: path = path
    filename = filename
    try:
        if path is None:
            struct = yaml.load(open((filename)).read(),Loader=yamlordereddictloader.Loader)
        else:
            struct= yaml.load(open((path+filename)).read(),Loader=yamlordereddictloader.Loader)
        return struct
    except :
        raise Exception ("[pyCGM2] : yaml syntax of file (%s) is incorrect. check it" %(filename))



def getTranslators(DATA_PATH, translatorType = "CGM1.translators"):
    #  translators management
    if os.path.isfile( (DATA_PATH + translatorType)):
       logging.warning("local translator found")

       sessionTranslators = openFile(DATA_PATH,translatorType)
       translators = sessionTranslators["Translators"]
       return translators
    else:
       return False

def getIKweightSet(DATA_PATH, ikwf):
    #  translators management
    if os.path.isfile( (DATA_PATH + ikwf)):
       logging.warning("local ik weightSet file found")
       ikWeight = openFile(DATA_PATH,ikwf)
       return ikWeight
    else:
       return False

def getMpFileContent(DATA_PATH,file,subject):

    if subject is not None:
        out = subject+"-" + file
    else:
        out = file

    if not os.path.isfile( (DATA_PATH + file)):
        copyfile((pyCGM2.PYCGM2_SETTINGS_FOLDER+file), (DATA_PATH + out))
        logging.warning("Copy of %s from pyCGM2 Settings folder"%(file))

    content = openFile(DATA_PATH,out)

    return content,out

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
    'RightSoleDelta' : mpInfo["MP"]["Required"]["RightSoleDelta"],
    'LeftShoulderOffset' : mpInfo["MP"]["Required"]["LeftShoulderOffset"],
    'RightShoulderOffset' : mpInfo["MP"]["Required"]["RightShoulderOffset"],
    'LeftElbowWidth' : mpInfo["MP"]["Required"]["LeftElbowWidth"],
    'LeftWristWidth' : mpInfo["MP"]["Required"]["LeftWristWidth"],
    'LeftHandThickness' : mpInfo["MP"]["Required"]["LeftHandThickness"],
    'RightElbowWidth' : mpInfo["MP"]["Required"]["RightElbowWidth"],
    'RightWristWidth' : mpInfo["MP"]["Required"]["RightWristWidth"],
    'RightHandThickness' : mpInfo["MP"]["Required"]["RightHandThickness"]

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
    mpInfo["MP"]["Required"][ "LeftShoulderOffset"] = model.mp["LeftShoulderOffset"]
    mpInfo["MP"]["Required"][ "RightShoulderOffset"] = model.mp["RightShoulderOffset"]
    mpInfo["MP"]["Required"][ "LeftElbowWidth"] = model.mp["LeftElbowWidth"]
    mpInfo["MP"]["Required"][ "LeftWristWidth"] = model.mp["LeftWristWidth"]
    mpInfo["MP"]["Required"][ "LeftHandThickness"] = model.mp["LeftHandThickness"]
    mpInfo["MP"]["Required"][ "RightElbowWidth"] = model.mp["RightElbowWidth"]
    mpInfo["MP"]["Required"][ "RightWristWidth"] = model.mp["RightWristWidth"]
    mpInfo["MP"]["Required"][ "RightHandThickness"] = model.mp["RightHandThickness"]


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

    if not os.path.isdir((folderPath+"\\"+newFolder)):
        os.makedirs((folderPath+"\\"+newFolder))

    for file in os.listdir((folderPath+"\\"+folder2copy)):
        if file.endswith(".Session.enf"):
            src = (folderPath+"\\"+folder2copy+"\\" +file)
            dst = (folderPath+"\\"+newFolder+"\\" +newFolder+".Session.enf")

            shutil.copyfile(src, dst)
        else:
            if selectedFiles is None:
                fileToCopy = file
                src = (folderPath+"\\"+folder2copy+"\\" +fileToCopy)
                dst = (folderPath+"\\"+newFolder+"\\" + fileToCopy)

                shutil.copyfile(src, dst)


            else:
                if file in selectedFiles:
                    fileToCopy = file

                    src = (folderPath+"\\"+folder2copy+"\\" +fileToCopy)
                    dst = (folderPath+"\\"+newFolder+"\\" + fileToCopy)

                    shutil.copyfile(src, dst)

def createDir(fullPathName):
    fullPathName = fullPathName
    pathOut = fullPathName[:-1] if fullPathName[-1:]=="\\" else fullPathName
    if not os.path.isdir((pathOut)):
        os.makedirs((pathOut))
    else:
        logging.warning("directory already exists")

def getDirs(folderPath):
    folderPath = folderPath
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

def copyPaste(src, dst):
    shutil.copyfile(src,
                    dst)

def copyPasteDirectory(src, dst):
    shutil.copytree(src,
                    dst)

def deleteDirectory(dir):
    shutil.rmtree(dir)


def readXml(DATA_PATH,filename):
    infile = open((DATA_PATH+filename),"r")
    contents = infile.read()
    soup = BeautifulSoup(contents,'xml')

    return soup


def getFileCreationDate(file):
    """
    str(getFileCreationDate(file).date())
    str(getFileCreationDate(file))
    """
    stat = os.stat(file)
    try:
        stamp =  stat.st_birthtime
    except AttributeError:
        stamp= stat.st_mtime

    return datetime.fromtimestamp(stamp)


def concatenateExcelFiles(DATA_PATH_OUT,outputFilename,sheetNames,xlsFiles):

    xlsxWriter = pd.ExcelWriter((DATA_PATH_OUT+outputFilename+".xlsx"))
    df_total = pd.DataFrame()
    for sheet in sheetNames:
        for file in xlsFiles:
            excel_file = pd.ExcelFile(file)
            sheets = excel_file.sheet_names
            if sheet in sheets:
                df = excel_file.parse(sheet_name = sheet)
                df_total = df_total.append(df)

        df_total.to_excel(xlsxWriter,sheet,index=False)

    xlsxWriter.save()
