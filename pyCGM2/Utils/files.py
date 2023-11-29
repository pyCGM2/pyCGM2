import pickle
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
LOGGER = pyCGM2.LOGGER
from pyCGM2.Model.model import Model
from pyCGM2.Processing.analysis import Analysis
from typing import List, Tuple, Dict, Optional


def loadSettings(DATA_PATH:str,settingFile:str,subfolder:str=""):
    """
    Load settings from a specified file. It first checks the data path for the settings file,
    if not found, it loads from the default pyCGM2 settings folder.

    Args:
        DATA_PATH (str): The path to the data directory.
        settingFile (str): The name of the settings file.
        subfolder (str, optional): Subfolder within the pyCGM2 settings folder. Defaults to "".

    Returns:
        dict: Loaded settings.
    """
    if os.path.isfile(DATA_PATH + settingFile):
        settings = openFile(DATA_PATH, settingFile)
        LOGGER.logger.warning(
            "[pyCGM2]: settings [%s] detected in the data folder"%(settingFile))
    else:
        settings = openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER+subfolder, settingFile)

    return settings

def loadModelSettings(DATA_PATH:str,expertsettings_filename:str):
    """
    Load pyCGM2 model settings from a specified file. It checks the data path first;
    if not found, it loads from the pyCGM2 settings folder.

    Args:
        DATA_PATH (str): The path to the data directory.
        expertsettings_filename (str): The name of the model settings file.

    Returns:
        dict: Loaded model settings.
    """
    if os.path.isfile(DATA_PATH+expertsettings_filename):
        settings = openFile(DATA_PATH,expertsettings_filename)
    else:
        settings = openFile(pyCGM2.PYCGM2_SETTINGS_FOLDER,expertsettings_filename)

    return settings

def openFile(path:str,filename:str):
    """
    Open a JSON or YAML file and return its contents as a dictionary. It detects the file format
    based on its content.

    Args:
        path (str): The directory path where the file is located.
        filename (str): The filename with extension.

    Returns:
        dict: The contents of the file.

    Raises:
        Exception: If the file is neither JSON nor YAML format.
    """
    if path is None:
        path =  getDirname(filename)
        filename =  getFilename(filename)

    if os.path.isfile( (path + filename)):
        content = open((path+filename)).read()

        jsonFlag = is_json(content)
        yamlFlag = is_yaml(content)
        if jsonFlag:
            LOGGER.logger.debug("your file (%s) matches json syntax"%filename)
            struct = openJson(path ,(filename))

        if yamlFlag:
            LOGGER.logger.debug("your file (%s) matches yaml syntax"%filename)
            struct = openYaml(path,filename)

        if not yamlFlag and not yamlFlag:
            raise Exception ("%s is neither a Yaml or a json file"%filename)

        return struct
    else:
        return False


def openJson(path:str,filename:str):
    """
    Open a JSON file and return its contents as a dictionary.

    Args:
        path (str): The directory path where the file is located.
        filename (str): The JSON filename.

    Returns:
        dict: The contents of the JSON file.

    Raises:
        Exception: If there is a JSON syntax error.
    """
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

def openYaml(path:str,filename:str):
    """
    Open a YAML file and return its contents as a dictionary.

    Args:
        path (str): The directory path where the file is located.
        filename (str): The YAML filename.

    Returns:
        dict: The contents of the YAML file.

    Raises:
        Exception: If there is a YAML syntax error.
    """
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

def openPickleFile(path:str,filename:str):
    """
    Open a serialized (pickle) file and return its contents.

    Args:
        path (str): The directory path where the file is located.
        filename (str): The filename with extension.

    Returns:
        object: The deserialized object from the pickle file.
    """


    with open(path+filename, 'rb') as f:
        content = pickle.load(f)

    return content

def savePickleFile(instance:object,path:str,filename:str):
    """
    Serialize an object and save it as a pickle file.

    Args:
        instance (object): The object to be serialized and saved.
        path (str): The directory path to save the file.
        filename (str): The filename for the saved file.
    """

    if os.path.isfile((path + filename)):
        LOGGER.logger.info("previous file removed")
        os.remove((path + filename))

    with open(path+filename, "wb") as FILE:
        pickle.dump(instance, FILE)


def readContent(stringContent:str):
    """
    Read a string content in JSON or YAML format and return it as a dictionary.

    Args:
        stringContent (str): The string content in JSON or YAML format.

    Returns:
        dict: The parsed content.

    Raises:
        Exception: If the content is neither JSON nor YAML.
    """

    jsonFlag = is_json(stringContent)
    yamlFlag = is_yaml(stringContent)
    if jsonFlag:
        LOGGER.logger.debug("your content  matches json syntax")
        struct = json.loads(stringContent,object_pairs_hook=OrderedDict)

    if yamlFlag:
        LOGGER.logger.debug("your content  matches yaml syntax")
        struct =  yaml.load(stringContent,Loader=yamlordereddictloader.Loader)

    if not yamlFlag and not yamlFlag:
        raise Exception ("content is neither a Yaml or a json")

    return struct



def loadModel(path:str,FilenameNoExt:str):
    """
    Load a pyCGM2 model instance from a file.

    Args:
        path (str): The path to the directory containing the model file.
        FilenameNoExt (str): The filename of the model file without extension.

    Returns:
        Model: The loaded pyCGM2 model instance.

    Raises:
        Exception: If the model file does not exist.
    """
    if FilenameNoExt is not None:
        filename = FilenameNoExt + "-pyCGM2.model"
    else:
        filename = "pyCGM2.model"

    # --------------------pyCGM2 MODEL ------------------------------
    if not os.path.isfile((path + filename)):
        raise Exception ("%s-pyCGM2.model file doesn't exist. Run CGM Calibration operation"%filename)
    else:
        with open(path+filename, 'rb') as f:
            model = pickle.load(f)

        return model

def saveModel(model:Model,path:str,FilenameNoExt:str):
    """
    Save a pyCGM2 model instance to a file.

    Args:
        model (Model): The pyCGM2 model instance to be saved.
        path (str): The directory path to save the model file.
        FilenameNoExt (str): The base filename to use for saving the model.
    """

    if FilenameNoExt is not None:
        filename = FilenameNoExt + "-pyCGM2.model"
    else:
        filename = "pyCGM2.model"

    #pyCGM2.model
    if os.path.isfile((path + filename)):
        LOGGER.logger.info("previous model removed")
        os.remove((path + filename))

    with open(path+filename, "wb") as modelFile:
        pickle.dump(model, modelFile)
    # modelFile.close()


def loadAnalysis(path:str,FilenameNoExt:str):
    """
    Load a pyCGM2 analysis instance from a file.

    Args:
        path (str): The path to the directory containing the analysis file.
        FilenameNoExt (str): The filename of the analysis file without extension.

    Returns:
        Analysis: The loaded pyCGM2 analysis instance.

    Raises:
        Exception: If the analysis file does not exist.
    """
    if FilenameNoExt is not None:
        filename = FilenameNoExt + "-pyCGM2.analysis"
    else:
        filename = "pyCGM2.analysis"

    # --------------------pyCGM2 MODEL ------------------------------
    if not os.path.isfile((path + filename)):
        raise Exception ("%s-pyCGM2.analysis file doesn't exist"%filename)
    else:
        with open(path+filename, 'rb') as f:
            analysis = pickle.load(f)

        return analysis

def saveAnalysis(analysisInstance:Analysis,path:str,FilenameNoExt:str):
    """
    Save a pyCGM2 analysis instance to a file.

    Args:
        analysisInstance (Analysis): The pyCGM2 analysis instance to be saved.
        path (str): The directory path to save the analysis file.
        FilenameNoExt (str): The base filename to use for saving the analysis.
    """
    if FilenameNoExt is not None:
        filename = FilenameNoExt + "-pyCGM2.analysis"
    else:
        filename = "pyCGM2.analysis"

    #pyCGM2.model
    if os.path.isfile((path + filename)):
        LOGGER.logger.info("previous analysis removed")
        os.remove((path + filename))

    with open(path+filename, "wb") as analysisFile:
        pickle.dump(analysisInstance, analysisFile)
    # modelFile.close()


def saveJson(path:str, filename:str, content:Dict,ensure_ascii:bool=False):
    """
    Save a dictionary as a JSON file.

    Args:
        path (str): The directory path to save the JSON file.
        filename (str): The name of the JSON file to be saved.
        content (Dict): The dictionary content to be saved as JSON.
        ensure_ascii (bool, optional): If set to False, non-ASCII characters will be saved as they are. Defaults to False.
    """

    if path is not None: path = path
    filename = filename
    if path is None:
        with open((filename), 'w') as outfile:
            json.dump(content, outfile,indent=4,ensure_ascii=ensure_ascii)
    else:
        with open((path+filename), 'w') as outfile:
            json.dump(content, outfile,indent=4,ensure_ascii=ensure_ascii)

def saveYaml(path:str, filename:str, content:Dict):
    """
    Save a dictionary as a YAML file.

    Args:
        path (str): The directory path to save the YAML file.
        filename (str): The name of the YAML file to be saved.
        content (Dict): The dictionary content to be saved as YAML.

    Note:
        This function may not work well with OrderedDict types.
    """

    if path is not None: path = path
    filename = filename
    if path is None:
        with open((filename), 'w') as outfile:
            yaml.dump(content, outfile,indent=4)
    else:
        with open((path+filename), 'w') as outfile:
            yaml.dump(content, outfile,indent=4)


def getTranslators(DATA_PATH:str, translatorType:str = "CGM1.translators"):
    """
    Retrieve CGM marker translators from a specified file.

    Args:
        DATA_PATH (str): The path to the data directory.
        translatorType (str, optional): The name of the translator file. Defaults to "CGM1.translators".

    Returns:
        dict: The loaded translators, or False if not found.
    """

    #  translators management
    if os.path.isfile( (DATA_PATH + translatorType)):
       LOGGER.logger.info("local translator found")

       sessionTranslators = openFile(DATA_PATH,translatorType)
       translators = sessionTranslators["Translators"]
       return translators
    else:
       return False

def getIKweightSet(DATA_PATH:str, ikwf:str):
    """
    Retrieve marker weights for kinematic fitting from a specified file.

    Args:
        DATA_PATH (str): The path to the data directory.
        ikwf (str): The name of the weights file.

    Returns:
        dict: The loaded marker weights, or False if not found.
    """

    if os.path.isfile( (DATA_PATH + ikwf)):
       LOGGER.logger.info("local ik weightSet file found")
       ikWeight = openFile(DATA_PATH,ikwf)
       return ikWeight
    else:
       return False

def getMpFileContent(DATA_PATH:str,file:str,subject:str):
    """
    Retrieve anthropometric data from a specified file.

    Args:
        DATA_PATH (str): The path to the data directory.
        file (str): The name of the file containing anthropometric data.
        subject (str): The name of the subject.

    Returns:
        dict: The loaded anthropometric data.
        str: The name of the output file.
    """

    if subject is not None:
        out = subject+"-" + file
    else:
        out = file

    if not os.path.isfile( (DATA_PATH + file)):
        copyfile((pyCGM2.PYCGM2_SETTINGS_FOLDER+file), (DATA_PATH + out))
        LOGGER.logger.info("Copy of %s from pyCGM2 Settings folder"%(file))

    content = openFile(DATA_PATH,out)

    return content,out

def getMp(mpInfo:Dict,resetFlag:bool=True):
    """
    Return required and optional anthropometric parameters from a given dictionary.

    Args:
        mpInfo (Dict): The global dictionary containing anthropometric parameters.
        resetFlag (bool, optional): If True, resets optional parameters. Defaults to True.

    Returns:
        dict: Required anthropometric parameters.
        dict: Optional anthropometric parameters.
    """

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


def loadMp(path:str,filename:str):
    """
    Load a set of anthropometric parameters (mp file).

    Args:
        path (str): The path to the directory containing the mp file.
        filename (str): The filename of the mp file to load.

    Returns:
        Dict: Content of the mp file including both required and optional anthropometric parameters.
        Dict: Required anthropometric parameters.
        Dict: Optional anthropometric parameters.
    """
    content = openFile(path, filename)

    for key in content["MP"]["Optional"]:
        if  content["MP"]["Optional"][key] is None:
            content["MP"]["Optional"][key] = 0

    required_mp = content["MP"]["Required"].copy()
    optional_mp = content["MP"]["Optional"].copy()

    return content, required_mp,optional_mp



def saveMp(mpInfo:Dict,model:Model,DATA_PATH:str,mpFilename:str):
    """
    Save anthropometric parameters as a JSON file.

    Args:
        mpInfo (Dict): Global anthropometric parameters dictionary.
        model (Model): A pyCGM2 model instance.
        DATA_PATH (str): The directory path to save the mp file.
        mpFilename (str): The filename for saving the anthropometric parameters.
    """

    # update optional mp and save a new info file
    mpInfo["MP"]["Required"][ "Bodymass"] = model.mp["Bodymass"]
    mpInfo["MP"]["Required"][ "Height"] = model.mp["Height"]
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



def getFiles(path:str, extension:str, ignore:Optional[bool]=None,raiseFlag:bool=False):
    """
    Retrieve all files with a specific extension from a directory.

    Args:
        path (str): The directory path to search for files.
        extension (str): The file extension to filter by.
        ignore (bool, optional): If specified, ignores files with certain criteria. Defaults to None.
        raiseFlag (bool, optional): If True, raises an exception if the directory is not found. Defaults to False.

    Returns:
        List[str]: A list of filenames with the specified extension.
    """

    try:
        out=[]
        for file in os.listdir(path):
            if ignore is None:
                if file.endswith(extension):
                    out.append(file)
            else:
                if file.endswith(extension) and ignore not in file:
                    out.append(file)
    except FileNotFoundError as e:
        LOGGER.logger.error(str(e))
        if raiseFlag: raise


    return out


def getC3dFiles(path:str, text:str="", ignore:Optional[bool]=None ):
    """
    Retrieve all C3D files from a directory, optionally filtering by text within filenames.

    Args:
        path (str): The directory path to search for C3D files.
        text (str, optional): Text to filter filenames by. Defaults to "".
        ignore (bool, optional): If specified, ignores files with certain criteria. Defaults to None.

    Returns:
        List[str]: A list of C3D filenames.
    """

    out=[]
    for file in os.listdir(path):
       if ignore is None:
           if file.endswith(".c3d"):
               if text in file:  out.append(file)
       else:
           if file.endswith(".c3d") and ignore not in file:
               if text in file:  out.append(file)

    return out

def copySessionFolder(folderPath:str, folder2copy:str, newFolder:str, selectedFiles:Optional[List]=None):
    """
    Copy a session folder, optionally including only selected files.

    Args:
        folderPath (str): The path to the parent directory.
        folder2copy (str): The name of the session folder to copy.
        newFolder (str): The name of the new folder to create.
        selectedFiles (List, optional): A list of specific files to include. Defaults to None.
    """


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

def createDir(fullPathName:str):
    """
    Create a new directory.

    Args:
        fullPathName (str): The full path name of the new directory.

    Returns:
        str: The full path of the created directory.
    """
    fullPathName = fullPathName
    pathOut = fullPathName[:-1] if fullPathName[-1:]=="\\" else fullPathName
    if not os.path.isdir((pathOut)):
        os.makedirs((pathOut))
    else:
        LOGGER.logger.info("directory already exists")
    return pathOut+"\\"

def getDirs(folderPath:str):
    """
    Get all subdirectories within a folder.

    Args:
        folderPath (str): The folder path to search for subdirectories.

    Returns:
        List[str]: A list of subdirectory names.
    """
    folderPath = folderPath
    pathOut = folderPath[:-1] if folderPath[-1:]=="\\" else folderPath
    dirs = [ name for name in os.listdir(pathOut) if os.path.isdir(os.path.join(pathOut, name)) ]
    return ( dirs)

def try_as(loader, s, on_error):
    """
    Attempt to parse a string with a given loader and catch specific errors.

    Args:
        loader (callable): The function used to load and parse the string.
        s (str): The string to be parsed.
        on_error (Exception): The exception type to catch.

    Returns:
        bool: True if parsing is successful, False otherwise.
    """
    try:
        loader(s)
        return True
    except on_error:
        return False

def is_json(s):
    """
    Check if a string is valid JSON.

    Args:
        s (str): The string to check.

    Returns:
        bool: True if the string is valid JSON, False otherwise.
    """
    return try_as(json.loads, s, ValueError)

def is_yaml(s):
    """
    Check if a string is valid YAML.

    Args:
        s (str): The string to check.

    Returns:
        bool: True if the string is valid YAML, False otherwise.
    """
    return try_as(yaml.safe_load, s, yaml.scanner.ScannerError)

def copyPaste(src:str, dst:str):
    """
    Copy and paste a file from a source to a destination.

    Args:
        src (str): The source file path.
        dst (str): The destination file path.
    """
    try:
        shutil.copyfile(src,
                        dst)
    except shutil.SameFileError:
        LOGGER.logger.debug(" source [%s] and destination [%s] are similar" %(src,dst))

    

def copyPasteDirectory(src:str, dst:str):
    """
    Copy and paste a directory from a source to a destination.

    Args:
        src (str): The source directory path.
        dst (str): The destination directory path.
    """
    try:
        shutil.copytree(src, dst)
    except FileExistsError:
        LOGGER.logger.warning("directory already exists. delete and overwritten ")
        shutil.rmtree(dst)
        shutil.copytree(src, dst)


def deleteDirectory(dir:str):
    """
    Delete a directory.

    Args:
        dir (str): The path of the directory to be deleted.
    """
    shutil.rmtree(dir)


def readXml(DATA_PATH:str,filename:str):
    """
    Read an XML file and parse its contents.

    Args:
        DATA_PATH (str): The path where the XML file is located.
        filename (str): The name of the XML file to read.

    Returns:
        BeautifulSoup object: Parsed content of the XML file.
    """
    with open((DATA_PATH+filename),"rb",) as f:
        content = f.read()

    soup = BeautifulSoup(content,'xml')

    return soup


def getFileCreationDate(file:str):
    """
    Get the creation date of a file.

    Args:
        file (str): The full path of the file.

    Returns:
        datetime: The creation date of the file.
    """
    stat = os.stat(file)
    try:
        stamp =  stat.st_birthtime
    except AttributeError:
        stamp= stat.st_mtime

    return datetime.fromtimestamp(stamp)


def concatenateExcelFiles(DATA_PATH_OUT:str,outputFilename:str,sheetNames:List[str],xlsFiles:List[str]):
    """
    Concatenate multiple Excel files into a single Excel file.

    Args:
        DATA_PATH_OUT (str): The output directory path.
        outputFilename (str): The name of the resulting Excel file.
        sheetNames (List[str]): A list of sheet names to include in the output file.
        xlsFiles (List[str]): A list of Excel files to concatenate.
    """
    xlsxWriter = pd.ExcelWriter((DATA_PATH_OUT+outputFilename+".xlsx"))
    for sheet in sheetNames:
        df_total = pd.DataFrame()
        for file in xlsFiles:
            excel_file = pd.ExcelFile(file)
            sheets = excel_file.sheet_names
            if sheet in sheets:
                df = excel_file.parse(sheetname = sheet)
                df_total = df_total.append(df)

        df_total.to_excel(xlsxWriter,sheet,index=False)

    xlsxWriter.save()


def getFilename(fullname:str):
    """
    Extract the filename from a full path.

    Args:
        fullname (str): The full path including the filename.

    Returns:
        str: The filename extracted from the full path.
    """
    return fullname[len(os.path.dirname(fullname))+1:]

def getDirname(fullname:str):
    """
    Extract the directory path from a full path.

    Args:
        fullname (str): The full path including the filename.

    Returns:
        str: The directory path extracted from the full path.
    """
    return fullname[0:len(os.path.dirname(fullname))+1]

def renameFile( fileToRename:str,renamedFile:str ):
    """
    Rename a file.

    Args:
        fileToRename (str): The current name (and path) of the file.
        renamedFile (str): The new name (and path) for the file.
    """
    try:       
        os.rename(fileToRename,renamedFile)                    
    except FileExistsError:
        os.remove(renamedFile)
        os.rename(fileToRename,renamedFile)  