import json
import os

from collections import OrderedDict
import yaml
import yamlordereddictloader

import pyCGM2
LOGGER = pyCGM2.LOGGER



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