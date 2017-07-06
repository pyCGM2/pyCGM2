# -*- coding: utf-8 -*-

import pyCGM2
import logging

import json
import os
from shutil import copyfile
from collections import OrderedDict
import shutil


def manage_pycgm2SessionInfos(DATA_PATH,subject):
    
    if not os.path.isfile( DATA_PATH + subject+"-pyCGM2.info"):
        copyfile(str(pyCGM2.CONFIG.PYCGM2_SESSION_SETTINGS_FOLDER+"pyCGM2.info"), str(DATA_PATH + subject+"-pyCGM2.info"))
        logging.warning("Copy of pyCGM2.info from pyCGM2 Settings folder")
        infoSettings = json.loads(open(DATA_PATH +subject+'-pyCGM2.info').read(),object_pairs_hook=OrderedDict)
    else:
        infoSettings = json.loads(open(DATA_PATH +subject+'-pyCGM2.info').read(),object_pairs_hook=OrderedDict)

    return infoSettings

def manage_pycgm2Translators(DATA_PATH, translatorType = "CGM1.translators"):
    #  translators management 
    if os.path.isfile( DATA_PATH + translatorType):
       logging.warning("local translator found")
       sessionTranslators = json.loads(open(DATA_PATH + translatorType).read(),object_pairs_hook=OrderedDict)
       translators = sessionTranslators["Translators"]
       return translators
    else:
       return False

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