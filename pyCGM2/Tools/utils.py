# -*- coding: utf-8 -*-
import os
import pdb
import shutil

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
    
    
def copySessionFolder(folderPath, folder2copy, newFolder):

    if not os.path.isdir(str(folderPath+"\\"+newFolder)):
        os.makedirs(str(folderPath+"\\"+newFolder)) 


    for file in os.listdir(folderPath+"\\"+folder2copy):
        if file.endswith(".Session.enf"):

            src = folderPath+"\\"+folder2copy+"\\" +file
            dst = folderPath+"\\"+newFolder+"\\" +newFolder+".Session.enf"            

            shutil.copyfile(src, dst)
        else:
            src = folderPath+"\\"+folder2copy+"\\" +file
            dst = folderPath+"\\"+newFolder+"\\" + file            

            shutil.copyfile(src, dst)