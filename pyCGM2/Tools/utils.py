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