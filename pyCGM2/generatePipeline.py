# -*- coding: utf-8 -*-
"""Helpers to generate Vicon Nexus pipeline files from templates."""
import os
import string

def scanViconTemplatePipeline(sourcePath,desPath,pyCGM2nexusAppsPath):
    """Copy Vicon Nexus pipeline templates, replacing the hard-coded Apps path.

    Each file of ``sourcePath`` is read, the template Apps path is replaced by
    ``pyCGM2nexusAppsPath`` and the result is written into ``desPath`` unless a
    file with the same name already exists.

    Args:
        sourcePath (str): Folder containing the template pipeline files.
        desPath (str): Destination folder.
        pyCGM2nexusAppsPath (str): Path of the pyCGM2 Nexus Apps folder to insert in the pipelines.
    """

    toreplace= "C:/Users/HLS501/Documents/Programming/API/pyCGM2/pyCGM2/Apps"

    sourcePath = sourcePath[:-1] if sourcePath[-1:]=="\\" else sourcePath
    desPath = desPath[:-1] if desPath[-1:]=="\\" else desPath
    pyCGM2nexusAppsPath = pyCGM2nexusAppsPath[:-1] if pyCGM2nexusAppsPath[-1:]=="\\" else pyCGM2nexusAppsPath

    pyCGM2nexusAppsPath_antislash = string.replace(pyCGM2nexusAppsPath, '\\', '/')

    for file in os.listdir(sourcePath):
        with open(sourcePath+"\\"+file, 'r') as f:
            file_contents = f.read()

        content = string.replace(file_contents, toreplace,pyCGM2nexusAppsPath_antislash)


        if not os.path.isfile( desPath +"\\"+ file):
            with open(desPath + "\\"+file, "w") as text_file:
                text_file.write(content)
