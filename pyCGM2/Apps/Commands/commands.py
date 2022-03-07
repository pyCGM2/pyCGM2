# -*- coding: utf-8 -*-
#APIDOC["Path"]=Executable Apps/Miscellaneous Commands
#APIDOC["Draft"]=False
#APIDOC["Import"]=False

#--end--

"""
Miscellaneous functions, each registered as a command script in `setup.py`.
Therefore, they could be call from the console through its associated executable file.
"""

from pyCGM2.Model.CGM2 import cgm
import sys
import os
from pyCGM2.Utils import files
import argparse
import pyCGM2
LOGGER = pyCGM2.LOGGER
#MODELS = ["CGM1", "CGM11", "CGM21",  "CGM22", "CGM23", "CGM24", "CGM25"]


def copyPasteEmgSettings():
    """ copy paste the global *emg.settings* file into your session folder

    Usage:

    ```bash
        pyCGM2-copyPasteEmgSettings.exe
    ```

    """

    files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"emg.settings",
                    os.getcwd()+"\\"+"emg.settings")
    os.startfile(os.getcwd()+"\\"+"emg.settings")


def copyPasteCgmSettings():
    """ copy paste the global *CGM#i-pyCGM2.settings* file into your session folder

    Usage:

    ```bash
        pyCGM2-copyPasteCgmSettings.exe -m  CGM1
        pyCGM2-copyPasteCgmSettings.exe --model  CGM1
    ```

    Args:
        -m, --model (str) : CGM model

    """

    parser = argparse.ArgumentParser(description='pyCGM2-copyPasteCGMSettings')
    parser.add_argument('-m', '--model', type=str, required=True,
                        help="choice is CGM1 CGM11 CGM21  CGM22 CGM23 CGM24 CGM25")

    args = parser.parse_args()

    if args.model not in cgm.CGM.VERSIONS:
        raise Exception(
            "[pyCGM2f] Model not known. Choice is CGM1, CGM11, CGM21.... CGM25")
    else:
        if args.model == "CGM1":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER
                            + "CGM1-pyCGM2.settings", os.getcwd()+"\\"+"CGM1-pyCGM2.settings")
            os.startfile(os.getcwd()+"\\"+"CGM1-pyCGM2.settings")
        if args.model == "CGM1.1":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM1_1-pyCGM2.settings",
                            os.getcwd()+"\\"+"CGM1_1-pyCGM2.settings")
            os.startfile(os.getcwd()+"\\"+"CGM1_1-pyCGM2.settings")
        if args.model == "CGM2.1":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_1-pyCGM2.settings",
                            os.getcwd()+"\\"+"CGM2_1-pyCGM2.settings")
            os.startfile(os.getcwd()+"\\"+"CGM2_1-pyCGM2.settings")
        if args.model == "CGM2.2":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_2-pyCGM2.settings",
                            os.getcwd()+"\\"+"CGM2_2-pyCGM2.settings")
            os.startfile(os.getcwd()+"\\"+"CGM2_2-pyCGM2.settings")
        if args.model == "CGM2.3":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_3-pyCGM2.settings",
                            os.getcwd()+"\\"+"CGM2_3-pyCGM2.settings")
            os.startfile(os.getcwd()+"\\"+"CGM2_3-pyCGM2.settings")
        if args.model == "CGM2.4":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_4-pyCGM2.settings",
                            os.getcwd()+"\\"+"CGM2_4-pyCGM2.settings")
            os.startfile(os.getcwd()+"\\"+"CGM2_4-pyCGM2.settings")
        if args.model == "CGM2.5":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_5-pyCGM2.settings",
                            os.getcwd()+"\\"+"CGM2_5-pyCGM2.settings")
            os.startfile(os.getcwd()+"\\"+"CGM2_5-pyCGM2.settings")


def displayAllScripts():
    """ display all available executable scripts

    Usage:

    ```bash
        pyCGM2-displayAllScripts.exe
    ```

    """

    PATH_TO_PYTHON_SCRIPTS = os.path.dirname(sys.executable)+"\\Scripts\\"

    fileList = list()
    for fileIt in os.listdir(PATH_TO_PYTHON_SCRIPTS):
        if fileIt.startswith("Nexus") and fileIt.endswith("exe"):
            fileList.append(fileIt)
        elif fileIt.startswith("QTM") and fileIt.endswith("exe"):
            fileList.append(fileIt)
        elif fileIt.startswith("pyCGM2-") and fileIt.endswith("exe"):
            fileList.append(fileIt)

    for it in fileList:
        print(it)
