# -*- coding: utf-8 -*-
#APIDOC: /Apps/Commands

"""
Useful commands
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
        python copyPasteEmgSettings.py
    ```

    """

    files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"emg.settings",
                    os.getcwd()+"\\"+"emg.settings")
    os.startfile(os.getcwd()+"\\"+"emg.settings")


def copyPasteCgmSettings():
    """ copy paste the global *CGM#i-pyCGM2.settings* file into your session folder

    Usage:

    ```bash
        python copyPasteCgmSettings.py -m  CGM1
        python copyPasteCgmSettings.py --model  CGM1
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
        python displayAllScripts.py
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
