# -*- coding: utf-8 -*-
import argparse
import logging
import os
import pyCGM2
from pyCGM2.Utils import files

MODELS =["CGM1","CGM11", "CGM21",  "CGM22", "CGM23", "CGM24", "CGM25"]

def copyPasteEmgSettings():
    """ copy paste the global emg.settings into the session folder

    :param -m, --model [str] - REQUIRED -:  CGM model (choice is CGM1 CGM11 CGM21  CGM22 CGM23 CGM24 CGM25)

    Examples:

        >>>
        """

    files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"emg.settings",
                    os.getcwd()+"\\"+"emg.settings")

def copyPasteCgmSettings():

    """ copy paste the global CGM#i-pyCGM2.settings into the session folder

    :param -m, --model [str] - REQUIRED -:  CGM model (choice is CGM1 CGM11 CGM21  CGM22 CGM23 CGM24 CGM25)

    Examples:

        >>>
    """

    parser = argparse.ArgumentParser(description='pyCGM2-copyPasteCGMSettings')
    parser.add_argument('-m','--model', type=str, required = True, help="choice is CGM1 CGM11 CGM21  CGM22 CGM23 CGM24 CGM25")

    args = parser.parse_args()

    if args.model not in MODELS:
        raise Exception ("[pyCGM2f] Model not known. Choice is CGM1, CGM11, CGM21.... CGM25")
    else:
        if args.model == "CGM1":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM1-pyCGM2.settings", os.getcwd()+"\\"+"CGM1-pyCGM2.settings")
        if args.model == "CGM11":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM1_1-pyCGM2.settings", os.getcwd()+"\\"+"CGM1_1-pyCGM2.settings")
        if args.model == "CGM21":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_1-pyCGM2.settings", os.getcwd()+"\\"+"CGM2_1-pyCGM2.settings")
        if args.model == "CGM22":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_2-pyCGM2.settings", os.getcwd()+"\\"+"CGM2_2-pyCGM2.settings")
        if args.model == "CGM23":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_3-pyCGM2.settings", os.getcwd()+"\\"+"CGM2_3-pyCGM2.settings")
        if args.model == "CGM24":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_4-pyCGM2.settings", os.getcwd()+"\\"+"CGM2_4-pyCGM2.settings")
        if args.model == "CGM25":
            files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_5-pyCGM2.settings", os.getcwd()+"\\"+"CGM2_5-pyCGM2.settings")
