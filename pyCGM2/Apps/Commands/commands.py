# -*- coding: utf-8 -*-

import pyCGM2
LOGGER = pyCGM2.LOGGER
from pyCGM2.Utils import files

import argparse
import os

def main():
    parser = argparse.ArgumentParser(prog='pyCGM2-preparation')
    # create sub-parser
    sub_parsers = parser.add_subparsers(help='',dest="Type")    
    
    parser_init = sub_parsers.add_parser('Settings', help='initiate settings in your data folder')
    parser_init.add_argument('-m', '--model', type=str, required=True,  help='CGM version')

    args = parser.parse_args()
    print(args)


    if args.Type == "Settings":
        init(args.model)
    else:
        raise Exception ("[pyCGM2] - command not known. (check out the command line help with -h")

def init(model):

    files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"emg.settings",
                        os.getcwd()+"\\"+"emg.settings")
    LOGGER.logger.info("[pyCGM2] file [emg.settings] copied in your data folder")

    if model == "CGM1":
        files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER
                        + "CGM1-pyCGM2.settings", os.getcwd()+"\\"+"CGM1-pyCGM2.settings")
        
    if model == "CGM1.1":
        files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM1_1-pyCGM2.settings",
                        os.getcwd()+"\\"+"CGM1_1-pyCGM2.settings")
    elif model == "CGM2.1":
        files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_1-pyCGM2.settings",
                        os.getcwd()+"\\"+"CGM2_1-pyCGM2.settings")
        # os.startfile(os.getcwd()+"\\"+"CGM2_1-pyCGM2.settings")
    elif model == "CGM2.2":
        files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_2-pyCGM2.settings",
                        os.getcwd()+"\\"+"CGM2_2-pyCGM2.settings")
    elif model == "CGM2.3":
        files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_3-pyCGM2.settings",
                        os.getcwd()+"\\"+"CGM2_3-pyCGM2.settings")

    elif model == "CGM2.4":
        files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_4-pyCGM2.settings",
                        os.getcwd()+"\\"+"CGM2_4-pyCGM2.settings")

    elif model == "CGM2.5":
        files.copyPaste(pyCGM2.PYCGM2_SETTINGS_FOLDER+"CGM2_5-pyCGM2.settings",
                        os.getcwd()+"\\"+"CGM2_5-pyCGM2.settings")
    else:
        LOGGER.logger.error("[pyCGM2] model version not know (CGM1, CGM1.1 ... CGM2.5)")
        raise
        




if __name__ == '__main__':
    main() 